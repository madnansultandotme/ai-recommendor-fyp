import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from models.base_recommender import BaseRecommender, RecommendationRequest, RecommendationResult
from database.database import get_db_context
from database.models import User, Startup, Position, ImplicitFeedback, ExplicitFeedback
from app.config import settings

logger = logging.getLogger(__name__)


class TwoTowerDataset(Dataset):
    """Dataset for Two-Tower model training."""
    
    def __init__(self, user_features, item_features, labels):
        self.user_features = torch.FloatTensor(user_features)
        self.item_features = torch.FloatTensor(item_features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'user_features': self.user_features[idx],
            'item_features': self.item_features[idx],
            'label': self.labels[idx]
        }


class TwoTowerModel(nn.Module):
    """Two-Tower Neural Network for recommendations."""
    
    def __init__(self, user_feature_dim, item_feature_dim, embedding_dim=128, hidden_dims=[256, 128]):
        super(TwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # User Tower
        user_layers = []
        in_dim = user_feature_dim
        for hidden_dim in hidden_dims:
            user_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            in_dim = hidden_dim
        
        user_layers.append(nn.Linear(in_dim, embedding_dim))
        self.user_tower = nn.Sequential(*user_layers)
        
        # Item Tower
        item_layers = []
        in_dim = item_feature_dim
        for hidden_dim in hidden_dims:
            item_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            in_dim = hidden_dim
        
        item_layers.append(nn.Linear(in_dim, embedding_dim))
        self.item_tower = nn.Sequential(*item_layers)
        
        # Final prediction layer
        self.prediction_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_features, item_features):
        # Get embeddings from both towers
        user_embedding = self.user_tower(user_features)
        item_embedding = self.item_tower(item_features)
        
        # L2 normalize embeddings
        user_embedding = nn.functional.normalize(user_embedding, p=2, dim=1)
        item_embedding = nn.functional.normalize(item_embedding, p=2, dim=1)
        
        # Concatenate embeddings for prediction
        combined = torch.cat([user_embedding, item_embedding], dim=1)
        prediction = self.prediction_layer(combined)
        
        return prediction.squeeze(), user_embedding, item_embedding
    
    def get_user_embedding(self, user_features):
        """Get user embedding from user tower."""
        with torch.no_grad():
            user_embedding = self.user_tower(user_features)
            return nn.functional.normalize(user_embedding, p=2, dim=1)
    
    def get_item_embedding(self, item_features):
        """Get item embedding from item tower."""
        with torch.no_grad():
            item_embedding = self.item_tower(item_features)
            return nn.functional.normalize(item_embedding, p=2, dim=1)


class TwoTowerRecommender(BaseRecommender):
    """Two-Tower Neural Network recommender."""
    
    def __init__(self):
        super().__init__("two_tower", "1.0")
        self.is_trained = False
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        self.feature_extractors = {}
        self.item_embeddings_cache = {}
        self.last_training_time = None
        
        self.metadata = {
            "description": "Two-Tower Neural Network using PyTorch",
            "requires_training": True,
            "supports_cold_start": False,
            "device": str(self.device)
        }
    
    def can_recommend(self, user_id: int, user_type: str) -> bool:
        """Check if model is trained and user has enough interactions."""
        if not self.is_trained:
            return False
        
        try:
            with get_db_context() as db:
                interaction_count = db.query(ImplicitFeedback).filter(
                    ImplicitFeedback.user_id == user_id
                ).count()
                return interaction_count >= 3  # Minimum interactions for neural model
        except Exception as e:
            logger.error(f"Error checking user interactions: {e}")
            return False
    
    async def recommend(
        self, 
        request: RecommendationRequest
    ) -> List[RecommendationResult]:
        """Generate Two-Tower neural network recommendations."""
        try:
            if not self.is_trained:
                logger.warning("Model not trained, cannot generate recommendations")
                return []
            
            # Get user features
            user_features = await self._extract_user_features(request.user_id)
            if user_features is None:
                return []
            
            # Get candidate items based on user type
            candidate_items = await self._get_candidate_items(request.user_type, request.filters)
            if not candidate_items:
                return []
            
            # Score all candidate items
            scored_items = []
            user_tensor = torch.FloatTensor(user_features).unsqueeze(0).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                user_embedding = self.model.get_user_embedding(user_tensor)
                
                for item_id, item_type, item_features in candidate_items:
                    item_tensor = torch.FloatTensor(item_features).unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    prediction, _, item_embedding = self.model(user_tensor, item_tensor)
                    score = float(prediction.cpu().numpy()[0])
                    
                    # Calculate similarity score for explanation
                    similarity = torch.cosine_similarity(user_embedding, item_embedding).item()
                    
                    scored_items.append({
                        'item_id': item_id,
                        'item_type': item_type,
                        'score': score,
                        'similarity': similarity
                    })
            
            # Sort by score and take top N
            scored_items.sort(key=lambda x: x['score'], reverse=True)
            top_items = scored_items[:request.limit]
            
            # Create results
            results = []
            for i, item in enumerate(top_items):
                match_reasons = await self._generate_neural_match_reasons(
                    request.user_id, item['item_id'], item['similarity']
                )
                
                results.append(RecommendationResult(
                    item_id=item['item_id'],
                    item_type=item['item_type'],
                    score=item['score'],
                    rank=i + 1,
                    match_reasons=match_reasons,
                    metadata={
                        "neural_score": item['score'],
                        "similarity_score": item['similarity'],
                        "algorithm": "two_tower_neural"
                    }
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Two-Tower recommendation: {e}")
            return []
    
    async def train(self, training_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train the Two-Tower neural network model."""
        start_time = datetime.now()
        
        try:
            logger.info("Starting Two-Tower neural network training...")
            
            # Build training dataset
            train_df, user_features_df, item_features_df = await self._build_training_data()
            
            if len(train_df) < 500:  # Need minimum training samples
                return {
                    "status": "failed",
                    "message": "Not enough training data for neural network",
                    "training_time": 0.0
                }
            
            # Prepare features
            X_user, X_item, y = self._prepare_features(train_df, user_features_df, item_features_df)
            
            # Split into train/validation
            indices = np.arange(len(y))
            train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
            
            # Create datasets
            train_dataset = TwoTowerDataset(
                X_user[train_idx], X_item[train_idx], y[train_idx]
            )
            val_dataset = TwoTowerDataset(
                X_user[val_idx], X_item[val_idx], y[val_idx]
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
            
            # Initialize model
            user_dim = X_user.shape[1]
            item_dim = X_item.shape[1]
            
            self.model = TwoTowerModel(
                user_feature_dim=user_dim,
                item_feature_dim=item_dim,
                embedding_dim=128,
                hidden_dims=[256, 128]
            ).to(self.device)
            
            # Training configuration
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.BCELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 10
            
            train_losses = []
            val_losses = []
            
            for epoch in range(settings.epochs):
                # Training phase
                self.model.train()
                train_loss = 0
                
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    user_features = batch['user_features'].to(self.device)
                    item_features = batch['item_features'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    predictions, _, _ = self.model(user_features, item_features)
                    loss = criterion(predictions, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation phase
                self.model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        user_features = batch['user_features'].to(self.device)
                        item_features = batch['item_features'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        predictions, _, _ = self.model(user_features, item_features)
                        loss = criterion(predictions, labels)
                        
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                scheduler.step(val_loss)
                
                logger.info(f"Epoch {epoch+1}/{settings.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    await self._save_model()
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Two-Tower training completed in {training_time:.2f}s")
            
            return {
                "status": "success",
                "message": "Two-Tower neural network trained successfully",
                "training_time": training_time,
                "training_samples": len(train_df),
                "best_val_loss": best_val_loss,
                "epochs_completed": epoch + 1,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1]
            }
            
        except Exception as e:
            logger.error(f"Error training Two-Tower model: {e}")
            return {
                "status": "failed",
                "message": f"Training failed: {str(e)}",
                "training_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _build_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Build training data from user interactions."""
        try:
            with get_db_context() as db:
                # Get all interactions
                interactions = []
                
                # Positive samples from feedback
                implicit_feedback = db.query(ImplicitFeedback).all()
                explicit_feedback = db.query(ExplicitFeedback).all()
                
                for feedback in implicit_feedback:
                    weight = self._get_interaction_weight(feedback.event_type, "implicit")
                    interactions.append({
                        'user_id': feedback.user_id,
                        'item_id': feedback.item_id,
                        'item_type': feedback.item_type,
                        'label': min(weight / 5.0, 1.0)  # Normalize to [0,1]
                    })
                
                for feedback in explicit_feedback:
                    weight = self._get_interaction_weight(feedback.feedback_type, "explicit")
                    label = 1.0 if weight > 0 else 0.0
                    interactions.append({
                        'user_id': feedback.user_id,
                        'item_id': feedback.item_id,
                        'item_type': feedback.item_type,
                        'label': label
                    })
                
                # Add negative samples (random user-item pairs without interactions)
                interactions_df = pd.DataFrame(interactions)
                interactions_set = set(zip(interactions_df['user_id'], interactions_df['item_id']))
                
                # Sample negative interactions
                all_users = interactions_df['user_id'].unique()
                all_items = interactions_df['item_id'].unique()
                
                negative_samples = []
                negative_count = min(len(interactions), 10000)  # Limit negative samples
                
                while len(negative_samples) < negative_count:
                    user_id = np.random.choice(all_users)
                    item_id = np.random.choice(all_items)
                    
                    if (user_id, item_id) not in interactions_set:
                        # Get item type
                        item_type = interactions_df[interactions_df['item_id'] == item_id]['item_type'].iloc[0]
                        
                        negative_samples.append({
                            'user_id': user_id,
                            'item_id': item_id,
                            'item_type': item_type,
                            'label': 0.0
                        })
                
                # Combine positive and negative samples
                all_interactions = interactions + negative_samples
                train_df = pd.DataFrame(all_interactions)
                
                # Build user and item feature matrices
                user_features_df = await self._build_user_features(all_users)
                item_features_df = await self._build_item_features(all_items)
                
                return train_df, user_features_df, item_features_df
                
        except Exception as e:
            logger.error(f"Error building training data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    async def _build_user_features(self, user_ids: np.ndarray) -> pd.DataFrame:
        """Build user feature matrix."""
        try:
            with get_db_context() as db:
                users = db.query(User).filter(User.user_id.in_(user_ids.tolist())).all()
                
                user_features = []
                for user in users:
                    features = self._extract_user_feature_vector(user)
                    features['user_id'] = user.user_id
                    user_features.append(features)
                
                return pd.DataFrame(user_features)
                
        except Exception as e:
            logger.error(f"Error building user features: {e}")
            return pd.DataFrame()
    
    async def _build_item_features(self, item_ids: np.ndarray) -> pd.DataFrame:
        """Build item feature matrix."""
        try:
            with get_db_context() as db:
                item_features = []
                
                for item_id in item_ids:
                    # Get item details from positions and startups
                    position = db.query(Position).filter(Position.position_id == item_id).first()
                    startup = db.query(Startup).filter(Startup.startup_id == item_id).first()
                    
                    if position:
                        features = self._extract_position_features(position, db)
                        features['item_id'] = item_id
                        features['item_type'] = 'position'
                        item_features.append(features)
                    elif startup:
                        features = self._extract_startup_features(startup)
                        features['item_id'] = item_id
                        features['item_type'] = 'startup'
                        item_features.append(features)
                
                return pd.DataFrame(item_features)
                
        except Exception as e:
            logger.error(f"Error building item features: {e}")
            return pd.DataFrame()
    
    def _extract_user_feature_vector(self, user: User) -> Dict[str, float]:
        """Extract numerical features from user profile."""
        profile = user.profile_data or {}
        
        features = {
            'user_type_developer': 1.0 if user.user_type == 'developer' else 0.0,
            'user_type_founder': 1.0 if user.user_type == 'founder' else 0.0,
            'user_type_investor': 1.0 if user.user_type == 'investor' else 0.0,
            'experience_years': float(profile.get('experience_years', 0)),
            'skills_count': float(len(profile.get('skills', []))),
            'preferred_industries_count': float(len(profile.get('preferred_industries', []))),
            'check_size_min': float(profile.get('check_size_min', 0)) / 1000000,  # Normalize to millions
            'check_size_max': float(profile.get('check_size_max', 0)) / 1000000,
        }
        
        # Add industry and skill one-hot encoding (simplified)
        all_industries = ["AI/ML", "FinTech", "HealthTech", "EdTech", "E-commerce"]
        for industry in all_industries:
            features[f'industry_{industry.lower().replace("/", "_")}'] = 1.0 if industry in profile.get('preferred_industries', []) else 0.0
        
        return features
    
    def _extract_position_features(self, position: Position, db) -> Dict[str, float]:
        """Extract numerical features from position."""
        requirements = position.requirements or {}
        
        # Get startup info
        startup = db.query(Startup).filter(Startup.startup_id == position.startup_id).first()
        startup_metadata = startup.startup_metadata if startup else {}
        
        features = {
            'required_skills_count': float(len(requirements.get('required_skills', []))),
            'preferred_skills_count': float(len(requirements.get('preferred_skills', []))),
            'experience_years_required': float(requirements.get('experience_years', 0)),
            'remote_ok': 1.0 if requirements.get('remote_ok') else 0.0,
            'salary_min': float(requirements.get('salary_range', {}).get('min', 0)) / 100000,  # Normalize
            'salary_max': float(requirements.get('salary_range', {}).get('max', 0)) / 100000,
            'startup_team_size': float(startup_metadata.get('team_size', 0)) / 100,  # Normalize
            'startup_funding': float(startup_metadata.get('funding_amount', 0)) / 1000000,  # Normalize to millions
        }
        
        # Add industry features
        all_industries = ["AI/ML", "FinTech", "HealthTech", "EdTech", "E-commerce"]
        startup_industries = startup_metadata.get('industry', [])
        for industry in all_industries:
            features[f'startup_industry_{industry.lower().replace("/", "_")}'] = 1.0 if industry in startup_industries else 0.0
        
        return features
    
    def _extract_startup_features(self, startup: Startup) -> Dict[str, float]:
        """Extract numerical features from startup."""
        metadata = startup.startup_metadata or {}
        
        features = {
            'team_size': float(metadata.get('team_size', 0)) / 100,  # Normalize
            'funding_amount': float(metadata.get('funding_amount', 0)) / 1000000,  # Normalize to millions
            'required_skills_count': float(len(metadata.get('required_skills', []))),
        }
        
        # Add stage features
        all_stages = ["Pre-Seed", "Seed", "Series A", "Series B", "Series C", "Growth"]
        stage = metadata.get('stage', '')
        for s in all_stages:
            features[f'stage_{s.lower().replace("-", "_")}'] = 1.0 if s == stage else 0.0
        
        # Add industry features
        all_industries = ["AI/ML", "FinTech", "HealthTech", "EdTech", "E-commerce"]
        startup_industries = metadata.get('industry', [])
        for industry in all_industries:
            features[f'industry_{industry.lower().replace("/", "_")}'] = 1.0 if industry in startup_industries else 0.0
        
        return features
    
    def _prepare_features(
        self, 
        train_df: pd.DataFrame, 
        user_features_df: pd.DataFrame, 
        item_features_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrices for training."""
        # Merge dataframes
        train_data = train_df.merge(user_features_df, on='user_id', how='left')
        train_data = train_data.merge(item_features_df, on='item_id', how='left')
        
        # Separate features and labels
        user_feature_cols = [col for col in user_features_df.columns if col != 'user_id']
        item_feature_cols = [col for col in item_features_df.columns if col not in ['item_id', 'item_type']]
        
        X_user = train_data[user_feature_cols].fillna(0).values
        X_item = train_data[item_feature_cols].fillna(0).values
        y = train_data['label'].values
        
        # Scale features
        X_user = self.user_scaler.fit_transform(X_user)
        X_item = self.item_scaler.fit_transform(X_item)
        
        return X_user, X_item, y
    
    def _get_interaction_weight(self, event_type: str, feedback_type: str) -> float:
        """Get weight for different types of interactions."""
        if feedback_type == "implicit":
            weights = {'view': 1.0, 'click': 2.0, 'save': 3.0}
            return weights.get(event_type, 1.0)
        else:  # explicit
            weights = {'like': 5.0, 'super_like': 10.0, 'pass': -2.0}
            return weights.get(event_type, 1.0)
    
    async def _extract_user_features(self, user_id: int) -> Optional[np.ndarray]:
        """Extract features for a specific user."""
        try:
            with get_db_context() as db:
                user = db.query(User).filter(User.user_id == user_id).first()
                if not user:
                    return None
                
                features = self._extract_user_feature_vector(user)
                
                # Convert to array in the same order as training
                user_features_df = pd.DataFrame([features])
                user_feature_cols = [col for col in features.keys()]
                
                X_user = user_features_df[user_feature_cols].fillna(0).values
                X_user = self.user_scaler.transform(X_user)
                
                return X_user[0]
                
        except Exception as e:
            logger.error(f"Error extracting user features: {e}")
            return None
    
    async def _get_candidate_items(self, user_type: str, filters: Dict[str, Any] = None) -> List[Tuple[int, str, np.ndarray]]:
        """Get candidate items for recommendation."""
        candidates = []
        
        try:
            with get_db_context() as db:
                if user_type == "developer":
                    positions = db.query(Position).all()
                    for position in positions:
                        features = self._extract_position_features(position, db)
                        features_df = pd.DataFrame([features])
                        item_feature_cols = [col for col in features.keys()]
                        
                        X_item = features_df[item_feature_cols].fillna(0).values
                        X_item = self.item_scaler.transform(X_item)
                        
                        candidates.append((position.position_id, "position", X_item[0]))
                
                elif user_type == "investor":
                    startups = db.query(Startup).all()
                    for startup in startups:
                        features = self._extract_startup_features(startup)
                        features_df = pd.DataFrame([features])
                        item_feature_cols = [col for col in features.keys()]
                        
                        X_item = features_df[item_feature_cols].fillna(0).values
                        X_item = self.item_scaler.transform(X_item)
                        
                        candidates.append((startup.startup_id, "startup", X_item[0]))
            
            return candidates[:500]  # Limit candidates for performance
            
        except Exception as e:
            logger.error(f"Error getting candidate items: {e}")
            return []
    
    async def _generate_neural_match_reasons(
        self, 
        user_id: int, 
        item_id: int, 
        similarity: float
    ) -> List[str]:
        """Generate match reasons for neural recommendations."""
        reasons = ["AI-powered personalized match"]
        
        if similarity > 0.8:
            reasons.append("High compatibility score from neural network")
        elif similarity > 0.6:
            reasons.append("Good compatibility based on learned preferences")
        else:
            reasons.append("Potential match based on user patterns")
        
        reasons.append(f"Neural similarity score: {similarity:.2f}")
        
        return reasons
    
    async def _save_model(self):
        """Save trained model to disk."""
        try:
            model_dir = os.path.join(settings.model_save_path, "two_tower")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model state
            model_path = os.path.join(model_dir, "model.pth")
            torch.save(self.model.state_dict(), model_path)
            
            # Save scalers and metadata
            metadata = {
                'user_scaler': self.user_scaler,
                'item_scaler': self.item_scaler,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'device': str(self.device)
            }
            
            metadata_path = os.path.join(model_dir, "metadata.pkl")
            with open(metadata_path, 'wb') as f:
                import pickle
                pickle.dump(metadata, f)
            
            logger.info(f"Two-Tower model saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving Two-Tower model: {e}")
    
    async def load_model(self) -> bool:
        """Load trained model from disk."""
        try:
            model_dir = os.path.join(settings.model_save_path, "two_tower")
            model_path = os.path.join(model_dir, "model.pth")
            metadata_path = os.path.join(model_dir, "metadata.pkl")
            
            if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                return False
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                import pickle
                metadata = pickle.load(f)
            
            self.user_scaler = metadata['user_scaler']
            self.item_scaler = metadata['item_scaler']
            self.last_training_time = datetime.fromisoformat(metadata['last_training_time']) if metadata['last_training_time'] else None
            
            # Initialize model (need to determine dimensions from saved data)
            # This is a simplified version - in practice, you'd save model architecture info
            self.model = TwoTowerModel(
                user_feature_dim=20,  # This should be saved in metadata
                item_feature_dim=15,  # This should be saved in metadata
                embedding_dim=128
            ).to(self.device)
            
            # Load model state
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            self.is_trained = True
            logger.info("Two-Tower model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Two-Tower model: {e}")
            return False