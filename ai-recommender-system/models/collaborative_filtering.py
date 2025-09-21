import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from scipy.sparse import csr_matrix
import implicit
import pickle
import os
from datetime import datetime, timedelta

from models.base_recommender import BaseRecommender, RecommendationRequest, RecommendationResult
from database.database import get_db_context
from database.models import User, Startup, Position, ImplicitFeedback, ExplicitFeedback
from app.config import settings

logger = logging.getLogger(__name__)


class CollaborativeFilteringRecommender(BaseRecommender):
    """Collaborative filtering recommender using matrix factorization (ALS)."""
    
    def __init__(self):
        super().__init__("collaborative_filtering", "1.0")
        self.is_trained = False
        self.model = None
        self.user_item_matrix = None
        self.user_mapping = {}  # user_id -> matrix index
        self.item_mapping = {}  # item_id -> matrix index
        self.reverse_user_mapping = {}  # matrix index -> user_id
        self.reverse_item_mapping = {}  # matrix index -> item_id
        self.item_types = {}  # item_id -> item_type
        self.last_training_time = None
        self.min_interactions = 5  # Minimum interactions needed to recommend
        
        self.metadata = {
            "description": "Collaborative filtering using Alternating Least Squares (ALS)",
            "requires_training": True,
            "supports_cold_start": False,
            "min_interactions": self.min_interactions
        }
    
    def can_recommend(self, user_id: int, user_type: str) -> bool:
        """Check if we have enough data to recommend for this user."""
        if not self.is_trained:
            return False
        
        # Check if user exists in our training data
        if user_id not in self.user_mapping:
            return False
        
        # Check if user has minimum interactions
        try:
            with get_db_context() as db:
                interaction_count = db.query(ImplicitFeedback).filter(
                    ImplicitFeedback.user_id == user_id
                ).count()
                return interaction_count >= self.min_interactions
        except Exception as e:
            logger.error(f"Error checking user interaction count: {e}")
            return False
    
    async def recommend(
        self, 
        request: RecommendationRequest
    ) -> List[RecommendationResult]:
        """Generate collaborative filtering recommendations."""
        try:
            if not self.is_trained:
                logger.warning("Model not trained, cannot generate recommendations")
                return []
            
            if not self.can_recommend(request.user_id, request.user_type):
                return []
            
            user_idx = self.user_mapping.get(request.user_id)
            if user_idx is None:
                return []
            
            # Get recommendations from the model
            item_indices, scores = self.model.recommend(
                userid=user_idx,
                user_items=self.user_item_matrix[user_idx],
                N=request.limit * 2,  # Get more to filter
                filter_already_liked_items=True
            )
            
            results = []
            for i, (item_idx, score) in enumerate(zip(item_indices, scores)):
                if len(results) >= request.limit:
                    break
                
                item_id = self.reverse_item_mapping.get(item_idx)
                if item_id is None:
                    continue
                
                item_type = self.item_types.get(item_id, "unknown")
                
                # Apply filters based on user type
                if not self._passes_user_type_filter(request.user_type, item_type):
                    continue
                
                # Apply custom filters
                if request.filters and not self._passes_custom_filters(item_id, item_type, request.filters):
                    continue
                
                # Get match reasons
                match_reasons = await self._generate_collaborative_match_reasons(
                    request.user_id, item_id, item_type
                )
                
                results.append(RecommendationResult(
                    item_id=item_id,
                    item_type=item_type,
                    score=float(score),
                    rank=len(results) + 1,
                    match_reasons=match_reasons,
                    metadata={
                        "cf_score": float(score),
                        "algorithm": "collaborative_filtering_als"
                    }
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in collaborative filtering recommendation: {e}")
            return []
    
    async def train(self, training_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train the collaborative filtering model."""
        start_time = datetime.now()
        
        try:
            logger.info("Starting collaborative filtering model training...")
            
            # Build user-item interaction matrix
            interactions_df = await self._build_interaction_matrix()
            
            if len(interactions_df) < 100:  # Need minimum interactions
                return {
                    "status": "failed",
                    "message": "Not enough interaction data for training",
                    "training_time": 0.0
                }
            
            # Create sparse matrix
            self.user_item_matrix, users_count, items_count = self._create_sparse_matrix(interactions_df)
            
            # Initialize and train ALS model
            self.model = implicit.als.AlternatingLeastSquares(
                factors=64,  # Number of latent factors
                regularization=0.01,
                iterations=20,
                alpha=40  # Confidence scaling
            )
            
            # Train the model
            self.model.fit(self.user_item_matrix)
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            # Save model to disk
            await self._save_model()
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Collaborative filtering training completed in {training_time:.2f}s")
            
            return {
                "status": "success",
                "message": "Collaborative filtering model trained successfully",
                "training_time": training_time,
                "users_count": users_count,
                "items_count": items_count,
                "interactions_count": len(interactions_df),
                "model_factors": 64
            }
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering model: {e}")
            return {
                "status": "failed",
                "message": f"Training failed: {str(e)}",
                "training_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _build_interaction_matrix(self) -> pd.DataFrame:
        """Build interaction matrix from database feedback."""
        try:
            with get_db_context() as db:
                # Get implicit feedback (views, clicks, saves)
                implicit_query = db.query(
                    ImplicitFeedback.user_id,
                    ImplicitFeedback.item_id,
                    ImplicitFeedback.item_type,
                    ImplicitFeedback.event_type,
                    ImplicitFeedback.timestamp
                ).all()
                
                # Get explicit feedback (likes, passes)
                explicit_query = db.query(
                    ExplicitFeedback.user_id,
                    ExplicitFeedback.item_id,
                    ExplicitFeedback.item_type,
                    ExplicitFeedback.feedback_type,
                    ExplicitFeedback.timestamp
                ).all()
                
                interactions = []
                
                # Process implicit feedback
                for feedback in implicit_query:
                    weight = self._get_implicit_weight(feedback.event_type)
                    interactions.append({
                        'user_id': feedback.user_id,
                        'item_id': feedback.item_id,
                        'item_type': feedback.item_type,
                        'weight': weight,
                        'timestamp': feedback.timestamp
                    })
                
                # Process explicit feedback
                for feedback in explicit_query:
                    weight = self._get_explicit_weight(feedback.feedback_type)
                    interactions.append({
                        'user_id': feedback.user_id,
                        'item_id': feedback.item_id,
                        'item_type': feedback.item_type,
                        'weight': weight,
                        'timestamp': feedback.timestamp
                    })
                
                df = pd.DataFrame(interactions)
                
                if len(df) == 0:
                    return df
                
                # Aggregate multiple interactions between same user-item pair
                df = df.groupby(['user_id', 'item_id', 'item_type']).agg({
                    'weight': 'sum',
                    'timestamp': 'max'
                }).reset_index()
                
                # Store item types for later filtering
                for _, row in df.iterrows():
                    self.item_types[row['item_id']] = row['item_type']
                
                return df
                
        except Exception as e:
            logger.error(f"Error building interaction matrix: {e}")
            return pd.DataFrame()
    
    def _create_sparse_matrix(self, interactions_df: pd.DataFrame) -> Tuple[csr_matrix, int, int]:
        """Create sparse user-item matrix."""
        # Create mappings
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['item_id'].unique()
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        
        # Create sparse matrix
        rows = [self.user_mapping[user_id] for user_id in interactions_df['user_id']]
        cols = [self.item_mapping[item_id] for item_id in interactions_df['item_id']]
        data = interactions_df['weight'].values
        
        matrix = csr_matrix(
            (data, (rows, cols)), 
            shape=(len(unique_users), len(unique_items))
        )
        
        return matrix, len(unique_users), len(unique_items)
    
    def _get_implicit_weight(self, event_type: str) -> float:
        """Get weight for implicit feedback events."""
        weights = {
            'view': 1.0,
            'click': 2.0,
            'save': 3.0
        }
        return weights.get(event_type, 1.0)
    
    def _get_explicit_weight(self, feedback_type: str) -> float:
        """Get weight for explicit feedback."""
        weights = {
            'like': 5.0,
            'super_like': 10.0,
            'pass': -2.0
        }
        return weights.get(feedback_type, 1.0)
    
    def _passes_user_type_filter(self, user_type: str, item_type: str) -> bool:
        """Check if item type is appropriate for user type."""
        valid_combinations = {
            'developer': ['position'],
            'founder': ['user'],  # Founders look for developers (users)
            'investor': ['startup']
        }
        return item_type in valid_combinations.get(user_type, [])
    
    def _passes_custom_filters(self, item_id: int, item_type: str, filters: Dict[str, Any]) -> bool:
        """Apply custom filters to recommendations."""
        try:
            with get_db_context() as db:
                if item_type == 'position':
                    position = db.query(Position).filter(Position.position_id == item_id).first()
                    if not position:
                        return False
                    
                    # Apply position-specific filters
                    requirements = position.requirements or {}
                    
                    if 'remote_ok' in filters and requirements.get('remote_ok') != filters['remote_ok']:
                        return False
                    
                elif item_type == 'startup':
                    startup = db.query(Startup).filter(Startup.startup_id == item_id).first()
                    if not startup:
                        return False
                    
                    # Apply startup-specific filters
                    metadata = startup.startup_metadata or {}
                    
                    if 'stage' in filters and metadata.get('stage') != filters['stage']:
                        return False
                    
                    if 'industry' in filters:
                        startup_industries = metadata.get('industry', [])
                        if isinstance(startup_industries, str):
                            startup_industries = [startup_industries]
                        if filters['industry'] not in startup_industries:
                            return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error applying custom filters: {e}")
            return True  # Default to including item if filter check fails
    
    async def _generate_collaborative_match_reasons(
        self, 
        user_id: int, 
        item_id: int, 
        item_type: str
    ) -> List[str]:
        """Generate reasons for collaborative filtering matches."""
        reasons = ["Users with similar preferences also liked this"]
        
        try:
            # Find similar users who also interacted with this item
            with get_db_context() as db:
                similar_interactions = db.query(ImplicitFeedback).filter(
                    ImplicitFeedback.item_id == item_id,
                    ImplicitFeedback.item_type == item_type,
                    ImplicitFeedback.user_id != user_id
                ).limit(5).all()
                
                if similar_interactions:
                    reasons.append(f"Recommended based on {len(similar_interactions)} similar users")
                
        except Exception as e:
            logger.error(f"Error generating match reasons: {e}")
        
        return reasons
    
    async def _save_model(self):
        """Save trained model to disk."""
        try:
            model_dir = os.path.join(settings.model_save_path, "collaborative_filtering")
            os.makedirs(model_dir, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'reverse_user_mapping': self.reverse_user_mapping,
                'reverse_item_mapping': self.reverse_item_mapping,
                'item_types': self.item_types,
                'user_item_matrix': self.user_item_matrix,
                'last_training_time': self.last_training_time
            }
            
            model_path = os.path.join(model_dir, "cf_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    async def load_model(self) -> bool:
        """Load trained model from disk."""
        try:
            model_path = os.path.join(settings.model_save_path, "collaborative_filtering", "cf_model.pkl")
            
            if not os.path.exists(model_path):
                return False
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.user_mapping = model_data['user_mapping']
            self.item_mapping = model_data['item_mapping']
            self.reverse_user_mapping = model_data['reverse_user_mapping']
            self.reverse_item_mapping = model_data['reverse_item_mapping']
            self.item_types = model_data['item_types']
            self.user_item_matrix = model_data['user_item_matrix']
            self.last_training_time = model_data['last_training_time']
            
            self.is_trained = True
            logger.info("Collaborative filtering model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "users_count": len(self.user_mapping),
            "items_count": len(self.item_mapping),
            "matrix_density": self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]) if self.user_item_matrix is not None else 0
        }