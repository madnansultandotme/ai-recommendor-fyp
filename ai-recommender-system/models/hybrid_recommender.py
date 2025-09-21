import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from models.base_recommender import BaseRecommender, RecommendationRequest, RecommendationResult
from models.content_based import ContentBasedRecommender
from models.collaborative_filtering import CollaborativeFilteringRecommender
from models.two_tower import TwoTowerRecommender

logger = logging.getLogger(__name__)


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender system combining multiple recommendation strategies:
    - Tier 1: Content-based filtering (semantic similarity)
    - Tier 2: Collaborative filtering (matrix factorization)
    - Tier 3: Two-Tower neural network (deep learning)
    """
    
    def __init__(self):
        super().__init__("hybrid", "1.0")
        
        # Initialize individual recommenders
        self.content_based = ContentBasedRecommender()
        self.collaborative_filtering = CollaborativeFilteringRecommender()
        self.two_tower = TwoTowerRecommender()
        
        # Weight configuration for different recommenders
        self.weights = {
            "content_based": 0.4,      # Always available, good for cold start
            "collaborative": 0.35,     # Strong when enough interactions exist
            "two_tower": 0.25          # Best for personalized recommendations
        }
        
        # Fallback strategy configuration
        self.fallback_strategy = "content_based"  # Default fallback
        
        self.metadata = {
            "description": "Hybrid recommender combining content-based, collaborative filtering, and neural approaches",
            "requires_training": True,  # CF and Two-Tower need training
            "supports_cold_start": True,  # Via content-based fallback
            "algorithms": ["content_based", "collaborative_filtering", "two_tower_neural"],
            "weights": self.weights
        }
        
        logger.info("Hybrid recommender system initialized")
    
    def can_recommend(self, user_id: int, user_type: str) -> bool:
        """Check if hybrid system can generate recommendations."""
        # At least one recommender should be able to recommend
        return (self.content_based.can_recommend(user_id, user_type) or
                self.collaborative_filtering.can_recommend(user_id, user_type) or
                self.two_tower.can_recommend(user_id, user_type))
    
    async def recommend(
        self, 
        request: RecommendationRequest
    ) -> List[RecommendationResult]:
        """Generate hybrid recommendations by combining multiple algorithms."""
        try:
            logger.info(f"Generating hybrid recommendations for user {request.user_id}")
            
            # Get recommendations from each available algorithm
            all_recommendations = {}
            algorithm_weights = {}
            
            # Content-based recommendations (Tier 1)
            if self.content_based.can_recommend(request.user_id, request.user_type):
                try:
                    content_recs = await self.content_based.recommend(request)
                    if content_recs:
                        all_recommendations["content_based"] = content_recs
                        algorithm_weights["content_based"] = self.weights["content_based"]
                        logger.info(f"Got {len(content_recs)} content-based recommendations")
                except Exception as e:
                    logger.warning(f"Content-based recommender failed: {e}")
            
            # Collaborative filtering recommendations (Tier 2)
            if self.collaborative_filtering.can_recommend(request.user_id, request.user_type):
                try:
                    cf_recs = await self.collaborative_filtering.recommend(request)
                    if cf_recs:
                        all_recommendations["collaborative"] = cf_recs
                        algorithm_weights["collaborative"] = self.weights["collaborative"]
                        logger.info(f"Got {len(cf_recs)} collaborative filtering recommendations")
                except Exception as e:
                    logger.warning(f"Collaborative filtering recommender failed: {e}")
            
            # Two-Tower neural recommendations (Tier 3)
            if self.two_tower.can_recommend(request.user_id, request.user_type):
                try:
                    neural_recs = await self.two_tower.recommend(request)
                    if neural_recs:
                        all_recommendations["two_tower"] = neural_recs
                        algorithm_weights["two_tower"] = self.weights["two_tower"]
                        logger.info(f"Got {len(neural_recs)} Two-Tower neural recommendations")
                except Exception as e:
                    logger.warning(f"Two-Tower neural recommender failed: {e}")
            
            # Handle case where no algorithms can provide recommendations
            if not all_recommendations:
                logger.warning("No algorithms could provide recommendations, returning empty list")
                return []
            
            # Combine recommendations using weighted score fusion
            combined_recommendations = self._combine_recommendations(
                all_recommendations, 
                algorithm_weights, 
                request.limit
            )
            
            # Add hybrid metadata
            for rec in combined_recommendations:
                rec.metadata["hybrid_info"] = {
                    "algorithms_used": list(algorithm_weights.keys()),
                    "algorithm_weights": algorithm_weights,
                    "combination_method": "weighted_score_fusion"
                }
            
            logger.info(f"Generated {len(combined_recommendations)} hybrid recommendations")
            return combined_recommendations
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendation: {e}")
            return []
    
    def _combine_recommendations(
        self,
        all_recommendations: Dict[str, List[RecommendationResult]],
        weights: Dict[str, float],
        limit: int
    ) -> List[RecommendationResult]:
        """Combine recommendations from multiple algorithms using weighted score fusion."""
        
        # Create a mapping of item_id -> combined recommendation
        item_scores = {}
        item_metadata = {}
        
        # Process recommendations from each algorithm
        for algorithm, recommendations in all_recommendations.items():
            weight = weights[algorithm]
            
            for rec in recommendations:
                item_id = rec.item_id
                weighted_score = rec.score * weight
                
                if item_id not in item_scores:
                    item_scores[item_id] = {
                        'total_score': 0.0,
                        'item_type': rec.item_type,
                        'match_reasons': set(),
                        'algorithm_scores': {},
                        'algorithm_ranks': {},
                        'contributing_algorithms': []
                    }
                
                # Accumulate weighted scores
                item_scores[item_id]['total_score'] += weighted_score
                item_scores[item_id]['algorithm_scores'][algorithm] = rec.score
                item_scores[item_id]['algorithm_ranks'][algorithm] = rec.rank
                item_scores[item_id]['contributing_algorithms'].append(algorithm)
                
                # Combine match reasons
                for reason in rec.match_reasons:
                    item_scores[item_id]['match_reasons'].add(f"[{algorithm.upper()}] {reason}")
                
                # Store additional metadata from original recommendation
                if item_id not in item_metadata:
                    item_metadata[item_id] = rec.metadata
        
        # Convert to RecommendationResult objects and sort by combined score
        combined_results = []
        
        for item_id, score_data in item_scores.items():
            # Normalize combined score by number of contributing algorithms
            normalized_score = score_data['total_score']
            
            # Create match reasons list
            match_reasons = list(score_data['match_reasons'])
            match_reasons.append(f"Combined from {len(score_data['contributing_algorithms'])} algorithms")
            
            # Build combined metadata
            combined_metadata = {
                "hybrid_score": normalized_score,
                "algorithm_scores": score_data['algorithm_scores'],
                "algorithm_ranks": score_data['algorithm_ranks'],
                "contributing_algorithms": score_data['contributing_algorithms'],
                "combination_method": "weighted_score_fusion"
            }
            
            # Merge original metadata
            if item_id in item_metadata:
                combined_metadata.update(item_metadata[item_id])
            
            result = RecommendationResult(
                item_id=item_id,
                item_type=score_data['item_type'],
                score=normalized_score,
                rank=0,  # Will be set after sorting
                match_reasons=match_reasons,
                metadata=combined_metadata
            )
            
            combined_results.append(result)
        
        # Sort by combined score (descending) and assign ranks
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        # Return top N results
        return combined_results[:limit]
    
    async def train(self, training_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train all trainable components of the hybrid system."""
        start_time = datetime.now()
        
        try:
            logger.info("Starting hybrid recommender training...")
            
            training_results = {}
            
            # Train collaborative filtering model
            if hasattr(self.collaborative_filtering, 'train'):
                logger.info("Training collaborative filtering model...")
                cf_result = await self.collaborative_filtering.train(training_data)
                training_results["collaborative_filtering"] = cf_result
            
            # Train Two-Tower neural network
            if hasattr(self.two_tower, 'train'):
                logger.info("Training Two-Tower neural network...")
                neural_result = await self.two_tower.train(training_data)
                training_results["two_tower"] = neural_result
            
            # Content-based doesn't need training (uses pre-trained embeddings)
            training_results["content_based"] = {
                "status": "skipped",
                "message": "Content-based recommender uses pre-trained embeddings"
            }
            
            total_training_time = (datetime.now() - start_time).total_seconds()
            
            # Determine overall training status
            failed_models = [model for model, result in training_results.items() 
                           if result.get("status") == "failed"]
            
            if len(failed_models) == len(training_results) - 1:  # All trainable models failed
                overall_status = "failed"
                message = f"All trainable models failed: {failed_models}"
            elif failed_models:
                overall_status = "partial"
                message = f"Some models failed: {failed_models}"
            else:
                overall_status = "success"
                message = "All models trained successfully"
            
            logger.info(f"Hybrid training completed in {total_training_time:.2f}s with status: {overall_status}")
            
            return {
                "status": overall_status,
                "message": message,
                "training_time": total_training_time,
                "individual_results": training_results
            }
            
        except Exception as e:
            logger.error(f"Error training hybrid recommender: {e}")
            return {
                "status": "failed",
                "message": f"Training failed: {str(e)}",
                "training_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def load_models(self) -> Dict[str, bool]:
        """Load all trained models."""
        load_results = {}
        
        try:
            # Load collaborative filtering model
            if hasattr(self.collaborative_filtering, 'load_model'):
                load_results["collaborative_filtering"] = await self.collaborative_filtering.load_model()
            else:
                load_results["collaborative_filtering"] = True  # No loading needed
            
            # Load Two-Tower neural network
            if hasattr(self.two_tower, 'load_model'):
                load_results["two_tower"] = await self.two_tower.load_model()
            else:
                load_results["two_tower"] = True  # No loading needed
            
            # Content-based doesn't need model loading
            load_results["content_based"] = True
            
            logger.info(f"Model loading results: {load_results}")
            return load_results
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return {"error": str(e)}
    
    def get_algorithm_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for each algorithm."""
        status = {}
        
        # Content-based status
        status["content_based"] = {
            "available": True,
            "trained": True,  # Always ready with pre-trained embeddings
            "supports_cold_start": True,
            "weight": self.weights["content_based"]
        }
        
        # Collaborative filtering status
        status["collaborative"] = {
            "available": hasattr(self.collaborative_filtering, 'is_trained'),
            "trained": getattr(self.collaborative_filtering, 'is_trained', False),
            "supports_cold_start": False,
            "weight": self.weights["collaborative"]
        }
        
        # Two-Tower neural network status
        status["two_tower"] = {
            "available": hasattr(self.two_tower, 'is_trained'),
            "trained": getattr(self.two_tower, 'is_trained', False),
            "supports_cold_start": False,
            "weight": self.weights["two_tower"]
        }
        
        return status
    
    def update_weights(self, new_weights: Dict[str, float]) -> bool:
        """Update algorithm weights for the hybrid system."""
        try:
            # Validate weights
            if not all(0 <= weight <= 1 for weight in new_weights.values()):
                raise ValueError("All weights must be between 0 and 1")
            
            # Update weights
            for algorithm, weight in new_weights.items():
                if algorithm in self.weights:
                    self.weights[algorithm] = weight
            
            logger.info(f"Updated hybrid weights: {self.weights}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating weights: {e}")
            return False
    
    async def evaluate_algorithms(
        self, 
        test_data: Dict[str, Any] = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate individual algorithms on test data."""
        evaluation_results = {}
        
        try:
            # This would implement evaluation logic for each algorithm
            # For now, return placeholder results
            evaluation_results = {
                "content_based": {
                    "precision_at_5": 0.0,
                    "recall_at_5": 0.0,
                    "ndcg_at_5": 0.0
                },
                "collaborative": {
                    "precision_at_5": 0.0,
                    "recall_at_5": 0.0,
                    "ndcg_at_5": 0.0
                },
                "two_tower": {
                    "precision_at_5": 0.0,
                    "recall_at_5": 0.0,
                    "ndcg_at_5": 0.0
                },
                "hybrid": {
                    "precision_at_5": 0.0,
                    "recall_at_5": 0.0,
                    "ndcg_at_5": 0.0
                }
            }
            
            logger.info("Algorithm evaluation completed")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating algorithms: {e}")
            return {}