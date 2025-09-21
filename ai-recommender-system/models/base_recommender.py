from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    user_id: int
    user_type: str  # 'founder', 'developer', 'investor'
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class RecommendationResult(BaseModel):
    """Result model for recommendations."""
    item_id: int
    item_type: str  # 'startup', 'position', 'user'
    score: float
    rank: int
    match_reasons: List[str] = []
    metadata: Optional[Dict[str, Any]] = None


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    request_id: str
    user_id: int
    algorithm_used: str
    results: List[RecommendationResult]
    total_results: int
    processing_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


class BaseRecommender(ABC):
    """Base class for all recommendation algorithms."""
    
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.is_trained = False
        self.model = None
        self.metadata = {}
    
    @abstractmethod
    async def recommend(
        self, 
        request: RecommendationRequest
    ) -> List[RecommendationResult]:
        """Generate recommendations for a user."""
        pass
    
    @abstractmethod
    def can_recommend(self, user_id: int, user_type: str) -> bool:
        """Check if the recommender can generate recommendations for this user."""
        pass
    
    @abstractmethod
    async def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the recommendation model."""
        pass
    
    def get_algorithm_name(self) -> str:
        """Get the full algorithm name including version."""
        return f"{self.name}_v{self.version}"
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get algorithm metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "is_trained": self.is_trained,
            **self.metadata
        }
    
    async def explain_recommendation(
        self, 
        user_id: int, 
        item_id: int,
        item_type: str
    ) -> List[str]:
        """Explain why an item was recommended."""
        return ["General content-based similarity"]


class HybridRecommender:
    """Hybrid recommender that combines multiple algorithms."""
    
    def __init__(self):
        self.recommenders: Dict[str, BaseRecommender] = {}
        self.weights: Dict[str, float] = {}
        self.fallback_order: List[str] = []
    
    def add_recommender(
        self, 
        name: str, 
        recommender: BaseRecommender, 
        weight: float = 1.0,
        is_fallback: bool = False
    ):
        """Add a recommender to the hybrid system."""
        self.recommenders[name] = recommender
        self.weights[name] = weight
        
        if is_fallback:
            self.fallback_order.append(name)
    
    async def recommend(
        self, 
        request: RecommendationRequest
    ) -> Tuple[List[RecommendationResult], str]:
        """Generate hybrid recommendations."""
        all_results = []
        used_algorithms = []
        
        # Try primary algorithms first
        for name, recommender in self.recommenders.items():
            if name in self.fallback_order:
                continue
                
            if recommender.can_recommend(request.user_id, request.user_type):
                try:
                    results = await recommender.recommend(request)
                    if results:
                        # Apply weight to scores
                        weight = self.weights.get(name, 1.0)
                        for result in results:
                            result.score *= weight
                        
                        all_results.extend(results)
                        used_algorithms.append(name)
                        
                except Exception as e:
                    logger.error(f"Error in {name} recommender: {e}")
                    continue
        
        # If no results from primary algorithms, use fallback
        if not all_results:
            for fallback_name in self.fallback_order:
                if fallback_name in self.recommenders:
                    recommender = self.recommenders[fallback_name]
                    if recommender.can_recommend(request.user_id, request.user_type):
                        try:
                            results = await recommender.recommend(request)
                            if results:
                                all_results.extend(results)
                                used_algorithms.append(f"{fallback_name}(fallback)")
                                break
                        except Exception as e:
                            logger.error(f"Error in fallback {fallback_name}: {e}")
                            continue
        
        # Combine and rank results
        combined_results = self._combine_results(all_results, request.limit)
        algorithm_used = "+".join(used_algorithms) if used_algorithms else "none"
        
        return combined_results, algorithm_used
    
    def _combine_results(
        self, 
        results: List[RecommendationResult], 
        limit: int
    ) -> List[RecommendationResult]:
        """Combine results from multiple algorithms."""
        # Group by item
        item_scores = {}
        item_reasons = {}
        item_metadata = {}
        
        for result in results:
            key = f"{result.item_type}_{result.item_id}"
            
            if key not in item_scores:
                item_scores[key] = []
                item_reasons[key] = set()
                item_metadata[key] = result.metadata or {}
            
            item_scores[key].append(result.score)
            item_reasons[key].update(result.match_reasons)
        
        # Calculate combined scores (average for now, could be more sophisticated)
        combined_results = []
        for key, scores in item_scores.items():
            item_type, item_id = key.split("_", 1)
            combined_score = sum(scores) / len(scores)
            
            combined_results.append(RecommendationResult(
                item_id=int(item_id),
                item_type=item_type,
                score=combined_score,
                rank=0,  # Will be set after sorting
                match_reasons=list(item_reasons[key]),
                metadata=item_metadata[key]
            ))
        
        # Sort by score and assign ranks
        combined_results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(combined_results[:limit]):
            result.rank = i + 1
        
        return combined_results[:limit]
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithm names."""
        return list(self.recommenders.keys())
    
    def get_algorithm_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all algorithms."""
        status = {}
        for name, recommender in self.recommenders.items():
            status[name] = {
                "metadata": recommender.get_metadata(),
                "weight": self.weights.get(name, 1.0),
                "is_fallback": name in self.fallback_order
            }
        return status