import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

from sqlalchemy.orm import Session
from database.database import get_db_context
from database.models import User, ImplicitFeedback, ExplicitFeedback, Startup, Position
from models.base_recommender import RecommendationResult

logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """Comprehensive metrics for evaluating recommendation quality."""
    
    def __init__(self):
        self.metrics_cache = {}
        self.last_computed = {}
    
    def precision_at_k(self, recommendations: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate Precision@K metric."""
        if not recommendations or k <= 0:
            return 0.0
        
        top_k = recommendations[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_items))
        return relevant_in_top_k / min(k, len(top_k))
    
    def recall_at_k(self, recommendations: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate Recall@K metric."""
        if not relevant_items or not recommendations or k <= 0:
            return 0.0
        
        top_k = recommendations[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_items))
        return relevant_in_top_k / len(relevant_items)
    
    def f1_at_k(self, recommendations: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate F1@K metric."""
        precision = self.precision_at_k(recommendations, relevant_items, k)
        recall = self.recall_at_k(recommendations, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(self, recommendations: List[int], relevant_items: Dict[int, float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""
        if not recommendations or k <= 0:
            return 0.0
        
        top_k = recommendations[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item_id in enumerate(top_k):
            relevance = relevant_items.get(item_id, 0.0)
            if relevance > 0:
                dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (Ideal DCG)
        sorted_relevance = sorted(relevant_items.values(), reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevance[:k]):
            if relevance > 0:
                idcg += relevance / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def mean_average_precision(self, recommendations: List[int], relevant_items: List[int]) -> float:
        """Calculate Mean Average Precision (MAP)."""
        if not relevant_items or not recommendations:
            return 0.0
        
        relevant_set = set(relevant_items)
        score = 0.0
        num_hits = 0.0
        
        for i, item_id in enumerate(recommendations):
            if item_id in relevant_set:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / len(relevant_items)
    
    def diversity_at_k(self, recommendations: List[int], item_features: Dict[int, List[str]], k: int) -> float:
        """Calculate diversity of recommendations based on item features."""
        if not recommendations or k <= 0:
            return 0.0
        
        top_k = recommendations[:k]
        if len(top_k) < 2:
            return 0.0
        
        # Calculate pairwise diversity
        total_pairs = 0
        diverse_pairs = 0
        
        for i in range(len(top_k)):
            for j in range(i + 1, len(top_k)):
                item1_features = set(item_features.get(top_k[i], []))
                item2_features = set(item_features.get(top_k[j], []))
                
                # Jaccard distance (1 - Jaccard similarity)
                if item1_features or item2_features:
                    intersection = len(item1_features & item2_features)
                    union = len(item1_features | item2_features)
                    jaccard_similarity = intersection / union if union > 0 else 0
                    diversity = 1 - jaccard_similarity
                    diverse_pairs += diversity
                
                total_pairs += 1
        
        return diverse_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def coverage(self, all_recommendations: List[List[int]], catalog_items: List[int]) -> float:
        """Calculate catalog coverage - fraction of items recommended to at least one user."""
        if not all_recommendations or not catalog_items:
            return 0.0
        
        recommended_items = set()
        for recommendations in all_recommendations:
            recommended_items.update(recommendations)
        
        return len(recommended_items) / len(catalog_items)
    
    def novelty(self, recommendations: List[int], item_popularity: Dict[int, float]) -> float:
        """Calculate novelty - average popularity of recommended items (lower is more novel)."""
        if not recommendations or not item_popularity:
            return 0.0
        
        novelty_scores = []
        for item_id in recommendations:
            popularity = item_popularity.get(item_id, 0.0)
            # Novelty is inverse of popularity
            novelty = -np.log2(popularity) if popularity > 0 else 0.0
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)
    
    async def get_relevant_items_for_user(self, user_id: int, item_type: str = None) -> Dict[int, float]:
        """Get relevant items for a user based on explicit feedback."""
        try:
            with get_db_context() as db:
                query = db.query(ExplicitFeedback).filter(ExplicitFeedback.user_id == user_id)
                
                if item_type:
                    query = query.filter(ExplicitFeedback.item_type == item_type)
                
                feedback = query.all()
                
                relevant_items = {}
                for fb in feedback:
                    # Assign relevance scores based on feedback type
                    relevance_scores = {
                        'super_like': 5.0,
                        'like': 3.0,
                        'pass': 0.0,
                        'dislike': 0.0
                    }
                    
                    score = relevance_scores.get(fb.feedback_type, 1.0)
                    if fb.rating:
                        score = fb.rating
                    
                    if score > 0:
                        relevant_items[fb.item_id] = score
                
                return relevant_items
                
        except Exception as e:
            logger.error(f"Error getting relevant items for user {user_id}: {e}")
            return {}
    
    async def get_item_popularity(self, item_type: str = None) -> Dict[int, float]:
        """Calculate item popularity based on implicit feedback."""
        try:
            with get_db_context() as db:
                query = db.query(ImplicitFeedback)
                
                if item_type:
                    query = query.filter(ImplicitFeedback.item_type == item_type)
                
                feedback = query.all()
                
                # Count interactions per item
                interaction_counts = defaultdict(int)
                total_interactions = 0
                
                for fb in feedback:
                    # Weight different event types
                    weights = {'view': 1, 'click': 2, 'save': 3, 'share': 2}
                    weight = weights.get(fb.event_type, 1)
                    interaction_counts[fb.item_id] += weight
                    total_interactions += weight
                
                # Convert to probabilities
                popularity = {}
                for item_id, count in interaction_counts.items():
                    popularity[item_id] = count / total_interactions if total_interactions > 0 else 0.0
                
                return popularity
                
        except Exception as e:
            logger.error(f"Error calculating item popularity: {e}")
            return {}
    
    async def get_item_features(self, item_type: str) -> Dict[int, List[str]]:
        """Get features for items (industries, skills, etc.)."""
        try:
            features = {}
            
            with get_db_context() as db:
                if item_type == "startup":
                    startups = db.query(Startup).all()
                    for startup in startups:
                        item_features = []
                        metadata = startup.startup_metadata or {}
                        
                        # Add industry features
                        industries = metadata.get('industry', [])
                        item_features.extend([f"industry_{ind.lower()}" for ind in industries])
                        
                        # Add stage feature
                        stage = metadata.get('stage', '')
                        if stage:
                            item_features.append(f"stage_{stage.lower()}")
                        
                        # Add funding range feature
                        funding = metadata.get('funding_amount', 0)
                        if funding > 0:
                            if funding < 1000000:
                                item_features.append("funding_seed")
                            elif funding < 10000000:
                                item_features.append("funding_series_a")
                            else:
                                item_features.append("funding_growth")
                        
                        features[startup.startup_id] = item_features
                
                elif item_type == "position":
                    positions = db.query(Position).all()
                    for position in positions:
                        item_features = []
                        requirements = position.requirements or {}
                        
                        # Add skill features
                        skills = requirements.get('required_skills', [])
                        item_features.extend([f"skill_{skill.lower()}" for skill in skills])
                        
                        # Add remote feature
                        if requirements.get('remote_ok'):
                            item_features.append("remote")
                        
                        # Add salary range feature
                        salary_range = requirements.get('salary_range', {})
                        salary_min = salary_range.get('min', 0)
                        if salary_min > 0:
                            if salary_min < 80000:
                                item_features.append("salary_entry")
                            elif salary_min < 150000:
                                item_features.append("salary_mid")
                            else:
                                item_features.append("salary_senior")
                        
                        features[position.position_id] = item_features
                
                return features
                
        except Exception as e:
            logger.error(f"Error getting item features: {e}")
            return {}
    
    async def evaluate_recommendations(
        self,
        user_recommendations: Dict[int, List[RecommendationResult]],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Dict[str, float]]:
        """Comprehensive evaluation of recommendations for multiple users."""
        
        try:
            results = {
                f"precision@{k}": [] for k in k_values
            }
            results.update({
                f"recall@{k}": [] for k in k_values
            })
            results.update({
                f"f1@{k}": [] for k in k_values
            })
            results.update({
                f"ndcg@{k}": [] for k in k_values
            })
            
            results.update({
                "map": [],
                "diversity": [],
                "novelty": []
            })
            
            # Get item popularity for novelty calculation
            startup_popularity = await self.get_item_popularity("startup")
            position_popularity = await self.get_item_popularity("position")
            all_popularity = {**startup_popularity, **position_popularity}
            
            # Get item features for diversity calculation
            startup_features = await self.get_item_features("startup")
            position_features = await self.get_item_features("position")
            all_features = {**startup_features, **position_features}
            
            # Evaluate each user's recommendations
            for user_id, recommendations in user_recommendations.items():
                # Extract item IDs from recommendations
                rec_item_ids = [rec.item_id for rec in recommendations]
                
                # Get relevant items for this user
                relevant_items_dict = await self.get_relevant_items_for_user(user_id)
                relevant_items_list = list(relevant_items_dict.keys())
                
                if not relevant_items_list:
                    # Skip users with no relevant items
                    continue
                
                # Calculate metrics for different k values
                for k in k_values:
                    precision = self.precision_at_k(rec_item_ids, relevant_items_list, k)
                    recall = self.recall_at_k(rec_item_ids, relevant_items_list, k)
                    f1 = self.f1_at_k(rec_item_ids, relevant_items_list, k)
                    ndcg = self.ndcg_at_k(rec_item_ids, relevant_items_dict, k)
                    
                    results[f"precision@{k}"].append(precision)
                    results[f"recall@{k}"].append(recall)
                    results[f"f1@{k}"].append(f1)
                    results[f"ndcg@{k}"].append(ndcg)
                
                # Calculate other metrics
                map_score = self.mean_average_precision(rec_item_ids, relevant_items_list)
                diversity = self.diversity_at_k(rec_item_ids, all_features, 10)
                novelty_score = self.novelty(rec_item_ids, all_popularity)
                
                results["map"].append(map_score)
                results["diversity"].append(diversity)
                results["novelty"].append(novelty_score)
            
            # Calculate averages
            avg_results = {}
            for metric, scores in results.items():
                if scores:
                    avg_results[metric] = {
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores)),
                        "count": len(scores)
                    }
                else:
                    avg_results[metric] = {"mean": 0.0, "std": 0.0, "count": 0}
            
            # Add coverage metric
            all_recommendations = [
                [rec.item_id for rec in recs] 
                for recs in user_recommendations.values()
            ]
            
            with get_db_context() as db:
                all_startup_ids = [s.startup_id for s in db.query(Startup).all()]
                all_position_ids = [p.position_id for p in db.query(Position).all()]
                catalog_items = all_startup_ids + all_position_ids
            
            coverage_score = self.coverage(all_recommendations, catalog_items)
            avg_results["coverage"] = {"mean": coverage_score, "std": 0.0, "count": 1}
            
            return avg_results
            
        except Exception as e:
            logger.error(f"Error evaluating recommendations: {e}")
            return {}


class AnalyticsDashboard:
    """Analytics dashboard for monitoring recommendation system performance."""
    
    def __init__(self):
        self.metrics = RecommendationMetrics()
    
    async def get_user_engagement_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get user engagement statistics."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            with get_db_context() as db:
                # Active users
                active_users = db.query(ImplicitFeedback.user_id.distinct()).filter(
                    ImplicitFeedback.timestamp >= cutoff_date
                ).count()
                
                # Total interactions
                total_implicit = db.query(ImplicitFeedback).filter(
                    ImplicitFeedback.timestamp >= cutoff_date
                ).count()
                
                total_explicit = db.query(ExplicitFeedback).filter(
                    ExplicitFeedback.timestamp >= cutoff_date
                ).count()
                
                # Interaction breakdown
                implicit_breakdown = {}
                implicit_feedback = db.query(ImplicitFeedback).filter(
                    ImplicitFeedback.timestamp >= cutoff_date
                ).all()
                
                for fb in implicit_feedback:
                    implicit_breakdown[fb.event_type] = implicit_breakdown.get(fb.event_type, 0) + 1
                
                explicit_breakdown = {}
                explicit_feedback = db.query(ExplicitFeedback).filter(
                    ExplicitFeedback.timestamp >= cutoff_date
                ).all()
                
                for fb in explicit_feedback:
                    explicit_breakdown[fb.feedback_type] = explicit_breakdown.get(fb.feedback_type, 0) + 1
                
                return {
                    "active_users": active_users,
                    "total_implicit_feedback": total_implicit,
                    "total_explicit_feedback": total_explicit,
                    "implicit_breakdown": implicit_breakdown,
                    "explicit_breakdown": explicit_breakdown,
                    "period_days": days
                }
                
        except Exception as e:
            logger.error(f"Error getting user engagement stats: {e}")
            return {}
    
    async def get_item_performance_stats(self, item_type: str = None) -> Dict[str, Any]:
        """Get item performance statistics."""
        try:
            with get_db_context() as db:
                query = db.query(ImplicitFeedback)
                if item_type:
                    query = query.filter(ImplicitFeedback.item_type == item_type)
                
                implicit_feedback = query.all()
                
                # Most viewed items
                view_counts = defaultdict(int)
                click_counts = defaultdict(int)
                save_counts = defaultdict(int)
                
                for fb in implicit_feedback:
                    if fb.event_type == 'view':
                        view_counts[fb.item_id] += 1
                    elif fb.event_type == 'click':
                        click_counts[fb.item_id] += 1
                    elif fb.event_type == 'save':
                        save_counts[fb.item_id] += 1
                
                # Top items by metric
                top_viewed = sorted(view_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                top_clicked = sorted(click_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                top_saved = sorted(save_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                
                return {
                    "top_viewed": top_viewed,
                    "top_clicked": top_clicked,
                    "top_saved": top_saved,
                    "total_items_with_views": len(view_counts),
                    "total_items_with_clicks": len(click_counts),
                    "total_items_with_saves": len(save_counts)
                }
                
        except Exception as e:
            logger.error(f"Error getting item performance stats: {e}")
            return {}
    
    async def get_algorithm_performance_comparison(self) -> Dict[str, Any]:
        """Compare performance of different recommendation algorithms."""
        # This would be implemented with A/B testing data
        # For now, return placeholder structure
        return {
            "content_based": {
                "precision@5": 0.0,
                "recall@5": 0.0,
                "ndcg@5": 0.0,
                "user_satisfaction": 0.0
            },
            "collaborative": {
                "precision@5": 0.0,
                "recall@5": 0.0,
                "ndcg@5": 0.0,
                "user_satisfaction": 0.0
            },
            "two_tower": {
                "precision@5": 0.0,
                "recall@5": 0.0,
                "ndcg@5": 0.0,
                "user_satisfaction": 0.0
            },
            "hybrid": {
                "precision@5": 0.0,
                "recall@5": 0.0,
                "ndcg@5": 0.0,
                "user_satisfaction": 0.0
            }
        }