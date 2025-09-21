# AI Recommender System - Caching Module

from .redis_cache import cache, cache_recommendations, cache_user_profile, cache_analytics, get_cache_stats

__all__ = [
    "cache",
    "cache_recommendations",
    "cache_user_profile", 
    "cache_analytics",
    "get_cache_stats"
]