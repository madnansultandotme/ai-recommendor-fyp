import asyncio
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.base_recommender import BaseRecommender, RecommendationRequest, RecommendationResult
from database.database import get_db_context
from database.models import User, Startup, Position
from database.vector_store import vector_store
from app.config import settings

logger = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender using semantic similarity via vector search."""
    
    def __init__(self):
        super().__init__("content_based", "1.0")
        self.is_trained = True  # Content-based doesn't require training
        self.metadata = {
            "description": "Semantic content-based recommendations using sentence transformers",
            "requires_training": False,
            "supports_cold_start": True
        }
    
    def can_recommend(self, user_id: int, user_type: str) -> bool:
        """Content-based can always recommend."""
        return True
    
    async def recommend(
        self, 
        request: RecommendationRequest
    ) -> List[RecommendationResult]:
        """Generate content-based recommendations."""
        try:
            # Get user profile
            user_profile = await self._get_user_profile(request.user_id)
            if not user_profile:
                return []
            
            # Generate recommendations based on user type
            if request.user_type == "developer":
                return await self._recommend_positions_for_developer(request, user_profile)
            elif request.user_type == "founder":
                return await self._recommend_developers_for_founder(request, user_profile)
            elif request.user_type == "investor":
                return await self._recommend_startups_for_investor(request, user_profile)
            else:
                logger.warning(f"Unknown user type: {request.user_type}")
                # Fallback: return some basic recommendations
                return await self._get_basic_recommendations(request)
        
        except Exception as e:
            logger.error(f"Error in content-based recommendation: {e}")
            # Fallback: return some basic recommendations
            return await self._get_basic_recommendations(request)
    
    async def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Content-based doesn't require training."""
        return {
            "status": "success",
            "message": "Content-based recommender doesn't require training",
            "training_time": 0.0
        }
    
    async def _get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user profile from database."""
        try:
            with get_db_context() as db:
                user = db.query(User).filter(User.user_id == user_id).first()
                if user:
                    return {
                        "user_id": user.user_id,
                        "user_type": user.user_type,
                        "profile_data": user.profile_data
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    async def _recommend_positions_for_developer(
        self,
        request: RecommendationRequest,
        user_profile: Dict[str, Any]
    ) -> List[RecommendationResult]:
        """Recommend positions for developers based on skills and preferences."""
        try:
            # Try vector store first
            profile_data = user_profile.get("profile_data", {})
            skills = profile_data.get("skills", [])
            preferences = {
                "preferred_industries": profile_data.get("preferred_industries", []),
                "preferred_roles": profile_data.get("preferred_roles", [])
            }
            
            try:
                # Search for similar positions using vector store
                similar_positions = vector_store.search_similar_positions(
                    user_skills=skills,
                    preferences=preferences,
                    n_results=request.limit * 2,  # Get more to filter
                    filters=request.filters
                )
                
                results = []
                for i, pos in enumerate(similar_positions[:request.limit]):
                    metadata = pos.get("metadata", {})
                    position_id = metadata.get("position_id")
                    if position_id is None:
                        continue

                    enriched_meta = {
                        "title": metadata.get("title", ""),
                        "startup_id": metadata.get("startup_id"),
                        "startup_name": metadata.get("startup_name"),
                        "location": metadata.get("location"),
                        "remote_ok": metadata.get("remote_ok"),
                        "salary_range": metadata.get("salary_range"),
                        "equity": metadata.get("equity"),
                        "skills_match": metadata.get("skills"),
                        "similarity_score": pos.get("similarity_score")
                    }
                    results.append(RecommendationResult(
                        item_id=position_id,
                        item_type="position",
                        score=pos["similarity_score"],
                        rank=i + 1,
                        match_reasons=["Skills and preferences match"],
                        metadata={k: v for k, v in enriched_meta.items() if v is not None}
                    ))
                
                if results:  # Return vector store results if available
                    return results
                    
            except Exception as vector_error:
                logger.warning(f"Vector store failed, falling back to database: {vector_error}")
            
            # Fallback to database query
            return await self._get_positions_from_database(request, skills, preferences)
            
        except Exception as e:
            logger.error(f"Error recommending positions for developer: {e}")
            return []
    
    async def _recommend_developers_for_founder(
        self,
        request: RecommendationRequest,
        user_profile: Dict[str, Any]
    ) -> List[RecommendationResult]:
        """Recommend developers for founders based on startup needs."""
        try:
            # Get founder's startup information
            startup_info = await self._get_founder_startup_info(request.user_id)
            if not startup_info:
                logger.warning(f"No startup info found for founder {request.user_id}, using fallback")
                return await self._get_recent_developers_from_database(request)

            # Create query based on startup requirements
            query_skills = startup_info.get("required_skills", [])
            industry = startup_info.get("industry", "")
            query_text = f"Skills needed: {', '.join(query_skills)} for {industry} startup".strip()

            # Attempt vector store search first
            similar_users = []
            try:
                similar_users = vector_store.search_similar_users(
                    query_text=query_text,
                    n_results=request.limit * 2,
                    filters={**request.filters, "user_type": "developer"} if request.filters else {"user_type": "developer"}
                )
            except Exception as vector_error:
                logger.warning(f"Vector store failed for founder {request.user_id}, falling back to database: {vector_error}")

            results = []
            for i, user in enumerate(similar_users[:request.limit]):
                metadata = user.get("metadata", {})
                user_id = metadata.get("user_id")
                if user_id is None:
                    continue

                profile = metadata.get("profile", {})
                results.append(RecommendationResult(
                    item_id=user_id,
                    item_type="user",
                    score=user["similarity_score"],
                    rank=i + 1,
                    match_reasons=["Skills match startup requirements"],
                    metadata={
                        "user_type": "developer",
                        "name": profile.get("name"),
                        "location": profile.get("location"),
                        "skills": metadata.get("skills", []),
                        "experience_years": profile.get("experience_years"),
                        "similarity_score": user.get("similarity_score")
                    }
                ))

            if results:
                return results

            logger.info(f"No vector-store matches for founder {request.user_id}, using recent developers fallback")
            return await self._get_recent_developers_from_database(request)

        except Exception as e:
            logger.error(f"Error recommending developers for founder: {e}")
            return await self._get_recent_developers_from_database(request)
    
    async def _recommend_startups_for_investor(
        self,
        request: RecommendationRequest,
        user_profile: Dict[str, Any]
    ) -> List[RecommendationResult]:
        """Recommend startups for investors based on investment preferences."""
        try:
            profile_data = user_profile.get("profile_data", {})
            interested_industries = profile_data.get("interested_industries", [])
            investment_stages = profile_data.get("investment_stages", [])

            # Create query based on investor preferences
            query_parts = []
            if interested_industries:
                query_parts.append(f"Industries: {', '.join(interested_industries)}")
            if investment_stages:
                query_parts.append(f"Stages: {', '.join(investment_stages)}")

            query_text = " ".join(query_parts) if query_parts else "innovative startups"

            # Attempt vector store search first
            similar_startups = []
            try:
                similar_startups = vector_store.search_similar_startups(
                    query_text=query_text,
                    n_results=request.limit * 2,
                    filters=request.filters
                )
            except Exception as vector_error:
                logger.warning(f"Vector store failed for investor {request.user_id}, falling back: {vector_error}")

            results = []
            for i, startup in enumerate(similar_startups[:request.limit]):
                metadata = startup.get("metadata", {})
                startup_id = metadata.get("startup_id")
                if startup_id is None:
                    continue

                results.append(RecommendationResult(
                    item_id=startup_id,
                    item_type="startup",
                    score=startup["similarity_score"],
                    rank=i + 1,
                    match_reasons=["Industry and stage alignment"],
                    metadata={
                        "name": metadata.get("name"),
                        "industry": metadata.get("industry", []),
                        "stage": metadata.get("stage", ""),
                        "location": metadata.get("location"),
                        "team_size": metadata.get("team_size"),
                        "funding_amount": metadata.get("funding_amount"),
                        "similarity_score": startup.get("similarity_score")
                    }
                ))

            if results:
                return results

            logger.info(f"No vector-store matches for investor {request.user_id}, using recent startups fallback")
            return await self._get_recent_startups_from_database(request)

        except Exception as e:
            logger.error(f"Error recommending startups for investor: {e}")
            return await self._get_recent_startups_from_database(request)
    
    async def _get_founder_startup_info(self, founder_id: int) -> Optional[Dict[str, Any]]:
        """Get startup information for a founder."""
        try:
            with get_db_context() as db:
                startup = db.query(Startup).filter(Startup.founder_id == founder_id).first()
                if startup:
                    startup_metadata = startup.startup_metadata or {}
                    return {
                        "startup_id": startup.startup_id,
                        "description": startup.description,
                        "industry": startup_metadata.get("industry", []),
                        "stage": startup_metadata.get("stage", ""),
                        "required_skills": startup_metadata.get("required_skills", [])
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting founder startup info: {e}")
            return None
    
    async def _get_basic_recommendations(self, request: RecommendationRequest) -> List[RecommendationResult]:
        """Get basic recommendations when other methods fail."""
        try:
            if request.user_type == "developer":
                return await self._get_recent_positions_from_database(request)
            elif request.user_type == "founder":
                return await self._get_recent_developers_from_database(request)
            elif request.user_type == "investor":
                return await self._get_recent_startups_from_database(request)
            else:
                return []
        except Exception as e:
            logger.error(f"Error in basic recommendations: {e}")
            return []
    
    async def _get_positions_from_database(
        self, 
        request: RecommendationRequest, 
        skills: List[str], 
        preferences: Dict[str, Any]
    ) -> List[RecommendationResult]:
        """Get position recommendations directly from database."""
        try:
            with get_db_context() as db:
                # Get all positions
                query = db.query(Position)
                
                # Apply basic filtering if needed
                positions = query.limit(request.limit * 2).all()
                
                results = []
                for i, position in enumerate(positions[:request.limit]):
                    # Simple scoring based on skill matches
                    score = self._calculate_position_score(position, skills)
                    
                    startup = position.startup
                    requirements = position.requirements or {}
                    fallback_meta = {
                        "title": position.title,
                        "startup_id": position.startup_id,
                        "startup_name": startup.name if startup else None,
                        "location": requirements.get("location"),
                        "remote_ok": requirements.get("remote_ok"),
                        "salary_range": requirements.get("salary_range"),
                        "equity": requirements.get("equity"),
                        "skills": requirements.get("skills", []),
                        "fallback_method": "database"
                    }

                    results.append(RecommendationResult(
                        item_id=position.position_id,
                        item_type="position",
                        score=score,
                        rank=i + 1,
                        match_reasons=["Database fallback"],
                        metadata={k: v for k, v in fallback_meta.items() if v not in (None, [])}
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting positions from database: {e}")
            return []
    
    async def _get_recent_positions_from_database(self, request: RecommendationRequest) -> List[RecommendationResult]:
        """Get recent positions from database."""
        try:
            with get_db_context() as db:
                positions = db.query(Position).limit(request.limit).all()
                
                results = []
                for i, position in enumerate(positions):
                    requirements = position.requirements or {}
                    startup = position.startup
                    meta = {
                        "title": position.title,
                        "startup_id": position.startup_id,
                        "startup_name": startup.name if startup else None,
                        "location": requirements.get("location"),
                        "remote_ok": requirements.get("remote_ok"),
                        "salary_range": requirements.get("salary_range"),
                        "equity": requirements.get("equity"),
                        "skills": requirements.get("skills", []),
                        "method": "recent_positions"
                    }

                    results.append(RecommendationResult(
                        item_id=position.position_id,
                        item_type="position",
                        score=0.5,  # Default score
                        rank=i + 1,
                        match_reasons=["Recent position"],
                        metadata={k: v for k, v in meta.items() if v not in (None, [])}
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting recent positions: {e}")
            return []
    
    async def _get_recent_developers_from_database(self, request: RecommendationRequest) -> List[RecommendationResult]:
        """Get recent developers from database."""
        try:
            with get_db_context() as db:
                developers = db.query(User).filter(User.user_type == "developer").limit(request.limit).all()
                
                results = []
                for i, user in enumerate(developers):
                    # Extract skills from user profile
                    profile_data = user.profile_data or {}
                    skills = profile_data.get("skills", [])
                    
                    meta = {
                        "user_type": "developer",
                        "name": profile_data.get("name"),
                        "location": profile_data.get("location"),
                        "skills": skills,
                        "experience_years": profile_data.get("experience_years"),
                        "method": "recent_developers"
                    }

                    results.append(RecommendationResult(
                        item_id=user.user_id,
                        item_type="user",
                        score=0.5,  # Default score
                        rank=i + 1,
                        match_reasons=["Recent developer"],
                        metadata={k: v for k, v in meta.items() if v not in (None, [])}
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting recent developers: {e}")
            return []
    
    async def _get_recent_startups_from_database(self, request: RecommendationRequest) -> List[RecommendationResult]:
        """Get recent startups from database."""
        try:
            with get_db_context() as db:
                startups = db.query(Startup).limit(request.limit).all()
                
                results = []
                for i, startup in enumerate(startups):
                    # Extract metadata from startup
                    startup_metadata = startup.startup_metadata or {}
                    industry = startup_metadata.get("industry", [])
                    stage = startup_metadata.get("stage", "")
                    
                    meta = {
                        "name": startup.name,
                        "industry": industry,
                        "stage": stage,
                        "location": startup_metadata.get("location"),
                        "team_size": startup_metadata.get("team_size"),
                        "funding_amount": startup_metadata.get("funding_amount"),
                        "method": "recent_startups"
                    }

                    results.append(RecommendationResult(
                        item_id=startup.startup_id,
                        item_type="startup",
                        score=0.5,  # Default score
                        rank=i + 1,
                        match_reasons=["Recent startup"],
                        metadata={k: v for k, v in meta.items() if v not in (None, [])}
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting recent startups: {e}")
            return []
    
    def _calculate_position_score(self, position: Position, user_skills: List[str]) -> float:
        """Calculate a simple score for position matching."""
        try:
            # Simple scoring: check if any user skills match position requirements
            position_text = f"{position.title} {position.description}"
            score = 0.3  # Base score
            
            for skill in user_skills:
                if skill.lower() in position_text.lower():
                    score += 0.1
            
            return min(score, 1.0)
            
        except Exception:
            return 0.3
