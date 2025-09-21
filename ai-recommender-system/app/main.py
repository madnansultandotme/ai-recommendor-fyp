import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging
from loguru import logger
from datetime import datetime

from app.config import settings
from database.database import get_db, check_db_health
from database.models import User, ImplicitFeedback, ExplicitFeedback
from models.base_recommender import (
    RecommendationRequest, 
    RecommendationResult, 
    RecommendationResponse
)
from models.hybrid_recommender import HybridRecommender
from analytics.metrics import RecommendationMetrics, AnalyticsDashboard
from ab_testing import ABTestingFramework, ExperimentConfig, ExperimentStatus
from caching import get_cache_stats
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Global recommender system, analytics, and A/B testing
hybrid_recommender = HybridRecommender()
analytics_dashboard = AnalyticsDashboard()
ab_testing = ABTestingFramework()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {settings.project_name}")
    
    # Initialize and load models for hybrid recommender
    try:
        load_results = await hybrid_recommender.load_models()
        logger.info(f"Model loading results: {load_results}")
    except Exception as e:
        logger.warning(f"Could not load trained models: {e}")
    
    logger.info("Hybrid recommender system initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")

# Create FastAPI app
app = FastAPI(
    title=settings.project_name,
    description="AI-powered recommender system for startups, developers, and investors",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for API
class RecommendationRequestAPI(BaseModel):
    user_id: int
    user_type: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    database: str
    version: str
    timestamp: str

class UserProfileResponse(BaseModel):
    user_id: int
    user_type: str
    profile_data: Dict[str, Any]

# Web UI Route
@app.get("/demo", tags=["UI"])
async def demo_ui():
    """Serve the demo web UI."""
    return FileResponse("static/index.html")

# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.project_name}",
        "version": "1.0.0",
        "docs": "/docs",
        "demo_ui": "/demo"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    db_status = "healthy" if check_db_health() else "unhealthy"
    
    return HealthResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        database=db_status,
        version="1.0.0",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

@app.get("/users/{user_id}/profile", response_model=UserProfileResponse, tags=["Users"])
async def get_user_profile(user_id: int, db: Session = Depends(get_db)):
    """Get user profile."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserProfileResponse(
        user_id=user.user_id,
        user_type=user.user_type,
        profile_data=user.profile_data or {}
    )

@app.post(f"{settings.api_v1_str}/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    request: RecommendationRequestAPI,
    db: Session = Depends(get_db)
):
    """Get personalized recommendations."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Validate user exists
        user = db.query(User).filter(User.user_id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Create recommendation request
        rec_request = RecommendationRequest(
            user_id=request.user_id,
            user_type=request.user_type,
            limit=request.limit,
            filters=request.filters,
            context=request.context
        )
        
        # Get recommendations from hybrid system
        results = await hybrid_recommender.recommend(rec_request)
        
        processing_time = (time.time() - start_time) * 1000
        
        # TODO: Log to recommendation_requests table
        
        return RecommendationResponse(
            request_id=request_id,
            user_id=request.user_id,
            algorithm_used="hybrid",
            results=results,
            total_results=len(results),
            processing_time_ms=processing_time,
            metadata={
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "filters_applied": request.filters or {},
                "context": request.context or {}
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get(f"{settings.api_v1_str}/recommendations/for-developers", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations_for_developers(
    user_id: int = Query(..., description="Developer user ID"),
    limit: int = Query(10, description="Number of recommendations to return"),
    industry: Optional[str] = Query(None, description="Filter by industry"),
    remote_ok: Optional[bool] = Query(None, description="Filter by remote work availability"),
    db: Session = Depends(get_db)
):
    """Get job position recommendations for developers."""
    filters = {}
    if industry:
        filters["industry"] = industry
    if remote_ok is not None:
        filters["remote_ok"] = remote_ok
    
    request = RecommendationRequestAPI(
        user_id=user_id,
        user_type="developer",
        limit=limit,
        filters=filters if filters else None
    )
    
    return await get_recommendations(request, db)

@app.get(f"{settings.api_v1_str}/recommendations/for-founders", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations_for_founders(
    user_id: int = Query(..., description="Founder user ID"),
    limit: int = Query(10, description="Number of recommendations to return"),
    skills: Optional[List[str]] = Query(None, description="Filter by required skills"),
    location: Optional[str] = Query(None, description="Filter by location"),
    db: Session = Depends(get_db)
):
    """Get developer recommendations for founders."""
    filters = {}
    if skills:
        filters["skills"] = skills
    if location:
        filters["location"] = location
    
    request = RecommendationRequestAPI(
        user_id=user_id,
        user_type="founder",
        limit=limit,
        filters=filters if filters else None
    )
    
    return await get_recommendations(request, db)

@app.get(f"{settings.api_v1_str}/recommendations/for-investors", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations_for_investors(
    user_id: int = Query(..., description="Investor user ID"),
    limit: int = Query(10, description="Number of recommendations to return"),
    stage: Optional[str] = Query(None, description="Filter by startup stage"),
    industry: Optional[str] = Query(None, description="Filter by industry"),
    db: Session = Depends(get_db)
):
    """Get startup recommendations for investors."""
    filters = {}
    if stage:
        filters["stage"] = stage
    if industry:
        filters["industry"] = industry
    
    request = RecommendationRequestAPI(
        user_id=user_id,
        user_type="investor",
        limit=limit,
        filters=filters if filters else None
    )
    
    return await get_recommendations(request, db)

# Training and Model Management Endpoints
class TrainingRequest(BaseModel):
    training_data: Optional[Dict[str, Any]] = None
    force_retrain: bool = False

@app.post(f"{settings.api_v1_str}/models/train", tags=["Models"])
async def train_models(request: TrainingRequest = TrainingRequest()):
    """Train all trainable models in the hybrid system."""
    try:
        logger.info("Starting model training...")
        result = await hybrid_recommender.train(request.training_data)
        return {
            "training_result": result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post(f"{settings.api_v1_str}/models/load", tags=["Models"])
async def load_models():
    """Load all trained models."""
    try:
        load_results = await hybrid_recommender.load_models()
        return {
            "load_results": load_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

class WeightUpdateRequest(BaseModel):
    weights: Dict[str, float]

class ImplicitFeedbackRequest(BaseModel):
    user_id: int
    item_type: str  # 'position' or 'startup'
    item_id: int
    event_type: str  # 'view', 'click', 'save'
    context: Optional[Dict[str, Any]] = None

class ExplicitFeedbackRequest(BaseModel):
    user_id: int
    item_type: str  # 'position' or 'startup'
    item_id: int
    feedback_type: str  # 'like', 'pass', 'super_like'
    rating: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

class ExperimentCreateRequest(BaseModel):
    name: str
    description: str
    algorithms: List[str]
    traffic_split: Dict[str, float]
    target_user_types: List[str]
    min_sample_size: int = 100
    max_duration_days: int = 30
    primary_metric: str = "precision@5"

@app.put(f"{settings.api_v1_str}/models/weights", tags=["Models"])
async def update_algorithm_weights(request: WeightUpdateRequest):
    """Update the weights for different algorithms in the hybrid system."""
    try:
        success = hybrid_recommender.update_weights(request.weights)
        if not success:
            raise HTTPException(status_code=400, detail="Invalid weights provided")
        
        return {
            "status": "success",
            "new_weights": hybrid_recommender.weights,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error updating weights: {e}")
        raise HTTPException(status_code=500, detail=f"Weight update failed: {str(e)}")

@app.get(f"{settings.api_v1_str}/models/evaluate", tags=["Models"])
async def evaluate_algorithms():
    """Evaluate all algorithms on test data."""
    try:
        evaluation_results = await hybrid_recommender.evaluate_algorithms()
        return {
            "evaluation_results": evaluation_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get(f"{settings.api_v1_str}/algorithms/status", tags=["System"])
async def get_algorithm_status():
    """Get status of all recommendation algorithms."""
    return {
        "algorithms": hybrid_recommender.get_algorithm_status(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post(f"{settings.api_v1_str}/feedback/implicit", tags=["Feedback"])
async def log_implicit_feedback(
    request: ImplicitFeedbackRequest,
    db: Session = Depends(get_db)
):
    """Log implicit user feedback (views, clicks, saves)."""
    try:
        # Validate user exists
        user = db.query(User).filter(User.user_id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Validate event type
        valid_events = ['view', 'click', 'save', 'share', 'bookmark']
        if request.event_type not in valid_events:
            raise HTTPException(status_code=400, detail=f"Invalid event type. Must be one of: {valid_events}")
        
        # Validate item type
        if request.item_type not in ['position', 'startup']:
            raise HTTPException(status_code=400, detail="Item type must be 'position' or 'startup'")
        
        # Create feedback record
        feedback = ImplicitFeedback(
            user_id=request.user_id,
            item_type=request.item_type,
            item_id=request.item_id,
            event_type=request.event_type,
            timestamp=datetime.utcnow(),
            context=request.context or {}
        )
        
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        
        logger.info(f"Logged implicit feedback: user {request.user_id} {request.event_type} {request.item_type} {request.item_id}")
        
        return {
            "status": "success",
            "message": "Implicit feedback logged successfully",
            "feedback_id": feedback.feedback_id,
            "timestamp": feedback.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging implicit feedback: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to log implicit feedback")

@app.post(f"{settings.api_v1_str}/feedback/explicit", tags=["Feedback"])
async def log_explicit_feedback(
    request: ExplicitFeedbackRequest,
    db: Session = Depends(get_db)
):
    """Log explicit user feedback (likes, passes, super_likes)."""
    try:
        # Validate user exists
        user = db.query(User).filter(User.user_id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Validate feedback type
        valid_feedback = ['like', 'pass', 'super_like', 'dislike', 'report']
        if request.feedback_type not in valid_feedback:
            raise HTTPException(status_code=400, detail=f"Invalid feedback type. Must be one of: {valid_feedback}")
        
        # Validate item type
        if request.item_type not in ['position', 'startup']:
            raise HTTPException(status_code=400, detail="Item type must be 'position' or 'startup'")
        
        # Check for existing feedback
        existing = db.query(ExplicitFeedback).filter(
            ExplicitFeedback.user_id == request.user_id,
            ExplicitFeedback.item_id == request.item_id,
            ExplicitFeedback.item_type == request.item_type
        ).first()
        
        if existing:
            # Update existing feedback
            existing.feedback_type = request.feedback_type
            existing.rating = request.rating
            existing.timestamp = datetime.utcnow()
            existing.context = request.context or {}
            db.commit()
            db.refresh(existing)
            feedback_id = existing.feedback_id
            action = "updated"
        else:
            # Create new feedback record
            feedback = ExplicitFeedback(
                user_id=request.user_id,
                item_type=request.item_type,
                item_id=request.item_id,
                feedback_type=request.feedback_type,
                rating=request.rating,
                timestamp=datetime.utcnow(),
                context=request.context or {}
            )
            
            db.add(feedback)
            db.commit()
            db.refresh(feedback)
            feedback_id = feedback.feedback_id
            action = "created"
        
        logger.info(f"Logged explicit feedback: user {request.user_id} {request.feedback_type} {request.item_type} {request.item_id} ({action})")
        
        return {
            "status": "success",
            "message": f"Explicit feedback {action} successfully",
            "feedback_id": feedback_id,
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging explicit feedback: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to log explicit feedback")

@app.get(f"{settings.api_v1_str}/stats", tags=["System"])
async def get_system_stats(db: Session = Depends(get_db)):
    """Get system statistics."""
    try:
        user_count = db.query(User).count()
        # TODO: Add more statistics
        
        return {
            "users": user_count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_status": "operational"
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Unable to fetch system statistics")

# Analytics Endpoints
@app.get(f"{settings.api_v1_str}/analytics/engagement", tags=["Analytics"])
async def get_engagement_analytics(
    days: int = Query(30, description="Number of days to analyze")
):
    """Get user engagement analytics."""
    try:
        stats = await analytics_dashboard.get_user_engagement_stats(days)
        return {
            "engagement_stats": stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error getting engagement analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch engagement analytics")

@app.get(f"{settings.api_v1_str}/analytics/items", tags=["Analytics"])
async def get_item_analytics(
    item_type: Optional[str] = Query(None, description="Filter by item type (startup/position)")
):
    """Get item performance analytics."""
    try:
        stats = await analytics_dashboard.get_item_performance_stats(item_type)
        return {
            "item_stats": stats,
            "item_type": item_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error getting item analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch item analytics")

@app.get(f"{settings.api_v1_str}/analytics/algorithms", tags=["Analytics"])
async def get_algorithm_analytics():
    """Get algorithm performance comparison."""
    try:
        comparison = await analytics_dashboard.get_algorithm_performance_comparison()
        return {
            "algorithm_comparison": comparison,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error getting algorithm analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch algorithm analytics")

@app.post(f"{settings.api_v1_str}/analytics/evaluate", tags=["Analytics"])
async def evaluate_recommendations(
    user_ids: List[int] = Query(..., description="List of user IDs to evaluate"),
    k_values: List[int] = Query([5, 10], description="K values for metrics")
):
    """Evaluate recommendation quality for specified users."""
    try:
        # Get recommendations for specified users
        user_recommendations = {}
        
        for user_id in user_ids:
            # Get user info
            with get_db_context() as db:
                user = db.query(User).filter(User.user_id == user_id).first()
                if not user:
                    continue
            
            # Get recommendations
            request = RecommendationRequest(
                user_id=user_id,
                user_type=user.user_type,
                limit=20
            )
            
            recommendations = await hybrid_recommender.recommend(request)
            if recommendations:
                user_recommendations[user_id] = recommendations
        
        # Evaluate using metrics
        metrics = RecommendationMetrics()
        evaluation_results = await metrics.evaluate_recommendations(user_recommendations, k_values)
        
        return {
            "evaluation_results": evaluation_results,
            "evaluated_users": len(user_recommendations),
            "k_values": k_values,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        logger.error(f"Error evaluating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to evaluate recommendations")

# A/B Testing Endpoints
@app.post(f"{settings.api_v1_str}/experiments", tags=["A/B Testing"])
async def create_experiment(request: ExperimentCreateRequest):
    """Create a new A/B test experiment."""
    try:
        config = ExperimentConfig(
            name=request.name,
            description=request.description,
            algorithms=request.algorithms,
            traffic_split=request.traffic_split,
            target_user_types=request.target_user_types,
            min_sample_size=request.min_sample_size,
            max_duration_days=request.max_duration_days,
            primary_metric=request.primary_metric
        )
        
        experiment_id = ab_testing.create_experiment(config)
        
        return {
            "experiment_id": experiment_id,
            "status": "created",
            "message": "Experiment created successfully",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post(f"{settings.api_v1_str}/experiments/{{experiment_id}}/start", tags=["A/B Testing"])
async def start_experiment(experiment_id: str):
    """Start an A/B test experiment."""
    try:
        success = ab_testing.start_experiment(experiment_id)
        
        if success:
            return {
                "experiment_id": experiment_id,
                "status": "started",
                "message": "Experiment started successfully",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to start experiment")
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to start experiment")

@app.post(f"{settings.api_v1_str}/experiments/{{experiment_id}}/stop", tags=["A/B Testing"])
async def stop_experiment(experiment_id: str):
    """Stop an A/B test experiment."""
    try:
        success = ab_testing.stop_experiment(experiment_id)
        
        if success:
            return {
                "experiment_id": experiment_id,
                "status": "stopped",
                "message": "Experiment stopped successfully",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to stop experiment")
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error stopping experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop experiment")

@app.get(f"{settings.api_v1_str}/experiments", tags=["A/B Testing"])
async def list_experiments():
    """List all A/B test experiments."""
    try:
        experiments = ab_testing.list_experiments()
        return {
            "experiments": experiments,
            "total_count": len(experiments),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail="Failed to list experiments")

@app.get(f"{settings.api_v1_str}/experiments/{{experiment_id}}", tags=["A/B Testing"])
async def get_experiment_status(experiment_id: str):
    """Get status and metrics for an A/B test experiment."""
    try:
        status = ab_testing.get_experiment_status(experiment_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return {
            "experiment_status": status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get experiment status")

# Caching Endpoints
@app.get(f"{settings.api_v1_str}/cache/stats", tags=["System"])
async def get_cache_statistics():
    """Get Redis cache statistics."""
    try:
        stats = await get_cache_stats()
        return {
            "cache_stats": stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache statistics")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level=settings.log_level.lower()
    )