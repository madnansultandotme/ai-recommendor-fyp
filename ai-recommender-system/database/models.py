from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

Base = declarative_base()


class User(Base):
    """User model for founders, developers, and investors."""
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)  # Optional if using external auth
    user_type = Column(String(50), nullable=False)  # 'founder', 'developer', 'investor'
    profile_data = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    startups = relationship("Startup", back_populates="founder")
    implicit_feedback = relationship("ImplicitFeedback", back_populates="user")
    explicit_feedback = relationship("ExplicitFeedback", back_populates="user")
    recommendation_requests = relationship("RecommendationRequest", back_populates="user")
    recommendation_engagements = relationship("RecommendationEngagement", back_populates="user")


class Startup(Base):
    """Startup model."""
    __tablename__ = "startups"
    
    startup_id = Column(Integer, primary_key=True, index=True)
    founder_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)  # Key field for semantic search
    startup_metadata = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    founder = relationship("User", back_populates="startups")
    positions = relationship("Position", back_populates="startup")
    text_embeddings = relationship("TextEmbedding", 
                                 primaryjoin="and_(foreign(TextEmbedding.item_id)==Startup.startup_id, "
                                           "TextEmbedding.item_type=='startup')")


class Position(Base):
    """Position/Job model."""
    __tablename__ = "positions"
    
    position_id = Column(Integer, primary_key=True, index=True)
    startup_id = Column(Integer, ForeignKey("startups.startup_id"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    requirements = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    startup = relationship("Startup", back_populates="positions")
    text_embeddings = relationship("TextEmbedding",
                                 primaryjoin="and_(foreign(TextEmbedding.item_id)==Position.position_id, "
                                           "TextEmbedding.item_type=='position')")


class ImplicitFeedback(Base):
    """Implicit feedback tracking (views, clicks, saves)."""
    __tablename__ = "implicit_feedback"
    
    feedback_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    item_type = Column(String(50), nullable=False)  # 'startup', 'position', 'investor'
    item_id = Column(Integer, nullable=False)
    event_type = Column(String(50), nullable=False)  # 'view', 'click', 'save'
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="implicit_feedback")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_item_timestamp', 'user_id', 'item_type', 'timestamp'),
    )


class ExplicitFeedback(Base):
    """Explicit feedback tracking (likes, passes, super_likes)."""
    __tablename__ = "explicit_feedback"
    
    feedback_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    item_type = Column(String(50), nullable=False)
    item_id = Column(Integer, nullable=False)
    feedback_type = Column(String(50), nullable=False)  # 'like', 'pass', 'super_like'
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="explicit_feedback")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_item_type', 'user_id', 'item_type'),
    )


class RecommendationRequest(Base):
    """Track recommendation requests for analytics and feedback loop."""
    __tablename__ = "recommendation_requests"
    
    request_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    endpoint_called = Column(String(255), nullable=False)  # e.g., '/for-developer'
    algorithm_used = Column(String(255), nullable=False)  # e.g., 'content_based', 'two_tower_v1'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="recommendation_requests")
    results = relationship("RecommendationResult", back_populates="request")
    engagements = relationship("RecommendationEngagement", back_populates="request")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_created_at', 'user_id', 'created_at'),
    )


class RecommendationResult(Base):
    """Store recommendation results for each request."""
    __tablename__ = "recommendation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(36), ForeignKey("recommendation_requests.request_id"), nullable=False)
    item_type = Column(String(50), nullable=False)
    item_id = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)
    match_reasons = Column(JSON, nullable=True)  # Array of reasons
    rank = Column(Integer, nullable=False)
    
    # Relationships
    request = relationship("RecommendationRequest", back_populates="results")
    
    # Indexes
    __table_args__ = (
        Index('idx_request_id', 'request_id'),
    )


class RecommendationEngagement(Base):
    """Track user engagement with recommendation results."""
    __tablename__ = "recommendation_engagement"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(36), ForeignKey("recommendation_requests.request_id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    item_id = Column(Integer, nullable=False)
    engagement_type = Column(String(50), nullable=False)  # 'click', 'like', 'apply'
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    request = relationship("RecommendationRequest", back_populates="engagements")
    user = relationship("User", back_populates="recommendation_engagements")
    
    # Indexes
    __table_args__ = (
        Index('idx_request_id_engagement', 'request_id'),
    )


class TextEmbedding(Base):
    """Store text embeddings for semantic search."""
    __tablename__ = "text_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    item_type = Column(String(50), nullable=False)  # 'startup', 'position', 'user'
    item_id = Column(Integer, nullable=False)
    field_name = Column(String(100), nullable=False)  # 'description', 'requirements', 'skills'
    embedding_vector = Column(JSON, nullable=False)  # Store as JSON array
    model_name = Column(String(255), nullable=False)  # Track which model generated this
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_item_type_id_field', 'item_type', 'item_id', 'field_name'),
        Index('idx_model_name', 'model_name'),
    )


class TrainingMetrics(Base):
    """Store training metrics and model performance."""
    __tablename__ = "training_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), nullable=False)
    model_version = Column(String(100), nullable=False)
    algorithm_type = Column(String(100), nullable=False)  # 'two_tower', 'collaborative_filtering', etc.
    metrics = Column(JSON, nullable=False)  # Store all metrics as JSON
    training_data_size = Column(Integer, nullable=False)
    training_duration_seconds = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_model_version', 'model_name', 'model_version'),
        Index('idx_created_at', 'created_at'),
    )


class ModelRegistry(Base):
    """Registry for deployed models."""
    __tablename__ = "model_registry"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), nullable=False)
    model_version = Column(String(100), nullable=False)
    algorithm_type = Column(String(100), nullable=False)
    model_path = Column(String(500), nullable=False)
    is_active = Column(Boolean, default=False)
    performance_metrics = Column(JSON, nullable=True)
    deployed_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String(255), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_model_active', 'model_name', 'is_active'),
        Index('idx_algorithm_active', 'algorithm_type', 'is_active'),
    )