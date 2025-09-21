import asyncio
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass
from collections import defaultdict
import math

from sqlalchemy.orm import Session
from database.database import get_db_context
from database.models import User, ImplicitFeedback, ExplicitFeedback
from models.base_recommender import RecommendationRequest, RecommendationResult
from models.content_based import ContentBasedRecommender
from models.collaborative_filtering import CollaborativeFilteringRecommender
from models.two_tower import TwoTowerRecommender
from models.hybrid_recommender import HybridRecommender
from analytics.metrics import RecommendationMetrics

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    name: str
    description: str
    algorithms: List[str]  # List of algorithm names to test
    traffic_split: Dict[str, float]  # Percentage of traffic for each algorithm
    target_user_types: List[str]  # User types to include in the test
    min_sample_size: int = 100  # Minimum users per variant
    max_duration_days: int = 30  # Maximum experiment duration
    significance_threshold: float = 0.05  # Statistical significance threshold
    primary_metric: str = "precision@5"  # Primary metric to optimize
    secondary_metrics: List[str] = None  # Additional metrics to track


@dataclass
class ExperimentResult:
    """Results from an A/B test experiment."""
    experiment_id: str
    algorithm: str
    users_assigned: int
    conversions: int
    conversion_rate: float
    metrics: Dict[str, float]
    confidence_interval: Tuple[float, float]
    statistical_significance: bool


class ABTestingFramework:
    """A/B testing framework for recommendation algorithms."""
    
    def __init__(self):
        self.experiments = {}
        self.user_assignments = {}  # user_id -> experiment_id -> algorithm
        self.metrics_calculator = RecommendationMetrics()
        
        # Initialize available algorithms
        self.available_algorithms = {
            "content_based": ContentBasedRecommender(),
            "collaborative": CollaborativeFilteringRecommender(),
            "two_tower": TwoTowerRecommender(),
            "hybrid": HybridRecommender()
        }
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B test experiment."""
        try:
            # Validate configuration
            self._validate_config(config)
            
            # Generate experiment ID
            experiment_id = self._generate_experiment_id(config.name)
            
            # Create experiment
            experiment = {
                "id": experiment_id,
                "config": config,
                "status": ExperimentStatus.DRAFT,
                "created_at": datetime.utcnow(),
                "started_at": None,
                "ended_at": None,
                "user_assignments": {},
                "results": {},
                "metrics_history": []
            }
            
            self.experiments[experiment_id] = experiment
            logger.info(f"Created experiment {experiment_id}: {config.name}")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            raise
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B test experiment."""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            
            if experiment["status"] != ExperimentStatus.DRAFT:
                raise ValueError(f"Cannot start experiment in status {experiment['status'].value}")
            
            # Update experiment status
            experiment["status"] = ExperimentStatus.RUNNING
            experiment["started_at"] = datetime.utcnow()
            
            logger.info(f"Started experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting experiment: {e}")
            return False
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an A/B test experiment."""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            
            if experiment["status"] != ExperimentStatus.RUNNING:
                raise ValueError(f"Cannot stop experiment in status {experiment['status'].value}")
            
            # Update experiment status
            experiment["status"] = ExperimentStatus.COMPLETED
            experiment["ended_at"] = datetime.utcnow()
            
            logger.info(f"Stopped experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping experiment: {e}")
            return False
    
    def _validate_config(self, config: ExperimentConfig):
        """Validate experiment configuration."""
        # Check algorithms exist
        for algorithm in config.algorithms:
            if algorithm not in self.available_algorithms:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Check traffic split sums to 1.0
        total_split = sum(config.traffic_split.values())
        if abs(total_split - 1.0) > 0.001:
            raise ValueError(f"Traffic split must sum to 1.0, got {total_split}")
        
        # Check traffic split matches algorithms
        for algorithm in config.algorithms:
            if algorithm not in config.traffic_split:
                raise ValueError(f"Traffic split missing for algorithm: {algorithm}")
        
        # Check user types
        valid_user_types = ["developer", "founder", "investor"]
        for user_type in config.target_user_types:
            if user_type not in valid_user_types:
                raise ValueError(f"Invalid user type: {user_type}")
        
        # Set default secondary metrics if not provided
        if config.secondary_metrics is None:
            config.secondary_metrics = ["recall@5", "ndcg@5", "diversity"]
    
    def _generate_experiment_id(self, name: str) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{name_hash}"
    
    def _assign_algorithm(self, user_id: int, traffic_split: Dict[str, float]) -> str:
        """Assign user to algorithm using deterministic hashing."""
        # Use hash of user ID for deterministic assignment
        user_hash = hashlib.md5(str(user_id).encode()).hexdigest()
        hash_value = int(user_hash[:8], 16) / 0xffffffff  # Normalize to [0, 1)
        
        # Assign based on traffic split
        cumulative = 0.0
        for algorithm, split in traffic_split.items():
            cumulative += split
            if hash_value < cumulative:
                return algorithm
        
        # Fallback to first algorithm
        return list(traffic_split.keys())[0]
    
    def assign_user_to_experiment(self, user_id: int, user_type: str, experiment_id: str) -> Optional[str]:
        """Assign a user to an algorithm variant in an experiment."""
        try:
            if experiment_id not in self.experiments:
                return None
            
            experiment = self.experiments[experiment_id]
            config = experiment["config"]
            
            # Check if experiment is running
            if experiment["status"] != ExperimentStatus.RUNNING:
                return None
            
            # Check if user type is eligible
            if user_type not in config.target_user_types:
                return None
            
            # Check if user is already assigned
            if user_id in experiment["user_assignments"]:
                return experiment["user_assignments"][user_id]
            
            # Assign user to algorithm variant using deterministic hash
            algorithm = self._assign_algorithm(user_id, config.traffic_split)
            
            # Store assignment
            experiment["user_assignments"][user_id] = algorithm
            
            # Also store in global user assignments
            if user_id not in self.user_assignments:
                self.user_assignments[user_id] = {}
            self.user_assignments[user_id][experiment_id] = algorithm
            
            logger.debug(f"Assigned user {user_id} to algorithm {algorithm} in experiment {experiment_id}")
            return algorithm
            
        except Exception as e:
            logger.error(f"Error assigning user to experiment: {e}")
            return None
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current status and metrics for an experiment."""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            config = experiment["config"]
            
            # Calculate runtime metrics
            users_assigned = len(experiment["user_assignments"])
            
            # Group by algorithm
            algorithm_counts = defaultdict(int)
            for user_id, algorithm in experiment["user_assignments"].items():
                algorithm_counts[algorithm] += 1
            
            # Calculate experiment duration
            if experiment["started_at"]:
                if experiment["ended_at"]:
                    duration = (experiment["ended_at"] - experiment["started_at"]).days
                else:
                    duration = (datetime.utcnow() - experiment["started_at"]).days
            else:
                duration = 0
            
            return {
                "experiment_id": experiment_id,
                "name": config.name,
                "status": experiment["status"].value,
                "created_at": experiment["created_at"].isoformat(),
                "started_at": experiment["started_at"].isoformat() if experiment["started_at"] else None,
                "ended_at": experiment["ended_at"].isoformat() if experiment["ended_at"] else None,
                "duration_days": duration,
                "users_assigned": users_assigned,
                "algorithm_distribution": dict(algorithm_counts),
                "target_user_types": config.target_user_types,
                "primary_metric": config.primary_metric,
                "min_sample_size": config.min_sample_size,
                "sample_size_reached": users_assigned >= config.min_sample_size
            }
            
        except Exception as e:
            logger.error(f"Error getting experiment status: {e}")
            return {}
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with their basic info."""
        try:
            experiments_list = []
            
            for exp_id, experiment in self.experiments.items():
                config = experiment["config"]
                
                experiments_list.append({
                    "experiment_id": exp_id,
                    "name": config.name,
                    "description": config.description,
                    "status": experiment["status"].value,
                    "algorithms": config.algorithms,
                    "created_at": experiment["created_at"].isoformat(),
                    "users_assigned": len(experiment["user_assignments"])
                })
            
            return experiments_list
            
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            return []