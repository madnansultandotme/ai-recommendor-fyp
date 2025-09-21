from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # Database Configuration
    database_url_postgres: str = Field(
        default="postgresql://user:password@localhost/ai_recommender",
        env="DATABASE_URL_POSTGRES"
    )
    database_url_sqlite: str = Field(
        default="sqlite:///./ai_recommender.db",
        env="DATABASE_URL_SQLITE"
    )
    use_postgres: bool = Field(default=False, env="USE_POSTGRES")
    
    # Vector Database Configuration
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        env="CHROMA_PERSIST_DIRECTORY"
    )
    vector_dimension: int = Field(default=384, env="VECTOR_DIMENSION")
    
    # ML Models Configuration
    sentence_transformer_model: str = Field(
        default="all-MiniLM-L6-v2",
        env="SENTENCE_TRANSFORMER_MODEL"
    )
    huggingface_cache_dir: str = Field(
        default="./models/cache",
        env="HUGGINGFACE_CACHE_DIR"
    )
    
    # API Configuration
    api_v1_str: str = Field(default="/api/v1", env="API_V1_STR")
    secret_key: str = Field(
        default="your-secret-key-here",
        env="SECRET_KEY"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    
    # Application Configuration
    project_name: str = Field(
        default="AI Recommender System",
        env="PROJECT_NAME"
    )
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Training Configuration
    training_batch_size: int = Field(default=32, env="TRAINING_BATCH_SIZE")
    learning_rate: float = Field(default=0.001, env="LEARNING_RATE")
    epochs: int = Field(default=10, env="EPOCHS")
    model_save_path: str = Field(
        default="./models/saved",
        env="MODEL_SAVE_PATH"
    )
    
    # Vector Search Configuration
    top_k_recommendations: int = Field(
        default=10,
        env="TOP_K_RECOMMENDATIONS"
    )
    similarity_threshold: float = Field(
        default=0.7,
        env="SIMILARITY_THRESHOLD"
    )
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_telemetry: bool = Field(default=False, env="ENABLE_TELEMETRY")
    
    @property
    def database_url(self) -> str:
        """Get the active database URL based on configuration."""
        return self.database_url_postgres if self.use_postgres else self.database_url_sqlite
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [
            self.chroma_persist_directory,
            self.huggingface_cache_dir,
            self.model_save_path,
            "./data",
            "./logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Create necessary directories
settings.create_directories()