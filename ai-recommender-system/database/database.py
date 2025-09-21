from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
import logging
from contextlib import contextmanager

from app.config import settings
from database.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database manager for handling connections and sessions."""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.setup_database()
    
    def setup_database(self):
        """Setup database engine and session factory."""
        try:
            if settings.use_postgres:
                # PostgreSQL setup
                self.engine = create_engine(
                    settings.database_url_postgres,
                    echo=settings.is_development,
                    pool_pre_ping=True,
                )
                logger.info("Using PostgreSQL database")
            else:
                # SQLite setup
                self.engine = create_engine(
                    settings.database_url_sqlite,
                    echo=settings.is_development,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool,
                )
                logger.info("Using SQLite database")
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create tables
            self.create_tables()
            
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            raise
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        db = self.SessionLocal()
        try:
            yield db
        except Exception as e:
            logger.error(f"Database session error: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    @contextmanager
    def get_session_context(self):
        """Context manager for database sessions."""
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception as e:
            logger.error(f"Database transaction error: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    def reset_database(self):
        """Reset database by dropping and recreating all tables."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.get_session_context() as db:
                from sqlalchemy import text
                db.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()

# Dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency to get database session."""
    yield from db_manager.get_session()


# Convenience functions
def get_db_session() -> Generator[Session, None, None]:
    """Get database session generator."""
    return db_manager.get_session()


def get_db_context():
    """Get database session context manager."""
    return db_manager.get_session_context()


def reset_database():
    """Reset the database."""
    db_manager.reset_database()


def check_db_health() -> bool:
    """Check database health."""
    return db_manager.health_check()