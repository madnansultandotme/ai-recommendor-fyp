#!/usr/bin/env python3
"""
Run script for the AI Recommender System.
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Run the FastAPI server."""
    import uvicorn
    from app.config import settings
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload and settings.is_development,
        log_level=settings.log_level.lower()
    )

def generate_data():
    """Generate sample data."""
    from scripts.generate_sample_data import main as generate_main
    generate_main()

def generate_meaningful_data():
    """Generate meaningful, realistic sample data."""
    from scripts.generate_meaningful_data import main as generate_meaningful_main
    generate_meaningful_main()

def reset_database():
    """Reset the database."""
    from database.database import reset_database as reset_db
    reset_db()
    print("Database reset successfully!")

def check_health():
    """Check system health."""
    from database.database import check_db_health
    from database.vector_store import vector_store
    
    print("Checking system health...")
    
    # Check database
    db_healthy = check_db_health()
    print(f"Database: {'✓ Healthy' if db_healthy else '✗ Unhealthy'}")
    
    # Check vector store collections
    try:
        startup_count = vector_store.get_collection_count("startups")
        position_count = vector_store.get_collection_count("positions")
        user_count = vector_store.get_collection_count("user_skills")
        
        print(f"Vector Store:")
        print(f"  Startups: {startup_count}")
        print(f"  Positions: {position_count}")
        print(f"  User Skills: {user_count}")
        
        vector_healthy = startup_count > 0 and position_count > 0
        print(f"  Status: {'✓ Healthy' if vector_healthy else '✗ Unhealthy'}")
    except Exception as e:
        print(f"Vector Store: ✗ Error - {e}")
    
    print("\nSystem status:", "✓ Ready" if db_healthy else "✗ Not Ready")

def main():
    parser = argparse.ArgumentParser(description="AI Recommender System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    server_parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    # Data commands
    subparsers.add_parser("generate-data", help="Generate sample data")
    subparsers.add_parser("generate-meaningful-data", help="Generate meaningful, realistic sample data")
    
    # Database command
    subparsers.add_parser("reset-db", help="Reset the database")
    
    # Health command
    subparsers.add_parser("health", help="Check system health")
    
    args = parser.parse_args()
    
    if args.command == "server":
        print("Starting AI Recommender System server...")
        run_server(
            host=args.host,
            port=args.port,
            reload=not args.no_reload
        )
    elif args.command == "generate-data":
        print("Generating sample data...")
        generate_data()
    elif args.command == "generate-meaningful-data":
        print("Generating meaningful, realistic sample data...")
        generate_meaningful_data()
    elif args.command == "reset-db":
        print("Resetting database...")
        reset_database()
    elif args.command == "health":
        check_health()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()