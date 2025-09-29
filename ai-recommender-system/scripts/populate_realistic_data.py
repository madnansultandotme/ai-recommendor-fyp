#!/usr/bin/env python3
"""Populate database with realistic data for better recommendations."""

import json
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.database import get_db_context
from database.models import User, Startup, Position

def load_data():
    """Load data from JSON file."""
    data_file = project_root / "data" / "realistic_data.json"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def clear_existing_data(db):
    """Clear existing data from database."""
    print("🗑️  Clearing existing data...")
    
    # Delete in reverse order due to foreign keys
    db.query(Position).delete()
    db.query(Startup).delete() 
    db.query(User).delete()
    
    db.commit()
    print("✅ Existing data cleared")

def populate_users(db, users_data):
    """Populate users table."""
    print("👥 Populating users...")
    
    for user_data in users_data:
        user = User(
            user_id=user_data["user_id"],
            email=user_data["email"],
            user_type=user_data["user_type"],
            profile_data=user_data["profile_data"]
        )
        db.add(user)
    
    db.commit()
    print(f"✅ Added {len(users_data)} users")

def populate_startups(db, startups_data):
    """Populate startups table."""
    print("🚀 Populating startups...")
    
    for startup_data in startups_data:
        startup = Startup(
            startup_id=startup_data["startup_id"],
            name=startup_data["name"],
            founder_id=startup_data["founder_id"],
            startup_metadata=startup_data["startup_metadata"]
        )
        db.add(startup)
    
    db.commit()
    print(f"✅ Added {len(startups_data)} startups")

def populate_positions(db, positions_data):
    """Populate positions table."""
    print("💼 Populating positions...")
    
    for position_data in positions_data:
        position = Position(
            position_id=position_data["position_id"],
            startup_id=position_data["startup_id"],
            title=position_data["title"],
            description=position_data["description"],
            requirements=position_data["requirements"],
            salary_range=position_data["salary_range"],
            equity=position_data["equity"],
            remote_ok=position_data["remote_ok"],
            location=position_data["location"]
        )
        db.add(position)
    
    db.commit()
    print(f"✅ Added {len(positions_data)} positions")

def verify_data(db):
    """Verify the populated data."""
    print("\n🔍 Verifying populated data...")
    
    user_count = db.query(User).count()
    startup_count = db.query(Startup).count()
    position_count = db.query(Position).count()
    
    print(f"  Users: {user_count}")
    print(f"  Startups: {startup_count}")
    print(f"  Positions: {position_count}")
    
    # Show some sample data
    print("\n📋 Sample data:")
    
    sample_users = db.query(User).limit(3).all()
    for user in sample_users:
        name = user.profile_data.get('name', 'No name') if user.profile_data else 'No name'
        print(f"  👤 {user.user_type.title()}: {name} (ID: {user.user_id})")
    
    sample_startups = db.query(Startup).limit(3).all()
    for startup in sample_startups:
        industry = ", ".join(startup.startup_metadata.get('industry', [])[:2]) if startup.startup_metadata else 'Unknown'
        print(f"  🚀 Startup: {startup.name} ({industry})")
    
    sample_positions = db.query(Position).limit(3).all()
    for position in sample_positions:
        startup_name = db.query(Startup).filter(Startup.startup_id == position.startup_id).first().name
        print(f"  💼 Position: {position.title} at {startup_name}")

def main():
    """Main function to populate database."""
    print("🚀 Starting database population with realistic data...\n")
    
    try:
        # Load data
        data = load_data()
        
        # Populate database
        with get_db_context() as db:
            # Clear existing data
            clear_existing_data(db)
            
            # Populate with new data
            populate_users(db, data["users"])
            populate_startups(db, data["startups"])
            populate_positions(db, data["positions"])
            
            # Verify
            verify_data(db)
        
        print(f"\n🎉 Database successfully populated with realistic data!")
        print("📊 Benefits of the new data:")
        print("  • Proper startup names (CodeFlow AI, HealthLink Pro, etc.)")
        print("  • Detailed user profiles with skills and preferences")
        print("  • Rich startup metadata with industry, stage, funding")
        print("  • Comprehensive job positions with requirements")
        print("  • Realistic data for better recommendation scoring")
        
        print(f"\n🔗 You can now test the demo at: http://localhost:8000/demo")
        
    except Exception as e:
        print(f"❌ Error populating database: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)