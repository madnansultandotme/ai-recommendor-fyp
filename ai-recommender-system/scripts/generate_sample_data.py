"""
Generate sample data for the AI recommender system.
This script creates realistic sample data for testing and development.
"""

import random
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta
from database.database import get_db_context
from database.models import User, Startup, Position, ImplicitFeedback, ExplicitFeedback
from database.vector_store import vector_store
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample data pools
INDUSTRIES = [
    "AI/ML", "FinTech", "HealthTech", "EdTech", "E-commerce", "SaaS", "Blockchain", 
    "IoT", "Cybersecurity", "CleanTech", "Gaming", "Social Media", "DeepTech",
    "AgriTech", "PropTech", "FoodTech", "Mobility", "AR/VR", "Analytics", "Cloud"
]

SKILLS = [
    "Python", "JavaScript", "React", "Node.js", "Go", "Java", "Rust", "TypeScript",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Data Science",
    "AWS", "Docker", "Kubernetes", "PostgreSQL", "MongoDB", "Redis", "GraphQL",
    "FastAPI", "Django", "Flask", "Vue.js", "Angular", "Swift", "Kotlin",
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Backend Development",
    "Frontend Development", "Full Stack", "DevOps", "Mobile Development", "UI/UX",
    "Product Management", "System Design", "Microservices", "API Design"
]

ROLES = [
    "Backend Developer", "Frontend Developer", "Full Stack Developer", "ML Engineer",
    "Data Scientist", "DevOps Engineer", "Mobile Developer", "Software Architect",
    "Tech Lead", "Senior Developer", "Junior Developer", "Product Engineer",
    "AI Researcher", "Data Engineer", "Platform Engineer", "Security Engineer"
]

STARTUP_STAGES = ["Pre-Seed", "Seed", "Series A", "Series B", "Series C", "Growth"]

INVESTMENT_STAGES = ["Pre-Seed", "Seed", "Series A", "Series B"]

LOCATIONS = [
    "San Francisco", "New York", "London", "Berlin", "Toronto", "Tel Aviv",
    "Singapore", "Austin", "Seattle", "Boston", "Amsterdam", "Paris", "Remote"
]

def generate_users(n_developers: int = 50, n_founders: int = 20, n_investors: int = 10) -> List[User]:
    """Generate sample users."""
    users = []
    
    # Generate developers
    for i in range(n_developers):
        skills = random.sample(SKILLS, random.randint(3, 8))
        preferred_industries = random.sample(INDUSTRIES, random.randint(1, 4))
        preferred_roles = random.sample(ROLES, random.randint(1, 3))
        
        user = User(
            email=f"developer{i+1}@example.com",
            user_type="developer",
            profile_data={
                "skills": skills,
                "preferred_industries": preferred_industries,
                "preferred_roles": preferred_roles,
                "experience_years": random.randint(1, 10),
                "location": random.choice(LOCATIONS)
            }
        )
        users.append(user)
    
    # Generate founders
    for i in range(n_founders):
        interested_industries = random.sample(INDUSTRIES, random.randint(1, 3))
        required_skills = random.sample(SKILLS, random.randint(2, 6))
        
        user = User(
            email=f"founder{i+1}@example.com",
            user_type="founder",
            profile_data={
                "interested_industries": interested_industries,
                "required_skills": required_skills,
                "location": random.choice(LOCATIONS),
                "previous_experience": random.choice([True, False])
            }
        )
        users.append(user)
    
    # Generate investors
    for i in range(n_investors):
        interested_industries = random.sample(INDUSTRIES, random.randint(2, 6))
        investment_stages = random.sample(INVESTMENT_STAGES, random.randint(1, 3))
        
        user = User(
            email=f"investor{i+1}@example.com",
            user_type="investor",
            profile_data={
                "interested_industries": interested_industries,
                "investment_stages": investment_stages,
                "check_size_min": random.randint(10000, 100000),
                "check_size_max": random.randint(200000, 2000000),
                "location": random.choice(LOCATIONS)
            }
        )
        users.append(user)
    
    return users

def generate_realistic_startups() -> List[Dict[str, Any]]:
    """Generate realistic startup data with names, descriptions, and metadata."""
    startups_data = [
        {
            "name": "CodeFlow AI",
            "description": "AI-powered platform for automated code review and software quality assurance. Helps development teams catch bugs early and improve code quality through intelligent analysis.",
            "industry": ["AI/ML", "SaaS"],
            "stage": "Seed",
            "tags": ["B2B", "Developer Tools", "AI"]
        },
        {
            "name": "HealthLink",
            "description": "Machine learning solution for personalized healthcare recommendations. Connects patients with the right treatments based on their medical history and genetic data.",
            "industry": ["HealthTech", "AI/ML"],
            "stage": "Series A",
            "tags": ["B2C", "Healthcare", "ML"]
        },
        {
            "name": "ChainTrust",
            "description": "Blockchain-based supply chain transparency platform. Enables brands to track products from source to consumer, ensuring authenticity and sustainability.",
            "industry": ["Blockchain", "SaaS"],
            "stage": "Pre-Seed",
            "tags": ["B2B", "Supply Chain", "Transparency"]
        },
        {
            "name": "EduVerse",
            "description": "Virtual reality platform for immersive learning experiences. Transforms education through interactive 3D environments for STEM subjects.",
            "industry": ["EdTech", "AR/VR"],
            "stage": "Seed",
            "tags": ["B2B", "B2C", "Education", "VR"]
        },
        {
            "name": "SwiftPay Global",
            "description": "Fintech solution for instant cross-border payments using cryptocurrency. Reduces transfer fees by 90% and settlement time to under 30 seconds.",
            "industry": ["FinTech", "Blockchain"],
            "stage": "Series B",
            "tags": ["B2B", "B2C", "Payments", "Crypto"]
        },
        {
            "name": "SmartCity IoT",
            "description": "IoT platform for intelligent urban infrastructure management. Optimizes traffic flow, energy usage, and waste collection through sensor networks.",
            "industry": ["IoT", "SaaS"],
            "stage": "Series A",
            "tags": ["B2B", "Government", "Smart City"]
        },
        {
            "name": "SkyDeliver",
            "description": "Autonomous drone delivery system with computer vision technology. Provides last-mile delivery solutions for e-commerce and medical supplies.",
            "industry": ["AI/ML", "Mobility"],
            "stage": "Seed",
            "tags": ["B2B", "Logistics", "Drones"]
        },
        {
            "name": "ChatGenius",
            "description": "NLP-powered customer service automation platform. Handles 80% of customer inquiries automatically while maintaining human-like conversations.",
            "industry": ["AI/ML", "SaaS"],
            "stage": "Series A",
            "tags": ["B2B", "Customer Service", "AI"]
        },
        {
            "name": "GreenPredict",
            "description": "Sustainable energy management system using predictive analytics. Helps businesses reduce energy costs by 40% through intelligent optimization.",
            "industry": ["CleanTech", "Analytics"],
            "stage": "Pre-Seed",
            "tags": ["B2B", "Sustainability", "Energy"]
        },
        {
            "name": "DevConnect",
            "description": "Social platform connecting remote developers globally. Features skill-based matching, project collaboration, and freelance opportunities.",
            "industry": ["Social Media", "SaaS"],
            "stage": "Seed",
            "tags": ["B2C", "Developer Community", "Remote Work"]
        },
        {
            "name": "CyberShield Pro",
            "description": "AI-driven cybersecurity threat detection and response system. Protects enterprise networks with real-time threat analysis and automated remediation.",
            "industry": ["Cybersecurity", "AI/ML"],
            "stage": "Series B",
            "tags": ["B2B", "Enterprise", "Security"]
        },
        {
            "name": "MindfulSpace",
            "description": "Digital mental health platform offering personalized therapy and wellness programs. Uses AI to match users with suitable mental health professionals.",
            "industry": ["HealthTech", "AI/ML"],
            "stage": "Series A",
            "tags": ["B2C", "Mental Health", "Therapy"]
        },
        {
            "name": "FarmSense",
            "description": "Smart agriculture platform using IoT sensors and machine learning. Optimizes crop yields while reducing water usage by 30% and pesticide use by 50%.",
            "industry": ["AgriTech", "IoT"],
            "stage": "Seed",
            "tags": ["B2B", "Agriculture", "Sustainability"]
        },
        {
            "name": "ShopMind",
            "description": "E-commerce personalization engine using deep learning. Increases conversion rates by 25% through intelligent product recommendations and dynamic pricing.",
            "industry": ["E-commerce", "AI/ML"],
            "stage": "Series A",
            "tags": ["B2B", "E-commerce", "Personalization"]
        },
        {
            "name": "DataVault Analytics",
            "description": "Cloud-native data analytics platform for enterprise clients. Processes petabytes of data with real-time insights and predictive modeling capabilities.",
            "industry": ["Analytics", "Cloud"],
            "stage": "Series B",
            "tags": ["B2B", "Enterprise", "Big Data"]
        },
        {
            "name": "NeoBank",
            "description": "Mobile-first banking solution for underbanked populations. Provides financial services through AI-powered credit scoring and micro-lending.",
            "industry": ["FinTech", "AI/ML"],
            "stage": "Series A",
            "tags": ["B2C", "Banking", "Financial Inclusion"]
        },
        {
            "name": "VirtualEstate",
            "description": "AR/VR platform for immersive virtual real estate tours. Reduces property viewing time by 60% while increasing buyer engagement.",
            "industry": ["PropTech", "AR/VR"],
            "stage": "Seed",
            "tags": ["B2B", "B2C", "Real Estate", "VR"]
        },
        {
            "name": "LegalMind AI",
            "description": "AI-powered legal document analysis and contract review platform. Reduces legal review time by 70% while improving accuracy and compliance.",
            "industry": ["AI/ML", "SaaS"],
            "stage": "Series A",
            "tags": ["B2B", "Legal Tech", "Document Analysis"]
        },
        {
            "name": "EcoRide",
            "description": "Sustainable transportation platform with electric vehicle fleet. Provides ride-sharing services with 100% renewable energy and carbon-neutral operations.",
            "industry": ["Mobility", "CleanTech"],
            "stage": "Series B",
            "tags": ["B2C", "Transportation", "Sustainability"]
        },
        {
            "name": "FoodFlow Optimizer",
            "description": "Food delivery optimization platform using machine learning algorithms. Reduces delivery time by 35% and operational costs by 20% for restaurants.",
            "industry": ["FoodTech", "AI/ML"],
            "stage": "Seed",
            "tags": ["B2B", "Food Delivery", "Optimization"]
        }
    ]
    return startups_data

def generate_startups(founders: List[User]) -> List[Startup]:
    """Generate sample startups."""
    descriptions = generate_startup_descriptions()
    startups = []
    
    for i, founder in enumerate(founders):
        if i >= len(descriptions):
            # Generate more descriptions if needed
            descriptions.extend(generate_startup_descriptions())
        
        founder_profile = founder.profile_data
        industries = founder_profile.get("interested_industries", random.sample(INDUSTRIES, 2))
        
        startup = Startup(
            founder_id=founder.user_id,
            name=f"TechCorp{i+1}",
            description=descriptions[i],
            startup_metadata={
                "industry": industries[:2],  # Pick first 2 industries
                "stage": random.choice(STARTUP_STAGES),
                "tags": random.sample(["B2B", "B2C", "SaaS", "Platform", "API", "Mobile", "Web"], random.randint(2, 4)),
                "location": founder_profile.get("location", random.choice(LOCATIONS)),
                "team_size": random.randint(2, 50),
                "funding_amount": random.randint(50000, 5000000),
                "required_skills": founder_profile.get("required_skills", random.sample(SKILLS, 4))
            }
        )
        startups.append(startup)
    
    return startups

def generate_positions(startups: List[Startup]) -> List[Position]:
    """Generate sample positions."""
    positions = []
    
    position_descriptions = [
        "We are looking for a passionate developer to join our growing team and help build the next generation of our platform.",
        "Join us in revolutionizing the industry with cutting-edge technology and innovative solutions.",
        "Seeking an experienced engineer to lead technical architecture and mentor junior developers.",
        "Help us scale our platform to handle millions of users with robust, efficient systems.",
        "Work on challenging problems in machine learning and artificial intelligence.",
        "Build beautiful, responsive user interfaces that delight our customers.",
        "Design and implement secure, scalable backend systems and APIs.",
        "Lead mobile development initiatives for iOS and Android platforms.",
        "Drive DevOps practices and cloud infrastructure optimization.",
        "Create data pipelines and analytics systems for business intelligence."
    ]
    
    for startup in startups:
        # Each startup has 1-4 positions
        n_positions = random.randint(1, 4)
        required_skills = startup.startup_metadata.get("required_skills", random.sample(SKILLS, 4))
        
        for i in range(n_positions):
            role = random.choice(ROLES)
            position_skills = random.sample(required_skills, random.randint(2, len(required_skills)))
            preferred_skills = random.sample(SKILLS, random.randint(1, 3))
            
            position = Position(
                startup_id=startup.startup_id,
                title=role,
                description=random.choice(position_descriptions),
                requirements={
                    "required_skills": position_skills,
                    "preferred_skills": preferred_skills,
                    "experience_years": random.randint(1, 8),
                    "remote_ok": random.choice([True, False]),
                    "salary_range": {
                        "min": random.randint(60000, 120000),
                        "max": random.randint(120000, 200000)
                    }
                }
            )
            positions.append(position)
    
    return positions

def generate_interactions(users: List[User], startups: List[Startup], positions: List[Position]):
    """Generate sample user interactions."""
    implicit_feedback = []
    explicit_feedback = []
    
    # Generate implicit feedback (views, clicks)
    for user in users:
        n_interactions = random.randint(5, 20)
        
        for _ in range(n_interactions):
            if user.user_type == "developer":
                # Developers interact with positions
                position = random.choice(positions)
                event_type = random.choice(["view", "click", "save"])
                
                feedback = ImplicitFeedback(
                    user_id=user.user_id,
                    item_type="position",
                    item_id=position.position_id,
                    event_type=event_type,
                    timestamp=datetime.now() - timedelta(days=random.randint(1, 30))
                )
                implicit_feedback.append(feedback)
            
            elif user.user_type == "investor":
                # Investors interact with startups
                startup = random.choice(startups)
                event_type = random.choice(["view", "click", "save"])
                
                feedback = ImplicitFeedback(
                    user_id=user.user_id,
                    item_type="startup",
                    item_id=startup.startup_id,
                    event_type=event_type,
                    timestamp=datetime.now() - timedelta(days=random.randint(1, 30))
                )
                implicit_feedback.append(feedback)
    
    # Generate explicit feedback (likes, passes)
    for user in users:
        n_explicit = random.randint(2, 10)
        
        for _ in range(n_explicit):
            feedback_type = random.choice(["like", "pass", "super_like"])
            
            if user.user_type == "developer":
                position = random.choice(positions)
                feedback = ExplicitFeedback(
                    user_id=user.user_id,
                    item_type="position",
                    item_id=position.position_id,
                    feedback_type=feedback_type,
                    timestamp=datetime.now() - timedelta(days=random.randint(1, 30))
                )
                explicit_feedback.append(feedback)
            
            elif user.user_type == "investor":
                startup = random.choice(startups)
                feedback = ExplicitFeedback(
                    user_id=user.user_id,
                    item_type="startup",
                    item_id=startup.startup_id,
                    feedback_type=feedback_type,
                    timestamp=datetime.now() - timedelta(days=random.randint(1, 30))
                )
                explicit_feedback.append(feedback)
    
    return implicit_feedback, explicit_feedback

def populate_vector_store(users: List[User], startups: List[Startup], positions: List[Position]):
    """Populate vector store with embeddings."""
    logger.info("Populating vector store...")
    
    # Add startup embeddings
    for startup in startups:
        try:
            vector_store.add_startup_embedding(
                startup_id=startup.startup_id,
                description=startup.description,
                metadata=startup.startup_metadata
            )
        except Exception as e:
            logger.error(f"Error adding startup embedding: {e}")
    
    # Add position embeddings
    for position in positions:
        try:
            # Get startup metadata for context
            startup = next((s for s in startups if s.startup_id == position.startup_id), None)
            startup_metadata = startup.startup_metadata if startup else {}
            
            vector_store.add_position_embedding(
                position_id=position.position_id,
                title=position.title,
                description=position.description,
                requirements=position.requirements,
                startup_metadata=startup_metadata
            )
        except Exception as e:
            logger.error(f"Error adding position embedding: {e}")
    
    # Add user skills embeddings (for developers)
    for user in users:
        if user.user_type == "developer":
            try:
                profile_data = user.profile_data
                skills = profile_data.get("skills", [])
                preferences = {
                    "preferred_industries": profile_data.get("preferred_industries", []),
                    "preferred_roles": profile_data.get("preferred_roles", [])
                }
                
                vector_store.add_user_skills_embedding(
                    user_id=user.user_id,
                    skills=skills,
                    preferences=preferences
                )
            except Exception as e:
                logger.error(f"Error adding user skills embedding: {e}")

def main():
    """Main function to generate all sample data."""
    logger.info("Starting sample data generation...")
    
    try:
        with get_db_context() as db:
            # Clear existing data
            logger.info("Clearing existing data...")
            db.query(ExplicitFeedback).delete()
            db.query(ImplicitFeedback).delete()
            db.query(Position).delete()
            db.query(Startup).delete()
            db.query(User).delete()
            
            # Generate users
            logger.info("Generating users...")
            users = generate_users(n_developers=50, n_founders=20, n_investors=10)
            db.add_all(users)
            db.flush()  # Get user IDs
            
            # Generate startups (only for founders)
            logger.info("Generating startups...")
            founders = [u for u in users if u.user_type == "founder"]
            startups = generate_startups(founders)
            db.add_all(startups)
            db.flush()  # Get startup IDs
            
            # Generate positions
            logger.info("Generating positions...")
            positions = generate_positions(startups)
            db.add_all(positions)
            db.flush()  # Get position IDs
            
            # Generate interactions
            logger.info("Generating user interactions...")
            implicit_feedback, explicit_feedback = generate_interactions(users, startups, positions)
            db.add_all(implicit_feedback)
            db.add_all(explicit_feedback)
            
            # Commit all changes
            db.commit()
            logger.info(f"Generated {len(users)} users, {len(startups)} startups, {len(positions)} positions")
            logger.info(f"Generated {len(implicit_feedback)} implicit and {len(explicit_feedback)} explicit feedback records")
    
        # Populate vector store
        populate_vector_store(users, startups, positions)
        
        logger.info("Sample data generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        raise

if __name__ == "__main__":
    main()