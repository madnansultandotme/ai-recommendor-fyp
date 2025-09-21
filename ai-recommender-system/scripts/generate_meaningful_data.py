"""
Generate meaningful, realistic sample data for the AI recommender system.
This script creates detailed, realistic profiles for developers, investors, and entrepreneurs.
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

# Realistic Developer Profiles
DEVELOPERS = [
    {
        "name": "Sarah Chen",
        "email": "sarah.chen@gmail.com",
        "skills": ["Python", "Machine Learning", "TensorFlow", "AWS", "Docker"],
        "experience_years": 5,
        "location": "San Francisco",
        "preferred_industries": ["AI/ML", "HealthTech"],
        "preferred_roles": ["ML Engineer", "Data Scientist"],
        "bio": "ML Engineer passionate about healthcare applications. Built predictive models for patient outcomes."
    },
    {
        "name": "Marcus Johnson",
        "email": "marcus.j.dev@gmail.com",
        "skills": ["JavaScript", "React", "Node.js", "GraphQL", "MongoDB"],
        "experience_years": 3,
        "location": "New York",
        "preferred_industries": ["FinTech", "E-commerce"],
        "preferred_roles": ["Frontend Developer", "Full Stack Developer"],
        "bio": "Frontend specialist with fintech experience. Love creating intuitive user experiences."
    },
    {
        "name": "Priya Patel",
        "email": "priya.patel.dev@gmail.com",
        "skills": ["Java", "Spring Boot", "Kubernetes", "PostgreSQL", "Microservices"],
        "experience_years": 7,
        "location": "Toronto",
        "preferred_industries": ["SaaS", "Enterprise"],
        "preferred_roles": ["Backend Developer", "Tech Lead"],
        "bio": "Senior backend engineer specializing in scalable microservices architecture."
    },
    {
        "name": "Alex Rodriguez",
        "email": "alex.rodriguez.code@gmail.com",
        "skills": ["Go", "Docker", "Kubernetes", "AWS", "DevOps"],
        "experience_years": 6,
        "location": "Austin",
        "preferred_industries": ["Cloud", "DevOps"],
        "preferred_roles": ["DevOps Engineer", "Platform Engineer"],
        "bio": "DevOps engineer focused on cloud infrastructure and CI/CD automation."
    },
    {
        "name": "Emma Thompson",
        "email": "emma.thompson.dev@gmail.com",
        "skills": ["Swift", "iOS", "React Native", "Firebase"],
        "experience_years": 4,
        "location": "London",
        "preferred_industries": ["Mobile", "Consumer Apps"],
        "preferred_roles": ["Mobile Developer", "iOS Developer"],
        "bio": "iOS developer with 50+ App Store apps. Specialist in consumer mobile experiences."
    },
    {
        "name": "David Kim",
        "email": "david.kim.security@gmail.com",
        "skills": ["Cybersecurity", "Penetration Testing", "Python", "Network Security"],
        "experience_years": 8,
        "location": "Seattle",
        "preferred_industries": ["Cybersecurity", "FinTech"],
        "preferred_roles": ["Security Engineer", "Cybersecurity Analyst"],
        "bio": "Cybersecurity expert with banking sector experience. Ethical hacker and security consultant."
    },
    {
        "name": "Lisa Wang",
        "email": "lisa.wang.data@gmail.com",
        "skills": ["Python", "Data Science", "Pandas", "Scikit-learn", "SQL"],
        "experience_years": 4,
        "location": "Berlin",
        "preferred_industries": ["Analytics", "E-commerce"],
        "preferred_roles": ["Data Scientist", "Data Analyst"],
        "bio": "Data scientist focused on customer analytics and recommendation systems."
    },
    {
        "name": "James Miller",
        "email": "james.miller.fullstack@gmail.com",
        "skills": ["TypeScript", "Vue.js", "Node.js", "PostgreSQL", "Docker"],
        "experience_years": 5,
        "location": "Remote",
        "preferred_industries": ["SaaS", "Startups"],
        "preferred_roles": ["Full Stack Developer", "Senior Developer"],
        "bio": "Full-stack developer with startup experience. Remote work advocate and open source contributor."
    },
    {
        "name": "Aisha Hassan",
        "email": "aisha.hassan.blockchain@gmail.com",
        "skills": ["Solidity", "Blockchain", "Web3", "JavaScript", "Smart Contracts"],
        "experience_years": 3,
        "location": "Dubai",
        "preferred_industries": ["Blockchain", "FinTech"],
        "preferred_roles": ["Blockchain Developer", "Web3 Developer"],
        "bio": "Blockchain developer specializing in DeFi applications and smart contract security."
    },
    {
        "name": "Tom Anderson",
        "email": "tom.anderson.ai@gmail.com",
        "skills": ["Python", "PyTorch", "Computer Vision", "Deep Learning", "MLOps"],
        "experience_years": 6,
        "location": "Boston",
        "preferred_industries": ["AI/ML", "Autonomous Vehicles"],
        "preferred_roles": ["AI Research Engineer", "Computer Vision Engineer"],
        "bio": "AI researcher focused on computer vision for autonomous systems. PhD in Computer Science."
    }
]

# Realistic Investor Profiles
INVESTORS = [
    {
        "name": "Jennifer Walsh",
        "email": "j.walsh@techventures.com",
        "firm": "Tech Ventures Capital",
        "interested_industries": ["AI/ML", "HealthTech", "SaaS"],
        "investment_stages": ["Seed", "Series A"],
        "check_size_min": 500000,
        "check_size_max": 3000000,
        "location": "San Francisco",
        "bio": "Partner at TVC focusing on AI-driven healthcare solutions. 15+ years investment experience."
    },
    {
        "name": "Michael Chen",
        "email": "m.chen@innovatefund.com",
        "firm": "Innovate Fund",
        "interested_industries": ["FinTech", "Blockchain", "Enterprise"],
        "investment_stages": ["Pre-Seed", "Seed"],
        "check_size_min": 250000,
        "check_size_max": 1500000,
        "location": "New York",
        "bio": "Early-stage investor with fintech background. Former Goldman Sachs VP, focuses on B2B."
    },
    {
        "name": "Sarah Martinez",
        "email": "s.martinez@greenfuture.vc",
        "firm": "Green Future VC",
        "interested_industries": ["CleanTech", "Sustainability", "AgriTech"],
        "investment_stages": ["Seed", "Series A"],
        "check_size_min": 1000000,
        "check_size_max": 5000000,
        "location": "Austin",
        "bio": "Impact investor focused on climate solutions. Former Tesla executive, cleantech specialist."
    },
    {
        "name": "Robert Kim",
        "email": "r.kim@digitalhorizons.com",
        "firm": "Digital Horizons",
        "interested_industries": ["E-commerce", "Consumer Apps", "Social Media"],
        "investment_stages": ["Series A", "Series B"],
        "check_size_min": 2000000,
        "check_size_max": 10000000,
        "location": "Los Angeles",
        "bio": "Growth-stage investor specializing in consumer internet. Ex-Google product manager."
    },
    {
        "name": "Dr. Emily Foster",
        "email": "e.foster@healthinnovate.vc",
        "firm": "Health Innovate VC",
        "interested_industries": ["HealthTech", "BioTech", "MedTech"],
        "investment_stages": ["Seed", "Series A", "Series B"],
        "check_size_min": 1500000,
        "check_size_max": 8000000,
        "location": "Boston",
        "bio": "MD/PhD investor focused on digital health innovations. Former CMO at major hospital system."
    }
]

# Realistic Entrepreneur/Founder Profiles
ENTREPRENEURS = [
    {
        "name": "Alex Rivera",
        "email": "alex@codeflowai.com",
        "startup": {
            "name": "CodeFlow AI",
            "description": "AI-powered code review platform that helps development teams catch bugs early and improve code quality through intelligent static analysis and machine learning.",
            "industry": ["AI/ML", "Developer Tools"],
            "stage": "Seed",
            "team_size": 8,
            "funding_amount": 2500000,
            "location": "San Francisco"
        },
        "background": "Former Google engineer with 10 years experience. Built developer tools at scale.",
        "required_skills": ["Python", "Machine Learning", "Backend Development", "DevOps"]
    },
    {
        "name": "Dr. Maya Patel",
        "email": "maya@healthlink.com",
        "startup": {
            "name": "HealthLink",
            "description": "Personalized healthcare platform using AI to connect patients with optimal treatments based on medical history, genetic data, and lifestyle factors.",
            "industry": ["HealthTech", "AI/ML"],
            "stage": "Series A",
            "team_size": 25,
            "funding_amount": 8000000,
            "location": "Boston"
        },
        "background": "MD with Stanford MBA. Former healthcare consultant at McKinsey.",
        "required_skills": ["Machine Learning", "Healthcare", "Data Science", "Regulatory Compliance"]
    },
    {
        "name": "James Wilson",
        "email": "james@chaintrustio.com",
        "startup": {
            "name": "ChainTrust",
            "description": "Blockchain-based supply chain transparency platform enabling brands to track products from source to consumer, ensuring authenticity and sustainability.",
            "industry": ["Blockchain", "Supply Chain"],
            "stage": "Pre-Seed",
            "team_size": 5,
            "funding_amount": 500000,
            "location": "Austin"
        },
        "background": "Supply chain expert from Amazon. Blockchain enthusiast and sustainability advocate.",
        "required_skills": ["Blockchain", "Solidity", "Supply Chain", "Web3"]
    },
    {
        "name": "Sofia Andersen",
        "email": "sofia@eduverse.com",
        "startup": {
            "name": "EduVerse",
            "description": "Virtual reality education platform creating immersive 3D learning experiences for STEM subjects, making complex concepts tangible and engaging.",
            "industry": ["EdTech", "VR/AR"],
            "stage": "Seed",
            "team_size": 12,
            "funding_amount": 3200000,
            "location": "Berlin"
        },
        "background": "Former educator turned tech entrepreneur. MIT Media Lab alum with VR expertise.",
        "required_skills": ["Unity", "VR Development", "Education", "3D Graphics"]
    },
    {
        "name": "Rajesh Kumar",
        "email": "raj@swiftpayglobal.com",
        "startup": {
            "name": "SwiftPay Global",
            "description": "Fintech platform for instant cross-border payments using cryptocurrency, reducing transfer fees by 90% and settlement time to under 30 seconds.",
            "industry": ["FinTech", "Cryptocurrency"],
            "stage": "Series B",
            "team_size": 45,
            "funding_amount": 15000000,
            "location": "London"
        },
        "background": "Former JPMorgan payments executive. Cryptocurrency pioneer with regulatory expertise.",
        "required_skills": ["Blockchain", "FinTech", "Regulatory", "Security"]
    },
    {
        "name": "Lisa Chang",
        "email": "lisa@smartcityiot.com",
        "startup": {
            "name": "SmartCity IoT",
            "description": "IoT platform for intelligent urban infrastructure management, optimizing traffic flow, energy usage, and waste collection through sensor networks.",
            "industry": ["IoT", "Smart Cities"],
            "stage": "Series A",
            "team_size": 20,
            "funding_amount": 6000000,
            "location": "Singapore"
        },
        "background": "Urban planning expert with IoT background. Former Cisco systems architect.",
        "required_skills": ["IoT", "Embedded Systems", "Data Analytics", "Urban Planning"]
    },
    {
        "name": "Mark Thompson",
        "email": "mark@skydeliver.com",
        "startup": {
            "name": "SkyDeliver",
            "description": "Autonomous drone delivery system with computer vision technology providing last-mile delivery solutions for e-commerce and medical supplies.",
            "industry": ["Logistics", "Drones"],
            "stage": "Seed",
            "team_size": 15,
            "funding_amount": 4000000,
            "location": "Seattle"
        },
        "background": "Aerospace engineer from Boeing. Drone technology specialist with pilot license.",
        "required_skills": ["Computer Vision", "Robotics", "Aerospace", "Machine Learning"]
    },
    {
        "name": "Natasha Volkov",
        "email": "natasha@chatgenius.ai",
        "startup": {
            "name": "ChatGenius",
            "description": "NLP-powered customer service automation platform handling 80% of customer inquiries automatically while maintaining human-like conversations.",
            "industry": ["AI/ML", "Customer Service"],
            "stage": "Series A",
            "team_size": 18,
            "funding_amount": 7500000,
            "location": "Toronto"
        },
        "background": "NLP researcher from University of Toronto. Former Shopify ML engineer.",
        "required_skills": ["NLP", "Machine Learning", "Customer Service", "Chatbots"]
    },
    {
        "name": "David Green",
        "email": "david@greenpredict.com",
        "startup": {
            "name": "GreenPredict",
            "description": "Sustainable energy management system using predictive analytics to help businesses reduce energy costs by 40% through intelligent optimization.",
            "industry": ["CleanTech", "Energy"],
            "stage": "Pre-Seed",
            "team_size": 6,
            "funding_amount": 750000,
            "location": "Denver"
        },
        "background": "Environmental engineer with energy sector experience. Former Tesla energy division.",
        "required_skills": ["Data Science", "Energy Systems", "Sustainability", "Predictive Analytics"]
    },
    {
        "name": "Kevin Park",
        "email": "kevin@devconnect.io",
        "startup": {
            "name": "DevConnect",
            "description": "Social platform connecting remote developers globally with skill-based matching, project collaboration tools, and freelance opportunities.",
            "industry": ["Developer Community", "Remote Work"],
            "stage": "Seed",
            "team_size": 10,
            "funding_amount": 2000000,
            "location": "Remote"
        },
        "background": "Former GitHub product manager. Remote work advocate and open source contributor.",
        "required_skills": ["Community Building", "Social Platforms", "Developer Tools", "Remote Work"]
    }
]

# Job Positions with realistic titles and descriptions
POSITIONS_DATA = [
    {
        "title": "Senior Machine Learning Engineer",
        "description": "Join our AI team to build next-generation ML models for healthcare diagnostics. Work with medical data to improve patient outcomes through predictive analytics.",
        "required_skills": ["Python", "Machine Learning", "TensorFlow", "Healthcare"],
        "preferred_skills": ["MLOps", "AWS", "Docker"],
        "experience_years": 5,
        "salary_range": {"min": 140000, "max": 180000}
    },
    {
        "title": "Frontend React Developer",
        "description": "Build beautiful, responsive user interfaces for our fintech platform. Work closely with designers to create intuitive financial tools for millions of users.",
        "required_skills": ["React", "JavaScript", "CSS", "TypeScript"],
        "preferred_skills": ["Next.js", "GraphQL", "Testing"],
        "experience_years": 3,
        "salary_range": {"min": 90000, "max": 130000}
    },
    {
        "title": "DevOps Platform Engineer",
        "description": "Scale our infrastructure to handle enterprise workloads. Build CI/CD pipelines and manage Kubernetes clusters across multiple cloud providers.",
        "required_skills": ["Kubernetes", "Docker", "AWS", "CI/CD"],
        "preferred_skills": ["Terraform", "Monitoring", "Security"],
        "experience_years": 4,
        "salary_range": {"min": 120000, "max": 160000}
    },
    {
        "title": "Full Stack Developer",
        "description": "Work across our entire tech stack to deliver features end-to-end. Collaborate with product teams to build scalable web applications.",
        "required_skills": ["JavaScript", "Node.js", "React", "PostgreSQL"],
        "preferred_skills": ["TypeScript", "GraphQL", "Redis"],
        "experience_years": 4,
        "salary_range": {"min": 110000, "max": 150000}
    },
    {
        "title": "Blockchain Developer",
        "description": "Design and implement smart contracts for our DeFi platform. Work on cutting-edge blockchain technology to revolutionize financial services.",
        "required_skills": ["Solidity", "Blockchain", "Web3", "Smart Contracts"],
        "preferred_skills": ["DeFi", "Security Auditing", "JavaScript"],
        "experience_years": 2,
        "salary_range": {"min": 100000, "max": 140000}
    },
    {
        "title": "iOS Mobile Developer",
        "description": "Create exceptional mobile experiences for our consumer app. Work with Swift and modern iOS frameworks to deliver polished applications.",
        "required_skills": ["Swift", "iOS", "UIKit", "Core Data"],
        "preferred_skills": ["SwiftUI", "Combine", "Testing"],
        "experience_years": 3,
        "salary_range": {"min": 105000, "max": 145000}
    },
    {
        "title": "Data Scientist",
        "description": "Analyze customer behavior to drive business insights. Build recommendation systems and predictive models using large datasets.",
        "required_skills": ["Python", "Data Science", "Machine Learning", "SQL"],
        "preferred_skills": ["Deep Learning", "Statistics", "Visualization"],
        "experience_years": 4,
        "salary_range": {"min": 115000, "max": 155000}
    },
    {
        "title": "Backend Engineer",
        "description": "Build robust APIs and microservices to power our platform. Work with high-scale distributed systems and modern cloud architecture.",
        "required_skills": ["Java", "Spring Boot", "PostgreSQL", "Microservices"],
        "preferred_skills": ["Kafka", "Redis", "Monitoring"],
        "experience_years": 5,
        "salary_range": {"min": 125000, "max": 165000}
    },
    {
        "title": "Security Engineer",
        "description": "Protect our platform and customer data from cyber threats. Implement security best practices and conduct penetration testing.",
        "required_skills": ["Cybersecurity", "Penetration Testing", "Network Security"],
        "preferred_skills": ["Cloud Security", "Compliance", "Incident Response"],
        "experience_years": 6,
        "salary_range": {"min": 135000, "max": 175000}
    },
    {
        "title": "AI Research Engineer",
        "description": "Push the boundaries of artificial intelligence research. Work on computer vision and natural language processing for autonomous systems.",
        "required_skills": ["Python", "Deep Learning", "Computer Vision", "Research"],
        "preferred_skills": ["PyTorch", "Publications", "PhD"],
        "experience_years": 5,
        "salary_range": {"min": 150000, "max": 200000}
    }
]

def generate_meaningful_users() -> List[User]:
    """Generate meaningful user profiles."""
    users = []
    
    # Create developers
    for i, dev_data in enumerate(DEVELOPERS):
        user = User(
            email=dev_data["email"],
            user_type="developer",
            profile_data={
                "name": dev_data["name"],
                "skills": dev_data["skills"],
                "experience_years": dev_data["experience_years"],
                "location": dev_data["location"],
                "preferred_industries": dev_data["preferred_industries"],
                "preferred_roles": dev_data["preferred_roles"],
                "bio": dev_data["bio"],
                "github_url": f"https://github.com/{dev_data['name'].lower().replace(' ', '')}",
                "linkedin_url": f"https://linkedin.com/in/{dev_data['name'].lower().replace(' ', '-')}"
            }
        )
        users.append(user)
    
    # Create entrepreneurs/founders
    for i, ent_data in enumerate(ENTREPRENEURS):
        user = User(
            email=ent_data["email"],
            user_type="founder",
            profile_data={
                "name": ent_data["name"],
                "background": ent_data["background"],
                "required_skills": ent_data["required_skills"],
                "location": ent_data["startup"]["location"],
                "previous_companies": ["Previous startup experience"],
                "linkedin_url": f"https://linkedin.com/in/{ent_data['name'].lower().replace(' ', '-')}",
                "company": ent_data["startup"]["name"]
            }
        )
        users.append(user)
    
    # Create investors
    for i, inv_data in enumerate(INVESTORS):
        user = User(
            email=inv_data["email"],
            user_type="investor",
            profile_data={
                "name": inv_data["name"],
                "firm": inv_data["firm"],
                "interested_industries": inv_data["interested_industries"],
                "investment_stages": inv_data["investment_stages"],
                "check_size_min": inv_data["check_size_min"],
                "check_size_max": inv_data["check_size_max"],
                "location": inv_data["location"],
                "bio": inv_data["bio"],
                "linkedin_url": f"https://linkedin.com/in/{inv_data['name'].lower().replace(' ', '-')}"
            }
        )
        users.append(user)
    
    return users

def generate_meaningful_startups(founders: List[User]) -> List[Startup]:
    """Generate meaningful startup profiles."""
    startups = []
    
    founder_index = 0
    for i, ent_data in enumerate(ENTREPRENEURS):
        # Find the corresponding founder user
        founder = None
        for user in founders:
            if user.profile_data.get("name") == ent_data["name"]:
                founder = user
                break
        
        if not founder:
            continue
            
        startup_data = ent_data["startup"]
        startup = Startup(
            founder_id=founder.user_id,
            name=startup_data["name"],
            description=startup_data["description"],
            startup_metadata={
                "industry": startup_data["industry"],
                "stage": startup_data["stage"],
                "team_size": startup_data["team_size"],
                "funding_amount": startup_data["funding_amount"],
                "location": startup_data["location"],
                "required_skills": ent_data["required_skills"],
                "website": f"https://{startup_data['name'].lower().replace(' ', '')}.com",
                "founded_year": random.randint(2018, 2023)
            }
        )
        startups.append(startup)
    
    return startups

def generate_meaningful_positions(startups: List[Startup]) -> List[Position]:
    """Generate meaningful job positions."""
    positions = []
    
    for startup in startups:
        # Each startup gets 1-3 relevant positions
        num_positions = random.randint(1, 3)
        startup_skills = startup.startup_metadata.get("required_skills", [])
        
        for i in range(num_positions):
            position_data = random.choice(POSITIONS_DATA)
            
            # Customize position based on startup
            position = Position(
                startup_id=startup.startup_id,
                title=position_data["title"],
                description=f"{position_data['description']} At {startup.name}, {startup.description[:100]}...",
                requirements={
                    "required_skills": position_data["required_skills"],
                    "preferred_skills": position_data["preferred_skills"],
                    "experience_years": position_data["experience_years"],
                    "remote_ok": startup.startup_metadata.get("location") == "Remote" or random.choice([True, False]),
                    "salary_range": position_data["salary_range"],
                    "equity": random.choice([True, False]),
                    "benefits": ["Health Insurance", "401k", "Flexible PTO", "Learning Budget"]
                }
            )
            positions.append(position)
    
    return positions

def generate_realistic_interactions(users: List[User], startups: List[Startup], positions: List[Position]):
    """Generate realistic user interactions based on profiles."""
    implicit_feedback = []
    explicit_feedback = []
    
    # Developers interact with relevant positions
    developers = [u for u in users if u.user_type == "developer"]
    for dev in developers:
        dev_skills = set(dev.profile_data.get("skills", []))
        preferred_industries = dev.profile_data.get("preferred_industries", [])
        
        # Find positions that match developer's skills/interests
        relevant_positions = []
        for pos in positions:
            pos_skills = set(pos.requirements.get("required_skills", []) + pos.requirements.get("preferred_skills", []))
            startup = next((s for s in startups if s.startup_id == pos.startup_id), None)
            
            skill_match = len(dev_skills.intersection(pos_skills)) > 0
            industry_match = startup and any(ind in startup.startup_metadata.get("industry", []) for ind in preferred_industries)
            
            if skill_match or industry_match:
                relevant_positions.append(pos)
        
        # Generate interactions for relevant positions
        for pos in relevant_positions[:random.randint(3, 8)]:
            # Implicit feedback
            for event in ["view", "click"]:
                feedback = ImplicitFeedback(
                    user_id=dev.user_id,
                    item_type="position",
                    item_id=pos.position_id,
                    event_type=event,
                    timestamp=datetime.now() - timedelta(days=random.randint(1, 30))
                )
                implicit_feedback.append(feedback)
            
            # Some explicit feedback
            if random.random() < 0.3:  # 30% chance of explicit feedback
                feedback_type = random.choices(
                    ["like", "pass", "super_like"], 
                    weights=[0.6, 0.3, 0.1]
                )[0]
                feedback = ExplicitFeedback(
                    user_id=dev.user_id,
                    item_type="position",
                    item_id=pos.position_id,
                    feedback_type=feedback_type,
                    timestamp=datetime.now() - timedelta(days=random.randint(1, 30))
                )
                explicit_feedback.append(feedback)
    
    # Investors interact with relevant startups
    investors = [u for u in users if u.user_type == "investor"]
    for inv in investors:
        interested_industries = inv.profile_data.get("interested_industries", [])
        investment_stages = inv.profile_data.get("investment_stages", [])
        
        # Find startups that match investor's interests
        relevant_startups = []
        for startup in startups:
            industry_match = any(ind in startup.startup_metadata.get("industry", []) for ind in interested_industries)
            stage_match = startup.startup_metadata.get("stage") in investment_stages
            
            if industry_match or stage_match:
                relevant_startups.append(startup)
        
        # Generate interactions for relevant startups
        for startup in relevant_startups[:random.randint(2, 6)]:
            # Implicit feedback
            for event in ["view", "click"]:
                feedback = ImplicitFeedback(
                    user_id=inv.user_id,
                    item_type="startup",
                    item_id=startup.startup_id,
                    event_type=event,
                    timestamp=datetime.now() - timedelta(days=random.randint(1, 30))
                )
                implicit_feedback.append(feedback)
            
            # Some explicit feedback
            if random.random() < 0.4:  # 40% chance of explicit feedback
                feedback_type = random.choices(
                    ["like", "pass", "super_like"],
                    weights=[0.5, 0.4, 0.1]
                )[0]
                feedback = ExplicitFeedback(
                    user_id=inv.user_id,
                    item_type="startup",
                    item_id=startup.startup_id,
                    feedback_type=feedback_type,
                    timestamp=datetime.now() - timedelta(days=random.randint(1, 30))
                )
                explicit_feedback.append(feedback)
    
    return implicit_feedback, explicit_feedback

def populate_vector_store_meaningful(users: List[User], startups: List[Startup], positions: List[Position]):
    """Populate vector store with meaningful embeddings."""
    logger.info("Populating vector store with meaningful data...")
    
    try:
        # Add startup embeddings
        for startup in startups:
            try:
                vector_store.add_startup_embedding(
                    startup_id=startup.startup_id,
                    description=startup.description,
                    metadata=startup.startup_metadata
                )
            except Exception as e:
                logger.error(f"Error adding startup embedding for {startup.name}: {e}")
        
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
                logger.error(f"Error adding position embedding for {position.title}: {e}")
        
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
                    logger.error(f"Error adding user skills embedding for {user.profile_data.get('name')}: {e}")
    
    except Exception as e:
        logger.warning(f"Vector store population failed: {e}")
        logger.info("Continuing without vector store...")

def main():
    """Generate all meaningful sample data."""
    logger.info("Starting meaningful sample data generation...")
    
    try:
        with get_db_context() as db:
            # Clear existing data
            logger.info("Clearing existing data...")
            db.query(ExplicitFeedback).delete()
            db.query(ImplicitFeedback).delete()
            db.query(Position).delete()
            db.query(Startup).delete()
            db.query(User).delete()
            
            # Generate meaningful users
            logger.info("Generating meaningful user profiles...")
            users = generate_meaningful_users()
            db.add_all(users)
            db.flush()  # Get user IDs
            logger.info(f"Generated {len(users)} users")
            for user in users:
                logger.info(f"  - {user.profile_data.get('name', 'Unknown')} ({user.user_type})")
            
            # Generate startups (only for founders)
            logger.info("Generating meaningful startups...")
            founders = [u for u in users if u.user_type == "founder"]
            startups = generate_meaningful_startups(founders)
            db.add_all(startups)
            db.flush()  # Get startup IDs
            logger.info(f"Generated {len(startups)} startups")
            for startup in startups:
                logger.info(f"  - {startup.name} ({startup.startup_metadata.get('stage')})")
            
            # Generate positions
            logger.info("Generating meaningful job positions...")
            positions = generate_meaningful_positions(startups)
            db.add_all(positions)
            db.flush()  # Get position IDs
            logger.info(f"Generated {len(positions)} positions")
            for position in positions:
                startup = next((s for s in startups if s.startup_id == position.startup_id), None)
                company = startup.name if startup else "Unknown"
                logger.info(f"  - {position.title} at {company}")
            
            # Generate realistic interactions
            logger.info("Generating realistic user interactions...")
            implicit_feedback, explicit_feedback = generate_realistic_interactions(users, startups, positions)
            db.add_all(implicit_feedback)
            db.add_all(explicit_feedback)
            
            # Commit all changes
            db.commit()
            logger.info(f"Generated {len(implicit_feedback)} implicit and {len(explicit_feedback)} explicit feedback records")
        
        # Populate vector store
        populate_vector_store_meaningful(users, startups, positions)
        
        logger.info("✅ Meaningful sample data generation completed successfully!")
        logger.info("\n🎯 Generated Data Summary:")
        logger.info(f"   👨‍💻 {len([u for u in users if u.user_type == 'developer'])} Developers")
        logger.info(f"   🚀 {len([u for u in users if u.user_type == 'founder'])} Founders")
        logger.info(f"   💰 {len([u for u in users if u.user_type == 'investor'])} Investors") 
        logger.info(f"   🏢 {len(startups)} Startups")
        logger.info(f"   💼 {len(positions)} Job Positions")
        logger.info(f"   📊 {len(implicit_feedback + explicit_feedback)} User Interactions")
        
    except Exception as e:
        logger.error(f"Error generating meaningful sample data: {e}")
        raise

if __name__ == "__main__":
    main()