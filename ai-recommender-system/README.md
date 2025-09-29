# AI Recommender System

A production-ready AI-powered recommender system for startups, developers, and investors using content-based filtering, collaborative filtering, and neural networks.

## Features

- **Multi-User Support**: Developers, Founders, and Investors
- **Advanced Algorithms**: Content-based, Collaborative Filtering, Two-Tower Neural Networks
- **Semantic Search**: Using Sentence Transformers and ChromaDB
- **Real-time API**: FastAPI with async support
- **Dual Database**: PostgreSQL and SQLite support
- **Vector Store**: ChromaDB for efficient similarity search
- **Production Ready**: Logging, monitoring, health checks

## Architecture

### Three-Tiered Recommendation Strategy

1. **Tier 1 - Cold Start (Content-Based)**
   - Semantic similarity using sentence transformers
   - Works immediately for new users
   - Industry and skill matching

2. **Tier 2 - Warm Start (Collaborative Filtering)**
   - User-based and item-based filtering
   - Requires some interaction data
   - "Users like you" recommendations

3. **Tier 3 - Hot Start (Neural Networks)**
   - Two-tower deep learning model
   - Personalized recommendations
   - Continuous learning from feedback

## Quick Start

This section is tailored for Windows PowerShell (pwsh). Linux/Mac commands are similar.

### 1. Create and activate a virtual environment

```powershell
# From the project root
python -m venv .venv
# Activate (PowerShell)
. .venv\Scripts\Activate.ps1
# If you get an execution policy error, run in an elevated pwsh:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Alternatively, you can skip activation and use the venv’s Python directly:

```powershell
& .venv\Scripts\python.exe -V
```

### 2. Install dependencies

```powershell
# If activated:
pip install -r requirements.txt
# Or without activation:
& .venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Configure environment (.env)

Create a `.env` file (or update your existing one) and set at least:

```bash
# Database (use SQLite for development)
USE_POSTGRES=false
DATABASE_URL_SQLITE=sqlite:///./ai_recommender.db

# Vector Database
CHROMA_PERSIST_DIRECTORY=./data/chroma_db

# ML Models
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2

# API / App
SECRET_KEY=your-secret-key-here
ENVIRONMENT=development
PLATFORM_API_KEY=demo-api-key-123
```

### 4. Initialize data and run the server

```powershell
# Health check
& .venv\Scripts\python.exe run.py health

# Generate realistic sample data (developers, founders, investors, startups, positions)
& .venv\Scripts\python.exe run.py generate-meaningful-data

# Start the server
& .venv\Scripts\python.exe run.py server
```

### 5. Use the demo and docs

- Demo UI: http://localhost:8000/demo (enter API Key: `demo-api-key-123`)  
- API Docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  
- Stats: http://localhost:8000/api/v1/stats

### 6. Calling the Use-Case endpoints (2.7.1)

All requests must include the header: `X-API-Key: demo-api-key-123` (or your configured key).

- UC1 Founder → Developers
```bash
curl -X POST "http://localhost:8000/api/v1/uc/founder/developers" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-123" \
  -d '{
    "founder_id": 52,
    "limit": 5,
    "filters": { "skills": ["Python","Machine Learning"], "location": "San Francisco" }
  }'
```

- UC3 Developer → Startups
```bash
curl -X POST "http://localhost:8000/api/v1/uc/developer/startups" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-123" \
  -d '{ "developer_id": 1, "limit": 5 }'
```

- UC4 Investor → Startups
```bash
curl -X POST "http://localhost:8000/api/v1/uc/investor/startups" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-123" \
  -d '{
    "investor_id": 71,
    "limit": 5,
    "filters": { "stage": "Seed", "industry": ["AI/ML","HealthTech"] }
  }'
```

- UC5 Trending (Non-personalized)
```bash
curl "http://localhost:8000/api/v1/uc/trending?item_type=startup&limit=5" \
  -H "X-API-Key: demo-api-key-123"
```

- UC6 Feedback
```bash
# Explicit like
curl -X POST "http://localhost:8000/api/v1/uc/feedback" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-123" \
  -d '{ "user_id": 1, "item_type": "position", "item_id": 101, "feedback": "like", "rating": 5 }'

# Implicit view
curl -X POST "http://localhost:8000/api/v1/uc/feedback" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-123" \
  -d '{ "user_id": 71, "item_type": "startup", "item_id": 15, "feedback": "view" }'
```

### 7. Frontend demo usage

- Enter API Key (`demo-api-key-123` by default)
- Select a User Type (Student/Developer, Founder/Entrepreneur, Investor)
- Fill optional Preferences (skills, industry, stage, location, check sizes)
- Click “Get Recommendations” → A clean ranked list is shown (no algorithm names)

### 8. Notes & Troubleshooting

- If activation fails in PowerShell, try: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` (in an elevated shell).  
- If you prefer not to activate, always call the venv python directly: `& .venv\Scripts\python.exe ...`  
- Default API key: `demo-api-key-123` (change via `PLATFORM_API_KEY` in `.env`).
- Ensure data is generated (`generate-meaningful-data`) before testing recommendations.

## API Endpoints

### Core Recommendations

```http
# Get recommendations for any user type
POST /api/v1/recommendations
{
  "user_id": 1,
  "user_type": "developer",
  "limit": 10,
  "filters": {"industry": "AI/ML"}
}

# Developer-specific (job positions)
GET /api/v1/recommendations/for-developers?user_id=1&limit=10&industry=AI/ML

# Founder-specific (developers to hire)
GET /api/v1/recommendations/for-founders?user_id=51&limit=10&skills=Python,React

# Investor-specific (startups to invest in)
GET /api/v1/recommendations/for-investors?user_id=71&limit=10&stage=Seed
```

### User Management

```http
# Get user profile
GET /users/{user_id}/profile
```

### Feedback

```http
# Log implicit feedback (views, clicks)
POST /api/v1/feedback/implicit
{
  "user_id": 1,
  "item_type": "position",
  "item_id": 101,
  "event_type": "click"
}

# Log explicit feedback (likes, passes)
POST /api/v1/feedback/explicit
{
  "user_id": 1,
  "item_type": "position", 
  "item_id": 101,
  "feedback_type": "like"
}
```

### System

```http
# Algorithm status
GET /api/v1/algorithms/status

# System statistics
GET /api/v1/stats
```

## Database Schema

### Core Tables

- **users**: User profiles (developers, founders, investors)
- **startups**: Startup information and metadata
- **positions**: Job positions with requirements
- **implicit_feedback**: Views, clicks, saves
- **explicit_feedback**: Likes, passes, super_likes

### Recommendation Tracking

- **recommendation_requests**: Track all recommendation requests
- **recommendation_results**: Store recommendation outputs
- **recommendation_engagement**: Track user engagement with recommendations

### ML Infrastructure

- **text_embeddings**: Store sentence transformer embeddings
- **training_metrics**: Model performance tracking
- **model_registry**: Deployed model versions

## Sample Data

The system generates realistic sample data including:

- **50 Developers** with diverse skills (Python, React, ML, etc.)
- **20 Startups** across industries (AI, FinTech, HealthTech, etc.)
- **60+ Job Positions** with realistic requirements
- **10 Investors** with different focus areas
- **500+ User Interactions** for collaborative filtering

## Development

### Project Structure

```
ai-recommender-system/
├── app/
│   ├── config.py          # Configuration settings
│   └── main.py           # FastAPI application
├── database/
│   ├── models.py         # SQLAlchemy models
│   ├── database.py       # Database connections
│   └── vector_store.py   # ChromaDB integration
├── models/
│   ├── base_recommender.py  # Base classes
│   └── content_based.py     # Content-based algorithm
├── scripts/
│   └── generate_sample_data.py  # Data generation
├── requirements.txt
└── run.py               # Main run script
```

### Adding New Algorithms

1. Inherit from `BaseRecommender`
2. Implement required methods: `recommend()`, `train()`, `can_recommend()`
3. Add to hybrid system in `main.py`

```python
from models.base_recommender import BaseRecommender

class MyRecommender(BaseRecommender):
    def __init__(self):
        super().__init__("my_algorithm", "1.0")
    
    async def recommend(self, request):
        # Your recommendation logic
        return []
    
    def can_recommend(self, user_id, user_type):
        return True
    
    async def train(self, training_data):
        # Your training logic
        return {"status": "success"}
```

### Database Commands

```bash
# Reset database (caution: deletes all data)
python run.py reset-db

# Generate fresh sample data
python run.py generate-data

# Check system health
python run.py health
```

### Configuration Options

Key environment variables:

- `USE_POSTGRES`: Switch between PostgreSQL and SQLite
- `SENTENCE_TRANSFORMER_MODEL`: Hugging Face model for embeddings
- `TOP_K_RECOMMENDATIONS`: Default number of recommendations
- `SIMILARITY_THRESHOLD`: Minimum similarity score
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

## Production Deployment

### Database Setup

For production, use PostgreSQL:

```bash
# Install PostgreSQL
# Create database: ai_recommender

# Update .env
USE_POSTGRES=true
DATABASE_URL_POSTGRES=postgresql://user:password@localhost/ai_recommender
```

### Performance Optimization

1. **Vector Store**: Use persistent ChromaDB storage
2. **Caching**: Add Redis for recommendation caching
3. **Async**: All I/O operations are async
4. **Indexing**: Database indexes on frequently queried columns
5. **Monitoring**: Built-in health checks and metrics

### Scaling

- **Horizontal**: Multiple FastAPI instances behind load balancer
- **Database**: Read replicas for recommendation queries
- **Vector Store**: Distributed ChromaDB setup
- **ML Models**: Separate training and inference services

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License.

## Support

For questions or issues:

1. Check the [API documentation](http://localhost:8000/docs)
2. Review the logs in `./logs/` directory
3. Use the health check endpoint for system status
4. Open an issue on GitHub

## Roadmap

- [ ] Two-Tower Neural Network implementation
- [ ] Collaborative filtering with Implicit library
- [ ] A/B testing framework
- [ ] Real-time model updates
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Graph-based recommendations