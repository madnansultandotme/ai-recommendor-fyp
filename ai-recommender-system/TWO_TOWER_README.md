# Two-Tower Neural Network Implementation

## Overview

This document provides detailed information about the Two-Tower neural network implementation in the AI Recommender System. The Two-Tower architecture represents the **Tier 3** (most sophisticated) level of our hybrid recommendation system.

## Architecture

### Model Design

The Two-Tower neural network consists of:

1. **User Tower**: Processes user features to generate user embeddings
2. **Item Tower**: Processes item features to generate item embeddings
3. **Prediction Layer**: Combines embeddings to predict user-item interactions

```python
TwoTowerModel(
    user_feature_dim=20,    # Number of user features
    item_feature_dim=15,    # Number of item features  
    embedding_dim=128,      # Embedding dimensionality
    hidden_dims=[256, 128]  # Hidden layer sizes
)
```

### Key Features

- **Deep Learning Architecture**: Uses PyTorch for neural network implementation
- **Feature Engineering**: Comprehensive feature extraction from user profiles and items
- **Training Pipeline**: Full training loop with validation, early stopping, and model persistence
- **Embedding Normalization**: L2 normalization for better similarity computation
- **Batch Processing**: Efficient batch processing with DataLoader

## Implementation Details

### 1. Feature Extraction

#### User Features (20 dimensions)
- User type encoding (developer, founder, investor)
- Experience years
- Skills count
- Industry preferences
- Investment check size (for investors)

#### Item Features (15+ dimensions)
- **For Positions**: Required skills, salary range, remote availability, startup info
- **For Startups**: Team size, funding amount, stage, industry, required skills

### 2. Training Process

```python
# Training configuration
optimizer = Adam(lr=0.001, weight_decay=1e-5)
criterion = BCELoss()
scheduler = ReduceLROnPlateau(patience=3)

# Features:
- Early stopping (patience=10)
- Learning rate scheduling  
- Model checkpointing
- Validation split (80/20)
- Negative sampling
```

### 3. Recommendation Generation

The model generates recommendations by:
1. Extracting user features for the target user
2. Computing item features for all candidates
3. Scoring user-item pairs through the neural network
4. Ranking by prediction scores
5. Generating explainable match reasons

## API Usage

### Training the Model

```bash
POST /api/v1/models/train
{
    "training_data": {},
    "force_retrain": false
}
```

### Getting Neural Recommendations

The Two-Tower model is automatically integrated into the hybrid system:

```bash
POST /api/v1/recommendations
{
    "user_id": 1,
    "user_type": "developer", 
    "limit": 10
}
```

### Model Management

```bash
# Check algorithm status
GET /api/v1/algorithms/status

# Load trained models
POST /api/v1/models/load

# Update hybrid weights
PUT /api/v1/models/weights
{
    "weights": {
        "content_based": 0.4,
        "collaborative": 0.35, 
        "two_tower": 0.25
    }
}
```

## Technical Specifications

### Model Architecture

```
User Tower:
Input(20) → Linear(256) → ReLU → Dropout(0.2) → BatchNorm 
         → Linear(128) → ReLU → Dropout(0.2) → BatchNorm
         → Linear(128) → L2Normalize

Item Tower: 
Input(15) → Linear(256) → ReLU → Dropout(0.2) → BatchNorm
         → Linear(128) → ReLU → Dropout(0.2) → BatchNorm  
         → Linear(128) → L2Normalize

Prediction:
Concat(User_Emb, Item_Emb) → Linear(64) → ReLU → Dropout(0.2)
                           → Linear(1) → Sigmoid
```

### Training Requirements

- **Minimum Data**: 500 user-item interactions
- **Hardware**: GPU support (CUDA) recommended but not required
- **Memory**: ~100MB for model parameters
- **Training Time**: 5-30 minutes depending on data size

### Performance Characteristics

- **Cold Start**: Not supported (requires user interaction history)
- **Scalability**: Good for up to 100K users/items
- **Accuracy**: High for users with sufficient interaction history
- **Explainability**: Neural similarity scores with match reasons

## Integration with Hybrid System

The Two-Tower model is seamlessly integrated into our hybrid recommender:

### Weight Configuration
```python
weights = {
    "content_based": 0.4,      # Always available, cold start
    "collaborative": 0.35,     # Matrix factorization 
    "two_tower": 0.25          # Deep learning personalization
}
```

### Fallback Strategy
1. **Primary**: All three algorithms if available
2. **Fallback 1**: Content-based + Collaborative if neural model fails
3. **Fallback 2**: Content-based only for cold start users

## File Structure

```
models/
├── two_tower.py           # Main implementation
├── hybrid_recommender.py  # Integration with hybrid system
└── base_recommender.py    # Base interfaces

Key Components:
- TwoTowerModel: PyTorch neural network
- TwoTowerDataset: Training data preparation  
- TwoTowerRecommender: Main recommender class
```

## Testing

Run the comprehensive test suite:

```bash
python test_two_tower.py
```

Tests cover:
- ✅ Model architecture and forward pass
- ✅ Feature extraction pipeline
- ✅ Training data preparation
- ✅ Integration with hybrid system

## Performance Monitoring

### Training Metrics
- Training/validation loss curves
- Early stopping triggers
- Model convergence monitoring
- Feature importance analysis

### Recommendation Quality
- Precision@K, Recall@K, NDCG@K
- Diversity scores
- Coverage analysis
- A/B testing framework (planned)

## Deployment Considerations

### Model Persistence
- Models saved to `./models/saved/two_tower/`
- PyTorch state dict format
- Metadata and scalers included
- Version control for model updates

### Production Settings
```python
# config.py settings for production
epochs = 50              # Longer training
learning_rate = 0.0005   # Lower learning rate  
batch_size = 1024        # Larger batches
embedding_dim = 256      # Higher capacity
```

## Next Steps

1. **A/B Testing Framework**: Compare neural vs collaborative performance
2. **Advanced Features**: Add temporal dynamics, session-based features
3. **Model Optimization**: Knowledge distillation, quantization
4. **Real-time Updates**: Online learning capabilities
5. **Multi-task Learning**: Joint optimization for different objectives

## Troubleshooting

### Common Issues

**CUDA/Memory Issues**
```python
# Force CPU usage
device = torch.device("cpu")

# Reduce batch size
batch_size = 256
```

**Training Convergence**
```python
# Adjust learning rate
learning_rate = 0.0001

# Increase regularization
weight_decay = 1e-4
```

**Feature Dimension Mismatch**
```python
# Check feature extraction consistency
user_features = extract_user_features()
print(f"User features shape: {len(user_features)}")
```

For more detailed debugging, check the application logs and model training outputs.