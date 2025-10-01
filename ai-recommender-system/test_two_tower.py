#!/usr/bin/env python3
"""
Test script for the Two-Tower Neural Network implementation.
This script tests the basic functionality without requiring a full database setup.
"""
import torch
import numpy as np
import pandas as pd
import pytest
from models.two_tower import TwoTowerModel, TwoTowerDataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_two_tower_model():
    """Test the Two-Tower model architecture."""
    logger.info("Testing Two-Tower model architecture...")
    
    # Create dummy data
    batch_size = 100
    user_feature_dim = 20
    item_feature_dim = 15
    
    # Generate random features
    user_features = torch.randn(batch_size, user_feature_dim)
    item_features = torch.randn(batch_size, item_feature_dim)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # Create model
    model = TwoTowerModel(
        user_feature_dim=user_feature_dim,
        item_feature_dim=item_feature_dim,
        embedding_dim=64,
        hidden_dims=[128, 64]
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        predictions, user_embeddings, item_embeddings = model(user_features, item_features)
        
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"User embeddings shape: {user_embeddings.shape}")
        logger.info(f"Item embeddings shape: {item_embeddings.shape}")
        logger.info(f"Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # Test training mode
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    
    # Single training step
    optimizer.zero_grad()
    predictions, _, _ = model(user_features, item_features)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    
    logger.info(f"Training step completed, loss: {loss.item():.4f}")
    
    # Test dataset
    dataset = TwoTowerDataset(user_features.numpy(), item_features.numpy(), labels.numpy())
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    # Test data loader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    batch = next(iter(dataloader))
    logger.info(f"DataLoader batch keys: {batch.keys()}")
    logger.info(f"Batch sizes: {[v.shape for v in batch.values()]}")
    
    logger.info("✅ Two-Tower model architecture test completed successfully!")
    
    assert predictions.shape == labels.shape
    assert user_embeddings.shape[0] == batch_size
    assert item_embeddings.shape[0] == batch_size

def test_feature_extraction():
    """Test feature extraction functions."""
    logger.info("Testing feature extraction...")
    
    # Mock user profile
    mock_profile = {
        'experience_years': 5,
        'skills': ['Python', 'JavaScript', 'React'],
        'preferred_industries': ['AI/ML', 'FinTech'],
        'check_size_min': 100000,
        'check_size_max': 500000
    }
    
    # Mock user object
    class MockUser:
        def __init__(self):
            self.user_type = 'developer'
            self.profile_data = mock_profile
    
    # Import the TwoTowerRecommender for feature extraction
    from models.two_tower import TwoTowerRecommender
    
    recommender = TwoTowerRecommender()
    user = MockUser()
    
    # Test user feature extraction
    user_features = recommender._extract_user_feature_vector(user)
    logger.info(f"Extracted user features: {len(user_features)} dimensions")
    logger.info(f"Feature names: {list(user_features.keys())}")
    logger.info(f"Feature values: {list(user_features.values())}")
    
    # Mock startup metadata
    mock_startup_metadata = {
        'team_size': 10,
        'funding_amount': 2000000,
        'required_skills': ['Python', 'Machine Learning'],
        'stage': 'Series A',
        'industry': ['AI/ML']
    }
    
    # Mock startup object
    class MockStartup:
        def __init__(self):
            self.startup_metadata = mock_startup_metadata
    
    startup = MockStartup()
    startup_features = recommender._extract_startup_features(startup)
    logger.info(f"Extracted startup features: {len(startup_features)} dimensions")
    logger.info(f"Feature names: {list(startup_features.keys())}")
    logger.info(f"Feature values: {list(startup_features.values())}")
    
    logger.info("✅ Feature extraction test completed successfully!")
    
    assert len(user_features) > 0
    assert len(startup_features) > 0

def test_training_data_preparation():
    """Test training data preparation."""
    logger.info("Testing training data preparation...")
    
    # Create mock training data
    interactions = [
        {'user_id': 1, 'item_id': 101, 'item_type': 'position', 'label': 1.0},
        {'user_id': 1, 'item_id': 102, 'item_type': 'position', 'label': 0.8},
        {'user_id': 2, 'item_id': 101, 'item_type': 'position', 'label': 0.0},
        {'user_id': 2, 'item_id': 103, 'item_type': 'startup', 'label': 1.0},
    ]
    
    train_df = pd.DataFrame(interactions)
    
    # Mock user features
    user_features_data = [
        {'user_id': 1, 'user_type_developer': 1.0, 'experience_years': 3.0, 'skills_count': 5.0},
        {'user_id': 2, 'user_type_investor': 1.0, 'experience_years': 10.0, 'skills_count': 3.0}
    ]
    user_features_df = pd.DataFrame(user_features_data)
    
    # Mock item features  
    item_features_data = [
        {'item_id': 101, 'required_skills_count': 3.0, 'remote_ok': 1.0, 'salary_min': 0.8},
        {'item_id': 102, 'required_skills_count': 4.0, 'remote_ok': 0.0, 'salary_min': 1.2},
        {'item_id': 103, 'team_size': 0.1, 'funding_amount': 2.0, 'stage_series_a': 1.0}
    ]
    item_features_df = pd.DataFrame(item_features_data)
    
    # Test feature preparation
    from models.two_tower import TwoTowerRecommender
    recommender = TwoTowerRecommender()
    
    try:
        X_user, X_item, y = recommender._prepare_features(train_df, user_features_df, item_features_df)

        logger.info(f"User features shape: {X_user.shape}")
        logger.info(f"Item features shape: {X_item.shape}")
        logger.info(f"Labels shape: {y.shape}")
        logger.info(f"User features sample: {X_user[0][:5]}")
        logger.info(f"Item features sample: {X_item[0][:5]}")
        logger.info(f"Labels sample: {y[:5]}")

        assert X_user.shape[0] == train_df.shape[0]
        assert X_item.shape[0] == train_df.shape[0]
        assert y.shape[0] == train_df.shape[0]
        assert X_user.ndim == 2 and X_item.ndim == 2

        logger.info("✅ Training data preparation test completed successfully!")

    except Exception as e:
        logger.error(f"❌ Training data preparation test failed: {e}")
        pytest.fail(f"Training data preparation test failed: {e}")

def main():
    """Run all tests."""
    logger.info("🚀 Starting Two-Tower Neural Network tests...")
    
    tests = [
        test_two_tower_model,
        test_feature_extraction,
        test_training_data_preparation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    logger.info(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Two-Tower implementation is working correctly.")
    else:
        logger.warning("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()