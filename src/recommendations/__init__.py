"""
week5/recommendations
=====================
Recommendation systems and dynamic styling matching algorithms.
"""
from src.recommendations.recommendation_engine import RecommendationEngine
from src.recommendations.style_recommender import StyleRecommender
from src.recommendations.brand_recommender import BrandRecommender
from src.recommendations.user_profile_manager import UserProfileManager, UserProfile

__all__ = [
    "RecommendationEngine",
    "StyleRecommender",
    "BrandRecommender",
    "UserProfileManager",
    "UserProfile"
]

