"""
week5/recommendations/user_profile_manager.py
============================================
User Preference Modeling and Profile Management System.
Tracks user favorite styles, brands, colors, search history, and recommendation history.
Persists state in user_profile.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Data model representing a user's style preferences and history."""
    user_id: str = Field(description="Unique alphanumeric identifier for the user.")
    favorite_styles: List[str] = Field(default_factory=list, description="Curated list of favorite styles.")
    favorite_brands: List[str] = Field(default_factory=list, description="Curated list of favorite brands.")
    favorite_colors: List[str] = Field(default_factory=list, description="Curated list of favorite colors.")
    search_history: List[str] = Field(default_factory=list, description="List of search queries executed.")
    recommendation_history: List[str] = Field(default_factory=list, description="List of recommended styles/brands/items.")
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    updated_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))


class UserProfileManager:
    """
    Manages loading, updating, logging, and saving user preference profiles.
    Persists data in a local JSON database.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the User Profile Manager.

        Parameters
        ----------
        db_path : Path or str, optional
            Path to the JSON database. Defaults to outputs/recommendations/user_profile.json.
        """
        if db_path:
            self.db_path = Path(db_path).resolve()
        else:
            self.db_path = Path("outputs/recommendations/user_profile.json").resolve()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.profiles: Dict[str, UserProfile] = {}
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load profiles from the JSON database file."""
        if not self.db_path.exists():
            self.profiles = {}
            logger.info(f"User Profile database not found at {self.db_path}. Starting clean.")
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.profiles = {}
            for user_id, profile_dict in data.items():
                try:
                    self.profiles[user_id] = UserProfile(**profile_dict)
                except Exception as err:
                    logger.error(f"Error parsing profile for user '{user_id}': {err}")
            
            logger.info(f"Loaded {len(self.profiles)} user profiles from {self.db_path}.")
        except Exception as err:
            logger.error(f"Failed to load user profiles from {self.db_path}: {err}")
            self.profiles = {}

    def save_profiles(self) -> None:
        """Persist profiles to the JSON database file."""
        try:
            data = {user_id: profile.model_dump() for user_id, profile in self.profiles.items()}
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.profiles)} user profiles to {self.db_path}.")
        except Exception as err:
            logger.error(f"Failed to save user profiles to {self.db_path}: {err}")
            raise

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Retrieve a user profile.

        Parameters
        ----------
        user_id : str

        Returns
        -------
        UserProfile, optional
        """
        return self.profiles.get(user_id)

    def create_profile(
        self,
        user_id: str,
        favorite_styles: Optional[List[str]] = None,
        favorite_brands: Optional[List[str]] = None,
        favorite_colors: Optional[List[str]] = None
    ) -> UserProfile:
        """
        Create a new user profile.

        Parameters
        ----------
        user_id : str
        favorite_styles : List[str], optional
        favorite_brands : List[str], optional
        favorite_colors : List[str], optional

        Returns
        -------
        UserProfile
        """
        if user_id in self.profiles:
            logger.warning(f"Profile for user '{user_id}' already exists. Returning existing profile.")
            return self.profiles[user_id]

        profile = UserProfile(
            user_id=user_id,
            favorite_styles=favorite_styles or [],
            favorite_brands=favorite_brands or [],
            favorite_colors=favorite_colors or []
        )
        self.profiles[user_id] = profile
        self.save_profiles()
        logger.info(f"Created new style profile for user '{user_id}'.")
        return profile

    def update_preferences(
        self,
        user_id: str,
        favorite_styles: Optional[List[str]] = None,
        favorite_brands: Optional[List[str]] = None,
        favorite_colors: Optional[List[str]] = None
    ) -> UserProfile:
        """
        Update user style preferences.

        Parameters
        ----------
        user_id : str
        favorite_styles : List[str], optional
        favorite_brands : List[str], optional
        favorite_colors : List[str], optional

        Returns
        -------
        UserProfile
        """
        profile = self.get_profile(user_id)
        if not profile:
            # Create profile if it does not exist
            return self.create_profile(user_id, favorite_styles, favorite_brands, favorite_colors)

        # Helper to update lists preserving uniqueness
        def merge_lists(existing: List[str], new_vals: Optional[List[str]]) -> List[str]:
            if new_vals is None:
                return existing
            merged = list(existing)
            for val in new_vals:
                clean_val = val.strip()
                if clean_val and clean_val not in merged:
                    merged.append(clean_val)
            return merged

        profile.favorite_styles = merge_lists(profile.favorite_styles, favorite_styles)
        profile.favorite_brands = merge_lists(profile.favorite_brands, favorite_brands)
        profile.favorite_colors = merge_lists(profile.favorite_colors, favorite_colors)
        
        profile.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self.save_profiles()
        logger.info(f"Updated preferences for user '{user_id}'.")
        return profile

    def record_search(self, user_id: str, query: str) -> None:
        """
        Log a query into the user's search history.

        Parameters
        ----------
        user_id : str
        query : str
        """
        profile = self.get_profile(user_id)
        if not profile:
            profile = self.create_profile(user_id)

        clean_query = query.strip()
        if not clean_query:
            return

        # Keep history list unique or append to end
        if clean_query in profile.search_history:
            profile.search_history.remove(clean_query)
        profile.search_history.append(clean_query)

        # Limit history length to 50
        if len(profile.search_history) > 50:
            profile.search_history = profile.search_history[-50:]

        profile.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self.save_profiles()
        logger.info(f"Logged search query '{clean_query}' for user '{user_id}'.")

    def record_recommendations(self, user_id: str, recommendations: List[str]) -> None:
        """
        Log recommendation output items into the user's recommendation history.

        Parameters
        ----------
        user_id : str
        recommendations : List[str]
        """
        profile = self.get_profile(user_id)
        if not profile:
            profile = self.create_profile(user_id)

        added_any = False
        for rec in recommendations:
            clean_rec = rec.strip()
            if clean_rec:
                if clean_rec in profile.recommendation_history:
                    profile.recommendation_history.remove(clean_rec)
                profile.recommendation_history.append(clean_rec)
                added_any = True

        if not added_any:
            return

        # Limit history length to 100
        if len(profile.recommendation_history) > 100:
            profile.recommendation_history = profile.recommendation_history[-100:]

        profile.updated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self.save_profiles()
        logger.info(f"Logged {len(recommendations)} recommended items for user '{user_id}'.")
