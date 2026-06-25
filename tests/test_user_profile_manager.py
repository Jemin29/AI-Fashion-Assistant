"""
week5/tests/test_user_profile_manager.py
========================================
Unit tests for the User Preference Modeling and Profile Management System.
"""

from __future__ import annotations

import json
import os
import pytest

from src.recommendations.user_profile_manager import UserProfileManager, UserProfile


@pytest.fixture
def temp_db_path(tmp_path):
    """Temporary JSON database path for profile manager testing."""
    return tmp_path / "user_profile.json"


class TestUserProfileManager:
    """Validate loading, saving, search/recommendation logging, preference updates, and json serialization validations."""

    def test_manager_initialization(self, temp_db_path):
        """Verify manager initializes with empty profiles if database doesn't exist."""
        manager = UserProfileManager(db_path=temp_db_path)
        assert len(manager.profiles) == 0
        assert manager.db_path == temp_db_path

    def test_create_and_get_profile(self, temp_db_path):
        """Verify profile creation and retrieval works."""
        manager = UserProfileManager(db_path=temp_db_path)

        # Create new profile
        profile = manager.create_profile(
            user_id="user_123",
            favorite_styles=["streetwear"],
            favorite_brands=["Nike"],
            favorite_colors=["black"]
        )

        assert profile.user_id == "user_123"
        assert profile.favorite_styles == ["streetwear"]
        assert profile.favorite_brands == ["Nike"]
        assert profile.favorite_colors == ["black"]
        assert len(profile.search_history) == 0
        assert len(profile.recommendation_history) == 0

        # Retrieve profile
        retrieved = manager.get_profile("user_123")
        assert retrieved is not None
        assert retrieved.user_id == "user_123"

        # Try to retrieve non-existent user profile
        assert manager.get_profile("non_existent") is None

        # Verify duplicate creation warning path returns existing
        duplicate = manager.create_profile(user_id="user_123")
        assert duplicate.favorite_styles == ["streetwear"]

    def test_update_preferences(self, temp_db_path):
        """Verify preference updating handles list merging and de-duplication."""
        manager = UserProfileManager(db_path=temp_db_path)

        # 1. Update on non-existent profile (should create it)
        profile = manager.update_preferences(
            user_id="user_456",
            favorite_styles=["streetwear"],
            favorite_brands=["Nike"]
        )
        assert profile.user_id == "user_456"
        assert profile.favorite_styles == ["streetwear"]

        # 2. Update existing preferences with duplicate and new values
        profile_updated = manager.update_preferences(
            user_id="user_456",
            favorite_styles=["streetwear", "minimalist"],
            favorite_brands=["Supreme", "Nike"],
            favorite_colors=["black"]
        )

        # Deduplicated sets
        assert profile_updated.favorite_styles == ["streetwear", "minimalist"]
        assert profile_updated.favorite_brands == ["Nike", "Supreme"]  # Nike already in list, Supreme appended
        assert profile_updated.favorite_colors == ["black"]

    def test_record_search_history(self, temp_db_path):
        """Verify queries are logged to search history and limited properly."""
        manager = UserProfileManager(db_path=temp_db_path)

        # 1. Log query for non-existent profile (creates profile)
        manager.record_search("user_789", "streetwear sneakers")
        profile = manager.get_profile("user_789")
        assert profile.search_history == ["streetwear sneakers"]

        # 2. Log empty query (should be ignored)
        manager.record_search("user_789", "")
        assert profile.search_history == ["streetwear sneakers"]

        # 3. Log duplicate query (should move to the end)
        manager.record_search("user_789", "cargo pants")
        manager.record_search("user_789", "streetwear sneakers")
        assert profile.search_history == ["cargo pants", "streetwear sneakers"]

        # 4. Limit search history to 50 entries
        for i in range(60):
            manager.record_search("user_789", f"query_{i}")

        assert len(profile.search_history) == 50
        assert profile.search_history[0] == "query_10"
        assert profile.search_history[-1] == "query_59"

    def test_record_recommendation_history(self, temp_db_path):
        """Verify recommendation outputs are logged and limited properly."""
        manager = UserProfileManager(db_path=temp_db_path)

        # 1. Log recommendations
        manager.record_recommendations("user_abc", ["Techwear", "Urban Minimal"])
        profile = manager.get_profile("user_abc")
        assert profile.recommendation_history == ["Techwear", "Urban Minimal"]

        # 2. Log duplicate recommendation (should move to the end)
        manager.record_recommendations("user_abc", ["Techwear"])
        assert profile.recommendation_history == ["Urban Minimal", "Techwear"]

        # 3. Enforce limit of 100 recommendations in history
        recs = [f"item_{i}" for i in range(120)]
        manager.record_recommendations("user_abc", recs)
        assert len(profile.recommendation_history) == 100
        assert profile.recommendation_history[0] == "item_20"
        assert profile.recommendation_history[-1] == "item_119"

    def test_save_and_load_database(self, temp_db_path):
        """Verify json persistence and database loading."""
        manager = UserProfileManager(db_path=temp_db_path)
        manager.create_profile("user_1", ["luxury"], ["Gucci"], ["red"])

        # Check database file was created
        assert temp_db_path.exists()

        # Reload database in a new manager instance
        new_manager = UserProfileManager(db_path=temp_db_path)
        profile = new_manager.get_profile("user_1")
        assert profile is not None
        assert profile.favorite_styles == ["luxury"]
        assert profile.favorite_brands == ["Gucci"]
        assert profile.favorite_colors == ["red"]

    def test_corrupted_json_handling(self, temp_db_path):
        """Verify corrupted JSON files are handled gracefully."""
        # Write invalid JSON content
        temp_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_db_path, "w", encoding="utf-8") as f:
            f.write("{invalid json content")

        manager = UserProfileManager(db_path=temp_db_path)
        assert len(manager.profiles) == 0

        # Try to parse valid JSON containing bad schemas
        bad_data = {
            "user_bad": {
                "user_id": "user_bad",
                "favorite_styles": "this should be a list, not string"
            }
        }
        with open(temp_db_path, "w", encoding="utf-8") as f:
            json.dump(bad_data, f)

        manager2 = UserProfileManager(db_path=temp_db_path)
        assert len(manager2.profiles) == 0  # Invalid schema profile ignored

    def test_default_db_path(self):
        """Verify that default db path is correctly constructed when db_path is not specified."""
        manager = UserProfileManager()
        assert manager.db_path is not None
        assert "outputs" in str(manager.db_path)

    def test_save_profiles_error(self, temp_db_path):
        """Verify that save_profiles handles exceptions gracefully and raises them."""
        manager = UserProfileManager(db_path=temp_db_path)
        # Force write error by pointing db_path to a directory that exists
        manager.db_path = temp_db_path.parent
        with pytest.raises(Exception):
            manager.save_profiles()

    def test_update_preferences_none_values(self, temp_db_path):
        """Verify merge_lists helper returns existing list when new_vals is None."""
        manager = UserProfileManager(db_path=temp_db_path)
        profile = manager.create_profile("user_x", ["streetwear"], ["Nike"], ["black"])
        # Update style only, leaving other preference parameters as None
        updated = manager.update_preferences("user_x", favorite_styles=["minimalist"])
        assert updated.favorite_styles == ["streetwear", "minimalist"]
        assert updated.favorite_brands == ["Nike"]
        assert updated.favorite_colors == ["black"]

    def test_record_recommendations_empty_list(self, temp_db_path):
        """Verify record_recommendations does nothing when no recommendations are passed or they are empty."""
        manager = UserProfileManager(db_path=temp_db_path)
        profile = manager.create_profile("user_y")
        
        # Log empty list
        manager.record_recommendations("user_y", [])
        assert len(profile.recommendation_history) == 0

        # Log blank items list
        manager.record_recommendations("user_y", ["", "   "])
        assert len(profile.recommendation_history) == 0
