"""
week5/tests/test_fashion_knowledge_base.py
==========================================
Unit tests for the FashionKnowledgeBase database and model validation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data.knowledge_base.fashion_knowledge_base import (
    FashionKnowledgeBase,
    KnowledgeItem,
    VALID_CATEGORIES,
)


class TestFashionKnowledgeBase:
    """Verify operations of the JSON-backed Fashion Knowledge Base."""

    @pytest.fixture
    def temp_kb(self):
        """Fixture initializing an isolated temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "knowledge_base.json"
            kb = FashionKnowledgeBase(db_path=db_file)
            yield kb

    def test_database_seeding_on_init(self, temp_kb):
        """Verify that default knowledge records are automatically seeded."""
        # Seeding creates multiple default concepts
        items = temp_kb.list_items()
        assert len(items) > 15
        
        # Check that we seeded elements across all 9 categories
        categories_represented = {item.category for item in items}
        assert len(categories_represented) == len(VALID_CATEGORIES)
        assert "fashion_styles" in categories_represented
        assert "color_theory" in categories_represented
        assert "fashion_seasons" in categories_represented
        assert "fabric_types" in categories_represented
        assert "fashion_categories" in categories_represented
        assert "brand_profiles" in categories_represented
        assert "streetwear_trends" in categories_represented
        assert "luxury_trends" in categories_represented
        assert "fashion_terminology" in categories_represented

    def test_invalid_category_raises_value_error(self, temp_kb):
        """Assert that creating an item with an invalid category raises ValueError."""
        with pytest.raises(ValueError):
            temp_kb.create_item(
                category="footwear_trends",  # Unsupported category
                name="Platform Shoes",
                content="Platform shoes trending in streetwear."
            )

        with pytest.raises(ValidationError):
            # Enforce validation directly at the Pydantic schema model layer
            KnowledgeItem(
                id="kb_invalid_item",
                category="invalid_category",
                name="Test",
                content="Test description"
            )

    def test_create_item_successfully(self, temp_kb):
        """Verify that creating a valid item adds it to the database."""
        new_item = temp_kb.create_item(
            category="fashion_terminology",
            name="Warp and Weft",
            content="The two directions of thread weave in fabrics.",
            tags=["weave", "textile"],
            metadata={"importance": "medium"}
        )

        assert new_item.id == "kb_fashion_terminology_warp_and_weft"
        assert new_item.name == "Warp and Weft"
        assert "weave" in new_item.tags
        assert new_item.metadata["importance"] == "medium"

        # Assert database persistence
        retrieved = temp_kb.get_item(new_item.id)
        assert retrieved is not None
        assert retrieved.name == "Warp and Weft"

    def test_create_duplicate_item_raises_value_error(self, temp_kb):
        """Verify that creating an item with a name that yields a duplicate ID fails."""
        temp_kb.create_item(
            category="fabric_types",
            name="Wool",
            content="Heavy winter textile."
        )

        with pytest.raises(ValueError):
            temp_kb.create_item(
                category="fabric_types",
                name="Wool",  # Duplicate name
                content="Alternative wool description."
            )

    def test_update_item_successfully(self, temp_kb):
        """Verify that updating item properties persists values and timestamps."""
        # Retrieve a seeded item
        item = temp_kb.list_items(category="fabric_types")[0]
        original_id = item.id
        original_update_time = item.updated_at

        # Apply update
        updated = temp_kb.update_item(
            item_id=original_id,
            content="Updated fabric description.",
            tags=["new_tag"],
            metadata={"source": "auditor"}
        )

        assert updated.id == original_id
        assert updated.content == "Updated fabric description."
        assert "new_tag" in updated.tags
        assert updated.metadata["source"] == "auditor"
        assert updated.updated_at >= original_update_time

        # Retrieve and verify database reload
        retrieved = temp_kb.get_item(original_id)
        assert retrieved.content == "Updated fabric description."
        assert "new_tag" in retrieved.tags

    def test_delete_item_successfully(self, temp_kb):
        """Verify deleting an item removes it from memory and disk."""
        item = temp_kb.create_item(
            category="color_theory",
            name="Triadic Colors",
            content="Three colors spaced evenly on the wheel."
        )
        item_id = item.id

        assert temp_kb.get_item(item_id) is not None
        
        # Delete item
        deleted = temp_kb.delete_item(item_id)
        assert deleted is True
        assert temp_kb.get_item(item_id) is None

        # Verify deleting a non-existent item returns False
        assert temp_kb.delete_item("non_existent_id") is False

    def test_list_items_with_category_and_tags_filtering(self, temp_kb):
        """Verify listing and filtering items works."""
        # Retrieve all fashion styles
        styles = temp_kb.list_items(category="fashion_styles")
        assert len(styles) >= 3
        for item in styles:
            assert item.category == "fashion_styles"

        # Retrieve items by tag filter
        casual_items = temp_kb.list_items(tags=["casual"])
        assert len(casual_items) >= 2
        
        # Verify case insensitivity and space trimming in tag matching
        luxury_items = temp_kb.list_items(tags=["  LUXURY  "])
        assert len(luxury_items) >= 1
