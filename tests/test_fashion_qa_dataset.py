"""
week5/tests/test_fashion_qa_dataset.py
======================================
Unit tests verifying the Fashion Q&A Dataset Builder and Q&A schema validations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data.knowledge_base.fashion_qa_dataset import (
    FashionQADatasetBuilder,
    FashionQARecord,
)


class TestFashionQADatasetBuilder:
    """Validate Q&A record schema checks, CRUD operations, generation count, and persistence."""

    def test_qa_record_validation(self):
        """Verify FashionQARecord validation constraints and default values."""
        # Valid record
        record = FashionQARecord(
            id="qa_style_advice_999",
            category="style_advice",
            question="What is style?",
            answer="Style is a way to express yourself."
        )
        assert record.category == "style_advice"
        assert record.question == "What is style?"
        assert isinstance(record.tags, list)
        assert isinstance(record.metadata, dict)
        assert record.created_at is not None
        assert record.updated_at is not None

        # Invalid category rejection
        with pytest.raises(ValidationError):
            FashionQARecord(
                id="qa_invalid",
                category="invalid_category",  # Must be in VALID_QA_CATEGORIES
                question="Q",
                answer="A"
            )

    def test_builder_seeding_and_crud(self):
        """Verify initialization seeds 500+ defaults and supports a complete CRUD cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_qa_dataset.json"

            # Initial load should trigger programmatic seeding
            builder = FashionQADatasetBuilder(db_path=db_path)
            assert db_path.exists()
            assert len(builder.records) >= 500

            # Verify all 5 categories are present
            categories = {item.category for item in builder.records.values()}
            assert categories == {"style_advice", "trend_advice", "fabrics", "brands", "fashion_terminology"}

            # Test Create (create_record)
            new_record = builder.create_record(
                category="fabrics",
                question="Is hemp a sustainable fabric option?",
                answer="Yes, hemp is highly sustainable as it requires minimal water and no pesticides to grow.",
                tags=["sustainability", "hemp"],
                metadata={"eco_friendly": True}
            )
            assert new_record.category == "fabrics"
            assert "hemp" in new_record.id
            assert builder.get_record(new_record.id) is not None

            # Test Read (list_by_category and get_record)
            fabrics_records = builder.list_by_category("fabrics")
            assert any(rec.id == new_record.id for rec in fabrics_records)

            # Test Tag Search (search_by_tags)
            search_results = builder.search_by_tags(["sustainability"])
            assert any(rec.id == new_record.id for rec in search_results)

            # Test search with empty list
            assert builder.search_by_tags([]) == []

            # Test Update (update_record)
            updated = builder.update_record(
                record_id=new_record.id,
                answer="Hemp is extremely sustainable and regenerates soil.",
                metadata={"eco_friendly": True, "organic": True}
            )
            assert updated.answer == "Hemp is extremely sustainable and regenerates soil."
            assert updated.metadata["organic"] is True

            # Verify update fails on missing keys
            with pytest.raises(KeyError):
                builder.update_record(record_id="non_existent", answer="No")

            # Test Delete (delete_record)
            target_id = new_record.id
            success = builder.delete_record(target_id)
            assert success is True
            assert builder.get_record(target_id) is None

            # Deleting non-existent returns False
            success_fail = builder.delete_record(target_id)
            assert success_fail is False

    def test_database_persistence(self):
        """Verify that saved dataset matches loaded dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "persist_qa_dataset.json"

            builder1 = FashionQADatasetBuilder(db_path=db_path)
            custom_rec = builder1.create_record(
                category="fashion_terminology",
                question="What is a yoke?",
                answer="A yoke is a shaped pattern piece that forms part of a garment, usually fitting around the neck and shoulders."
            )

            # Load in a second instance
            builder2 = FashionQADatasetBuilder(db_path=db_path)
            loaded_rec = builder2.get_record(custom_rec.id)
            assert loaded_rec is not None
            assert loaded_rec.question == "What is a yoke?"
            assert loaded_rec.answer == custom_rec.answer
