import pytest
from chains.clause_extractor import ExtractedClause, deduplicate_clauses


class TestExtractedClause:
    def test_create_clause(self):
        clause = ExtractedClause(
            clause_number="Art. 1",
            clause_text="Test clause text",
            topic="general"
        )
        assert clause.clause_number == "Art. 1"
        assert clause.clause_text == "Test clause text"
        assert clause.topic == "general"

    def test_clause_with_different_topics(self):
        topics = ["pets", "common_areas", "fees", "quorum", "visitors",
                  "property_use", "fines", "general"]
        for topic in topics:
            clause = ExtractedClause(
                clause_number="Art. 1",
                clause_text="Test",
                topic=topic
            )
            assert clause.topic == topic


class TestDeduplicateClauses:
    def test_removes_duplicates(self, sample_clauses):
        unique = deduplicate_clauses(sample_clauses)
        assert len(unique) == 2  # 3 clauses, 1 duplicate

    def test_preserves_first_occurrence(self, sample_clauses):
        unique = deduplicate_clauses(sample_clauses)
        clause_numbers = [c.clause_number for c in unique]
        assert clause_numbers == ["Art. 1", "Art. 2"]

    def test_empty_list(self):
        result = deduplicate_clauses([])
        assert result == []

    def test_no_duplicates(self):
        clauses = [
            ExtractedClause(
                clause_number="Art. 1",
                clause_text="First clause",
                topic="general"
            ),
            ExtractedClause(
                clause_number="Art. 2",
                clause_text="Second clause",
                topic="general"
            ),
        ]
        result = deduplicate_clauses(clauses)
        assert len(result) == 2

    def test_all_duplicates(self):
        clauses = [
            ExtractedClause(
                clause_number="Art. 1",
                clause_text="Same clause",
                topic="general"
            ),
            ExtractedClause(
                clause_number="Art. 1",
                clause_text="Same clause different text",
                topic="pets"
            ),
            ExtractedClause(
                clause_number="Art. 1",
                clause_text="Yet another",
                topic="fees"
            ),
        ]
        result = deduplicate_clauses(clauses)
        assert len(result) == 1
        assert result[0].clause_text == "Same clause"

    def test_single_clause(self):
        clauses = [
            ExtractedClause(
                clause_number="Art. 1",
                clause_text="Only clause",
                topic="general"
            ),
        ]
        result = deduplicate_clauses(clauses)
        assert len(result) == 1

    def test_preserves_clause_content(self):
        clauses = [
            ExtractedClause(
                clause_number="Art. 15, Par. 2",
                clause_text="E proibido fazer barulho apos as 22h.",
                topic="property_use"
            ),
        ]
        result = deduplicate_clauses(clauses)
        assert result[0].clause_number == "Art. 15, Par. 2"
        assert result[0].clause_text == "E proibido fazer barulho apos as 22h."
        assert result[0].topic == "property_use"
