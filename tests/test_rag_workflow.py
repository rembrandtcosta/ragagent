import pytest
from datetime import datetime
from rag_workflow import (
    ClauseAnalysisResult,
    DocumentAnalysisReport,
    clear_internal_documents,
    get_internal_document_names,
)


class TestClauseAnalysisResult:
    def test_create_result(self):
        result = ClauseAnalysisResult(
            clause_number="Art. 1",
            clause_text="Proibido animais de estimacao.",
            topic="pets",
            is_potentially_illegal=True,
            confidence="alta",
            conflicting_articles=["Art. 1228", "Art. 1336 IV"],
            explanation="Proibicao total de animais viola direito de propriedade.",
            legal_principle_violated="Direito de propriedade",
            recommendation="Reformular para permitir animais com restricoes."
        )
        assert result.clause_number == "Art. 1"
        assert result.is_potentially_illegal is True
        assert result.confidence == "alta"
        assert len(result.conflicting_articles) == 2

    def test_result_with_no_illegality(self):
        result = ClauseAnalysisResult(
            clause_number="Art. 5",
            clause_text="O horario de silencio e das 22h as 8h.",
            topic="property_use",
            is_potentially_illegal=False,
            confidence="alta",
            conflicting_articles=[],
            explanation="Clausula em conformidade com o Codigo Civil.",
            legal_principle_violated=None,
            recommendation="Manter clausula como esta."
        )
        assert result.is_potentially_illegal is False
        assert result.legal_principle_violated is None
        assert result.conflicting_articles == []


class TestDocumentAnalysisReport:
    @pytest.fixture
    def sample_report(self):
        clauses = [
            ClauseAnalysisResult(
                clause_number="Art. 1",
                clause_text="Proibido animais.",
                topic="pets",
                is_potentially_illegal=True,
                confidence="alta",
                conflicting_articles=["Art. 1228"],
                explanation="Viola direito de propriedade.",
                legal_principle_violated="Propriedade",
                recommendation="Reformular."
            ),
            ClauseAnalysisResult(
                clause_number="Art. 2",
                clause_text="Silencio das 22h as 8h.",
                topic="property_use",
                is_potentially_illegal=False,
                confidence="alta",
                conflicting_articles=[],
                explanation="Conforme.",
                legal_principle_violated=None,
                recommendation="Manter."
            ),
        ]
        return DocumentAnalysisReport(
            document_name="convencao_teste.pdf",
            analysis_date="2024-01-15T10:30:00",
            total_clauses_analyzed=2,
            potentially_illegal_count=1,
            clauses=clauses
        )

    def test_create_report(self, sample_report):
        assert sample_report.document_name == "convencao_teste.pdf"
        assert sample_report.total_clauses_analyzed == 2
        assert sample_report.potentially_illegal_count == 1
        assert len(sample_report.clauses) == 2

    def test_to_dict(self, sample_report):
        result = sample_report.to_dict()
        assert isinstance(result, dict)
        assert result["document_name"] == "convencao_teste.pdf"
        assert result["total_clauses_analyzed"] == 2
        assert result["potentially_illegal_count"] == 1
        assert len(result["clauses"]) == 2

    def test_to_dict_clauses_are_dicts(self, sample_report):
        result = sample_report.to_dict()
        for clause in result["clauses"]:
            assert isinstance(clause, dict)
            assert "clause_number" in clause
            assert "is_potentially_illegal" in clause

    def test_empty_report(self):
        report = DocumentAnalysisReport(
            document_name="empty.pdf",
            analysis_date="2024-01-15T10:30:00",
            total_clauses_analyzed=0,
            potentially_illegal_count=0,
            clauses=[]
        )
        result = report.to_dict()
        assert result["total_clauses_analyzed"] == 0
        assert result["clauses"] == []


class TestInternalDocumentManagement:
    def test_clear_internal_documents(self):
        # Call clear and verify it doesn't raise errors
        clear_internal_documents()
        names = get_internal_document_names()
        assert names == []

    def test_get_internal_document_names_returns_list(self):
        result = get_internal_document_names()
        assert isinstance(result, list)

    def test_get_internal_document_names_returns_copy(self):
        # Ensure we get a copy, not the original list
        result1 = get_internal_document_names()
        result2 = get_internal_document_names()
        assert result1 is not result2
