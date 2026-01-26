import os
import sys

import pytest

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_markdown_text():
    """Sample text with markdown formatting."""
    return """# Title Here

This is **bold text** and this is *italic text*.

## Another Heading

More content with __underscored bold__ and _underscored italic_.

---

Final paragraph.
"""


@pytest.fixture
def sample_plain_text():
    """Sample plain text document."""
    return """NOTIFICACAO DE BARULHO

Prezado Morador,

Vimos por meio desta notificar que no dia 15 de janeiro de 2024,
foi registrada reclamacao de barulho proveniente de sua unidade.

Solicitamos que observe o horario de silencio estabelecido no
regimento interno.

Atenciosamente,
Maria Santos
Sindica
"""


@pytest.fixture
def sample_clauses():
    """Sample clause data for testing."""
    from chains.clause_extractor import ExtractedClause
    return [
        ExtractedClause(
            clause_number="Art. 1",
            clause_text="Os moradores devem respeitar o silencio.",
            topic="property_use"
        ),
        ExtractedClause(
            clause_number="Art. 2",
            clause_text="E proibido fumar nas areas comuns.",
            topic="common_areas"
        ),
        ExtractedClause(
            clause_number="Art. 1",  # Duplicate
            clause_text="Os moradores devem respeitar o silencio.",
            topic="property_use"
        ),
    ]
