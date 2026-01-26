import pytest
from utils.document_formatter import (
    strip_markdown,
    is_title_line,
    sanitize_text,
    text_to_docx,
    text_to_pdf,
    format_document
)


class TestStripMarkdown:
    def test_removes_bold_asterisks(self):
        text = "This is **bold** text"
        result = strip_markdown(text)
        assert result == "This is bold text"

    def test_removes_bold_underscores(self):
        text = "This is __bold__ text"
        result = strip_markdown(text)
        assert result == "This is bold text"

    def test_removes_italic_asterisks(self):
        text = "This is *italic* text"
        result = strip_markdown(text)
        assert result == "This is italic text"

    def test_removes_italic_underscores(self):
        text = "This is _italic_ text"
        result = strip_markdown(text)
        assert result == "This is italic text"

    def test_removes_heading_markers(self):
        text = "# Heading 1\n## Heading 2\n### Heading 3"
        result = strip_markdown(text)
        assert "Heading 1" in result
        assert "#" not in result

    def test_removes_horizontal_rules(self):
        text = "Before\n---\nAfter"
        result = strip_markdown(text)
        assert "---" not in result

    def test_handles_empty_string(self):
        assert strip_markdown("") == ""

    def test_handles_text_without_markdown(self):
        text = "Plain text without any formatting"
        assert strip_markdown(text) == text


class TestIsTitleLine:
    def test_uppercase_title(self):
        assert is_title_line("NOTIFICACAO DE BARULHO") is True

    def test_mostly_uppercase(self):
        # Needs >70% uppercase to be considered a title
        assert is_title_line("TITULO DO DOCUMENTO") is True
        # This one has too many lowercase letters
        assert is_title_line("TITULO Com Algumas Minusculas") is False

    def test_lowercase_not_title(self):
        assert is_title_line("this is not a title") is False

    def test_short_line_not_title(self):
        assert is_title_line("AB") is False

    def test_empty_line_not_title(self):
        assert is_title_line("") is False
        assert is_title_line("   ") is False

    def test_long_line_not_title(self):
        long_text = "A" * 100
        assert is_title_line(long_text) is False

    def test_numbers_only_not_title(self):
        assert is_title_line("12345") is False


class TestSanitizeText:
    def test_replaces_smart_quotes(self):
        text = "He said \u201chello\u201d"
        result = sanitize_text(text)
        assert '"' in result
        assert '\u201c' not in result
        assert '\u201d' not in result

    def test_replaces_em_dash(self):
        text = "word\u2014word"
        result = sanitize_text(text)
        assert "-" in result

    def test_replaces_ellipsis(self):
        text = "and so\u2026"
        result = sanitize_text(text)
        assert "..." in result

    def test_replaces_ordinal_indicators(self):
        text = "1\u00ba lugar"
        result = sanitize_text(text)
        assert "1o lugar" in result

    def test_handles_normal_text(self):
        text = "Normal ASCII text"
        result = sanitize_text(text)
        assert result == "Normal ASCII text"


class TestTextToDocx:
    def test_returns_bytes(self, sample_plain_text):
        result = text_to_docx(sample_plain_text)
        assert isinstance(result, bytes)

    def test_returns_non_empty(self, sample_plain_text):
        result = text_to_docx(sample_plain_text)
        assert len(result) > 0

    def test_produces_valid_docx_header(self, sample_plain_text):
        result = text_to_docx(sample_plain_text)
        # DOCX files are ZIP files starting with PK
        assert result[:2] == b'PK'

    def test_handles_empty_text(self):
        result = text_to_docx("")
        assert isinstance(result, bytes)


class TestTextToPdf:
    def test_returns_bytes(self, sample_plain_text):
        result = text_to_pdf(sample_plain_text)
        assert isinstance(result, bytes)

    def test_returns_non_empty(self, sample_plain_text):
        result = text_to_pdf(sample_plain_text)
        assert len(result) > 0

    def test_produces_valid_pdf_header(self, sample_plain_text):
        result = text_to_pdf(sample_plain_text)
        # PDF files start with %PDF
        assert result[:4] == b'%PDF'

    def test_handles_empty_text(self):
        result = text_to_pdf("")
        assert isinstance(result, bytes)

    def test_handles_special_characters(self):
        text = "Texto com caracteres: \u00e9, \u00e3, \u00f5"
        result = text_to_pdf(text)
        assert isinstance(result, bytes)


class TestFormatDocument:
    def test_txt_format(self, sample_plain_text):
        data, mime, ext = format_document(sample_plain_text, "txt")
        assert isinstance(data, bytes)
        assert mime == "text/plain"
        assert ext == ".txt"

    def test_docx_format(self, sample_plain_text):
        data, mime, ext = format_document(sample_plain_text, "docx")
        assert isinstance(data, bytes)
        assert mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert ext == ".docx"

    def test_pdf_format(self, sample_plain_text):
        data, mime, ext = format_document(sample_plain_text, "pdf")
        assert isinstance(data, bytes)
        assert mime == "application/pdf"
        assert ext == ".pdf"

    def test_unknown_format_defaults_to_txt(self, sample_plain_text):
        data, mime, ext = format_document(sample_plain_text, "unknown")
        assert mime == "text/plain"
        assert ext == ".txt"

    def test_strips_markdown_from_input(self, sample_markdown_text):
        data, _, _ = format_document(sample_markdown_text, "txt")
        text = data.decode('utf-8')
        assert "**" not in text
        assert "##" not in text
