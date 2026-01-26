import pytest
from chains.document_fields import DOCUMENT_FIELDS, get_document_fields


class TestDocumentFieldsDict:
    def test_has_expected_document_types(self):
        expected_types = [
            "notificacao_barulho",
            "notificacao_inadimplencia",
            "advertencia",
            "convocacao_assembleia",
            "ata_assembleia",
            "comunicado_geral"
        ]
        for doc_type in expected_types:
            assert doc_type in DOCUMENT_FIELDS

    def test_each_type_has_fields(self):
        for doc_type, fields in DOCUMENT_FIELDS.items():
            assert isinstance(fields, list)
            assert len(fields) > 0

    def test_fields_have_required_keys(self):
        required_keys = ["field_id", "label", "field_type", "required", "placeholder"]
        for doc_type, fields in DOCUMENT_FIELDS.items():
            for field in fields:
                for key in required_keys:
                    assert key in field, f"Missing '{key}' in {doc_type}"

    def test_all_types_have_nome_condominio(self):
        for doc_type, fields in DOCUMENT_FIELDS.items():
            field_ids = [f["field_id"] for f in fields]
            assert "nome_condominio" in field_ids, f"Missing 'nome_condominio' in {doc_type}"

    def test_all_types_have_nome_sindico(self):
        for doc_type, fields in DOCUMENT_FIELDS.items():
            field_ids = [f["field_id"] for f in fields]
            assert "nome_sindico" in field_ids, f"Missing 'nome_sindico' in {doc_type}"

    def test_field_types_are_valid(self):
        valid_types = ["text", "textarea", "date", "number", "select"]
        for doc_type, fields in DOCUMENT_FIELDS.items():
            for field in fields:
                assert field["field_type"] in valid_types, \
                    f"Invalid field_type '{field['field_type']}' in {doc_type}"

    def test_select_fields_have_options(self):
        for doc_type, fields in DOCUMENT_FIELDS.items():
            for field in fields:
                if field["field_type"] == "select":
                    assert "options" in field and field["options"], \
                        f"Select field without options in {doc_type}"


class TestGetDocumentFields:
    def test_returns_predefined_fields_for_known_types(self):
        for doc_type in DOCUMENT_FIELDS.keys():
            fields = get_document_fields(doc_type)
            assert fields == DOCUMENT_FIELDS[doc_type]

    def test_notificacao_barulho_fields(self):
        fields = get_document_fields("notificacao_barulho")
        field_ids = [f["field_id"] for f in fields]
        assert "unidade_infratora" in field_ids
        assert "descricao_barulho" in field_ids
        assert "data_ocorrencia" in field_ids

    def test_notificacao_inadimplencia_fields(self):
        fields = get_document_fields("notificacao_inadimplencia")
        field_ids = [f["field_id"] for f in fields]
        assert "unidade_devedora" in field_ids
        assert "valor_total" in field_ids
        assert "meses_devidos" in field_ids

    def test_convocacao_assembleia_fields(self):
        fields = get_document_fields("convocacao_assembleia")
        field_ids = [f["field_id"] for f in fields]
        assert "tipo_assembleia" in field_ids
        assert "data_assembleia" in field_ids
        assert "pauta" in field_ids

    def test_returns_list(self):
        for doc_type in DOCUMENT_FIELDS.keys():
            fields = get_document_fields(doc_type)
            assert isinstance(fields, list)


class TestNotificacaoBarulhoFields:
    def test_has_all_expected_fields(self):
        fields = DOCUMENT_FIELDS["notificacao_barulho"]
        field_ids = [f["field_id"] for f in fields]
        expected = [
            "nome_condominio",
            "unidade_infratora",
            "nome_morador",
            "data_ocorrencia",
            "hora_ocorrencia",
            "descricao_barulho",
            "nome_sindico"
        ]
        for expected_id in expected:
            assert expected_id in field_ids

    def test_required_fields_are_marked(self):
        fields = DOCUMENT_FIELDS["notificacao_barulho"]
        required_fields = [f for f in fields if f["required"]]
        required_ids = [f["field_id"] for f in required_fields]
        assert "nome_condominio" in required_ids
        assert "unidade_infratora" in required_ids
        assert "descricao_barulho" in required_ids
