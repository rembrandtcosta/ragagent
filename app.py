import asyncio
import json

import streamlit as st

from rag_workflow import (
    analyze_document,
    check_document_suggestion,
    clear_internal_documents,
    detect_explicit_document_request,
    get_document_fields,
    get_internal_document_names,
    identify_used_sources,
    process_question,
    recreate_graph,
    set_internal_retriever,
    write_document,
)
from utils.document_formatter import format_document

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

st.title("Assistente de Legislação Condominial Brasileira")


def render_download_buttons(document_text: str, base_filename: str, title: str = "Documento"):
    """Render download buttons for multiple formats."""
    st.markdown("**Download do documento:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        txt_data, txt_mime, txt_ext = format_document(document_text, "txt", title)
        st.download_button(
            label="TXT",
            data=txt_data,
            file_name=f"{base_filename}{txt_ext}",
            mime=txt_mime,
            key=f"txt_{base_filename}"
        )

    with col2:
        docx_data, docx_mime, docx_ext = format_document(document_text, "docx", title)
        st.download_button(
            label="DOCX",
            data=docx_data,
            file_name=f"{base_filename}{docx_ext}",
            mime=docx_mime,
            key=f"docx_{base_filename}"
        )

    with col3:
        pdf_data, pdf_mime, pdf_ext = format_document(document_text, "pdf", title)
        st.download_button(
            label="PDF",
            data=pdf_data,
            file_name=f"{base_filename}{pdf_ext}",
            mime=pdf_mime,
            key=f"pdf_{base_filename}"
        )

# Initialize session state for document suggestion flow
if "pending_suggestion" not in st.session_state:
    st.session_state.pending_suggestion = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "generated_document" not in st.session_state:
    st.session_state.generated_document = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_form" not in st.session_state:
    st.session_state.document_form = None  # Stores form data when collecting info
if "loaded_documents" not in st.session_state:
    st.session_state.loaded_documents = []  # List of loaded document names

# Mode toggle
mode = st.radio(
    "Selecione o modo:",
    ["Consulta", "Análise de Legalidade"],
    horizontal=True
)

if mode == "Consulta":
    # Display chat history
    for idx, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                st.write("**Fonte:**")
                for source in message["sources"]:
                    st.write(f'- {source}')
            # Render download buttons for document messages
            if message.get("document_data"):
                render_download_buttons(
                    message["document_data"]["text"],
                    f"documento_{message['document_data']['type']}_{idx}",
                    message["document_data"].get("name", "Documento")
                )

    # Handle document generation if user accepted (from suggestion)
    if st.session_state.generated_document:
        doc_type = st.session_state.pending_suggestion['document_type']
        doc_name = st.session_state.pending_suggestion.get('document_name', doc_type)

        # Add to chat history with document data for persistent downloads
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"### Documento Gerado\n\n{st.session_state.generated_document}",
            "document_data": {
                "text": st.session_state.generated_document,
                "type": doc_type,
                "name": doc_name
            }
        })
        st.session_state.generated_document = None
        st.session_state.pending_suggestion = None
        st.rerun()

    # Show document form if collecting info
    if st.session_state.document_form:
        form_data = st.session_state.document_form
        st.markdown(f"### Preencha as informações para: {form_data['document_name']}")

        with st.form(key="document_info_form"):
            field_values = {}
            for field in form_data["fields"]:
                field_id = field["field_id"]
                label = field["label"] + (" *" if field.get("required") else "")

                if field["field_type"] == "textarea":
                    field_values[field_id] = st.text_area(
                        label,
                        placeholder=field.get("placeholder", ""),
                        key=f"form_{field_id}"
                    )
                elif field["field_type"] == "date":
                    field_values[field_id] = st.date_input(
                        label,
                        value=None,
                        key=f"form_{field_id}"
                    )
                elif field["field_type"] == "number":
                    field_values[field_id] = st.number_input(
                        label,
                        min_value=0,
                        key=f"form_{field_id}"
                    )
                elif field["field_type"] == "select" and field.get("options"):
                    field_values[field_id] = st.selectbox(
                        label,
                        options=field["options"],
                        key=f"form_{field_id}"
                    )
                else:
                    field_values[field_id] = st.text_input(
                        label,
                        placeholder=field.get("placeholder", ""),
                        key=f"form_{field_id}"
                    )

            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Gerar Documento", type="primary")
            with col2:
                cancel = st.form_submit_button("Cancelar")

            if submit:
                # Format field values for the document
                info_text = "\n".join([
                    f"{field['label']}: {field_values[field['field_id']]}"
                    for field in form_data["fields"]
                    if field_values.get(field["field_id"])
                ])

                with st.spinner("Gerando documento..."):
                    document = write_document(
                        document_type=form_data["document_type"],
                        document_name=form_data["document_name"],
                        original_question=form_data.get("original_question", ""),
                        previous_answer=form_data.get("previous_answer", ""),
                        additional_info=info_text
                    )

                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"### Documento Gerado\n\n{document}",
                    "document_data": {
                        "text": document,
                        "type": form_data["document_type"],
                        "name": form_data["document_name"]
                    }
                })
                st.session_state.document_form = None
                st.session_state.pending_suggestion = None
                st.rerun()

            if cancel:
                st.session_state.document_form = None
                st.session_state.pending_suggestion = None
                st.rerun()

    # Show document suggestion if pending (and no form active)
    elif st.session_state.pending_suggestion and not st.session_state.generated_document:
        suggestion = st.session_state.pending_suggestion
        st.info(f"**Sugestão:** {suggestion['suggestion_message']}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sim, redigir documento", type="primary"):
                # Get fields for the document type
                fields = get_document_fields(
                    suggestion["document_type"],
                    suggestion.get("document_name", "")
                )
                st.session_state.document_form = {
                    "document_type": suggestion["document_type"],
                    "document_name": suggestion.get("document_name", suggestion["document_type"]),
                    "fields": fields,
                    "original_question": st.session_state.last_question,
                    "previous_answer": st.session_state.last_answer
                }
                st.rerun()
        with col2:
            if st.button("Não, obrigado"):
                st.session_state.pending_suggestion = None
                st.rerun()

    # Original query mode
    query = st.chat_input(placeholder="O que você gostaria de saber?")
    if query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        # Check if this is an explicit document request
        doc_request = detect_explicit_document_request(query)

        if doc_request.get("is_explicit_request") and doc_request.get("document_type"):
            # Show form to collect information before generating
            doc_type = doc_request["document_type"]
            doc_name = doc_request.get("document_name") or doc_type

            fields = get_document_fields(doc_type, doc_name, doc_request.get("extracted_context", ""))

            st.session_state.document_form = {
                "document_type": doc_type,
                "document_name": doc_name,
                "fields": fields,
                "original_question": query,
                "previous_answer": "",
                "extracted_context": doc_request.get("extracted_context", "")
            }
            st.rerun()
        else:
            # Normal RAG flow
            result, _ = process_question(query)
            answer = result["solution"]

            # Identify which documents were actually used in the answer
            used_docs = identify_used_sources(answer, result["documents"], query)

            # Deduplicate sources by page_content
            seen_sources = set()
            sources = []
            for doc in used_docs:
                if doc.page_content not in seen_sources:
                    seen_sources.add(doc.page_content)
                    sources.append(doc.page_content)

            # Add assistant message to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

            with st.chat_message("assistant"):
                st.write(answer)
                if sources:
                    st.write("**Fonte:**")
                    for source in sources:
                        st.write(f'- {source}')

            # Check for document suggestion
            suggestion = check_document_suggestion(query, answer)
            if suggestion.get("should_suggest"):
                st.session_state.pending_suggestion = suggestion
                st.session_state.last_question = query
                st.session_state.last_answer = answer
                st.rerun()

    # Document management section
    st.sidebar.markdown("### Documentos Indexados")

    if st.session_state.loaded_documents:
        st.sidebar.success(f"{len(st.session_state.loaded_documents)} documento(s) carregado(s)")
        for doc_name in st.session_state.loaded_documents:
            st.sidebar.markdown(f"- {doc_name}")

        if st.sidebar.button("Limpar todos os documentos", key="clear_docs", type="secondary"):
            clear_internal_documents()
            recreate_graph()
            st.session_state.loaded_documents = []
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.caption("Para substituir os documentos, basta fazer upload de novos arquivos.")
    else:
        st.sidebar.info("Nenhum documento carregado")

    # Upload section in main area
    with st.expander("Upload de Documentos Internos", expanded=not st.session_state.loaded_documents):
        st.markdown("Faça upload dos documentos internos do condomínio (convenção, regimento interno, atas, etc.)")
        st.caption("Os documentos anteriores serão substituídos ao fazer novo upload.")

        user_files = st.file_uploader(
            "Escolha os arquivos",
            type="pdf",
            help="Upload de convenção, regimento interno ou outros documentos para consultas contextualizadas.",
            label_visibility="collapsed",
            key="internal_docs",
            accept_multiple_files=True
        )

        if user_files:
            # Check if these are new files
            new_file_names = [f.name for f in user_files]
            if set(new_file_names) != set(st.session_state.loaded_documents):
                with st.spinner("Indexando documentos..."):
                    # Prepare list of (filename, bytes) tuples
                    documents = [(f.name, f.read()) for f in user_files]
                    count = set_internal_retriever(documents)
                    recreate_graph()
                    st.session_state.loaded_documents = new_file_names

                st.success(f"{count} documento(s) indexado(s) com sucesso!")
                st.rerun()

else:
    # Analysis mode
    st.markdown("### Análise de Legalidade de Documentos Condominiais")
    st.markdown(
        "Faça upload de uma convenção ou regimento interno para identificar "
        "cláusulas potencialmente ilegais segundo o Código Civil brasileiro."
    )

    analysis_file = st.file_uploader(
        "Upload do documento para análise",
        type="pdf",
        help="Faça upload da convenção ou regimento interno em PDF.",
        key="analysis_doc"
    )

    if analysis_file is not None:
        if st.button("Analisar Documento", type="primary"):
            with st.spinner("Analisando documento... Isso pode levar alguns minutos."):
                file_bytes = analysis_file.read()
                report = analyze_document(file_bytes, analysis_file.name)

            # Display summary metrics
            st.markdown("---")
            st.markdown("## Resultado da Análise")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Cláusulas", report.total_clauses_analyzed)
            with col2:
                st.metric("Potencialmente Ilegais", report.potentially_illegal_count)
            with col3:
                if report.total_clauses_analyzed > 0:
                    conformity = ((report.total_clauses_analyzed - report.potentially_illegal_count)
                                  / report.total_clauses_analyzed * 100)
                    st.metric("Taxa de Conformidade", f"{conformity:.1f}%")
                else:
                    st.metric("Taxa de Conformidade", "N/A")

            # Display flagged clauses
            if report.potentially_illegal_count > 0:
                st.markdown("### Cláusulas Potencialmente Ilegais")

                illegal_clauses = [c for c in report.clauses if c.is_potentially_illegal]
                for clause in illegal_clauses:
                    with st.expander(
                        f"⚠️ {clause.clause_number} - {clause.topic.upper()} "
                        f"(Confiança: {clause.confidence})"
                    ):
                        st.markdown("**Texto da Cláusula:**")
                        st.info(clause.clause_text)

                        st.markdown("**Artigos do Código Civil Conflitantes:**")
                        for article in clause.conflicting_articles:
                            st.warning(f"• {article}")

                        if clause.legal_principle_violated:
                            st.markdown(f"**Princípio Violado:** {clause.legal_principle_violated}")

                        st.markdown("**Explicação:**")
                        st.write(clause.explanation)

                        st.markdown("**Recomendação:**")
                        st.success(clause.recommendation)
            else:
                st.success("Nenhuma cláusula potencialmente ilegal foi identificada.")

            # Display all analyzed clauses
            with st.expander("Ver todas as cláusulas analisadas"):
                for clause in report.clauses:
                    status = "⚠️" if clause.is_potentially_illegal else "✅"
                    st.markdown(f"{status} **{clause.clause_number}** ({clause.topic})")
                    st.caption(clause.clause_text[:200] + "..." if len(clause.clause_text) > 200 else clause.clause_text)
                    st.divider()

            # JSON download
            st.markdown("---")
            report_json = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 Download Relatório JSON",
                data=report_json,
                file_name=f"analise_{analysis_file.name.replace('.pdf', '')}.json",
                mime="application/json"
            )
