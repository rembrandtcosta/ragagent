import json
import streamlit as st
from rag_workflow import (
    process_question, set_internal_retriever, analyze_document,
    check_document_suggestion, write_document, detect_explicit_document_request
)

import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

st.title("Assistente de Legislação Condominial Brasileira")

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

# Mode toggle
mode = st.radio(
    "Selecione o modo:",
    ["Consulta", "Análise de Legalidade"],
    horizontal=True
)

if mode == "Consulta":
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                st.write("**Fonte:**")
                for source in message["sources"]:
                    st.write(f'- {source}')

    # Handle document generation if user accepted
    if st.session_state.generated_document:
        with st.chat_message("assistant"):
            st.markdown("### Documento Gerado")
            st.markdown(st.session_state.generated_document)

            # Download button for the document
            st.download_button(
                label="Download Documento (.txt)",
                data=st.session_state.generated_document,
                file_name=f"documento_{st.session_state.pending_suggestion['document_type']}.txt",
                mime="text/plain"
            )
        st.session_state.generated_document = None
        st.session_state.pending_suggestion = None

    # Show document suggestion if pending
    if st.session_state.pending_suggestion and not st.session_state.generated_document:
        suggestion = st.session_state.pending_suggestion
        st.info(f"**Sugestão:** {suggestion['suggestion_message']}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sim, redigir documento", type="primary"):
                with st.spinner("Gerando documento..."):
                    document = write_document(
                        document_type=suggestion["document_type"],
                        document_name=suggestion["document_name"],
                        original_question=st.session_state.last_question,
                        previous_answer=st.session_state.last_answer
                    )
                    st.session_state.generated_document = document
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"### Documento Gerado\n\n{document}"
                    })
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
            # Bypass RAG and go straight to document generation
            with st.spinner("Gerando documento..."):
                document = write_document(
                    document_type=doc_request["document_type"],
                    document_name=doc_request["document_name"] or doc_request["document_type"],
                    original_question=query,
                    previous_answer="",
                    additional_info=doc_request.get("extracted_context") or ""
                )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"### Documento Gerado\n\n{document}"
            })

            with st.chat_message("assistant"):
                st.markdown("### Documento Gerado")
                st.markdown(document)
                st.download_button(
                    label="Download Documento (.txt)",
                    data=document,
                    file_name=f"documento_{doc_request['document_type']}.txt",
                    mime="text/plain"
                )
        else:
            # Normal RAG flow
            result, _ = process_question(query)
            answer = result["solution"]
            sources = [doc.page_content for doc in result["documents"]]

            # Add assistant message to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

            with st.chat_message("assistant"):
                st.write(answer)
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

    def render_upload_section():
        """Shows the document upload section for internal docs"""
        st.info("**Faça upload do documento interno do condomínio**\n\n")
        user_file = st.file_uploader(
            "Escolha um arquivo",
            type="pdf",
            help="Upload de convenção ou regimento interno para consultas contextualizadas.",
            label_visibility="collapsed",
            key="internal_doc"
        )
        return user_file

    user_file = render_upload_section()
    if user_file is not None:
        file_bytes = user_file.read()
        set_internal_retriever(file_bytes)
        st.success("Documento carregado com sucesso!")

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
