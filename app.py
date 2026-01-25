import json
import streamlit as st
from rag_workflow import process_question, set_internal_retriever, analyze_document

import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

st.title("Assistente de Legislação Condominial Brasileira")

# Mode toggle
mode = st.radio(
    "Selecione o modo:",
    ["Consulta", "Análise de Legalidade"],
    horizontal=True
)

if mode == "Consulta":
    # Original query mode
    query = st.chat_input(placeholder="O que você gostaria de saber?")
    if query:
        with st.chat_message("user"):
            st.write(query)

        result, _ = process_question(query)
        print(result)
        with st.chat_message("assistant"):
            st.write(result["solution"])
            st.write("**Fonte:**")
            for doc in result["documents"]:
                st.write(f'- {doc.page_content}')

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
