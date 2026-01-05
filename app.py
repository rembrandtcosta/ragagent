import streamlit as st
from rag_workflow import process_question, set_internal_retriever

import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

st.title("Assistente de Legislação Condominial Brasileira")

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
    """Shows the document upload section"""
    # Upload area with simple styling
    st.info("**Drag & Drop Your Document**\n\n")

    # File uploader
    user_file = st.file_uploader(
        "Choose a file",
        type="pdf",
        help="Upload any supported document type.",
        label_visibility="collapsed"
    )

    return user_file


user_file = render_upload_section()
if user_file is not None:
    file_bytes = user_file.read()
    set_internal_retriever(file_bytes)
