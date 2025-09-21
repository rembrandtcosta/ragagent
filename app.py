import streamlit as st
from rag_workflow import process_question

import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

st.title("Assistente de Legislação Condominial Brasileira")

query = st.chat_input(placeholder="O que você gostaria de saber?")
if query:
    with st.chat_message("user"):
        st.write(query)

    result = process_question(query)
    print(result)
    with st.chat_message("assistant"):
        st.write(result["solution"])
        st.write("**Fonte:**")
        for doc in result["documents"]:
            st.write(f'- {doc.page_content}')
