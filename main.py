from backend.core import run_llm
import streamlit as st
from typing import Set


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i + 1}. {source}\n"
    return sources_string


st.header("Asistente de procedimientos de la sede electrónica del Ayuntamiento de Murcia")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if prompt:
        with st.spinner("Generation response.."):
            generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])

            sources = set(
                [doc.metadata["source"] for doc in generated_response["context"]]
            )

            response = (
                f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
            )
            st.session_state["chat_history"].extend([prompt, response])

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Botón para borrar la conversación
if st.button("Borrar conversación"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    # Opcional: Recargar la página para reflejar el cambio
    st.rerun()