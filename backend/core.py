import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from typing import Any, Dict, List
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from mistralai import Mistral

load_dotenv()

MISTRAL_API_KEY=os.environ["MISTRAL_API_KEY"]
PERSIST_DIRECTORY="./.chroma"
COLLECTION_NAME="rag-chroma"
EMBEDDING_MODEL="mistral-embed"
SIMILARITY_THRESHOLD=0.5

api_key=os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL, mistral_api_key=MISTRAL_API_KEY)
llm = ChatMistralAI(model="open-mistral-nemo-2407", mistral_api_key=MISTRAL_API_KEY)

retriever = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
).as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,
        "k": 1, # coger sólo un documento
    }
)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    template = """ 
        Eres un asistente para tareas de pregunta-respuesta. Usa el contexto indicado para responder la pregunta.
        Si no conoces la respuesta, simplemente di que no lo sabes.
        Responde sólo a preguntas sobre procedimientos de la sede electrónica del Ayuntamiento de Murcia.
        Para cualquier otra pregunta, responde que sólo estás entrenado para responder sobre la sede electrónica del Ayuntamiento de Murcia. 
        Pregunta:
        {input}
        Historial:
        {chat_history}
        Contexto:
        {context}
        """
    retrieval_qa_chat_prompt = PromptTemplate(template=template, input_variables=["input", "chat_history", "context"])

    template_rephrase = """
        Dada la conversación y pregunta que aparecen a continuación, reformula la pregunta teniendo en cuenta la conversación.  Trata de que la pregunta no sea breve para incluir el mayor contexto posible.
        
        Converesación: {chat_history}
        Pregunta: {input}
        Pregunta reformulada:
        """
    rephrase_prompt = PromptTemplate(template=template_rephrase, input_variables=["input", "chat_history"])

    def get_relevant_documents(query):
        """Recupera documentos relevantes y filtra por similitud."""
        docs_and_scores = retriever.get_relevant_documents(query)
        return [doc for doc, score in docs_and_scores if score >= SIMILARITY_THRESHOLD]

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=rephrase_prompt,
    )

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=combine_docs_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result


if __name__ == "__main__":
    #res = run_llm(query="tengo un perro peligroso. Tengo que informar al Ayuntamiento?")
    res = run_llm(query="Soy funcionario del Ayuntamiento.  Cómo puedo solicitar la carrera profesional?")

    print(res)
