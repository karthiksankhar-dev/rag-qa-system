import os
import streamlit as st
from langchain_community.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_ollama import ChatOllama
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.callbacks.streamlit.streamlit_callback_handler import LLMThoughtLabeler
from id import nvapi_key, SERPER_API_KEY

# --- API keys ---
os.environ["SERPER_API_KEY"] = SERPER_API_KEY
os.environ["NVIDIA_API_KEY"] = nvapi_key

# --- Data sources ---
search = GoogleSerperAPIWrapper()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# --- Streaming LLM ---
llm = ChatOllama(model="llama3.1", temperature=0, streaming=True)

# --- Embeddings / Vector DB ---
embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
persist_directory = "./chroma_db"
loaded_vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# --- Text splitter ---
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# --- Custom labeler to replace 'Thinking...' ---
class GeneratingLabeler(LLMThoughtLabeler):
    @staticmethod
    def get_initial_label() -> str:
        return "Generating answer…"

    @staticmethod
    def get_final_agent_thought_label() -> str:
        return "Generating answer…"

def rag_pipeline(user_query, st_callback):
    # Retrieve external content
    google_results = search.run(user_query)
    wiki_results = wikipedia.run(user_query)

    # Prepare docs
    combined_docs = [google_results, wiki_results]
    docs_chunked = splitter.create_documents(combined_docs)

    # Upsert to vectorstore
    loaded_vectorstore.add_documents(docs_chunked)
    retriever = loaded_vectorstore.as_retriever()

    # Build RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    # Stream tokens
    answer = rag_chain.run(user_query, callbacks=[st_callback])

    # Ensure the container stays open/expanded on all versions
    try:
        st_callback._current_thought._container.update(
            label="Done",
            state="complete",
            expanded=True,
        )
    except Exception:
        pass

    return answer

# --- UI ---
st.set_page_config(page_title="RAG-Powered Question Answering System", page_icon="✨", layout="centered")
st.title("RAG-Powered Question Answering System")
st.write("Ask me anything!")

user_question = st.text_input("Enter your question:")

# Single output area (no chat bubble/icon)
output_area = st.container()

if user_question:
    st_callback = StreamlitCallbackHandler(
        parent_container=output_area,
        expand_new_thoughts=True,
        collapse_completed_thoughts=False,   # keep open
        thought_labeler=GeneratingLabeler(), # custom label text
    )
    rag_pipeline(user_question, st_callback)
