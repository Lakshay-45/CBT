import streamlit as st
import openai
from dotenv import load_dotenv
import os
import tempfile
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Load environment variables from .env file
# Create the env file with mistral 7B instruct key from openrouter
load_dotenv()

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME = "mistralai/mistral-7b-instruct:free"

CHROMA_PATH = "chroma_db_session"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# --- LLM and Vector Store Initialization ---
@st.cache_resource
def load_llm_client():
    """Initialize the LLM client (OpenRouter)."""
    client = openai.OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_API_BASE,
    )
    return client


@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


client = load_llm_client()
embeddings = load_embedding_model()


# --- Document Processing Function ---
def process_documents(uploaded_files):
    """
    Loads, splits, and embeds the uploaded documents, then creates a vector store.
    """
    with st.spinner("Processing documents... This may take a moment."):
        # Create a temporary directory to store uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            docs = []
            for uploaded_file in uploaded_files:
                temp_filepath = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_filepath, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Use the appropriate loader based on file extension
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(temp_filepath)
                elif uploaded_file.name.endswith('.docx'):
                    loader = Docx2txtLoader(temp_filepath)
                elif uploaded_file.name.endswith('.txt'):
                    loader = TextLoader(temp_filepath)
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue

                docs.extend(loader.load())

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)

            # Create an in-memory Chroma vector store
            vector_store = Chroma.from_documents(chunks, embeddings)
            st.session_state.vector_store = vector_store

    st.success("Documents processed successfully! You can now ask questions.")


# --- RAG and Chat Logic ---
def get_llm_response(query):
    """
    Generate a response from the LLM using the retrieved context.
    """
    vector_store = st.session_state.vector_store
    retrieved_docs = vector_store.similarity_search(query, k=5)
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    You are an expert Q&A assistant. Use the following retrieved context to answer the user's question.
    If you don't know the answer from the context, state that the information is not available in the provided documents.
    Do not make up information. Be concise and clear.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


# --- Streamlit UI ---
st.set_page_config(page_title="Dynamic RAG Chatbot", page_icon="📄")
st.title("📄 Dynamic RAG Q&A Chatbot")
st.write("Upload your documents and ask questions about their content.")

# Sidebar for document upload
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your .txt, .pdf, or .docx files",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents"):
            process_documents(uploaded_files)

# Main chat interface
st.header("Ask a Question")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages only if documents have been processed
if "vector_store" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            response = get_llm_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
else:
    st.info("Please upload and process your documents in the sidebar to begin.")