import streamlit as st
from langchain.document_loaders import GithubFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import spacy
from spacy.cli import download as spacy_download

# Function to load spaCy model
# @st.cache_resource
# def load_spacy_model():
#     model_name = "en_core_web_sm"
#     try:
#         nlp = spacy.load(model_name)
#     except OSError:
#         spacy_download(model_name)
#         nlp = spacy.load(model_name)
#     return nlp

# Function to load documents from GitHub
@st.cache_data
def load_documents():
    loader = GithubFileLoader(
        repo="panaversity/learn-applied-generative-ai-fundamentals",
        branch="main",
        access_token=st.secrets["github"]["access_token"],
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(".md")
    )
    docs = loader.load()
    return docs

# Function to split documents
@st.cache_data
def split_documents(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(_docs)
    return splits

# Function to create embeddings
@st.cache_resource
def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

# Function to create vector store
@st.cache_resource
def create_vector_store(_splits, _embeddings):
    vectorstore = FAISS.from_documents(documents=_splits, embedding=_embeddings)
    return vectorstore

# Function to initialize LLM
@st.cache_resource
def initialize_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=st.secrets["google"]["api_key"]
    )
    return llm

# Function to create RAG chain
def create_rag_chain(retriever, llm):
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def main():
    st.set_page_config(page_title="LangChain QA App", layout="wide")
    st.title("📚 QA with panaversity 'learn applied generative ai' repository")
    
    # Sidebar for app settings
    st.sidebar.header("🔧 Settings")
    with st.sidebar.expander("📄 API Keys"):
        st.write("Ensure your `secrets.toml` is configured properly.")
    
    # Load spaCy model
    # nlp = load_spacy_model()
    
    # Load and process documents
    with st.spinner("📥 Loading documents from GitHub..."):
        docs = load_documents()
    st.success(f"✅ Loaded {len(docs)} documents.")
    
    with st.spinner("✂️ Splitting documents into chunks..."):
        splits = split_documents(docs)
    st.success(f"✅ Split into {len(splits)} chunks.")
    
    # Display a sample chunk
    if splits:
        st.subheader("📝 Sample Document Chunk")
        with st.expander("View Sample"):
            st.write(splits[0].page_content)
    
    # Create embeddings
    with st.spinner("🔗 Creating embeddings..."):
        embeddings = create_embeddings()
    st.success("✅ Embeddings created successfully.")
    
    # Display embedding details
    st.write("**Embedding Model:** all-MiniLM-L6-v2")
    
    # Create vector store
    with st.spinner("🗄️ Creating vector store..."):
        vectorstore = create_vector_store(splits, embeddings)
    st.success(f"✅ Vector store created with {vectorstore.index.ntotal} embeddings.")
    
    # Initialize retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 similar documents
    )
    
    # Initialize LLM
    with st.spinner("🤖 Initializing Language Model..."):
        llm = initialize_llm()
    st.success("✅ Language Model initialized.")
    
    # Create RAG chain
    rag_chain = create_rag_chain(retriever, llm)
    
    # User input
    st.subheader("❓ Ask a Question")
    user_question = st.text_input("Enter your question here:", "")
    
    if st.button("📬 Get Answer") and user_question:
        with st.spinner("🔍 Processing your question..."):
            try:
                response = rag_chain.invoke(user_question)
                st.success("✅ Response:")
                st.write(response)
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
