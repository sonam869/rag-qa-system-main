"""
Streamlit Frontend for RAG System
Upload documents, ask questions, and get answers with citations.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline
from chunking import TextChunker
import tempfile
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .citation-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ecc71;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

@st.cache_resource
def initialize_pipeline():
    """Initialize the RAG pipeline (cached)"""
    try:
        with st.spinner("🔄 Initializing RAG pipeline..."):
            pipeline = RAGPipeline()
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        return None

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract text"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            # Process PDF file
            pdf_reader = PdfReader(uploaded_file)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
            return content
        else:
            # Process text/markdown files
            content = uploaded_file.read().decode("utf-8")
            return content
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def ingest_documents(pipeline, documents):
    """Ingest documents into the RAG system"""
    try:
        with st.spinner("📚 Processing and ingesting documents..."):
            num_chunks = pipeline.ingest_documents(documents)
        return num_chunks
    except Exception as e:
        st.error(f"Error ingesting documents: {str(e)}")
        return 0

def main():
    # Header
    st.markdown('<p class="main-header">🤖 RAG Q&A System</p>', unsafe_allow_html=True)
    st.markdown("**Upload documents, ask questions, and get answers with citations**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check API keys
        pinecone_key = os.getenv("PINECONE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        
        if pinecone_key:
            st.success("✅ Pinecone API Key loaded")
        else:
            st.error("❌ Pinecone API Key not found")
        
        if groq_key:
            st.success("✅ Groq API Key loaded")
        else:
            st.error("❌ Groq API Key not found")
        
        st.markdown("---")
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider("Number of chunks to retrieve", 5, 30, 10)
        rerank_top_k = st.slider("Top chunks after reranking", 3, 10, 5)
        use_mmr = st.checkbox("Use MMR (diversity)", value=True)
        lambda_param = st.slider("MMR Lambda (relevance vs diversity)", 0.0, 1.0, 0.5, 0.1)
        
        st.markdown("---")
        
        # LLM settings
        st.subheader("🤖 LLM Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        max_tokens = st.slider("Max tokens", 256, 2048, 1024, 128)
        
        st.markdown("---")
        
        # Stats
        if st.session_state.pipeline and st.session_state.documents_loaded:
            st.subheader("📊 Stats")
            try:
                stats = st.session_state.pipeline.get_stats()
                st.metric("Total Chunks", stats.get("total_chunks", 0))
                st.metric("Embedding Dim", stats.get("embedding_dimension", 0))
            except:
                pass
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "💬 Ask Questions", "📜 History"])
    
    # Tab 1: Document Upload
    with tab1:
        st.header("📄 Upload or Paste Documents")
        
        # Initialize pipeline
        if st.session_state.pipeline is None:
            if not pinecone_key or not groq_key:
                st.error("Please configure API keys in .env file")
                st.info("Required keys: PINECONE_API_KEY, GROQ_API_KEY")
                return
            
            st.session_state.pipeline = initialize_pipeline()
            
            if st.session_state.pipeline is None:
                return
        
        # Upload method selection
        upload_method = st.radio("Choose input method:", ["Upload File", "Paste Text"])
        
        documents_to_ingest = []
        
        if upload_method == "Upload File":
            uploaded_files = st.file_uploader(
                "Upload documents (.txt, .md, .pdf)",
                type=["txt", "md", "pdf"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    content = process_uploaded_file(uploaded_file)
                    if content:
                        documents_to_ingest.append({
                            "text": content,
                            "source": uploaded_file.name,
                            "title": uploaded_file.name,
                            "section": "Main"
                        })
                        st.success(f"✅ Loaded: {uploaded_file.name}")
        
        else:  # Paste Text
            col1, col2 = st.columns(2)
            
            with col1:
                doc_title = st.text_input("Document Title", "Pasted Document")
            with col2:
                doc_section = st.text_input("Section (optional)", "Main")
            
            pasted_text = st.text_area(
                "Paste your text here:",
                height=300,
                placeholder="Enter or paste your document text..."
            )
            
            if pasted_text.strip():
                documents_to_ingest.append({
                    "text": pasted_text,
                    "source": f"{doc_title}.txt",
                    "title": doc_title,
                    "section": doc_section if doc_section else "Main"
                })
        
        # Ingest button
        if documents_to_ingest:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("📚 Process & Ingest Documents", type="primary", use_container_width=True):
                    num_chunks = ingest_documents(st.session_state.pipeline, documents_to_ingest)
                    
                    if num_chunks > 0:
                        st.success(f"✅ Successfully ingested {num_chunks} chunks from {len(documents_to_ingest)} document(s)")
                        st.session_state.documents_loaded = True
                        st.balloons()
                    else:
                        st.error("Failed to ingest documents")
    
    # Tab 2: Ask Questions
    with tab2:
        st.header("💬 Ask Questions")
        
        if not st.session_state.documents_loaded:
            st.warning("⚠️ Please upload and ingest documents first in the 'Document Upload' tab")
            return
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="What is machine learning?",
            key="query_input"
        )
        
        # Search button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            search_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
        
        if search_button and query.strip():
            with st.spinner("🤔 Thinking..."):
                try:
                    # Query the pipeline
                    response = st.session_state.pipeline.query(
                        question=query,
                        top_k=top_k,
                        rerank_top_k=rerank_top_k,
                        use_mmr=use_mmr,
                        lambda_param=lambda_param
                    )
                    
                    # Store in history
                    st.session_state.query_history.append({
                        "query": query,
                        "response": response
                    })
                    
                    # Display answer
                    if response.get("has_answer"):
                        st.markdown("### 📝 Answer")
                        st.markdown(f'<div class="answer-box">{response["answer"]}</div>', unsafe_allow_html=True)
                        
                        # Display sources
                        if response.get("sources"):
                            st.markdown("### 📚 Sources & Citations")
                            
                            for source in response["sources"]:
                                with st.expander(f"[{source['citation']}] {source['title']}", expanded=False):
                                    st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                                    
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        st.markdown(f'<span class="citation-badge">[{source["citation"]}]</span>', unsafe_allow_html=True)
                                    with col2:
                                        st.markdown(f"**Score:** {source['score']:.4f}")
                                    
                                    if source.get("section"):
                                        st.markdown(f"**Section:** {source['section']}")
                                    
                                    st.markdown(f"**Source:** `{source['source']}`")
                                    st.markdown("**Text:**")
                                    st.text_area(
                                        "Source text",
                                        source['text'],
                                        height=150,
                                        key=f"source_{source['citation']}",
                                        label_visibility="collapsed"
                                    )
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ " + response["answer"])
                
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
    
    # Tab 3: History
    with tab3:
        st.header("📜 Query History")
        
        if not st.session_state.query_history:
            st.info("No queries yet. Ask a question in the 'Ask Questions' tab!")
        else:
            for i, item in enumerate(reversed(st.session_state.query_history)):
                with st.expander(f"Q{len(st.session_state.query_history) - i}: {item['query']}", expanded=False):
                    response = item['response']
                    st.markdown("**Answer:**")
                    st.write(response['answer'])
                    
                    if response.get('sources'):
                        st.markdown("**Citations:**")
                        for source in response['sources']:
                            st.markdown(f"- [{source['citation']}] {source['title']} (Score: {source['score']:.4f})")
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.query_history = []
                st.rerun()

if __name__ == "__main__":
    main()
