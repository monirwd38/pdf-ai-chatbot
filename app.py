import os
import json
import tempfile
import shutil
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Constants for Persistence ---
DB_DIR = "faiss_index_db"
HISTORY_FILE = "chat_history.json"
FILES_REGISTRY = "processed_files.json"

# --- Helper Functions ---
def load_json_file(filepath, default_val):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default_val

def save_json_file(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clear_all_data():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    if os.path.exists(FILES_REGISTRY):
        os.remove(FILES_REGISTRY)
    st.session_state.vector_store = None
    st.session_state.chat_history = [{"role": "assistant", "content": "Hello! I am your AI PDF Assistant. You can now upload multiple PDFs. Please upload your documents and ask any questions!"}]
    st.session_state.processed_files = []

# --- Page Configuration ---
st.set_page_config(page_title="DocuMind AI | PDF Assistant", page_icon="🧠", layout="wide")

# Custom UI Styling (Enhanced for Smart Look)
st.markdown("""
<style>

/* ===== GOOGLE FONT ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ===== GLOBAL ===== */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(to bottom right, #F8FAFC, #EEF2FF);
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1350px;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

/* ===== TITLE ===== */
.main-title {
    font-size: 3rem;
    font-weight: 800;
    color: #111827;
    margin-bottom: 0;
    letter-spacing: -1px;
}

.gradient-text {
    background: linear-gradient(90deg, #4F46E5, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #6B7280;
    font-size: 1.05rem;
    margin-top: -10px;
    margin-bottom: 30px;
}

/* ===== GLASS CARD ===== */
.glass-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.06);
}

/* ===== CHAT AREA ===== */
[data-testid="stChatMessage"] {
    border-radius: 20px;
    padding: 16px 18px;
    margin-bottom: 14px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 4px 18px rgba(0,0,0,0.03);
}

[data-testid="stChatMessage"]:nth-child(odd) {
    background: #FFFFFF;
}

[data-testid="stChatMessage"]:nth-child(even) {
    background: #F8FAFC;
}

/* ===== INPUT ===== */
[data-testid="stChatInput"] {
    background: white;
    border-radius: 18px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 6px 24px rgba(0,0,0,0.05);
    padding: 8px;
}

.stTextInput input {
    border-radius: 14px !important;
    border: 1px solid #D1D5DB !important;
    padding: 0.8rem !important;
}

/* ===== BUTTONS ===== */
.stButton button {
    border-radius: 14px !important;
    border: none !important;
    padding: 0.7rem 1rem !important;
    font-weight: 600 !important;
    transition: all 0.25s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}

.file-badge {
    background: linear-gradient(to right, #EEF2FF, #F5F3FF);
    color: #4338CA;
    padding: 12px 14px;
    border-radius: 14px;
    margin-bottom: 10px;
    border: 1px solid #E0E7FF;
    font-size: 0.85rem;
    font-weight: 600;
}

.sidebar-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    color: #9CA3AF;
    margin-top: 22px;
    margin-bottom: 8px;
}

.status-card {
    background: linear-gradient(135deg, #111827, #1F2937);
    color: white;
    padding: 18px;
    border-radius: 18px;
    margin-top: 18px;
}

.status-title {
    font-size: 0.8rem;
    opacity: 0.7;
}

.status-value {
    font-size: 1.7rem;
    font-weight: 800;
    margin-top: 4px;
}
/* ===== BUTTON ROW FIX ===== */
div[data-testid="stHorizontalBlock"] {
    gap: 12px;
}

/* ===== ALL BUTTONS ===== */
.stButton > button {
    width: 100%;
    height: 54px;
    border-radius: 14px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    transition: all 0.25s ease;
}

/* ===== INDEX BUTTON ===== */
div[data-testid="column"]:first-child .stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 8px 20px rgba(99,102,241,0.25);
}

/* ===== CLEAR BUTTON ===== */
div[data-testid="column"]:last-child .stButton > button {
    background: white !important;
    color: #374151 !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}

/* ===== HOVER ===== */
.stButton > button:hover {
    transform: translateY(-2px);
}

/* ===== REMOVE RED BACKGROUND ===== */
div[data-testid="column"] {
    background: transparent !important;
}
/* ===== HIDE STREAMLIT BRANDING ===== */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}         
""", unsafe_allow_html=True)

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_json_file(HISTORY_FILE, [{"role": "assistant", "content": "Hello! I am your AI PDF Assistant. You can now upload multiple PDFs. Please upload your documents and ask any questions!"}])
if "processed_files" not in st.session_state:
    st.session_state.processed_files = load_json_file(FILES_REGISTRY, [])
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=70) 
    st.markdown("""
    <div style="margin-top:10px; margin-bottom:25px;">
        <div style="font-size:2rem; font-weight:800; color:#111827;">
            DocuMind AI
        </div>
        <div style="color:#6B7280; font-size:0.9rem; margin-top:-5px;">
            Smart Enterprise PDF Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="sidebar-label">AUTHENTICATION</p>', unsafe_allow_html=True)
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", help="Your key is never stored on our servers.")
    
    # Auto-load existing database
    if api_key and st.session_state.vector_store is None:
        os.environ["OPENAI_API_KEY"] = api_key
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        if os.path.exists(DB_DIR):
            try:
                st.session_state.vector_store = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
                st.toast("✅ Knowledge base loaded from local storage!")
            except Exception as e:
                st.error(f"Failed to load DB: {e}")

    st.markdown('<p class="sidebar-label">DOCUMENT MANAGEMENT</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    
    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button(
            "🚀 Index PDFs", 
            type="primary", 
            use_container_width=True
        )
    with col2:
        reset_btn = st.button(
            "🗑️ Clear All", 
            type="secondary", 
            use_container_width=True
        )

    if process_btn:
        if not api_key:
            st.error("Please provide an OpenAI API Key.")
        elif not uploaded_files:
            st.warning("Please upload some PDF files first.")
        else:
            with st.status("Building your knowledge base...", expanded=True) as status_container:
                try:
                    os.environ["OPENAI_API_KEY"] = api_key
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    all_chunks = []
                    new_files_list = []
                    
                    for uploaded_file in uploaded_files:
                        if uploaded_file.name in st.session_state.processed_files:
                            st.write(f"Skipping {uploaded_file.name} (Already indexed)")
                            continue
                            
                        st.write(f"Processing: {uploaded_file.name}")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        loader = PyPDFLoader(tmp_file_path)
                        documents = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_documents(documents)
                        
                        all_chunks.extend(chunks)
                        st.session_state.processed_files.append(uploaded_file.name)
                        new_files_list.append(uploaded_file.name)
                        os.unlink(tmp_file_path)
                    
                    if all_chunks:
                        if st.session_state.vector_store is None:
                            st.session_state.vector_store = FAISS.from_documents(all_chunks, embeddings)
                        else:
                            st.session_state.vector_store.add_documents(all_chunks)
                        
                        st.session_state.vector_store.save_local(DB_DIR)
                        save_json_file(FILES_REGISTRY, st.session_state.processed_files)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": f"✅ Successfully added {len(new_files_list)} documents: {', '.join(new_files_list)}. How can I help you with them?"
                        })
                        save_json_file(HISTORY_FILE, st.session_state.chat_history)
                        status_container.update(label="Index Complete!", state="complete", expanded=False)
                        st.rerun()
                    else:
                        status_container.update(label="No new files to process.", state="complete")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if reset_btn:
        clear_all_data()
        st.toast("Everything has been reset.")
        st.rerun()

    # Indexed Files Display
    if st.session_state.processed_files:
        if st.session_state.processed_files:
            st.markdown(f"""
            <div class="status-card">
                <div class="status-title">DOCUMENTS INDEXED</div>
                <div class="status-value">{len(st.session_state.processed_files)}</div>
            </div>
            """, unsafe_allow_html=True)

# --- Main Interface ---
# Header Section
st.markdown("""
<div class="glass-card" style="margin-top: 40px;">
    <div class="main-title">
        🧠 <span class="gradient-text">DocuMind AI</span>
    </div>
    <p class="subtitle">
        Enterprise-grade AI assistant for intelligent PDF analysis using advanced RAG architecture.
    </p>
</div>
""", unsafe_allow_html=True)

# Chat Display Container 
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) 

# Chat Input
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    # 1. Add user query to UI
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Generate and display AI response
    with st.chat_message("assistant"):
        if not api_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
        elif st.session_state.vector_store is None:
            st.warning("Please upload and process a PDF first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    os.environ["OPENAI_API_KEY"] = api_key
                    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                    
                    # History handling
                    chat_history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.chat_history[-5:-1]])
                    loaded_docs_str = ", ".join(st.session_state.processed_files)
                    
                    template = """You are a professional AI assistant. 
                    Knowledge Base: [{loaded_docs}]

                    Retrieved Context:
                    {context}

                    Recent History:
                    {chat_history}

                    Question: {question}
                    
                    Instructions: 
                    - Use ONLY the provided context. 
                    - If missing, say you don't know. 
                    - Answer in the user's language (Bengali/English).
                    - Use Markdown for structure.

                    Answer:"""
                    
                    prompt = ChatPromptTemplate.from_template(template)
                    
                    def format_docs(docs):
                        return "\n\n".join(f"--- Source: {doc.metadata.get('source', 'Unknown')} ---\n{doc.page_content}" for doc in docs)

                    rag_chain = (
                        {
                            "context": retriever | format_docs, 
                            "question": RunnablePassthrough(),
                            "chat_history": lambda x: chat_history_str,
                            "loaded_docs": lambda x: loaded_docs_str
                        }
                        | prompt | llm | StrOutputParser()
                    )
                    
                    response = rag_chain.invoke(user_query)
                    st.markdown(response)
                    
                    # Update History
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    save_json_file(HISTORY_FILE, st.session_state.chat_history)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.markdown("""
<div style="text-align:center; padding:30px 10px 10px 10px; color:#6B7280; font-size:0.9rem;">
    © 2025 DocuMind AI • Professional Multi-PDF RAG Assistant
</div>
""", unsafe_allow_html=True)