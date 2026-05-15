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
    """Loads a JSON file or returns default value."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default_val

def save_json_file(filepath, data):
    """Saves data to a local JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clear_all_data():
    """Deletes vector DB and all JSON files for a fresh start."""
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    if os.path.exists(FILES_REGISTRY):
        os.remove(FILES_REGISTRY)
    st.session_state.vector_store = None
    st.session_state.chat_history = load_json_file(HISTORY_FILE, [{"role": "assistant", "content": "Hello! Ami apnar AI PDF Assistant. Apni ekhon ekadik (multiple) PDF upload korte parben. PDF upload korun ebong proshno korun!"}])
    st.session_state.processed_files = load_json_file(FILES_REGISTRY, [])

# --- Page Configuration ---
st.set_page_config(page_title="DocuMind AI | PDF Assistant", page_icon="🧠", layout="wide")

# Custom UI Styling
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h1 { color: #1E3A8A; font-weight: 700; }
    .stChatFloatingInputContainer { padding-bottom: 1rem; }
    .sidebar-text { font-size: 0.9rem; color: #4B5563; }
    .file-list { font-size: 0.85rem; color: #10B981; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "chat_history" not in st.session_state:
    default_greeting = [{"role": "assistant", "content": "Hello! Ami apnar AI PDF Assistant. Apni ekhon ekadik (multiple) PDF upload korte parben. PDF upload korun ebong proshno korun!"}]
    st.session_state.chat_history = load_json_file(HISTORY_FILE, default_greeting)
if "processed_files" not in st.session_state:
    st.session_state.processed_files = load_json_file(FILES_REGISTRY, [])
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar UI ---
with st.sidebar:
    st.title("⚙️ Settings & Upload")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    
    # Auto-load existing database if API key is provided
    if api_key and st.session_state.vector_store is None:
        os.environ["OPENAI_API_KEY"] = api_key
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        if os.path.exists(DB_DIR):
            try:
                st.session_state.vector_store = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
                st.success("✅ Ager save kora database load hoyeche!")
            except Exception as e:
                st.error(f"Failed to load DB: {e}")

    # Display currently loaded files
    if len(st.session_state.processed_files) > 0:
        st.markdown("### 📚 Indexed Documents:")
        for f in st.session_state.processed_files:
            st.markdown(f"<div class='file-list'>✔️ {f}</div>", unsafe_allow_html=True)

    st.divider()
    # Accept multiple files
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process PDFs", type="primary", use_container_width=True):
        if not api_key:
            st.error("Doya kore OpenAI API Key din!")
        elif not uploaded_files:
            st.error("Doya kore PDF file upload korun!")
        else:
            with st.spinner("Extracting & Embedding PDFs..."):
                try:
                    os.environ["OPENAI_API_KEY"] = api_key
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    
                    all_chunks = []
                    new_files_processed = 0
                    
                    for uploaded_file in uploaded_files:
                        # Skip if already processed
                        if uploaded_file.name in st.session_state.processed_files:
                            continue
                            
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        # Load and Split PDF
                        loader = PyPDFLoader(tmp_file_path)
                        documents = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_documents(documents)
                        
                        all_chunks.extend(chunks)
                        st.session_state.processed_files.append(uploaded_file.name)
                        new_files_processed += 1
                        
                        os.unlink(tmp_file_path) # Cleanup temp file
                    
                    if all_chunks:
                        # Create or Update Vector Store
                        if st.session_state.vector_store is None:
                            vector_store = FAISS.from_documents(all_chunks, embeddings)
                            st.session_state.vector_store = vector_store
                        else:
                            st.session_state.vector_store.add_documents(all_chunks)
                        
                        # Save to disk
                        st.session_state.vector_store.save_local(DB_DIR)
                        save_json_file(FILES_REGISTRY, st.session_state.processed_files)
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": f"✅ {new_files_processed} ti notun PDF successfully process ebong save kora hoyeche! Ebar proshno korun."})
                        save_json_file(HISTORY_FILE, st.session_state.chat_history)
                        st.rerun()
                    else:
                        st.info("Uploaded PDFs are already in the database.")
                        
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")

    st.divider()
    if st.button("🗑️ Reset Everything", type="secondary", use_container_width=True):
        clear_all_data()
        st.rerun()

    st.markdown("---")
    st.markdown('<p class="sidebar-text"><b>DocuMind AI</b> now supports multiple documents, cross-document queries, and topic summarization.</p>', unsafe_allow_html=True)

# --- Main Interface ---
st.title("🧠 DocuMind: Multi-PDF AI Chatbot")

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
user_query = st.chat_input("Ask anything about your documents...")

if user_query:
    # 1. Show user query
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Generate response
    with st.chat_message("assistant"):
        if not api_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
            st.session_state.chat_history.pop() # Remove un-answered query
        elif st.session_state.vector_store is None:
            st.warning("Please upload a PDF and click 'Process PDFs' first.")
            st.session_state.chat_history.pop()
        else:
            with st.spinner("Analyzing knowledge base..."):
                try:
                    os.environ["OPENAI_API_KEY"] = api_key
                    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}) # Increased K for multi-doc
                    
                    # Formatting previous chat history
                    chat_history_str = ""
                    recent_history = st.session_state.chat_history[-7:-1] 
                    for msg in recent_history:
                        role_name = "User" if msg["role"] == "user" else "AI"
                        chat_history_str += f"{role_name}: {msg['content']}\n"
                        
                    # Get list of loaded files
                    loaded_docs_str = ", ".join(st.session_state.processed_files)
                    
                    # Modern RAG Prompt tailored for Multiple Documents
                    template = """You are a highly intelligent and professional AI assistant. 
                    You have access to a knowledge base containing information from the following uploaded documents:
                    [{loaded_docs}]

                    Use the retrieved context below and the previous chat history to answer the user's question accurately.
                    
                    CRITICAL INSTRUCTIONS:
                    1. If the user asks "what information is available", "what are the topics", or similar global questions, use the list of uploaded documents and the context provided to create a well-structured list of topics/summaries.
                    2. If the answer requires combining information from multiple documents, do so seamlessly.
                    3. If the answer is completely missing from the context, state clearly that you don't know based on the uploaded documents. Do not hallucinate.
                    4. Answer in the same language as the user's question (e.g., Bengali or English). Provide the response using markdown styling (bullet points, bold text) for readability.

                    Chat History:
                    {chat_history}

                    Context from PDFs:
                    {context}

                    Current Question: {question}
                    Answer:"""
                    
                    prompt = ChatPromptTemplate.from_template(template)
                    
                    def format_docs(docs):
                        # Join chunks with metadata (if available) to help LLM separate context
                        return "\n\n---\n\n".join(doc.page_content for doc in docs)

                    # Building LCEL Chain
                    rag_chain = (
                        {
                            "context": retriever | format_docs, 
                            "question": RunnablePassthrough(),
                            "chat_history": lambda x: chat_history_str,
                            "loaded_docs": lambda x: loaded_docs_str
                        }
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Get response and display
                    response = rag_chain.invoke(user_query)
                    st.markdown(response)
                    
                    # Save to history & file
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    save_json_file(HISTORY_FILE, st.session_state.chat_history)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")