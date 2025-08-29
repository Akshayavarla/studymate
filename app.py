import streamlit as st
import os
import json
from datetime import datetime
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from utils import load_document, create_vectorstore
from dotenv import load_dotenv
import csv
import io

load_dotenv()

# Configure page
st.set_page_config(
    page_title="StudyMate - AI-Powered Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main background with gradient */
    .main > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Container styling */
    .stContainer > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    /* Answer card styling */
    .answer-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Status banners */
    .status-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #2d5016;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #8b0000;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #8b4513;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    /* File upload area */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin: 1rem 0;
    }
    
    /* Q&A history styling */
    .qa-history {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #764ba2;
    }
    
    /* Referenced paragraphs */
    .ref-paragraph {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None



# Main header
st.markdown('<h1 class="main-header">🎓 StudyMate</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">AI-Powered Multi-Document Learning Assistant</p>', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # File upload section
    st.markdown("### 📁 Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF or Text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more documents to create your knowledge base"
    )
    
    # Process uploaded files
    if uploaded_files:
        if set([f.name for f in uploaded_files]) != set(st.session_state.processed_files):
            st.markdown('<div class="status-warning">🔄 Processing new documents...</div>', unsafe_allow_html=True)
            
            # Clear previous data
            st.session_state.qa_chain = None
            st.session_state.vectorstore = None
            st.session_state.processed_files = []
            
            # Ensure documents folder exists
            os.makedirs("documents", exist_ok=True)
            
            all_docs = []
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Save file
                    save_path = os.path.join("documents", uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Create file object
                    class File:
                        name = save_path
                    
                    # Load document
                    docs = load_document(File())
                    all_docs.extend(docs)
                    st.session_state.processed_files.append(uploaded_file.name)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                except Exception as e:
                    st.markdown(f'<div class="status-error">❌ Error processing {uploaded_file.name}: {str(e)}</div>', unsafe_allow_html=True)
            
            if all_docs:
                try:
                    # Create vectorstore and QA chain
                    vectorstore = create_vectorstore(all_docs)
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                    llm = Ollama(model="llama3.2")
                    
                    prompt = ChatPromptTemplate.from_template(
                        """
                        You are StudyMate, an AI learning assistant. Use the following context from the uploaded documents to answer the question accurately and helpfully.
                        
                        Instructions:
                        - Provide clear, comprehensive answers based on the context
                        - If you don't know something based on the context, say so
                        - When possible, reference which document or section the information comes from
                        - Format your answer in a student-friendly way
                        
                        Context:
                        {context}
                        
                        Question: {question}
                        
                        Answer:
                        """
                    )
                    
                    llm_chain = LLMChain(llm=llm, prompt=prompt)
                    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
                    qa_chain = RetrievalQA(
                        combine_documents_chain=stuff_chain, 
                        retriever=retriever,
                        return_source_documents=True
                    )
                    
                    st.session_state.qa_chain = qa_chain
                    st.session_state.vectorstore = vectorstore
                    
                    st.markdown(f'<div class="status-success">✅ Successfully processed {len(uploaded_files)} files with {len(all_docs)} document chunks!</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f'<div class="status-error">❌ Error creating knowledge base: {str(e)}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-success">✅ {len(uploaded_files)} files ready for questions!</div>', unsafe_allow_html=True)
    
    # Question input section
    st.markdown("### 💭 Ask Your Question")
    question = st.text_input(
        "What would you like to know about your documents?",
        placeholder="e.g., What are the main concepts discussed in the documents?",
        help="Ask any question about the content in your uploaded documents"
    )
    
    # Answer generation
    if st.button("🚀 Get Answer", disabled=not (uploaded_files and question and st.session_state.qa_chain)):
        if question.strip():
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get answer with source documents
                    result = st.session_state.qa_chain({"query": question})
                    answer = result['result']
                    source_docs = result.get('source_documents', [])
                    
                    # Display answer
                    st.markdown("### 🎯 Answer")
                    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
                    
                    # Display referenced paragraphs
                    if source_docs:
                        with st.expander("📚 Referenced Paragraphs", expanded=False):
                            for i, doc in enumerate(source_docs, 1):
                                st.markdown(f'<div class="ref-paragraph"><strong>Reference {i}:</strong><br>{doc.page_content[:500]}...</div>', unsafe_allow_html=True)
                    
                    # Add to history
                    qa_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": question,
                        "answer": answer,
                        "sources": len(source_docs)
                    }
                    st.session_state.qa_history.insert(0, qa_entry)
                    
                except Exception as e:
                    st.markdown(f'<div class="status-error">❌ Error generating answer: {str(e)}</div>', unsafe_allow_html=True)

with col2:
    # Sidebar content
    st.markdown("### 📊 Session Statistics")
    
    if st.session_state.processed_files:
        st.metric("📄 Files Processed", len(st.session_state.processed_files))
        st.metric("❓ Questions Asked", len(st.session_state.qa_history))
        
        # Display processed files
        with st.expander("📂 Processed Files"):
            for file in st.session_state.processed_files:
                st.write(f"• {file}")
    
    # Q&A History
    st.markdown("### 📝 Q&A History")
    
    if st.session_state.qa_history:
        # Download as TXT
        txt_lines = []
        for i, qa in enumerate(st.session_state.qa_history, 1):
            txt_lines.append(f"Q{i}: {qa['question']}")
            txt_lines.append(f"Asked: {qa['timestamp']}")
            txt_lines.append(f"Answer: {qa['answer']}")
            txt_lines.append(f"Sources: {qa['sources']} references")
            txt_lines.append("-" * 40)
        history_txt = "\n".join(txt_lines)
        st.download_button(
            label="📥 Download Q&A Log (TXT)",
            data=history_txt,
            file_name=f"studymate_qa_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

        # Download as CSV
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["Timestamp", "Question", "Answer", "Sources"])
        for qa in st.session_state.qa_history:
            writer.writerow([qa["timestamp"], qa["question"], qa["answer"], qa["sources"]])
        st.download_button(
            label="📥 Download Q&A Log (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"studymate_qa_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Display recent Q&As
        for i, qa in enumerate(st.session_state.qa_history[:5]):  # Show last 5
            with st.expander(f"Q{len(st.session_state.qa_history)-i}: {qa['question'][:50]}..."):
                st.markdown(f"**Asked:** {qa['timestamp']}")
                st.markdown(f"**Question:** {qa['question']}")
                st.markdown(f"**Answer:** {qa['answer'][:200]}...")
                st.markdown(f"**Sources:** {qa['sources']} references")
        
        if len(st.session_state.qa_history) > 5:
            st.info(f"... and {len(st.session_state.qa_history) - 5} more questions")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.qa_history = []
            st.rerun()

    else:
        st.info("No questions asked yet. Start by uploading documents and asking questions!")

# Bottom section with usage tips
st.markdown("---")
st.markdown("### 💡 Tips for Better Results")

tips_col1, tips_col2, tips_col3 = st.columns(3)

with tips_col1:
    st.markdown("""
    **📋 Question Types:**
    - Summarize key concepts
    - Compare different topics
    - Explain specific terms
    - Find relevant examples
    """)

with tips_col2:
    st.markdown("""
    **📄 Document Tips:**
    - Upload multiple related files
    - Use clear, readable PDFs
    - Include diverse sources
    - Keep files well-organized
    """)

with tips_col3:
    st.markdown("""
    **🎯 Best Practices:**
    - Ask specific questions
    - Review referenced paragraphs
    - Download your Q&A history
    - Build on previous answers
    """)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 0.9rem;">🎓 StudyMate - Powered by AI • Built with Streamlit </p>', 
    unsafe_allow_html=True
)
