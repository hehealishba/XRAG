import streamlit as st
import json
from pathlib import Path
from rag_neo4j import Neo4jExplainableRAG

# Page configuration
st.set_page_config(
    page_title="Financial Autoleasing RAG",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def initialize_rag():
    """Initialize the RAG system"""
    try:
        with st.spinner("Initializing RAG system..."):
            rag = Neo4jExplainableRAG()
            st.session_state.rag = rag
            st.session_state.initialized = True
            return True
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return False


def seed_graph():
    """Seed the Neo4j graph with data"""
    try:
        with st.spinner("Seeding graph database..."):
            st.session_state.rag.seed_graph()
            st.success("Graph seeded successfully!")
            return True
    except Exception as e:
        st.error(f"Failed to seed graph: {str(e)}")
        return False


# Sidebar
with st.sidebar:
    st.title("🚗 Financial Autoleasing RAG")
    st.markdown("---")
    
    # Initialize button
    if st.button("🔄 Initialize RAG System", use_container_width=True):
        if initialize_rag():
            st.rerun()
    
    if st.session_state.initialized:
        st.success("✅ RAG System Ready")
        
        # Seed graph button
        st.markdown("---")
        if st.button("🌱 Seed Graph Database", use_container_width=True):
            seed_graph()
    else:
        st.warning("⚠️ RAG System Not Initialized")
        st.info("Click 'Initialize RAG System' to start")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This RAG system uses Neo4j to store and retrieve 
    financial autoleasing offers. Ask questions about 
    lease terms, risk controls, pricing, and more.
    """)


# Main content
st.title("🚗 Financial Autoleasing RAG System")
st.markdown("Ask questions about lease offers, terms, risk controls, and pricing.")

# Initialize if not done
if not st.session_state.initialized:
    st.info("👈 Please initialize the RAG system from the sidebar first.")
    if st.button("Initialize Now"):
        initialize_rag()
        st.rerun()
else:
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    for idx, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat["question"])
        
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            
            # Show sources in expander
            if chat["sources"]:
                with st.expander(f"📚 Sources ({len(chat['sources'])})"):
                    for source in chat["sources"]:
                        st.markdown(f"**{source['title']}**")
                        st.markdown(f"- Score: {source['score']:.4f}")
                        st.markdown(f"- Source: {source.get('source', 'N/A')}")
                        st.markdown(f"- Doc ID: `{source['doc_id']}`")
                        st.markdown("---")
            
            # Show evidence in expander
            if chat.get("evidence"):
                with st.expander(f"🔍 Evidence Graph ({len(chat['evidence'])} nodes)"):
                    for i, ev in enumerate(chat["evidence"][:10], 1):  # Show first 10
                        st.markdown(f"**Node {i}**")
                        st.markdown(f"- Labels: {', '.join(ev.get('labels', []))}")
                        if ev.get('relationship_path'):
                            st.markdown(f"- Relationships: {' → '.join(ev['relationship_path'])}")
                        if ev.get('properties'):
                            props = {k: v for k, v in ev['properties'].items() if k not in ['text']}
                            if props:
                                st.json(props)
                        st.markdown("---")
    
    # Question input
    question = st.chat_input("Ask a question about lease offers...")
    
    if question:
        # Add user question to chat
        st.session_state.chat_history.append({
            "question": question,
            "answer": "",
            "sources": [],
            "evidence": []
        })
        
        # Get answer
        try:
            with st.spinner("Searching and generating answer..."):
                response = st.session_state.rag.answer(question)
                
                # Update chat history
                st.session_state.chat_history[-1] = response
                
                # Rerun to display
                st.rerun()
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            st.session_state.chat_history.pop()

    # Clear chat button
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

