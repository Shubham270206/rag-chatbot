import streamlit as st
import tempfile
import os
from rag_engine import build_qa_chain

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("RAG PDF Chatbot")
st.caption("Upload one or more PDFs and ask questions about them.")

uploaded_files = st.file_uploader(
    "Upload your PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    file_names = [f.name for f in uploaded_files]

    with st.spinner("Processing PDFs..."):
        if "qa_chain" not in st.session_state or \
           st.session_state.get("file_names") != file_names:
            tmp_paths = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_paths.append(tmp.name)

            st.session_state.qa_chain = build_qa_chain(tmp_paths)
            st.session_state.file_names = file_names
            st.session_state.messages = []

            for path in tmp_paths:
                os.unlink(path)

    st.success(f"Ready! Loaded {len(uploaded_files)} PDF(s): {', '.join(file_names)}")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ask something about your PDFs..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = ""
                for msg in st.session_state.messages[:-1]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history += f"{role}: {msg['content']}\n"

                result = st.session_state.qa_chain(question, history=history)
                answer = result["result"]
                sources = result["source_documents"]

            st.write(answer)

            with st.expander("Source passages"):
                for doc in sources:
                    page = doc.metadata.get("page", "?")
                    source = doc.metadata.get("source", "unknown")
                    filename = os.path.basename(source)
                    st.markdown(f"**{filename} — Page {page+1}:** {doc.page_content[:300]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Upload one or more PDFs to get started.")