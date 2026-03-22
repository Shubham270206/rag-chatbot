import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

def build_qa_chain(pdf_paths):
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    def qa_chain(question: str, history: str = ""):
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""You are a helpful assistant. Use the context below to answer the question.
If you don't know the answer from the context, say so.

Context from document:
{context}

Conversation so far:[]
{history}

User: {question}
Assistant:"""
        response = llm.invoke(prompt)
        return {"result": response.content, "source_documents": docs}

    return qa_chain