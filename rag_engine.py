import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# load the groq api key from .env - took me a while to get this right
load_dotenv()

def build_qa_chain(pdf_paths):
    # handle both single and multiple pdfs
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    # read all the pdfs and combine them
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    # split into smaller chunks - 500 worked better than 1000 for my notes
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50  # overlap so context doesnt get cut off at boundaries
    )
    chunks = splitter.split_documents(documents)

    # free local embedding model - no api key needed here
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # store vectors in faiss for fast similarity search
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # using groq because its free and faster than openai
    # struggled a lot with api key errors during deployment
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2  # low temp = more factual answers
    )

    def qa_chain(question: str, history: str = ""):
        # fetch top 4 relevant chunks for the question
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])

        # pass context + history so the bot remembers previous questions
        prompt = f"""You are a helpful assistant. Use the context below to answer the question.
If you don't know the answer from the context, say so.

Context from document:
{context}

Conversation so far:
{history}

User: {question}
Assistant:"""
        response = llm.invoke(prompt)
        return {"result": response.content, "source_documents": docs}

    return qa_chain