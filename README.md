🧠 DocMind AI — RAG PDF Chatbot
A professional RAG (Retrieval-Augmented Generation) chatbot that lets you upload PDF documents and have a conversation with them. Built from scratch to improve my understanding of how LLMs and vector search work together.

Live Demo
🔗 Try it live on Streamlit Cloud
💻 GitHub Repository

How It Works
This project uses a two-phase RAG pipeline:
Ingestion Pipeline (runs once per upload):

Loads and parses PDFs using PyPDFLoader
Splits text into overlapping chunks using RecursiveCharacterTextSplitter (chunk size: 500, overlap: 50)
Converts chunks into vector embeddings using sentence-transformers (all-MiniLM-L6-v2) — runs locally, no API needed
Stores embeddings in a FAISS vector index for fast similarity search

Query Pipeline (runs on every question):

Embeds the user's question using the same model
Retrieves the top-4 most relevant chunks from FAISS
Passes the chunks + full conversation history to Llama 3.3 70B via Groq API
Returns a grounded answer with source page and file references


Features

Upload and query multiple PDFs at once
Multi-turn conversation memory — remembers previous questions
Source attribution — shows exactly which file and page each answer came from
Progress bar during PDF processing
Professional purple dark theme UI with chat bubbles
File upload in sidebar to keep chat area clean
Deployed live on Streamlit Cloud with secrets management


Tech Stack
ComponentTechnologyFrontendStreamlitLLMLlama 3.3 70B via Groq APIEmbeddingssentence-transformers (all-MiniLM-L6-v2)Vector StoreFAISS (Facebook AI Similarity Search)PDF LoaderLangChain PyPDFLoaderText SplittingLangChain RecursiveCharacterTextSplitterSecretspython-dotenv (local) / Streamlit secrets (cloud)

Project Structure
rag-chatbot/
├── app.py            # Streamlit UI, chat logic, custom CSS
├── rag_engine.py     # RAG pipeline (ingestion + retrieval + LLM)
├── requirements.txt  # Python dependencies
├── .gitignore        # Excludes venv and .env from GitHub
└── .env              # API keys (not committed to GitHub)

Getting Started
1. Clone the repository
bashgit clone https://github.com/Shubham270206/rag-chatbot.git
cd rag-chatbot
2. Create and activate a virtual environment
bashpython -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
3. Install dependencies
bashpip install -r requirements.txt
4. Get a free Groq API key
Sign up at console.groq.com and create an API key.
Create a .env file in the project root:
GROQ_API_KEY=your_groq_api_key_here
5. Run the app
bashstreamlit run app.py
Open your browser at http://localhost:8501.

Usage

Upload one or more PDF files using the sidebar uploader
Wait for the progress bar to complete
Type your question in the chat input at the bottom
View the answer and expand "View source passages" to verify


Key Design Decisions

FAISS over ChromaDB — runs locally with no server needed, simpler setup
all-MiniLM-L6-v2 — lightweight embedding model, free, strong semantic search
Chunk overlap of 50 tokens — prevents context from being cut off at boundaries
Groq over OpenAI — free tier, faster inference via Groq's LPU architecture
Conversation history as plain text — simpler than LangChain memory abstractions, easier to debug
Sidebar layout — keeps the main area focused on the chat experience


What I Learned

How RAG pipelines work end-to-end (chunking → embeddings → vector search → LLM)
Debugging real LangChain version changes (not just tutorial code)
Managing API keys securely in local and cloud deployments
Deploying a live ML app on Streamlit Cloud with secrets management


Future Improvements

Support for DOCX, TXT, and CSV files
Re-ranking retrieved chunks for better accuracy
Persistent FAISS index to avoid re-processing on reload
Document summary view on upload
Authentication for private document storage


Author
Shubham Sunil Sinha
2nd Year CSE AIML — VIT Bhopal
GitHub
