# GenAI-RAG: Retrieval-Augmented Generation System

A powerful Retrieval-Augmented Generation (RAG) system that combines LangChain, MistralAI, and Chroma vector database to enable intelligent document querying and Q&A capabilities.

## 🌟 Features

- **Dual Interface**: Command-line and Streamlit web UI for maximum flexibility
- **Intelligent Document Processing**: Automatically chunks and indexes PDF documents
- **MistralAI Integration**: Leverages state-of-the-art embeddings and language models
- **Smart Retrieval**: Uses Maximum Marginal Relevance (MMR) search for diverse context retrieval
- **Vector Database**: Persistent Chroma database for efficient document storage and retrieval
- **Context-Aware Q&A**: Generates answers based on document context with fallback messages for out-of-context questions
- **Session Management**: Maintains chat history in the Streamlit interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MistralAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GENAI-RAG
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1     # On Windows
   source .venv/bin/activate      # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root and add your API keys:
   ```
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

## 📁 Project Structure

```
GENAI-RAG/
├── main.py                 # CLI RAG system for interactive Q&A
├── app.py                  # Streamlit web interface
├── create_database.py      # Script to build vector database from PDFs
├── requirements.txt        # Python dependencies
├── Document_Loader/        # Directory containing PDF documents
│   └── deep-learning.pdf   # Sample document
└── chroma_db/              # Persistent vector database storage
```

## 💻 Usage

### 1. Prepare Your Documents

Place your PDF documents in the `Document_Loader/` directory. Then, build the vector database:

```bash
python create_database.py
```

This script will:
- Load all PDF documents
- Split them into chunks (1000 characters with 200 character overlap)
- Generate embeddings using MistralAI
- Store vectors in Chroma database

### 2. CLI Interface

Run the command-line RAG system:

```bash
python main.py
```

**Usage:**
- Enter your questions at the prompt
- The system retrieves relevant context and generates answers
- Type `0` to exit

**Example:**
```
you: What is deep learning?
AI: Deep learning is a subset of machine learning that uses neural networks with multiple layers...

you: 0
```

### 3. Streamlit Web UI

Launch the interactive web interface:

```bash
streamlit run app.py
```

**Features:**
- Upload PDF documents directly through the sidebar
- Real-time document processing and indexing
- Beautiful chat interface with message styling
- Visual indicators for retrieved documents
- Persistent chat history during session

## 🔧 Configuration

### Retrieval Settings (in main.py)

Modify the retriever configuration to fine-tune search behavior:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",              # Search strategy
    search_kwargs={
        "k": 3,                     # Number of documents to return
        "fetch_k": 10,              # Fetch more candidates for MMR
        "lambda_mult": 0.5          # Diversity parameter (0-1)
    }
)
```

### Text Splitting Settings (in create_database.py)

Adjust chunk size and overlap for different document types:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Characters per chunk
    chunk_overlap=200       # Overlap between chunks
)
```

## 📦 Dependencies

- **LangChain**: Orchestration and prompt management
- **MistralAI**: State-of-the-art embeddings and language models
- **Chroma**: Vector database for efficient storage and retrieval
- **Streamlit**: Web UI framework
- **FastAPI**: Optional API layer support
- **PyPDF**: PDF document loading
- **python-docx**: Word document support
- **python-dotenv**: Environment variable management

See `requirements.txt` for the complete list.

## 🔐 API Keys

This project requires a [MistralAI API key](https://console.mistral.ai/):

1. Sign up at Mistral AI
2. Generate an API key in your account dashboard
3. Add it to your `.env` file as `MISTRAL_API_KEY`

## 📝 System Prompt

The system uses the following prompt for context-aware responses:

```
You are a helpful AI assistant that answers questions based on the provided context.
If the answer is not in the context, say: "I could not find the answer in the document."
```

Modify this in `main.py` or `app.py` to customize the AI's behavior.

## 🎯 How It Works

1. **Document Loading**: PDFs are loaded and split into manageable chunks
2. **Embedding Generation**: MistralAI creates embeddings for each chunk
3. **Storage**: Embeddings and documents are stored in Chroma database
4. **Query Processing**: User queries are embedded and searched against the database
5. **Context Retrieval**: MMR search retrieves the most relevant documents
6. **Response Generation**: MistralAI LLM generates answers based on retrieved context

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

**Q: "I could not find the answer in the document." for all queries**
- Ensure your PDF is in `Document_Loader/` 
- Run `create_database.py` to rebuild the database
- Check that the PDF contains searchable text (not scanned images)

**Q: MistralAI API errors**
- Verify your API key is correct in `.env`
- Check your account has sufficient credits
- Ensure internet connection is stable

**Q: Slow retrieval**
- Reduce `fetch_k` parameter in retriever settings
- Use smaller `chunk_size` values
- Clear old `chroma_db/` folder and rebuild database

## 📧 Support

For issues and questions, please open an issue on GitHub or contact the maintainers.

---

**Made with ❤️ using LangChain, MistralAI, and Chroma**
