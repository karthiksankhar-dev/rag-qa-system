
# RAG Question Answering System

A powerful Retrieval Augmented Generation (RAG) application that combines real-time web search, Wikipedia knowledge, and local language models to provide comprehensive answers to user questions.

## 🚀 Features

- **Multi-Source Knowledge Retrieval**: Integrates Google Search and Wikipedia for comprehensive information gathering
- **Local LLM Processing**: Uses Ollama with Llama 3.1 for intelligent response generation
- **Vector Database Storage**: Implements Chroma DB with NVIDIA embeddings for semantic search
- **Real-time Streaming**: Provides live response generation with custom UI feedback
- **Persistent Knowledge Base**: Maintains conversation history and learned information
- **User-Friendly Interface**: Clean Streamlit web application with expandable output containers


## 🛠️ Prerequisites

Before running this application, ensure you have:

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- Google Serper API key
- NVIDIA API key
- At least 8GB RAM (recommended for local LLM)


## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-qa-system.git
cd rag-qa-system
```


### 2. Create a conda Environment

```bash
conda create -n env_name python=3.8
conda activate env_name
```


### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


### 4. Install and Setup Ollama


```bash
# Install Ollama (visit https://ollama.ai/ for OS-specific instructions)
# After installation on your machine, run
ollama --version
# Pull the required model
ollama pull llama3.1
```


### 5. Setup API Keys

Edit the file named `id.py` in the project root directory:

```python
# id.py
SERPER_API_KEY = "your_serper_api_key_here"
nvapi_key = "your_nvidia_api_key_here"
```


## 🔑 API Key Setup

### Google Serper API

1. Visit [Serper.dev](https://serper.dev/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Add it to your `id.py` file

### NVIDIA API

1. Visit [NVIDIA AI Foundation Models](https://www.nvidia.com/en-us/ai-data-science/foundation-models/)
2. Sign up and get your API key
3. Add it to your `id.py` file

## 🚀 Usage

### 1. Start the Application

```bash
streamlit run RAG.py
```


### 2. Access the Interface

Open your browser and navigate to `http://localhost:8501`

### 3. Ask Questions

- Enter your question in the text input field
- Watch as the system retrieves information and generates answers in real-time
- Expand the output container to see the complete response


## 📁 Project Structure

```
rag-qa-system/
├── app.py                 # Main application file
├── id.py                  # API keys configuration
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── chroma_db/            # Vector database storage (created automatically)
└── .gitignore           # Git ignore file
```


## 🔧 Configuration

### Customizing the LLM

To use a different Ollama model, modify the `llm` initialization in `RAG.py`:

```python
llm = ChatOllama(model="your-preferred-model", temperature=0, streaming=True)
```


### Adjusting Text Chunking

Modify the text splitter parameters for different chunk sizes:

```python
splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
```


### Vector Database Location

Change the persistence directory for the Chroma database:

```python
persist_directory = "./your-custom-db-path"
```


## 📋 Requirements.txt

```txt
streamlit>=1.28.0
langchain-community>=0.0.20
langchain-ollama>=0.1.0
langchain-nvidia-ai-endpoints>=0.1.0
chromadb>=0.4.15
google-search-results>=2.4.2
wikipedia>=1.4.0
langchain>=0.1.0
```


## 🐛 Troubleshooting

### Common Issues

**1. Ollama Connection Error**

```bash
# Ensure Ollama is running
ollama serve
# Pull the required model
ollama pull llama3.1
```

**2. API Key Errors**

- Verify your API keys are correctly set in `id.py`
- Check API key quotas and permissions

**3. ChromaDB Persistence Issues**

- Delete the `chroma_db` folder and restart the application
- Ensure proper write permissions in the project directory

**4. Memory Issues**

- Reduce chunk size in the text splitter
- Close other memory-intensive applications


## 🔒 Security Notes

- Never commit your `id.py` file to version control
- Add `id.py` to your `.gitignore` file
- Use environment variables in production deployments


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) for the RAG framework
- [Ollama](https://ollama.ai/) for local LLM hosting
- [Streamlit](https://streamlit.io/) for the web interface
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [NVIDIA](https://www.nvidia.com/) for embedding models



**Note**: Make sure to keep your API keys secure and never share them publicly!

