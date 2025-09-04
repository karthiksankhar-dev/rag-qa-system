
# RAG Question Answering System

A powerful Retrieval Augmented Generation (RAG) application that combines real-time web search, Wikipedia knowledge, and local language models to provide comprehensive answers to user questions.
🚀 Features
Multi-Source Knowledge Retrieval: Integrates Google Search and Wikipedia for comprehensive information gathering
Local LLM Processing: Uses Ollama with Llama 3.1 for intelligent response generation
Vector Database Storage: Implements Chroma DB with NVIDIA embeddings for semantic search
Real-time Streaming: Provides live response generation with custom UI feedback
Persistent Knowledge Base: Maintains conversation history and learned information
User-Friendly Interface: Clean Streamlit web application with expandable output containers
🛠️ Prerequisites
Before running this application, ensure you have:
Python 3.8 or higher
[Ollama](https://ollama.ai/) installed and running
Google Serper API key
NVIDIA API key
At least 8GB RAM (recommended for local LLM)
📦 Installation

1. Clone the Repository
bash
git clone [https://github.com/yourusername/rag-qa-system.git](https://github.com/yourusername/rag-qa-system.git)
cd rag-qa-system
2. Create Virtual Environment
bash
python -m venv venv
source venv/bin/activate  \# On Windows: venv\Scripts\activate
3. Install Dependencies
bash
pip install streamlit
pip install langchain-community
pip install langchain-ollama
pip install langchain-nvidia-ai-endpoints
pip install chromadb
pip install google-search-results
pip install wikipedia
Or install from requirements file:
bash
pip install -r requirements.txt
4. Install and Setup Ollama
bash

# Install Ollama (visit [https://ollama.ai/](https://ollama.ai/) for OS-specific instructions)

# Pull the required model

ollama pull llama3.1
5. Setup API Keys
Create a file named id.py in the project root directory:
python

# id.py

SERPER_API_KEY = "your_serper_api_key_here"
nvapi_key = "your_nvidia_api_key_here"
🔑 API Key Setup
Google Serper API
Visit [Serper.dev](https://serper.dev/)
Sign up for a free account
Get your API key from the dashboard
Add it to your id.py file
NVIDIA API
Visit [NVIDIA AI Foundation Models](https://www.nvidia.com/en-us/ai-data-science/foundation-models/)
Sign up and get your API key
Add it to your id.py file
🚀 Usage

1. Start the Application
bash
streamlit run app.py
2. Access the Interface
Open your browser and navigate to http://localhost:8501
3. Ask Questions
Enter your question in the text input field
Watch as the system retrieves information and generates answers in real-time
Expand the output container to see the complete response
📁 Project Structure
text
rag-qa-system/
├── app.py                 \# Main application file
├── id.py                  \# API keys configuration
├── requirements.txt       \# Python dependencies
├── README.md             \# Project documentation
├── chroma_db/            \# Vector database storage (created automatically)
└── .gitignore           \# Git ignore file
🔧 Configuration
Customizing the LLM
To use a different Ollama model, modify the llm initialization in app.py:
python
llm = ChatOllama(model="your-preferred-model", temperature=0, streaming=True)
Adjusting Text Chunking
Modify the text splitter parameters for different chunk sizes:
python
splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
Vector Database Location
Change the persistence directory for the Chroma database:
python
persist_directory = "./your-custom-db-path"
📋 Requirements.txt
text
streamlit>=1.28.0
langchain-community>=0.0.20
langchain-ollama>=0.1.0
langchain-nvidia-ai-endpoints>=0.1.0
chromadb>=0.4.15
google-search-results>=2.4.2
wikipedia>=1.4.0
langchain>=0.1.0
🐛 Troubleshooting
Common Issues
4. Ollama Connection Error
bash

# Ensure Ollama is running

ollama serve

# Pull the required model

ollama pull llama3.1
2. API Key Errors
Verify your API keys are correctly set in id.py
Check API key quotas and permissions
3. ChromaDB Persistence Issues
Delete the chroma_db folder and restart the application
Ensure proper write permissions in the project directory
4. Memory Issues
Reduce chunk size in the text splitter
Close other memory-intensive applications
🔒 Security Notes
Never commit your id.py file to version control
Add id.py to your .gitignore file
Use environment variables in production deployments
🤝 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
📄 License
This project is licensed under the MIT License - see the [LICENSE](https://www.perplexity.ai/search/LICENSE) file for details.
🙏 Acknowledgments
[LangChain](https://python.langchain.com/) for the RAG framework
[Ollama](https://ollama.ai/) for local LLM hosting
[Streamlit](https://streamlit.io/) for the web interface
[ChromaDB](https://www.trychroma.com/) for vector storage
[NVIDIA](https://www.nvidia.com/) for embedding models
📞 Support
If you encounter any issues or have questions:
Open an issue on GitHub
Check the troubleshooting section above
Review the official documentation for each technology used
I need the above to be in a single .md code file

Got it ✅ — you want the entire project documentation written as a single Markdown (`.md`) file.
Here’s the properly formatted `README.md` content you can save directly as a Markdown file:

```markdown
# 🚀 RAG Question Answering System

A powerful **Retrieval Augmented Generation (RAG)** application that combines real-time web search, Wikipedia knowledge, and local language models to provide **comprehensive, intelligent answers** to user questions.

---

## ✨ Features
- **Multi-Source Knowledge Retrieval**: Integrates Google Search and Wikipedia for comprehensive information gathering  
- **Local LLM Processing**: Uses [Ollama](https://ollama.ai/) with *Llama 3.1* for intelligent response generation  
- **Vector Database Storage**: Implements **Chroma DB** with NVIDIA embeddings for semantic search  
- **Real-time Streaming**: Provides live response generation with custom UI feedback  
- **Persistent Knowledge Base**: Maintains conversation history and learned information  
- **User-Friendly Interface**: Clean **Streamlit** web application with expandable output containers  

---

## 🛠️ Prerequisites
Before running this application, ensure you have:
- **Python 3.8+**
- **Ollama** installed and running
- **Google Serper API key**
- **NVIDIA API key**
- At least **8GB RAM** (recommended for local LLM)

---

## 📦 Installation

### 1. Clone the Repository
```

git clone https://github.com/yourusername/rag-qa-system.git
cd rag-qa-system

```

### 2. Create Virtual Environment
```

python -m venv venv
source venv/bin/activate   \# On Windows: venv\Scripts\activate

```

### 3. Install Dependencies
```

pip install streamlit
pip install langchain-community
pip install langchain-ollama
pip install langchain-nvidia-ai-endpoints
pip install chromadb
pip install google-search-results
pip install wikipedia

```

Or install everything from requirements:
```

pip install -r requirements.txt

```

### 4. Install and Setup Ollama
```


# Install Ollama (see https://ollama.ai/ for OS-specific instructions)

# Pull the required model

ollama pull llama3.1

```

### 5. Setup API Keys
Create a file named **id.py** in the project root:
```


# id.py

SERPER_API_KEY = "your_serper_api_key_here"
nvapi_key = "your_nvidia_api_key_here"

```

---

## 🔑 API Key Setup

### Google Serper API
1. Visit [Serper.dev](https://serper.dev/)  
2. Sign up for a free account  
3. Get your API key from the dashboard  
4. Add it to `id.py`  

### NVIDIA API
1. Visit [NVIDIA AI Foundation Models](https://www.nvidia.com/en-us/ai-data-science/foundation-models/)  
2. Sign up and get your API key  
3. Add it to `id.py`  

---

## 🚀 Usage

### 1. Start the Application
```

streamlit run app.py

```

### 2. Access the Interface
Open your browser: **http://localhost:8501**

### 3. Ask Questions
- Enter your question in the input field  
- Watch as the system retrieves and generates answers in real-time  
- Expand the output to view the detailed response  

---

## 📁 Project Structure
```

rag-qa-system/
├── app.py                 \# Main application file
├── id.py                  \# API keys configuration
├── requirements.txt       \# Python dependencies
├── README.md              \# Project documentation
├── chroma_db/             \# Vector database storage (auto-generated)
└── .gitignore             \# Git ignore file

```

---

## 🔧 Configuration

### Customizing the LLM
```

llm = ChatOllama(model="your-preferred-model", temperature=0, streaming=True)

```

### Adjusting Text Chunking
```

splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

```

### Vector Database Location
```

persist_directory = "./your-custom-db-path"

```

---

## 📋 requirements.txt
```

streamlit>=1.28.0
langchain-community>=0.0.20
langchain-ollama>=0.1.0
langchain-nvidia-ai-endpoints>=0.1.0
chromadb>=0.4.15
google-search-results>=2.4.2
wikipedia>=1.4.0
langchain>=0.1.0

```

---

## 🐛 Troubleshooting

### 1. Ollama Connection Error
```


# Ensure Ollama is running

ollama serve

# Pull required model

ollama pull llama3.1

```

### 2. API Key Errors
- Verify keys are in `id.py`  
- Check API key quotas and permissions  

### 3. ChromaDB Issues
- Delete the `chroma_db` folder and restart  
- Ensure the project directory has write permissions  

### 4. Memory Issues
- Reduce chunk size in the text splitter  
- Close other memory-heavy programs  

---

## 🔒 Security Notes
- ❌ Never commit your `id.py` file to GitHub  
- Add `id.py` to `.gitignore`  
- 🔑 Use environment variables for deployment  

---

## 🤝 Contributing
1. Fork the repo  
2. Create a feature branch:  
```

git checkout -b feature/amazing-feature

```
3. Commit changes:  
```

git commit -m "Add some amazing feature"

```
4. Push branch:  
```

git push origin feature/amazing-feature

```
5. Open a Pull Request  

---

## 📄 License
This project is licensed under the **MIT License** – see the LICENSE file for details.

---

## 🙏 Acknowledgments
- [LangChain](https://python.langchain.com/) – for the RAG framework  
- [Ollama](https://ollama.ai/) – for local LLM hosting  
- [Streamlit](https://streamlit.io/) – for UI  
- [ChromaDB](https://www.trychroma.com/) – for vector storage  
- [NVIDIA](https://www.nvidia.com/) – for embeddings  

---

## 📞 Support
- Open an **issue on GitHub**  
- Check the **Troubleshooting** section above  
- Review the official **documentation** of each dependency  

```


***

Would you like me to also create a **minimal ready-to-run `requirements.txt` file** separately along with this `README.md`? That way you can just copy both into your repo and start immediately.

