# Chat with AI: Streamlit Web Application

## Overview
This Streamlit-based web application offers an interactive AI-powered chat interface, leveraging advanced NLP techniques. It also features capabilities for fetching and processing blog content, integrating various libraries like Hugging Face, FAISS, and BeautifulSoup.

## Key Features
- AI-driven chat functionality.
- Text processing with sentence chunking for large texts.
- Blog content extraction and analysis.
- Embedding generation using Hugging Face models.
- Vector storage and retrieval using FAISS.
- Easy-to-use web interface built with Streamlit.

## System Requirements
- Python 3.x
- Streamlit
- LangChain
- HuggingFace Transformers
- NLTK
- Requests
- BeautifulSoup4
- dotenv (for environment variable management)

## Installation Guide
### Clone the Repository:
```bash
git clone https://github.com/manikantpandey/Chatbot-Application-using-Streamlit-and-AI.git
cd Chatbot-Application-using-Streamlit-and-AI
```

## Install Required Packages:
```bash
pip install -r requirements.txt
```

## Setting Up Environment
Create a .env file in the root directory and add necessary environment variables (if any).

## Running the Application
### Launch the application by running:
```bash
streamlit run web.py
```
Access the web interface via the provided URL after execution.

## Usage Instructions
-Chat Interaction: Type your questions in the text input field to converse with the AI.
-Processing Blog Content:
-Paste a blog URL in the sidebar input field.
-Click 'Process' to fetch and analyze the blog content.
-The application breaks down the blog text into manageable chunks for processing.

## Contributing
Contributions to this project are welcomed. Please follow standard open-source contribution guidelines.

## License
MIT
