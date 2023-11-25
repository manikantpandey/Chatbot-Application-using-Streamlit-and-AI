import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import requests
from bs4 import BeautifulSoup
from transformers import T5Tokenizer
from nltk.tokenize import sent_tokenize

def split_text_into_sentences(text, max_length):
    tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=max_length)
    sentences = sent_tokenize(text)
    chunks = []

    current_chunk = ""
    for sentence in sentences:
        new_chunk = current_chunk + sentence + " "
        if len(tokenizer.encode(new_chunk)) <= max_length:
            current_chunk = new_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def process_text_chunks(chunks):
    processed_texts = []
    for chunk in chunks:
        processed_chunk = get_vectorstore(chunk)  
        processed_texts.append(processed_chunk)
    return processed_texts



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")   # Use instructor of Your choice
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})   # Use LLM of Your choice

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def fetch_blog_links(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        latest_container = soup.find('div', {'id': 'nlatest'})               # Set division and classes according to website
        list_items = latest_container.find_all('li') if latest_container else []
        blog_links = [item.find('a').get('href') for item in list_items]
        return blog_links
    else:
        print(f"Failed to retrieve the webpage, status code: {response.status_code}")
        return []

def extract_blog_content(blog_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(blog_url, headers=headers)
    
    print(f"Status Code for {blog_url}: {response.status_code}")

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        content = {}
        paragraphs = soup.find_all('p')           # Set division and classes according to website
        content['text'] = '\n\n'.join(paragraph.text for paragraph in paragraphs)

        return content
    else:
        print("Failed to retrieve the blog")
        return {}

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with AI")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with AI")
    user_question = st.text_input("Ask your question:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Link")
        website_url = st.text_input("Paste your link here and click on 'Process'")
        if st.button("Process"):
            with st.spinner("Processing"):
                if 'blog_links' not in st.session_state or st.session_state.website_url != website_url:
                    st.session_state.website_url = website_url
                    st.session_state.blog_links = fetch_blog_links(website_url)

                all_blog_texts = ""

                for blog_link in st.session_state.blog_links:
                    content = extract_blog_content(blog_link)
                    all_blog_texts += content['text'] + "\n\n"

                max_length = 2048
                text_chunks = split_text_into_sentences(all_blog_texts, max_length)

                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()