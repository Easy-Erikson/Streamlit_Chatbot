import streamlit as st
import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI as OpenAI
from langchain.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from dotenv import load_dotenv

### https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

# Load Secrets and Configuration from .env File
load_dotenv(override=True)

# Configuration parameters
DEFAULT_DIRECTORY = "./data"
DEFAULT_MODEL_NAME = "gpt-4o"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_K = 10
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_PROMPT_TEMPLATE = """Du bist ein unternehmensspezifischer Chatbot, der darauf ausgelegt ist,
präzise und relevante Informationen aus internen Datenquellen und externen Wissensbasen zu liefern.
Dein Ziel ist es, die Fragen der Nutzer effizient und verständlich zu beantworten,
dabei einen professionellen Ton beizubehalten und die Unternehmenswerte sowie -richtlinien zu berücksichtigen.
Stelle sicher, bei jeder Information die Quelle anzugeben.

Kontext aus Dokumenten:
{context}

Chat Verlauf:
{chat_history}

Nutzer Anfrage:
{user_request}
"""


# Function to load and process documents
def load_docs(directory, chunk_size=1000, chunk_overlap=200):
    try:
        # Print the directory being loaded
        print(f"Loading documents from directory: {directory}")

        # Get list of markdown files
        files = glob.glob(os.path.join(directory, "**/*.*"), recursive=True)
        print(f"Found {len(files)} files")

        # Load the documents using TextLoader
        documents = []
        for i, file in enumerate(files):
            print(file)
            if file.endswith('.pdf'):
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                doc = Document(page_content=text, metadata={"source": file})
                documents.append(doc)
            elif file.endswith('.txt'):
                text = file.getvalue().decode("utf-8")
                doc = Document(page_content=text, metadata={"source": file})
                documents.append(doc)
            elif file.endswith('.md'):
                doc = TextLoader(file)
                documents.extend(doc.load())
        
        print(f"Loaded {len(documents)} documents")

        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        print(f"Split documents into {len(texts)} chunks")
        
        return texts
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
        return []

# Function to create vector store
def create_vector_store(texts, embedding_model):
    try:
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vector_store = Chroma.from_documents(texts, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

# Function to get top_k relevant documents
def retrieve_top_documents(vector_store, user_request, top_k=5):
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    return retriever.get_relevant_documents(user_request)


# Function to handle user input
def handle_user_input(user_request, vector_store, chat_history, model_name, prompt_template, temperature, max_tokens, top_k):
    top_docs = retrieve_top_documents(vector_store, user_request, top_k)
    combined_docs = "\n\n".join([f"Quelle: {doc.metadata['source']}\n{doc.page_content}" for doc in top_docs])
    
    # Format the prompt with the context and user request
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Initialize the OpenAI model
    llm = OpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens, stream_usage=True)
    
    chain = prompt | llm
    
    # Generate the response from the model
    response = chain.stream({"context": combined_docs, "chat_history": chat_history, "user_request": user_request})
    
    # Extract the content from the AIMessage object
    ai_response_content = response
    
    return ai_response_content


# Main function
def main():
    st.set_page_config(page_title="Chat with Markdown Files", page_icon=":books:")
    st.title("_DEVK-GPT_")
    st.logo("https://upload.wikimedia.org/wikipedia/de/thumb/9/92/DEVK_201x_logo.svg/1200px-DEVK_201x_logo.svg.png")
    
        # Configuration inputs
    directory = DEFAULT_DIRECTORY
    model_name = DEFAULT_MODEL_NAME
    embedding_model = DEFAULT_EMBEDDING_MODEL
    prompt_template = DEFAULT_PROMPT_TEMPLATE
    temperature = DEFAULT_TEMPERATURE
    max_tokens = DEFAULT_MAX_TOKENS
    chunk_size = DEFAULT_CHUNK_SIZE
    chunk_overlap = DEFAULT_CHUNK_OVERLAP

    # Initialize vector store on first run
    if "vector_store" not in st.session_state:
        if directory and os.path.isdir(directory):
            texts = load_docs(directory, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.session_state.vector_store = create_vector_store(texts, embedding_model)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    with st.sidebar:
        top_k = st.slider("Anzahl Ergebnisse Dokumentensuche", min_value=1, max_value=25, value=DEFAULT_TOP_K)
            
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_request = st.chat_input("Stelle eine Frage:")
    if user_request and st.session_state.vector_store:
        st.chat_message("user").write(user_request)
        response = st.chat_message("ai").write_stream(handle_user_input(user_request, st.session_state.vector_store, st.session_state.chat_history, model_name, prompt_template, temperature, max_tokens, top_k))
        st.session_state.messages.append({"role": "user", "content": user_request})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append(("Human", user_request))
        st.session_state.chat_history.append(("AI", response))
        
if __name__ == "__main__":
    main()