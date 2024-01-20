import os
import re
import random
import pinecone
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from utils import update_chat_history_and_get_answer, clear_history

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT=os.getenv("PINECONE_ENVIRONMENT")



def extract_letters(input_string):
    """
    Extracts only the letters from the input string.

    Args:
    input_string (str): The input string from which to extract letters.

    Returns:
    str: A string containing only letters from the input string.
    """
    # Use regular expression to find all letters and join them
    return ''.join(re.findall("[a-zA-Z]+", input_string))


# print(pinecone.list_indexes())

def get_index(index_name):
    indexes = pinecone.list_indexes()
    if len(indexes)==0:
        pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
        # index = pinecone.Index(index_name) # 
        # return index
    else:
        if index_name not in indexes:
            pinecone.delete_index(indexes[0])
            pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
        # if index_name in indexes:
        #     index = pinecone.Index(index_name) # connect to pinecone index
        #     return index
        # else:
        #     pinecone.delete_index(indexes[0])
        #     pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
        #     index = pinecone.Index(index_name) # 
        #     return index

    
# get_index('yam')
# loader = PyPDFLoader("./James Clear - Atomic Habits (2022).pdf")
# # print("loader is ", loader)
# documents = loader.load()
# # print("docments is", loader.load())
# # print("documents is ")
# text_splitter = CharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=0)
# texts = text_splitter.split_documents(documents)
# print("lenghts ius", len(texts))
# embeddings = OpenAIEmbeddings()
# docsearch = Pinecone.from_documents(texts, embeddings, index_name='yam')
# query = "What are the stages of habit formation"
# # docs = docsearch.similarity_search(query)
# # print(docs)
# # print(docs[0].page_content)
# retriever = docsearch.as_retriever(search_type='similarity',search_kwargs={"k":2})

def chatbot():
    if 'qabot' not in st.session_state:
        st.error("Please Upload a PDF File")
    else:
        st.header("Q & A Assistant")
        qa = st.session_state['qabot']
        st.markdown(f"Ask questions related to the uploaded file here : {st.session_state['pdf_name']}")
        message("Hiiiiii!!, My name is Jefe, your personal Q & A Assistant  😊 😊 😊!!!!")
        if "past" not in st.session_state:
            st.session_state['past'] = []
        if "generated" not in st.session_state:
            st.session_state["generated"] = []
        if "input_message_key" not in st.session_state:
            st.session_state["input_message_key"] = str(random.random())
        chat_container = st.container()
        user_input = st.text_input("Type your question here.", key=st.session_state["input_message_key"])
        if st.button("Send"):
            response = update_chat_history_and_get_answer(user_input,qa)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(response)
            st.session_state["input_message_key"] = str(random.random())
            print("jdfbdvfd hf")
            print(st.session_state['input_message_key'])
            st.rerun()
        if st.session_state["generated"]:
             with chat_container:
                  for i in range(len(st.session_state["generated"])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                    message(st.session_state["generated"][i], key=str(i))


def homepage():
    st.title('PDF Q&A Assistant')

    # Adding a brief description
    st.write("Upload your PDF and ask any question - Get answers instantly! Our AI-powered tool reads through your PDF and provides answers to your queries.")

    # Upload Section
    st.header('Upload Your PDF')
    st.write("Drag and drop your PDF file here or click to upload. Please ensure that the text in the PDF is selectable and not a scanned image.")
    uploaded_file = st.file_uploader("", type="pdf")

    if uploaded_file is not None:
        # Check if the uploaded file is a PDF
        if "pdf_name" in st.session_state:
            clear_history()
            del st.session_state.past[:]
            del st.session_state.generated[:]
           


        pdf = uploaded_file.name
        with open(pdf, mode='wb') as f:
            f.write(uploaded_file.getbuffer()) # save pdf to disk
        st.success("Uploading File.....")
        print(pdf)
        name = extract_letters(pdf.lower())
        name = name[:40]
        get_index(name)
        loader = PyPDFLoader(pdf)
        # print("loader is ", loader)
        documents = loader.load()
        # print("docments is", loader.load())
        # print("documents is ")
        text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        print("lenghts ius", len(texts))
        # print("Texts is", texts)
        if len(texts) == 0:
            st.error("Please ensure your uploaded  document is selectable (i.e not scanned)")
        else:
            st.success("File uploaded successfully!")
            st.write("Processing Uploaded PDF..........")
            embeddings = OpenAIEmbeddings()
            docsearch = Pinecone.from_documents(texts, embeddings, index_name=name)
            retriever = docsearch.as_retriever(search_type='similarity',search_kwargs={"k":2})
            # if "pdf_name" in st.session_state:
            #     db.delete_collection()
           
            # db = Chroma.from_documents(texts,embeddings,collection_name="test_collection")
            # print("yam")
            # print(db._collection.count())
            # retriever = db.as_retriever(search_type='similarity',search_kwargs={"k":2})

            qa = ConversationalRetrievalChain.from_llm(OpenAI(),retriever)
            st.success("PDF processed Successfully!!!")
            st.session_state['qabot'] = qa
            st.session_state['pdf_name'] = pdf
            st.write("Proceed to the Assistant Please")



def main():
    pinecone.init(
    api_key=PINECONE_API_KEY , # find at app.pinecone.io
    environment=PINECONE_ENVIRONMENT,  # next to api key in console
    )
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home Page","Q&A Assistant"])
    if selection == "Home Page":
        homepage()
    elif selection == "Q&A Assistant":
        chatbot()


if __name__ == "__main__":
    main()
