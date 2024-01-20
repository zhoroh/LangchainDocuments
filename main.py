import os
import random
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from utils import update_chat_history_and_get_answer, clear_history

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setting the title of the app


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
        loader = PyPDFLoader(pdf)
        # print("loader is ", loader)
        documents = loader.load()
        # print("docments is", loader.load())
        text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        # print("Texts is", texts)
        if len(texts) == 0:
            st.error("Please ensure your uploaded  document is selectable (i.e not scanned)")
        else:
            st.success("File uploaded successfully!")
            st.write("Processing Uploaded PDF..........")
            embeddings = OpenAIEmbeddings()
            db = Chroma.from_documents(texts,embeddings)
            retriever = db.as_retriever(search_type='similarity',search_kwargs={"k":2})
            qa = ConversationalRetrievalChain.from_llm(OpenAI(),retriever)
            st.success("PDF processed Successfully!!!")
            st.session_state['qabot'] = qa
            st.session_state['pdf_name'] = pdf



def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home Page","Q&A Assistant"])
    if selection == "Home Page":
        homepage()
    elif selection == "Q&A Assistant":
        chatbot()


if __name__ == "__main__":
    main()