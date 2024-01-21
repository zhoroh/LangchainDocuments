import os
import random
import string
import pinecone
import traceback
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(    
        openai_api_key=OPENAI_API_KEY, 
        model_name="gpt-4", 
        temperature=0.0
    )

class Agency:
    def __init__(self):
        self.pdf_agent=None
        self.tools = None
        self.agent = None
    
    def initialize_tools(self,pdf_agent,pdf_name):
        self.pdf_agent = pdf_agent
        tools = [
            Tool(
                name = "PDF Question and Answering Assistant",
                func=self.pdf_agent.run,
                description=f"""
                Useful for answering questions related to the uploaded pdf, the name of the pdf file is {pdf_name}
                """
            )
        ]
        self.tools = tools

    def load_agent_details(self,memory):
        agent_kwargs = {'prefix': f'You are a friendly PDF Question and Answering Assistant. You are tasked to assist the current user on questions related to the uploaded PDF file. You have access to the following tools:{self.tools}. Try as much as possible to get your answers only from the tools provided to you. If you can not answer correctly, respond appropriately'}
        # initialize the LLM agent
        agent = initialize_agent(self.tools, 
                                llm, 
                                agent="chat-conversational-react-description", 
                                verbose=True, 
                                agent_kwargs=agent_kwargs,
                                handle_parsing_errors=True,
                                memory=memory
                                )
        self.agent = agent

    def get_response(self,user_input):
        response = self.agent({"input":user_input,"chat_history":[]})
        return response

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
            # try:

            response = st.session_state["qabot"].get_response(user_input)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(response['output'])
            st.session_state["input_message_key"] = str(random.random())
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
            del st.session_state.past[:]
            del st.session_state.generated[:]
           


        pdf = uploaded_file.name
        pdf_name = '.'.join(pdf.split('.')[:-1])
        with open(pdf, mode='wb') as f:
            f.write(uploaded_file.getbuffer()) # save pdf to disk
        st.success("Uploading File.....")
        name = generate_random_string()
        
        loader = PyPDFLoader(pdf)
        # print("loader is ", loader)
        documents = loader.load()
        # print("docments is", loader.load())
        # print("documents is ")
        text_splitter = CharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=20)
        texts = text_splitter.split_documents(documents)
        if len(texts) == 0:
            st.error("Please ensure your uploaded  document is selectable (i.e not scanned)")
        else:
            st.success("File uploaded successfully!")
            embeddings = OpenAIEmbeddings()
            try:
                db = FAISS.from_documents(texts,embeddings)
                retriever = db.as_retriever()
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                )
                st.write("Processing Uploaded PDF..........")
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                agency = Agency()
                agency.initialize_tools(qa,pdf_name)
                agency.load_agent_details(memory)
                st.success("PDF processed Successfully!!!")
                st.session_state['qabot'] = agency
                st.session_state['pdf_name'] = pdf
                st.write("Proceed to the Assistant Please")
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()
                st.error("A network error occured, Please check your internet connection and try again")



def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home Page","Q&A Assistant"])
    if selection == "Home Page":
        homepage()
    elif selection == "Q&A Assistant":
        chatbot()


if __name__ == "__main__":
    main()
