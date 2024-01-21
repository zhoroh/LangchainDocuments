import os
import random
import string
import pinecone
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT=os.getenv("PINECONE_ENVIRONMENT")

llm = ChatOpenAI(    
        openai_api_key=OPENAI_API_KEY, 
        model_name="gpt-4", 
        temperature=0.0
    )

class Agency:
    def __init__(self):
        self.pdf_agent=None
        # self.bookTitle = None
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

# change the value of the prefix argument in the initialize_agent function. This will overwrite the default prompt template of the zero shot agent type
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

    # def update_history(self):
    #     global chat_history
    def get_response(self,user_input):
        global chat_history
        response = self.agent({"input":user_input,"chat_history":[]})
       
        return response

def generate_random_string(length=10):
    """
    Generates a random string of a given length with only lowercase letters.

    Args:
    length (int): The length of the random string to generate. Default is 10.

    Returns:
    str: A random string of lowercase letters of the specified length.
    """
    # Use only lowercase letters
    characters = string.ascii_lowercase

    # Randomly select characters and join them into a string
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string





# print(pinecone.list_indexes())

def get_index(index_name):
    try:

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
    except:
        st.error("A network error occured, Please Try again")

    
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
            response = st.session_state["qabot"].get_response(user_input)
            print(response)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(response['output'])
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
            # clear_history()
            del st.session_state.past[:]
            del st.session_state.generated[:]
           


        pdf = uploaded_file.name
        pdf_name = '.'.join(pdf.split('.')[:-1])
        with open(pdf, mode='wb') as f:
            f.write(uploaded_file.getbuffer()) # save pdf to disk
        st.success("Uploading File.....")
        print(pdf)
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
        print("lenghts ius", len(texts))
        # print("Texts is", texts)
        if len(texts) == 0:
            st.error("Please ensure your uploaded  document is selectable (i.e not scanned)")
        else:
            get_index(name)
            st.success("File uploaded successfully!")
            st.write("Processing Uploaded PDF..........")
            # try:
            print(1)
            embeddings = OpenAIEmbeddings()
            print(2)
            try:

                docsearch = Pinecone.from_documents(texts, embeddings, index_name=name)
                print(3)
                retriever = docsearch.as_retriever()
                # if "pdf_name" in st.session_state:
                #     db.delete_collection()
            
                # db = Chroma.from_documents(texts,embeddings,collection_name="test_collection")
                # print("yam")
                # print(db._collection.count())
                # retriever = db.as_retriever(search_type='similarity',search_kwargs={"k":2})
                print(4)
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                )
                print(5)
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                agency = Agency()
                print("yam")
                agency.initialize_tools(qa,pdf_name)
                print("work")
                agency.load_agent_details(memory)
                print("worked")
                # qa = ConversationalRetrievalChain.from_llm(OpenAI(),retriever)
                st.success("PDF processed Successfully!!!")
                st.session_state['qabot'] = agency
                st.session_state['pdf_name'] = pdf
                st.write("Proceed to the Assistant Please")
            except:
                st.error("A network error occured, Please check your internet connection and try again")



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
