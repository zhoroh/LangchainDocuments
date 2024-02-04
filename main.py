import streamlit as st
import os
import re
import random
import traceback
import copy
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OBJECTIVE="Objective"
TRUE_OR_FALSE="True/False"
OBJECTIVE_DEFAULT_VALUE=5
TRUE_OR_FALSE_DEFAULT_VALUE=5

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
        agent_kwargs = {
            'prefix': f"""
            You are a friendly PDF Question and Answering Assistant.
            You are tasked to assist the current user on questions related to the uploaded PDF file.
            You have access to the following tools:{self.tools}.
             Try as much as possible to get your answers only from the tools provided to you. If you can not answer correctly, respond appropriately"""}
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

# Upload Section
def upload():
    st.header('Upload Your PDF')
    st.write("Drag and drop your PDF file here or click to upload. Please ensure that the text in the PDF is selectable and not a scanned image.")
    uploaded_file = st.file_uploader("", type="pdf")

    if uploaded_file is not None:
        # Check if the uploaded file is a PDF
        if "pdf_name" in st.session_state:
            if 'past' in st.session_state:
                del st.session_state.past[:]
            if 'generated' in st.session_state:
                del st.session_state.generated[:]
        if 'questions' in st.session_state:
            del st.session_state['questions']


        pdf = uploaded_file.name
        pdf_name = '.'.join(pdf.split('.')[:-1])
        with open(pdf, mode='wb') as f:
            f.write(uploaded_file.getbuffer()) # save pdf to disk
        st.success("Uploading File.....")
        loader = PyPDFLoader(pdf)
        st.session_state['pages'] = loader.load_and_split()
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=20)
        texts = text_splitter.split_documents(documents)
        if len(texts) == 0:
            st.error("Please ensure your uploaded  document is selectable (i.e not scanned)")
        else:
            st.success("File uploaded successfully!")
            st.write("Processing Uploaded PDF..........")
            embeddings = OpenAIEmbeddings()
            try:
                db = FAISS.from_documents(texts,embeddings)
                retriever = db.as_retriever()
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                )
                
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                agency = Agency()
                agency.initialize_tools(qa,pdf_name)
                agency.load_agent_details(memory)
                st.session_state['qabot'] = agency
                st.session_state['pdf_name'] = pdf
                #generate summary
                response = agency.get_response("Generate a short, concise and useful summary of the uploaded document")
                summary = response['output']
                st.success("PDF processed Successfully!!!")
                st.info("Summary of Uploaded Document is shown below")
                st.write(summary)
                # action = st.radio("What do you want to do?", ["INTERACT WITH UPLOADED PDF", "QUIZZ GENERATION"],horizontal=True)
                st.info("Please Proceed By Clicking on any of the Options on the Left Sidebar")
                # if st.button("Proceed"):
                #     if action == "INTERACT WITH UPLOADED PDF":
                #         chatbot()
                #     else:
                #         quizz_generation()
               
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()
                st.error("A network error occured, Please check your internet connection and try again")


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
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(response['output'])
            st.session_state["input_message_key"] = str(random.random())
            st.rerun()
        if st.session_state["generated"]:
             with chat_container:
                  for i in range(len(st.session_state["generated"])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                    message(st.session_state["generated"][i], key=str(i))


def parse_questions(input_text):
    question_blocks = input_text.split('QUESTION')

    questions = []

    for block in question_blocks:
        print("BLOCK is", block )
        if block.strip() and "CHOICE_A" in block:
            
            question_num = int(re.search(r'\d+', block).group())

          
            question_text = re.search(r'(?<=:\s).*?(?=\n)', block).group()

         
            choices = re.findall(r'(CHOICE_[A-E]):\s(.*?)\n', block)
            options = {choice[0][-1]: choice[1] for choice in choices}

         
            answer = re.search(r'Answer:\s([A-E])', block).group(1)

        
            question_obj = {
                "question_num": question_num,
                "question": question_text,
                "options": options,
                "answer": answer
            }

          
            questions.append(question_obj)

    return questions


def generate_quizz(document,type,num_of_questions):
    if type == OBJECTIVE:
        

        prompt_template =  """You are a teacher preparing questions for a quiz. Given the following document, please generate {num} multiple-choice questions (MCQs) with 5 options and a corresponding answer letter based on the document except for CHOICE_A which should ALWAYS BE defaulted to NONE

            Example question:

            QUESTION 1: question here
            CHOICE_A: NONE
            CHOICE_B: choice here
            CHOICE_C: choice here
            CHOICE_D: choice here
            CHOICE_E: choice here
            Answer: C
            
            QUESTION 2: question here
            CHOICE_A: NONE
            CHOICE_B: choice here
            CHOICE_C: choice here
            CHOICE_D: choice here
            CHOICE_E: choice here
            Answer: B



            These questions should be detailed and solely based on the information provided in the document.

            <Begin Document>
            {doc}
            <End Document>"""
    elif type == TRUE_OR_FALSE:
        prompt_template =  """You are a teacher preparing True or False questions for a quiz. Given the following document, please generate {num} True or False type questions  with 3 options and a corresponding True or False answer except for CHOICE_A which should ALWAYS BE defaulted to NONE
            For each question, randomly select a section or sentence from the document, and then create a question that is directly based on the text .

            Remember to vary the type of information each question is based on. Also, ensure each question has a different structure or focus to avoid repetition.


            Example question:

            QUESTION 1: question here
            CHOICE_A: NONE
            CHOICE_B: choice here
            CHOICE_C: choice here
            Answer: B

            QUESTION 2: question here
            CHOICE_A: NONE
            CHOICE_B: choice here
            CHOICE_C: choice here
            Answer: C



            These questions should be detailed and solely based on the information provided in the document.


            <Begin Document>
            {doc}
            <End Document>"""

    
    prompt = PromptTemplate(
    input_variables=["doc","num"], template=prompt_template
    )
    llm = LLMChain(llm=ChatOpenAI(temperature=0.5, model_name="gpt-4"), prompt=prompt)
    questions = (llm.run(num=num_of_questions,doc=document))
    cleaned_questions = parse_questions(questions)
    return cleaned_questions
    
def generate_questions(quiz_type:str,pages,no_of_questions):
    questions = []
    for page in pages:
        document = page.page_content
        cleaned_quest = generate_quizz(document,quiz_type,no_of_questions)
        questions.extend(cleaned_quest)

    return questions
    
def generate_page_ranges(n):
    ranges = {}
    for i in range(1, n, 10):
        end_page = min(i + 9, n)
        ranges[f"Pages {i} - {end_page}"] = [i-1,end_page]
    return ranges

def get_pages(selected_page_ranges,page_ranges_2_index,total_pages):
    if len(selected_page_ranges) == page_ranges_2_index:
        return total_pages
    
    pages_to_query = []
    for page_range in selected_page_ranges:
        from_index, to_index = page_ranges_2_index[page_range]
        for i in range(from_index,to_index):
            pages_to_query.append(total_pages[i])
    return pages_to_query
        
def get_selected_page_ranges(page_ranges):
    
    st.write("What Pages do you want to test your knowledge in?")
    # Create a row for 'Select All' checkbox
    
    col1, col2 = st.columns([1, 3])  # Adjust the ratio if needed
    select_all = col1.checkbox('Select All')

    if select_all:
        st.info("We advise you to test your knowledge per Page Range so your questions gets generated faster 😢😰")
        st.info("Should you want to still test your self on multiple Page Ranges, Try to Limit them as much as possible to get your questions generated faster 😒😪")
        selected_page_ranges = page_ranges
    else:
        selected_page_ranges = []
      
        for i in range(0, len(page_ranges), 6):
            cols = st.columns(6)  # Create 6 columns
            # Iterate over the columns
            for col, page_range in zip(cols, page_ranges[i:i+6]):
                with col:
                    if st.checkbox(page_range):
                        selected_page_ranges.append(page_range)

    return selected_page_ranges




    
def quizz_generation():
    if 'qabot' not in st.session_state:
        st.error("A PDF File has to be uploaded to generate quizz")
        return
    st.header("🌟 Welcome to StuddyBuddy! Ready to Test Your Knowledge? 🚀")
    
    pages = st.session_state['pages']
    # print("PAEGS is ",pages)
    page_ranges_2_index= generate_page_ranges(len(pages))

    page_ranges = list(page_ranges_2_index.keys())
    selected_page_ranges = get_selected_page_ranges(page_ranges)
    st.write('Selected Page Range:', ', '.join(selected_page_ranges) if selected_page_ranges else 'None')
    selection = st.radio("What Format do you want?", [OBJECTIVE, TRUE_OR_FALSE],horizontal=True)
    user_input = None
    if selection == OBJECTIVE:
        user_input = st.number_input('How many Questions Do you want Generated per Page', min_value=2, value=OBJECTIVE_DEFAULT_VALUE, step=1, format='%d')
        if user_input > OBJECTIVE_DEFAULT_VALUE:
            st.error(f"Error: value should not be greater than {OBJECTIVE_DEFAULT_VALUE}.")
            return
        st.info(f"You'll be tested on a maximum of {OBJECTIVE_DEFAULT_VALUE} questions per Page")
    if selection == TRUE_OR_FALSE:
        user_input = st.number_input('How many Questions Do you want Generated per Page', min_value=2, value=TRUE_OR_FALSE_DEFAULT_VALUE, step=1, format='%d')
        if user_input > TRUE_OR_FALSE_DEFAULT_VALUE:
            st.error(f"Error: value should not be greater than {TRUE_OR_FALSE_DEFAULT_VALUE}.")
            return
        st.info(f"You'll be tested on a maximum of {TRUE_OR_FALSE_DEFAULT_VALUE} questions per Page")
    if st.button('generate quizz'):
        if len(selected_page_ranges) == 0:
            st.error("Please select Page Ranges you want to get tested on")
            return  
        pages_to_query = get_pages(selected_page_ranges,page_ranges_2_index,pages)
        st.info("Generating Quizz Questions")
        st.session_state['questions'] = generate_questions(selection,pages_to_query,user_input)
        st.session_state['selection'] = selection
        st.success("Quizz generated Successfukky")
        st.info("Please Click on the Display Quizz Button to Proceed")
    
if 'quiz_state' not in st.session_state:
    st.session_state.quiz_state = {'submitted': False, 'user_answers': {}, 'score': 0}


def get_state():
    return {'submitted': False, 'user_answers': {}, 'score': 0}


def display_on_streamlit():
    if 'questions' not in st.session_state:
        st.error("Generate Quizz please")
        return
    data = st.session_state['questions']
    type = st.session_state['selection']
    if type == OBJECTIVE:
        st.title("Multiple Choice Quiz")
    elif type == TRUE_OR_FALSE:
        st.title("True or False Type Quiz")

    state = get_state()

    if not state['submitted']:
        form_placeholder = st.empty()
        with form_placeholder.form(key='quiz_form'):
            user_answers = {}
            for idx, question in enumerate(data, start=1):
                st.write(f"Question {idx}: {question['question']}")
                options = question['options']
                user_answers[idx] = st.radio(f"Options for Question {idx}", options.keys(), format_func=lambda x: options[x])
            
            submit_button = st.form_submit_button(label='Submit Answers')

        if submit_button:
            state['submitted'] = True
            state['user_answers'] = user_answers
            # Calculate score
            state['score'] = sum(1 for idx, q in enumerate(data, start=1) if user_answers[idx] == q['answer'])
            form_placeholder.empty() # Clear the form

    if state['submitted']:
        for idx, question in enumerate(data, start=1):
            user_answer = state['user_answers'][idx]
            correct_answer = question['options'][question['answer']]
            if user_answer == question['answer']:
                feedback = "Correct"
            else:
                feedback = f"Incorrect, Correct answer is {correct_answer}"

            st.write(f"Question {idx}: {question['question']}")
            st.write(f"Your answer: {question['options'][user_answer]} - {feedback}")
            st.write("\n")
            st.write("\n")

        # Display score
        st.write(f"Your Score: {state['score']}/{len(data)}")

        if st.button('Try Again'):
            state['submitted']
            st.experimental_rerun()


def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Upload File","Interact With Uploaded PDF", "Quizz Generation","Display Quizz"])
    if selection == "Upload File":
        upload()
    elif selection == "Interact With Uploaded PDF":
        chatbot()
    elif selection == "Quizz Generation":
        quizz_generation()
    elif selection == "Display Quizz":
        display_on_streamlit()

main()