chat_history =[]
def update_chat_history_and_get_answer(prompt,qa):
    global chat_history
    result = qa({"question": prompt, "chat_history": chat_history})
    response = result['answer']
    chat_history.append((prompt,response))
    return response

def clear_history():
    global chat_history
    chat_history = []