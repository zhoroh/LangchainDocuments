#some code was here
import re
def extract_qa(text):
    """
    Extracts questions and answers from a given text and returns them as a list of dictionaries.
    Each dictionary contains 'question' and 'answer' as keys.
    """
    # Pattern to match questions and answers
    pattern = r"QUESTION \d+: (.*?)\nAnswer: (.*?)\n"
    # Finding all matches
    matches = re.findall(pattern, text, re.DOTALL)
    # Converting matches to the desired format
    qa_list = [{"question": match[0], "answer": match[1]} for match in matches]
    
    return qa_list

# Test the function with the provided text
provided_text = """
QUESTION 1: Why is it important for the activities in an algorithm to be clearly defined?
Answer: To avoid ambiguities.




QUESTION 2: What is the condition for getting output from an algorithm?
Answer: The algorithm must stop after a finite time.
"""

# Calling the function and displaying the result
extracted_qa_list = extract_qa(provided_text)
print(extracted_qa_list)
print(provided_text.split("QUESTION"))