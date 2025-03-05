from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# read your code
with open('main.go', 'r') as file:
    # Read the entire content of the file
    code_snippet = file.read()



prompt_template = PromptTemplate.from_template(
    """
    You are a senior software developer.
    Your job is to review the codes written by other software developers
    Given below is the code written by one of the team member
    Review this code for code quality and give your feedback.\n\n
    
    ```golang
    {code_snippet}
    ```
    
    Give you review in points. 
    Between points leave 2 new line space.
    Mention part of the code in your points if required.
    In your review at the max keep {number_of_points} points
    """
)

prompt_input = prompt_template.format(
    code_snippet=code_snippet,
    number_of_points=5)


llm = ChatOllama(model="qwen2.5-coder",temperature=0.1,)
llm_resp = llm.invoke(prompt_input)
print(llm_resp.content)