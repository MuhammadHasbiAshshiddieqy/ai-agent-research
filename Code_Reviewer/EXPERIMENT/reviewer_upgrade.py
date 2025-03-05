from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# read your rules
with open('code_rules.txt', 'r') as file:
    # Read the entire content of the file
    code_rules = file.read()



prompt_template = PromptTemplate.from_template(
    """
    You are an AI assistant specialized in code review.  
    Your task is to review the provided Golang code snippet **strictly based on the given rules**.  

    🚨 **Do not introduce any additional rules beyond those stated below.** 🚨  

    ## **Rules for Code Review**  
    1️⃣ Identify and report only the following issues:  
        - Code redundancy  
        - Inefficient logic  
        - Security vulnerabilities  
        - Bad coding practices  
        - Bad coding complexity
        - Unnecessary dependencies
        {code_rules}  
      
    2️⃣ Do **not** comment on:  
        - Variable naming unless it violates best practices  
        - Code style unless it affects readability  
        - Any issues outside the categories above  

    3️⃣ Your review **must** follow this JSON format:  
    ```json
    {{
        "review": [
            {{
                "file_name": <name_of_file>,
                "line": <line_number>,
                "point": "<issue_category>",
                "description": "<brief_explanation>",
                "suggested_fix": "<how_to_fix>",
                "code_change": "<modified_code_if_needed>"
            }}
        ]
    }}
    ```

    - **If no code modification is needed**, set `"code_change": null`.  
    - **If a code modification is required**, provide the updated code snippet inside `"code_change"`.

    **Example Review Output:**  
    ```json
    {{
        "review": [
            {{
                "line": 8,
                "point": "Redundant Code",
                "description": "The function contains duplicate logic for error handling.",
                "suggested_fix": "Refactor error handling into a single function.",
                "code_change": "func handleError(err error) {{ log.Println(err) }}"
            }},
            {{
                "line": 15,
                "point": "Security Vulnerability",
                "description": "User input is not sanitized before database query execution.",
                "suggested_fix": "Use parameterized queries to prevent SQL injection.",
                "code_change": null
            }}
        ]
    }}
    ```

    Now, review the following Golang code based **only on the rules above**:  

    ```golang
    {code_snippet}
    ```

    **Your output must strictly follow the JSON format provided.**
    """
)

def read_file_in_batches(file_path, batch_size):
    with open(file_path, "r") as f:
        batch = []
        for line in f:
            batch.append(line.strip())  # Tambahkan ke batch
            if len(batch) == batch_size:
                yield batch  # Kembalikan batch ke caller
                batch = []  # Reset batch

        if batch:
            yield batch  # Kembalikan sisa batch terakhir jika ada

# Contoh penggunaan:
for batch in read_file_in_batches("review_code/main.go", 5):
    prompt_input = prompt_template.format(
    code_rules=code_rules,
    code_snippet=batch,
    number_of_points=5)

    llm = ChatOllama(model="qwen2.5-coder",temperature=0.1,)
    llm_resp = llm.invoke(prompt_input)
    print(llm_resp.content)

