from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import os

# Membaca aturan dari file
with open('code_rules.txt', 'r') as file:
    code_rules = file.read().strip()

# Template Prompt
prompt_template = PromptTemplate.from_template(
    """
    You are an AI assistant specialized in code review.  
    Your task is to review the provided Golang code snippet **strictly based on the given rules**.  

    🚨 **Do not introduce any additional rules beyond those stated below.** 🚨  

    ## **Rules for Code Review**  
    {code_rules}

    1️⃣ Identify and report only the following issues:  
        - Code redundancy  
        - Inefficient logic  
        - Security vulnerabilities  
        - Bad coding practices  
        - Bad coding complexity  
        - Unnecessary dependencies  

    2️⃣ Do **not** comment on:  
        - Variable naming unless it violates best practices  
        - Code style unless it affects readability  
        - Any issues outside the categories above  

    3️⃣ Your review **must** follow this JSON format:  
    ```json
    {{
        "review": [
            {{
                "file_name": "{file_name}",
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

    Now, review the following Golang code from `{file_name}` based **only on the rules above**:  

    ```golang
    {code_snippet}
    ```

    **Your output must strictly follow the JSON format provided.**
    """
)

# Fungsi membaca file dalam batch dengan sliding window dan mencegah review ulang
def read_file_in_batches(file_path, batch_size, overlap):
    with open(file_path, "r") as f:
        lines = f.readlines()  # Baca semua baris
        total_lines = len(lines)
        reviewed_lines = set()  # Set untuk menyimpan baris yang sudah direview

        for i in range(0, total_lines, batch_size - overlap):
            # Ambil batch dengan sliding window
            batch = [(i + j + 1, lines[i + j].rstrip()) for j in range(min(batch_size, total_lines - i))]

            # Hanya simpan baris yang belum direview
            batch = [(ln, code) for ln, code in batch if ln not in reviewed_lines]

            # Lewati batch kosong
            if not batch:
                continue

            # Tandai baris dalam batch sebagai sudah direview
            reviewed_lines.update(ln for ln, _ in batch)
            
            yield batch

# Fungsi untuk mendapatkan semua file .go dalam direktori (termasuk subfolder)
def list_go_files(directory):
    go_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".go"):
                go_files.append(os.path.join(root, file))
    return go_files

# Proses review dengan batching
for file_name in list_go_files("example_code"):
    for batch in read_file_in_batches(file_name, batch_size=5, overlap=2):
        # Format batch menjadi string dengan nomor baris
        formatted_code = "\n".join(f"{ln}: {code}" for ln, code in batch)

        prompt_input = prompt_template.format(
            code_rules=code_rules,
            file_name=os.path.basename(file_name),
            code_snippet=formatted_code,
        )

        llm = ChatOllama(model="qwen2.5-coder", temperature=0.1)
        llm_resp = llm.invoke(prompt_input)
        print(llm_resp.content)
