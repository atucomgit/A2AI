import os
import sys
import requests
from typing import List

def main():
    if len(sys.argv) == 1:
        root_dir = "../utils"
    else:
        root_dir = sys.argv[1]

    if os.path.isfile(root_dir):
        refactor_file(root_dir)
    else:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if is_program_file(filename):
                    file_path = os.path.join(dirpath, filename)
                    refactor_file(file_path)


def is_program_file(filename: str) -> bool:
    extensions = ('.py', '.html', '.css', '.js', '.c', '.cpp', '.cs', '.java', '.vb', '.vbs')
    return any(filename.endswith(ext) for ext in extensions)

def load_refactoring_specifications() -> str:
    with open("refactoring_specifications.md", "r", encoding="utf-8") as file:
        return file.read()

def refactor_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        original_code = file.read()

    print(f"---- リファクタリング実施 ----")
    print(file_path)
    print("作業中...")
    refactored_code = run_chatgpt(original_code)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(refactored_code)
    print("完了！")

def run_chatgpt(code: str) -> str:
    openai_api_key = os.environ["OPENAI_API_KEY"]
    refactoring_specifications = load_refactoring_specifications()
    
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": refactoring_specifications},
            {"role": "user", "content": code},
        ],
        "max_tokens": 4000
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        },
        json=data
    )

    choice = response.json()['choices'][0]
    return choice['message']['content'].strip()

if __name__ == "__main__":
    main()