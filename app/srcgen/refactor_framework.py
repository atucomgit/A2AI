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


def refactor_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        original_code = file.read()

    refactored_code = communicate_with_chatgpt(original_code)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(refactored_code)


def communicate_with_chatgpt(code: str) -> str:
    openai_api_key = os.environ["OPENAI_API_KEY"]

    refactoring_specifications = """
Policy:
  Brainstorm to fill all methods as far as you know with mutually exclusive and collectively exhaustive options, step-by-step.
  Only output "ACTION" to preserve the token. Do not output Instruction, Policy, Abstract, Fact, Process.
  Please output an exactly executable code only. Output a perfect code only. Never output any commentary.
  Never output "```python" block.
Abstract:
  This program has the functionality to apply refactoring to all source code within the application.
  It covers all directories under the specified root directory and targets all source code.
Fact:
  Language Script for source code Generation
  4. Refactoring Specifications
    1. Rename class names, variable names, and method names to be more understandable, write in English
    2. For comments
      1. Add missing ones
      2. Correct the wrong ones
      3. Delete unnecessary ones
      4. Please add any missing class comments, method comments, and line comments
      5. For method comments, please include a summary of the parameters and how to use them
      6. Write a brief summary of the class's processing logic in the class comment
      7. Write your source code comments in Japanese. If you have comments in languages other than Japanese, please translate them into Japanese.
    3. Remove unnecessary imports and variables
    4. For line breaks
      1. Add missing ones
      2. Delete unnecessary ones
    5. Simplify complex logic
    6. Implement security measures
    7. Implement performance measures
    8. If the execution log is missing, please insert it
    9. If a compilation error or a runtime error occurs, please make the necessary fixes
    """
    
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