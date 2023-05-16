import argparse
import requests
import os
import glob
import re

def read_definition_files(file_root):
    """Reads all definition files with a .md extension."""
    definition_files = []
    for file_path in glob.glob(f"{file_root}/**/*md", recursive=True):
        definition_files.append(file_path)
    return definition_files

def read_definition_file(file_path):
    """Reads a specific definition file."""
    with open(file_path, "r") as f:
        definition = f.read()
    return definition

def extract_model_name(definition):
    match = re.search(r'Model\s=\s(.+)', definition)
    if match:
        return match.group(1).strip()
    else:
        return None

def extract_interactive_mode(definition):
    match = re.search(r'Interactive\s=\s(.+)', definition)
    if match:
        return match.group(1).strip().lower() == "true"
    else:
        return False

def chat_gpt_interaction(definition, prompt, openai_api_key, force=False, existing_source=False, model_name="gpt-3.5-turbo"):

    """Interact with GPT via OpenAI API."""
    if existing_source and not force:
        print("SkipGenerate: Because file already exists and not force.")
        return "SkipGenerate: Because file already exists and not force."

    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": definition},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2000,
        "n": 1,
        "stop": None,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}"
            },
            json=data
        )
        response.raise_for_status()
        choice = response.json()['choices'][0]
        generated_code = choice['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        generated_code = ""

    return generated_code

def save_response_to_file(response, output_path):
    """Saves generated code to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as outfile:
        outfile.write(response)

def read_existing_code(file_path):
    """Reads the existing code from a file."""
    with open(file_path, "r", encoding='utf-8') as f:
        existing_code = f.read()
    return existing_code

def main(force, filter_regex, file_root, output_dir_root, model, interactive):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    

    definition_files = read_definition_files(file_root)
    for file_path in definition_files:
        if not re.search(filter_regex, file_path):
            continue

        definition = read_definition_file(file_path)
        model_name = "gpt-4"
        if model:
            model_name = "gpt-3.5-turbo" if model == 3 else "gpt-4"
        else:
            model_in_definition = extract_model_name(definition)
            if model_in_definition:
                model_name = model_in_definition

        interactive_mode = extract_interactive_mode(definition) if not interactive else interactive

        user_input = "この定義に従って、コードを生成してください。"

        output_path = file_path.replace(file_root, output_dir_root)
        output_path = output_path.replace(".md", "")

        print("------------------------------------")
        print(f"コード生成開始: {file_path}")
        print(f"利用Model: {model_name}")
        print("------------------------------------")

        generated_code = ""
        while True:
            generated_code_part = chat_gpt_interaction(definition, user_input, openai_api_key, force, os.path.exists(output_path), model_name=model_name)
            if not generated_code_part:
                print("エラーが発生したため、これ以上コードを生成できません。")
                break
            print("Generated code part:\n", generated_code_part)
            generated_code += generated_code_part
            if generated_code_part != "SkipGenerate: Because file already exists and not force.":
                save_response_to_file(generated_code, output_path)
            if interactive_mode:
                user_input = f"Please generate the continuation of this code and only reply with the continuation code. Only output ACTION to preserve the token. Do not output Instruction, Policy, Abstract, Fact, Process. code={generated_code}"
                continuation = input("継続して追加でコード生成を実行しますか？（y/n）: ")

                if continuation.lower() != 'y':
                    break
            else:
                break

        print("Complete generated code:\n", generated_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate code from definitions with OpenAI GPT')
    parser.add_argument('-f', '--force', action="store_true", help='Force to regenerate code.')
    parser.add_argument('-r', '--filter-regex', type=str, default="", help='Regex to filter definition files.')
    parser.add_argument('-m', type=int, help='Select the GPT model: 3 for GPT-3.5-turbo, 4 for GPT-4')
    parser.add_argument('-i', '--interactive', action="store_true", help='Enter interactive code generation mode.')

    args = parser.parse_args()

    definition_file_root = "../../definition"
    output_dir_root = "../"

    main(args.force, args.filter_regex, definition_file_root, output_dir_root, args.m, args.interactive)