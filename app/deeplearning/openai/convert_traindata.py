# ムジュンスメールをOpenAIのファインチューニングに使えるjson形式に変換するツール
import json

def convert_to_json(input_file, output_file):
    output_data = []

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    prompt = ""
    ignore_line = False

    for line in lines:
        line = line.lstrip()
        if line.startswith("「"):
            line = line.lstrip()
            if prompt != "" and not ignore_line:
                output_data.append({"prompt": prompt, "completion": completion.strip()})
            prompt, completion = line[1:].split("」", 1)
            ignore_line = False
        elif line != "":
            prompt = line[:20].strip()
            completion = line
            output_data.append({"prompt": prompt, "completion": completion.strip()})
            ignore_line = True

    with open(output_file, "w", encoding="utf-8") as f:
        for data in output_data:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

# 実行例
input_file_path = "./data_sets/train.txt"
output_file_path = "./data_sets/train.json"
convert_to_json(input_file_path, output_file_path)
