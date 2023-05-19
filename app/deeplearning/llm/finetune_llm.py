import os
import argparse
from transformers import T5Tokenizer, AutoModelForCausalLM, AutoTokenizer

MODELS = {
    # tramsformersは"4.29.2"が必要。利用する場合はインストールしなおしか、vmを切り替える
    # pip unistall transformers
    # pip install transformers
    "rinna": {
        "framework": "tensorflow",
        "base_model": "rinna/japanese-gpt2-medium",
        "output_dir": "finetuned/rinna-gpt2/"
    },
    # transformersは"4.30.0.dev0"が必要。利用する場合はインストールしなおしか、vmを切り替える
    # pip install git+https://github.com/huggingface/transformers
    "tokodai": {
        "framework": "pytorch",
        "base_model": "okazaki-lab/japanese-gpt2-medium-unidic",
        "output_dir": "finetuned/tokodai-gpt2/"
    }
}

# 学習量の定義
EPOCHS = 1

# 利用するモデルの切り替え
model_type = "rinna"
framework = MODELS[model_type]["framework"]
base_model = MODELS[model_type]["base_model"]
output_dir = MODELS[model_type]["output_dir"]

def finetune_and_save_model(path_to_dataset):
    # TODO 以下、ターミナルで実行しないとうまくいかない
    setenv = "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"
    os.system(setenv)

    print("---- トレーニング開始 ----")
    print(f"データ：{path_to_dataset}")
    print(f"Model：{model_type}")

    command = f'python ../../../../transformers/examples/{framework}/language-modeling/run_clm.py ' \
        f'--model_name_or_path={base_model} ' \
        f'--train_file={path_to_dataset} ' \
        f'--validation_file={path_to_dataset} ' \
        '--do_train ' \
        '--do_eval ' \
        f'--num_train_epochs={EPOCHS} ' \
        '--save_steps=5000 ' \
        '--save_total_limit=3 ' \
        '--per_device_train_batch_size=1 ' \
        '--per_device_eval_batch_size=1 ' \
        f'--output_dir={output_dir}'
    
    # お試し実装。Q&A対応モデルになるようにファインチューニングする場合
    qa_command = "python ../../../../transformers/examples/tensorflow/question-answering/run_qa.py " \
        f"--model_name_or_path={base_model} " \
        f'--train_file={path_to_dataset} ' \
        f'--validation_file={path_to_dataset} ' \
        '--do_train ' \
        '--do_eval ' \
        f'--num_train_epochs={EPOCHS} ' \
        '--save_steps=5000 ' \
        '--save_total_limit=3 ' \
        '--per_device_train_batch_size=1 ' \
        '--per_device_eval_batch_size=1 ' \
        f'--output_dir={output_dir}'

    os.system(command)

def send_prompt_and_run(prompt):
    """
    この関数はプロンプトに対する応答を生成し、表示します。
    prompt: 応答を生成するためのプロンプト
    """

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    if framework == "tensorflow":
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(output_dir, from_tf=True)
    elif framework == "pytorch":
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(output_dir)

    # 推論の実行
    import time
    start_time = time.time()
    print("----推論開始-------------------------------")
    print(f"Model: {model_type}")
    print("-------------------------------------------")

    input = tokenizer.encode(prompt, return_tensors="pt")
    output = fine_tuned_model.generate(input, do_sample=True, max_length=200, num_return_sequences=1)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
    result_text = '\n'.join(decoded_output)
    print(result_text)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("----推論終了-------------------------------")
    print(f"処理時間: {elapsed_time:.3f}秒")

def create_qa_train_data():
    """question-answeri"""
    import json

    """ダミーのトレーニングデータを作成"""
    file_path = "./data_sets/question-answering/qa_data.json"

    data = []
    for i in range(100):
        question = f"質問{i+1}"
        answer = f"質問{i+1}の回答は回答{i+1}です"
        qa_pair = {"question": question, "answer": answer}
        data.append(qa_pair)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)

if __name__ == "__main__":
    # 引数のパーサを作成
    parser = argparse.ArgumentParser(description='Train or run a fine-tuned GPT-2 model.')
    parser.add_argument('-t', '--train', action='store_true', help='Fine-tune language-modeling the model. ')
    parser.add_argument('-qt', '--question_train', action='store_true', help='Fine-tune question-answering the model.')
    parser.add_argument('-r', '--run', action='store_true', help='Generate a response to a prompt.')
    parser.add_argument('-c', '--create_data', action='store_true', help='Generate training data.')
    args = parser.parse_args()

    if args.create_data:
        create_qa_train_data()
    else:
        if args.train:
            finetune_and_save_model('./data_sets/language-modeling/mujunss_mail.txt')
        elif args.question_train:
            finetune_and_save_model('./data_sets/question-answering/qa_data.json')             
        elif args.run:
            prompt = input("文章生成を行う冒頭の文を与えてください：")
            if prompt:
                send_prompt_and_run(prompt)
            else:
                print("冒頭の文が未指定だったので、処理をキャンセルしました。")
        else:
            print("Specify either -t for training or -r PROMPT for running.")
