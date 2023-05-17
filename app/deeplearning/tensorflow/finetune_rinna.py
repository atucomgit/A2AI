import os
import argparse
from transformers import T5Tokenizer, AutoModelForCausalLM

EPOCHS = 3

def finetune_and_save_model(path_to_dataset):
    # TODO 以下、ターミナルで実行しないとうまくいかない
    setenv = "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python"
    os.system(setenv)

    print("トレーニング開始")
    print(f"データ：{path_to_dataset}")

    command = 'python ../../../../transformers/examples/tensorflow/language-modeling/run_clm.py ' \
        '--model_name_or_path=rinna/japanese-gpt2-medium ' \
        f'--train_file={path_to_dataset} ' \
        f'--validation_file={path_to_dataset} ' \
        '--do_train ' \
        '--do_eval ' \
        f'--num_train_epochs={EPOCHS} ' \
        '--save_steps=5000 ' \
        '--save_total_limit=3 ' \
        '--per_device_train_batch_size=1 ' \
        '--per_device_eval_batch_size=1 ' \
        '--output_dir=finetuned_model/'

    os.system(command)

def send_prompt_and_run(prompt):
    """
    この関数はプロンプトに対する応答を生成し、表示します。
    prompt: 応答を生成するためのプロンプト
    """

    # トークナイザーとモデルの準備
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
    model = AutoModelForCausalLM.from_pretrained("finetuned_model/", from_tf=True)

    # 推論の実行
    import time
    start_time = time.time()
    print("----推論開始-------------------------------")

    input = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input, do_sample=True, max_length=200, num_return_sequences=3)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
    result_text = '\n'.join(decoded_output)
    print(result_text)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("----推論終了-------------------------------")
    print(f"処理時間: {elapsed_time:.3f}秒")

def create_train_data():
    """ダミーのトレーニングデータを作成"""
    file_path = "./data_sets/qa_data.txt"

    with open(file_path, "w") as file:
        for i in range(100):
            question = f"質問{i+1}"
            answer = f"質問{i+1}の回答は回答{i+1}です"
            line = f"今何時？そうねだいたいね〜\n"
            file.write(line)

if __name__ == "__main__":
    # 引数のパーサを作成
    parser = argparse.ArgumentParser(description='Train or run a fine-tuned GPT-2 model.')
    parser.add_argument('-t', '--train', action='store_true', help='Fine-tune the model.')
    parser.add_argument('-r', '--run', action='store_true', help='Generate a response to a prompt.')
    parser.add_argument('-c', '--create_data', action='store_true', help='Generate training data.')
    args = parser.parse_args()

    if args.create_data:
        create_train_data()
    else:
        if args.train:
            finetune_and_save_model('./data_sets/mujunss_mail.txt')
            # finetune_and_save_model('./data_sets/qa_data.txt')            
        elif args.run:
            prompt = input("文章生成を行う冒頭の文を与えてください：")
            if prompt:
                send_prompt_and_run(prompt)
            else:
                print("冒頭の文が未指定だったので、処理をキャンセルしました。")
        else:
            print("Specify either -t for training or -r PROMPT for running.")
