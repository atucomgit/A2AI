import os
import argparse
import shutil
import torch
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# 以下、頭良い順に並べる
MODELS = {
    # トレーニングはtransformersは"4.30.0.dev0"で大丈夫
    "rinna-instruct": {
        "framework": "pytorch",
        "base_model": "rinna/japanese-gpt-neox-3.6b-instruction-sft",
        "output_dir": "finetuned/rinna-instruct/"
    },
    # トレーニングする場合は、tramsformersは"4.29.2"が必要。
    # バージョンが高いとNo module named 'keras.__internal__'エラーが出る。
    # 利用する場合はインストールしなおしか、vmを切り替える
    # pip uninstall transformers
    # pip install transformers
    # それでも、No module named 'keras.engine'エラーが出る場合は、kerasも再インストール
    # pip uninstall keras
    # pip install keras
    "rinna": {
        "framework": "tensorflow",
        "base_model": "rinna/japanese-gpt2-medium",
        "output_dir": "finetuned/rinna-gpt2/"
    },
    # トレーニングする場合は、transformersは"4.30.0.dev0"が必要。利用する場合はインストールしなおしか、vmを切り替える
    # pip install git+https://github.com/huggingface/transformers
    "tokodai": {
        "framework": "pytorch",
        "base_model": "okazaki-lab/japanese-gpt2-medium-unidic",
        "output_dir": "finetuned/tokodai-gpt2/"
    },
    # トレーニングする場合は、transformersは"4.30.0.dev0"が必要。
    "waseda": {
        "framework": "pytorch",
        "base_model": "nlp-waseda/gpt2-small-japanese",
        "output_dir": "finetuned/waseda-gpt2/"
    }   
}

"""
ファインチューニングでの賢さランキング
チャット
1. GPT3-ADAでのembed（会話は自然）
2. rinna-neox-3.6b-instruction (内容は本物の狙った人格っぽい)　*1位と2位は肉薄
3. （その他は未チューニング）

文書生成
1. rinna-neox-3.6b-instruction（ラジオ番組の回など、特殊なエッセイ形式も生成できた）
2. GPT3-ADAでのembed
3. rinna / tokodai
4. GPT3-Davinciでのファインチューニング（データの作りに問題があったかも）
5. waseda small
"""

# 学習量の定義
EPOCHS = 3

# 利用するモデルの切り替え
model_type = "rinna"
# model_type = "rinna-instruct"
# model_type = "tokodai"
# model_type = "waseda"
framework = MODELS[model_type]["framework"]
base_model = MODELS[model_type]["base_model"]
output_dir = MODELS[model_type]["output_dir"]
tmp_out_dir = "tmp_finetuned" 

def train(path_to_dataset):

    print("---- トレーニング開始 ----")
    print(f"データ：{path_to_dataset}")
    print(f"Model：{model_type}")

    # 
    # 1. トークナイザーの準備
    #
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # トークナイズ関数の定義
    CUTOFF_LEN = 256  # 最大長
    def tokenize(prompt, tokenizer):
        result = tokenizer(
            prompt+"<|endoftext|>",
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=False,
        )
        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
        }

    # トークナイズの動作確認：input_idsの最後にEOS「0」が追加されてることを確認
    print(tokenize("hi there", tokenizer))

    # プロンプトテンプレートの準備
    def generate_prompt(data_point):
        return data_point["text"]

    #
    # 2. データセットの準備
    data = load_dataset(path_to_dataset)

    # データセットの確認
    print(data)
    print(data["train"][6])

    # プロンプトテンプレートの確認
    print(generate_prompt(data["train"][6]))

    # 学習データと検証データの準備
    VAL_SET_SIZE = 100

    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"]
    val_data = train_val["test"]
    train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
    val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))

    #
    # 3. モデルの準備
    #
    model = AutoModelForCausalLM.from_pretrained(base_model)

    #
    # 4. トレーニング
    #

    # ログのクレンジング
    is_delete_log = input("前回処理で残っているログを削除しますか？(y/n): ")
    if is_delete_log.upper() == "Y":
        shutil.rmtree("./log", ignore_errors=True)
        print("ログを削除しました")
    else:
        print("ログ削除はキャンセルされました")

    # トレーナーの準備
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            num_train_epochs=EPOCHS,
            learning_rate=3e-4,
            logging_steps=1,
            logging_dir="./log",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            # max_steps=200, 
            optim="adamw_torch",
            output_dir=tmp_out_dir,
            report_to="tensorboard",
            save_total_limit=3,
            push_to_hub=False,
            auto_find_batch_size=True,
            use_mps_device=True  # Macの場合、この引数を足すとGPUを利用してくれる
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 学習の実行
    model.config.use_cache = False  # 警告を黙らせる
    trainer.train() 
    model.config.use_cache = True

    # LoRAモデルの保存
    trainer.model.save_pretrained(output_dir)

    # 不要なディレクトリの削除
    shutil.rmtree(tmp_out_dir)

def question_train(path_to_dataset):
    # 以下、環境変数が必要なので設定。他に影響が出ないように現在のコンソールのみに適用
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    print("---- トレーニング開始 ----")
    print(f"データ：{path_to_dataset}")
    print(f"Model：{model_type}")

    # お試し実装。Q&A対応モデルになるようにファインチューニングする場合
    # 参考）
    # https://note.com/npaka/n/na8721fdc3e24
    # 残念ながら、以下はM2MaxのGPUは利用されない模様。ものすごく時間がかかる
    # （残念ながら、legacyではuse_mps_device引数が使えない）
    # 解決！！！ run_squad.pyの708行目がcudaしか対応していないため、mpsに修正すると、M2MAXのGPUを使ってくれる
    # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # 　↓
    # device = torch.device("mps")
    # （ただし！ファインチューニング後のモデルが350GBになってしまうので、何かがおかしい気がする・・・）
    command = "python ../../../../transformers/examples/legacy/question-answering/run_squad.py " \
        "--model_type=bert " \
        f"--model_name_or_path=colorfulscoop/bert-base-ja " \
        f'--output_dir=./finetuned/qa-test' \
        "--train_file=./data_sets/question-answering/DDQA-1.0_RC-QA_train.json " \
        "--predict_file=./data_sets/question-answering/DDQA-1.0_RC-QA_dev.json " \
        "--per_gpu_train_batch_size 1 " \
        "--learning_rate 3e-5 " \
        "--num_train_epochs 1 " \
        "--max_seq_length 384 " \
        "--doc_stride 128 " \
        "--do_train " \
        "--do_eval " \
        "--overwrite_output_dir "

    # お試し実装。tensorflowでQ&A対応モデルになるようにファインチューニングする場合
    # 悩み1) QAデータをどういうふうに食わせれば良いか不明。train_fileを指定すると落ちる。
    # 悩み2) 最新のkerasだと遅くなる警告が出る・・・（macでやるにはしんどいか・・・）→(解決！)use_mps_device引数でGPU使ってくれる。でも元データのsquadが大きいので、超遅い。
    # 悩み3）trainデータをsquad形式のjsonを用意してdataset_nameに配置しても、なぜか落ちる。データ形式が違うと言われてしまう。
    # command = "python ../../../../transformers/examples/tensorflow/question-answering/run_qa.py " \
    #     f"--model_name_or_path=colorfulscoop/bert-base-ja " \
    #     f'--output_dir=./finetuned/qa ' \
    #     "--dataset_name=squad " \
    #     "--num_train_epochs 1 " \
    #     "--per_gpu_train_batch_size 1 " \
    #     "--do_train " \
    #     "--do_eval " \
    #     "--use_mps_device=True "

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
    output = fine_tuned_model.generate(input, do_sample=True, max_length=300, num_return_sequences=1)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
    result_text = '\n'.join(decoded_output)
    result_text = result_text.replace(" ", "")
    print(result_text)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("----推論終了-------------------------------")
    print(f"処理時間: {elapsed_time:.3f}秒")

def chat_with_rinna_instruct():
    """
    rinna_instructとチャットするメソッドです
    後ほど↑のメソッドと合流して一つに整理します。
    """
    print(f"TORCHの状況：{torch.backends.mps.is_available()}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    if framework == "tensorflow":
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(output_dir, from_tf=True)
    elif framework == "pytorch":
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(output_dir)

    while True:
        user_input = input("りんなとチャットしましょう：")
        prompt = ""
        if user_input:
            prompt = f"ユーザー: {user_input}<NL>システム:"
        else:
            print("冒頭の文が未指定だったので、処理をキャンセルしました。")
            continue

        # 推論の実行
        import time
        start_time = time.time()
        print("----推論開始-------------------------------")
        print(f"Model: {model_type}")
        print("-------------------------------------------")

        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
        output = fine_tuned_model.generate(tokenized_prompt, do_sample=True, max_length=1000, num_return_sequences=2)
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
        result_text = '\n'.join(decoded_output)
        result_text = result_text.replace(" ", "")
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

def save_tokenizer():
    # トークナイザーのインスタンス化
    base_model = "cyberagent/open-calm-small"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # トークナイザーの保存
    tokenizer.save_pretrained("./tokenizer")

def convert_to_ggml():
    from llm_rs.convert import AutoConverter

    # Specify the model to be converted and an output directory
    export_folder = "./ggml" 
    base_model = "./finetuned/open-calm/"

    # Perform the model conversion
    converted_model = AutoConverter.convert(base_model, export_folder)
    print(f"ggml形式に変換完了：{converted_model}")

    from llm_rs import Mpt, QuantizationType, ContainerType

    # Mpt.quantize(converted_model,
    #     "./ggml/-f16_4_0.bin",
    #     quantization=QuantizationType.Q4_0,
    #     container=ContainerType.GGJT
    # )

    print("Modelをロード")
    model = Mpt(converted_model)

    print("トークを開始")
    # Initiate a text generation
    result = model.generate("The meaning of life is")

    # Display the generated text
    print(result.text)


if __name__ == "__main__":
    # 引数のパーサを作成
    parser = argparse.ArgumentParser(description='Train or run a fine-tuned GPT-2 model.')
    parser.add_argument('-t', '--train', action='store_true', help='Fine-tune language-modeling the model. ')
    parser.add_argument('-qt', '--question_train', action='store_true', help='Fine-tune question-answering the model.')
    parser.add_argument('-r', '--run', action='store_true', help='Generate a response to a prompt.')
    parser.add_argument('-c', '--create_data', action='store_true', help='Generate training data.')
    parser.add_argument('-ri', '--rinna_instruct', action='store_true', help='rinna-instructでチャットする場合.')
    parser.add_argument('-st', '--save_tokenizer', action='store_true', help='そのモデルが利用するトークナイザーを保存します.')
    parser.add_argument('-cg', '--convert_to_ggml', action='store_true', help='ggml形式にコンバートします.')
    args = parser.parse_args()

    if args.create_data:
        create_qa_train_data()
    else:
        if args.train:
            if model_type == "rinna-instruct":
                print("すごい時間がかかりますが本当に学習しますか？学習する場合は、この部分の分岐をソース修正してください")
            else:
                train('./data_sets/language-modeling/')
        elif args.question_train:
            question_train('./data_sets/question-answering/qa_data.json')             
        elif args.run:
            prompt = input("文章生成を行う冒頭の文を与えてください：")
            if prompt:
                send_prompt_and_run(prompt)
            else:
                print("冒頭の文が未指定だったので、処理をキャンセルしました。")
        elif args.rinna_instruct:
            print("**** チャットモードに入ります。終了する場合は、ctrl＋cを押してください。 ****")
            chat_with_rinna_instruct()
        elif args.save_tokenizer:
            save_tokenizer()
        elif args.convert_to_ggml:
            convert_to_ggml()
        else:
            print("Specify either -t for training or -r PROMPT for running.")
