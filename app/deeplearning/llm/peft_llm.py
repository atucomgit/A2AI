from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
import transformers

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# 基本パラメータ
BASE_MODEL = "cyberagent/open-calm-medium"  # LoRAのベースとなるモデル
DATA_SET = "kunishou/databricks-dolly-15k-ja"  # LoRAに学習させたいデータセット
PEFT_MODEL = "finetuned/lora-calm-medium"  # LoRAしたモデルの出力先
output_dir = "tmp_finetuned-LoRA"  # ここに出力されるものは、トレーニング完了後に削除してしまってOK

def train():
    # 
    # 1. トークナイザーの準備
    #
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

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
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            {data_point["instruction"]}

            ### Input:
            {data_point["input"]}

            ### Response:
            {data_point["output"]}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

            ### Instruction:
            {data_point["instruction"]}

            ### Response:
            {data_point["output"]}"""

    #
    # 2. データセットの準備
    #
    # (参考) "kunishou/databricks-dolly-15k-ja"の中身
    # https://raw.githubusercontent.com/kunishou/databricks-dolly-15k-ja/main/databricks-dolly-15k-ja.json
    data = load_dataset(DATA_SET)

    # データセットの確認
    print(data["train"][5])

    # プロンプトテンプレートの確認
    print(generate_prompt(data["train"][5]))

    # 学習データと検証データの準備
    VAL_SET_SIZE = 2000

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
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        # load_in_8bit=True,  # Macだとこれ無理
        device_map="auto",
        llm_int8_enable_fp32_cpu_offload=True,  # Macだとこれ必要
        offload_folder="offload"  # Macだとこれ必要
    )

    # LoRAのパラメータ
    # target_modules: https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L202
    lora_config = LoraConfig(
        r= 8, 
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # モデルの前処理
    model = prepare_model_for_int8_training(model)

    # LoRAモデルの準備
    model = get_peft_model(model, lora_config)

    # 学習可能パラメータの確認
    model.print_trainable_parameters()

    #
    # 4. トレーニング
    #
    epochs = 3
    max_steps = 200    # GPUを使えて、時間がある場合はこれを除外するとたくさん訓練するが、Macでは厳しい。
    eval_steps = 200
    save_steps = 200
    logging_steps = 20

    # トレーナーの準備
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            num_train_epochs=epochs,
            learning_rate=3e-4,
            logging_steps=logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            max_steps=max_steps, 
            optim="adamw_torch",
            output_dir=output_dir,
            report_to="none",
            save_total_limit=3,
            push_to_hub=False,
            auto_find_batch_size=True
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 学習の実行
    model.config.use_cache = False  # 警告を黙らせる
    trainer.train() 
    model.config.use_cache = True

    # LoRAモデルの保存
    trainer.model.save_pretrained(PEFT_MODEL)

def run(prompt):

    # モデルの準備
    # LoRAを8bit量子化で作っているので、実行時はベースモデルも8bit量子化する
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        # load_in_8bit=True,  # MacだとこれはNG
        device_map="auto",
        llm_int8_enable_fp32_cpu_offload=True,  # Macだとこれ必要
        # offload_folder="offload"  # Macだとこれ必要（要らんかも？）
    )

    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # LoRAモデルの準備
    # ベースのモデルとLoRAモデルを結合させて取得しているのかと思ったら、LoRAモデルだけを取得していた
    model = PeftModel.from_pretrained(
        model, 
        PEFT_MODEL, 
        device_map="auto"
    )

    # 評価モード
    model.eval()

    # プロンプトテンプレートの準備
    def generate_prompt(data_point):
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            {data_point["instruction"]}

            ### Input:
            {data_point["input"]}

            ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

            ### Instruction:
            {data_point["instruction"]}

            ### Response:"""
    
    # テキスト生成関数の定義
    def generate(instruction, input=None, maxTokens=256):
        # 推論
        prompt = generate_prompt({'instruction':instruction,'input':input})
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids  # Macでは、ラストの.cuda()は削除

        outputs = model.generate(
            input_ids=input_ids, 
            max_new_tokens=maxTokens, 
            do_sample=True,
            temperature=0.7, 
            top_p=0.75, 
            top_k=40,         
            no_repeat_ngram_size=2,
        )
        outputs = outputs[0].tolist()

        # EOSトークンにヒットしたらデコード完了
        if tokenizer.eos_token_id in outputs:
            eos_index = outputs.index(tokenizer.eos_token_id)
            decoded = tokenizer.decode(outputs[:eos_index])

            # レスポンス内容のみ抽出
            sentinel = "### Response:"
            sentinelLoc = decoded.find(sentinel)
            if sentinelLoc >= 0:
                print("AIの回答：")
                print(decoded[sentinelLoc+len(sentinel):])
            else:
                print('Warning: Expected prompt template to be emitted.  Ignoring output.')
        else:
            print('Warning: no <eos> detected ignoring output')

    generate(prompt)

if __name__ == "__main__":
    # 引数のパーサを作成
    parser = argparse.ArgumentParser(description='Train or run a fine-tuned GPT-2 model.')
    parser.add_argument('-t', '--train', action='store_true', help='train the model by LoRA.')
    parser.add_argument('-r', '--run', action='store_true', help='Run the model.')
    args = parser.parse_args()

    if args.train:
        train()
    elif args.run:
        prompt = input("質問：")
        if not prompt:
            prompt = "まどか☆マギカで一番かわいいのは？"
        print(prompt)
        run(prompt)
