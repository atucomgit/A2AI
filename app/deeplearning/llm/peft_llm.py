import argparse
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
import transformers
import finetune_utils

# 基本パラメータ
BASE_MODEL = "cyberagent/open-calm-medium"  # LoRAのベースとなるモデル
DATA_SET = "kunishou/databricks-dolly-15k-ja"  # LoRAに学習させたいデータセット

# 以下は個別に定義する必要なし（あとでどこかに隠す）
FINETUNED_MODEL = "finetuned/" + BASE_MODEL.split("/")[1]
PEFT_MODEL = "finetuned/lora/" + BASE_MODEL.split("/")[1]
MERGED_MODEL = "finetuned/lora-merged/" + BASE_MODEL.split("/")[1]
tmp_output_dir = "tmp_finetuned"  # ここに出力されるものは、トレーニング完了後に削除してしまってOKなので、自動で消しています

def train(is_lora):
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
    print(data["train"][0])

    # プロンプトテンプレートの確認
    print(generate_prompt(data["train"][0]))

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
        llm_int8_enable_fp32_cpu_offload=True  # Macの場合の8bit量子化はこの引数
        # load_in_8bit=True,  # Macだとこれ無理
        # device_map="auto",  # Macの場合、use_mps_deviceと相反するためauto指定は厳禁！
    )

    print("------------------------------------------")
    print(f"TargetModel: {BASE_MODEL}")
    print(f"DataSet: {DATA_SET}")
    if is_lora:
        print("TrainingMode: LoRA")
        # LoRAのパラメータ
        # target_modules: https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L202
        from peft.utils import other
        if model.config.model_type not in other.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
            print(f"{model.config.model_type}はLoRAで想定されていないモデルタイプです。チューニングを中断します。")
            return

        lora_config = LoraConfig(
            r= 8, 
            lora_alpha=16,
            target_modules=other.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model.config.model_type],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # モデルの前処理
        model = prepare_model_for_kbit_training(model)

        # LoRAモデルの準備
        model = get_peft_model(model, lora_config)

        # 学習可能パラメータの確認
        model.print_trainable_parameters()
    else:
        print("TrainingMode: Full fine-tuning")
    print("------------------------------------------")

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

    epochs = 1
    max_steps = 10
    logging_steps = 1

    # トレーナーの準備
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            num_train_epochs=epochs,
            learning_rate=3e-4,
            logging_steps=logging_steps,
            logging_dir="./log",
            evaluation_strategy="no",
            save_strategy="epoch",
            max_steps=max_steps, 
            optim="adamw_torch",
            output_dir=tmp_output_dir,
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

    # トレーニングしたモデルの保存
    if is_lora:
        trainer.model.save_pretrained(PEFT_MODEL)
        save_model_to_automatic1111 = input("Auomatic1111に生成したLoRAモデルを転送しますか？(y/n): ")
        if save_model_to_automatic1111.upper() == "Y":
            finetune_utils.save_safetensors(BASE_MODEL, PEFT_MODEL)
    else:
        trainer.model.save_pretrained(FINETUNED_MODEL)

    # 不要なディレクトリの削除
    shutil.rmtree(tmp_output_dir)

def run(prompt, is_lora, is_merged):

    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if is_lora:
        print("------------------------------------------")
        print(f"TargetModel: {PEFT_MODEL}")
        # モデルの準備
        # LoRAを8bit量子化で作っているので、実行時はベースモデルも8bit量子化する
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            llm_int8_enable_fp32_cpu_offload=True
            )

        # LoRAモデルの準備
        # 以下の処理では、ベースのモデルとLoRAモデルを両方ともメモリにロードしている
        # ちなみに、以下で取得するmodelをsave_pretrained()すると、LoRAモデルだけ保存される
        model = PeftModel.from_pretrained(model, PEFT_MODEL)
    elif is_merged:
        print("------------------------------------------")
        print(f"TargetModel: {MERGED_MODEL}")

        # モデルの準備
        # 8bit量子化で作っているので、実行時はベースモデルも8bit量子化する
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL,
            llm_int8_enable_fp32_cpu_offload=True
            )
    else:
        print("------------------------------------------")
        print(f"TargetModel: {FINETUNED_MODEL}")

        # モデルの準備
        # 8bit量子化で作っているので、実行時はベースモデルも8bit量子化する
        model = AutoModelForCausalLM.from_pretrained(
            FINETUNED_MODEL,
            llm_int8_enable_fp32_cpu_offload=True
            )
    print("------------------------------------------")

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
        print(f"DEBUG: {tokenizer.decode(outputs)}")

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
    parser.add_argument('-t', '--train', action='store_true', help='fine tune the model.')
    parser.add_argument('-l', '--lora', action='store_true', help='train the model by LoRA.')
    parser.add_argument('-r', '--run', action='store_true', help='Run the model.')
    parser.add_argument('-m', '--merge', action='store_true', help='Automatic1111でマージしたモデルをlora-mergedに保存します.')
    args = parser.parse_args()

    if args.train:
        train(args.lora)
    elif args.run:
        prompt = input("質問：")
        if not prompt:
            prompt = "まどか☆マギカで一番かわいいのは？"
        print(prompt)
        run(prompt, args.lora, args.merge)
    elif args.merge:
        finetune_utils.load_safetensors_and_save_as_model(BASE_MODEL)