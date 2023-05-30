import argparse
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
import transformers
import utils_for_tuning
from training_prompt_generator import TrainingPromptGenerator

"""---- ユーザーによるデータカスタマイズブロック：以下、自由に変更可能です。 ----"""

# トレーニングする対象のモデル
BASE_MODEL = "rinna/japanese-gpt2-medium"      # huggingFaceで検索できるモデル名（ローカルのファイルを指定してもOK）
DATA_SET = "kunishou/databricks-dolly-15k-ja"  # 学習させたいデータセット（ローカルのファイルを指定してもOK）
DATA_SET_TYPE = TrainingPromptGenerator.DATA_TYPE_INSTRUCT_CHAT  # DataSetの形式に合致するデータタイプ。詳しくはTrainingPromptGenerator参照。

# トレーニング用のハイパーパラメータ
EPOCHS = 1            # トレーニング反復回数
LEARNING_RATE=3e-4    # 学習レート
MAX_STEPS = 200       # 学習最大ステップ数（上限を設定したい場合に設定する）
LOGGING_STEPS = 1     # ログを保存する単位。小さい数字にすると、TensorBoardから細かくデータが見える

"""---- ユーザーによるデータカスタマイズブロック：ここまで ----"""


# 以下は個別に定義する必要なし（あとでどこかに隠す）
FINETUNED_MODEL = "finetuned/" + "-".join(BASE_MODEL.split("/"))
PEFT_MODEL = "finetuned-lora/" + "-".join(BASE_MODEL.split("/"))
MERGED_MODEL = "finetuned-lora-merged/" + "-".join(BASE_MODEL.split("/"))
tmp_output_dir = "tmp_finetuned"  # ここに出力されるものは、トレーニング完了後に削除してしまってOKなので、自動で消しています

def train(is_lora):
    # 
    # 1. トークナイザーの準備
    #
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # トークナイズ関数の定義
    CUTOFF_LEN = 256  # 最大長
    def tokenize(prompt, tokenizer):
        if prompt is None:
            prompt = ""

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
        return TrainingPromptGenerator.get_training_data(data_point, DATA_SET_TYPE)

    #
    # 2. データセットの準備
    #
    # (参考) "kunishou/databricks-dolly-15k-ja"の中身
    # https://raw.githubusercontent.com/kunishou/databricks-dolly-15k-ja/main/databricks-dolly-15k-ja.json
    data = load_dataset(DATA_SET)

    # データセットの確認
    print("データセットの確認")
    print(data["train"][0])

    # プロンプトテンプレートの確認
    print("プロンプトテンプレートの確認")
    print(generate_prompt(data["train"][0]))

    # 学習データと検証データの準備
    data_size = len(data["train"])
    VAL_SET_RATIO = 0.3  # 検証データの割合
    VAL_SET_SIZE = int(data_size * VAL_SET_RATIO)

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

    # トレーナーの準備
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=LOGGING_STEPS,
            logging_dir="./log",
            evaluation_strategy="no",
            save_strategy="epoch",
            max_steps=MAX_STEPS, 
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
            utils_for_tuning.save_safetensors(BASE_MODEL, PEFT_MODEL)
    else:
        trainer.model.save_pretrained(FINETUNED_MODEL)

    # 不要なディレクトリの削除
    shutil.rmtree(tmp_output_dir)

def run(prompt, is_fine_tuned, is_lora, is_merged):

    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 引数によってロードするモデルを切り替える
    if is_fine_tuned:
        print("------------------------------------------")
        print(f"TargetModel: {FINETUNED_MODEL}")

        # モデルの準備
        # 8bit量子化で作っているので、実行時はベースモデルも8bit量子化する
        model = AutoModelForCausalLM.from_pretrained(
            FINETUNED_MODEL,
            llm_int8_enable_fp32_cpu_offload=True
            )
    elif is_lora:
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
        print(f"TargetModel: {BASE_MODEL}")

        # モデルの準備
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            llm_int8_enable_fp32_cpu_offload=True
            )
    print("------------------------------------------")

    # 評価モード
    model.eval()

    # プロンプトテンプレートの準備
    def generate_prompt(prompt):
        return TrainingPromptGenerator.get_prompt(prompt, DATA_SET_TYPE)
    
    # テキスト生成関数の定義
    def generate(prompt, sub_prompt=None, maxTokens=256):
        # 推論
        prompt = generate_prompt({'prompt':prompt,'sub_prompt':sub_prompt})
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
                print('Warning: Expected prompt template to be emitted.')
                print("AIの回答：")
                print(decoded)
        else:
            print('Warning: no <eos> detected ignoring output')
            print("AIの回答：")
            print({tokenizer.decode(outputs)})

    generate(prompt)

if __name__ == "__main__":
    # 引数のパーサを作成
    parser = argparse.ArgumentParser(description='Train or run a fine-tuned GPT-2 model.')
    parser.add_argument('-t', '--train', action='store_true', help='fine tune the model.')
    parser.add_argument('-r', '--run', action='store_true', help='Run the model.')
    parser.add_argument('-f', '--finetuned', action='store_true', help='Finetuned option.')
    parser.add_argument('-l', '--lora', action='store_true', help='LoRA option.')
    parser.add_argument('-m', '--merge', action='store_true', help='merge option.')
    args = parser.parse_args()

    if args.train:
        train(args.lora)
    elif args.run:
        prompt = input("質問：")
        if not prompt:
            prompt = "まどか☆マギカで一番かわいいのは？"
        print(prompt)
        run(prompt, args.finetuned, args.lora, args.merge)
    elif args.merge:
        utils_for_tuning.load_safetensors_and_save_as_model(BASE_MODEL, MERGED_MODEL)