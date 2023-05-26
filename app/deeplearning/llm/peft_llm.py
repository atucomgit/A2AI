# 基本パラメータ
model_name = "cyberagent/open-calm-medium"
dataset = "kunishou/databricks-dolly-15k-ja"
peft_name = "lora-calm-medium"
output_dir = "finetuned-LoRA"

from transformers import AutoTokenizer

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_name)

CUTOFF_LEN = 256  # 最大長

# トークナイズ関数の定義
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

from datasets import load_dataset

# データセットの準備
data = load_dataset(dataset)

VAL_SET_SIZE = 2000

# 学習データと検証データの準備
train_val = data["train"].train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]
train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))


from transformers import AutoModelForCausalLM

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # load_in_8bit=True,  # Macだとこれ無理
    device_map="auto",
    llm_int8_enable_fp32_cpu_offload=True,  # Macだとこれ必要
    offload_folder="offload"  # Macだとこれ必要
)

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# LoRAのパラメータ
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

import transformers
max_steps = 200    # GPUで時間がある場合はこれを除外するとたくさん訓練する、
eval_steps = 200
save_steps = 200
logging_steps = 20

# トレーナーの準備
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        num_train_epochs=3,
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
model.config.use_cache = False
trainer.train() 
model.config.use_cache = True

# LoRAモデルの保存
trainer.model.save_pretrained(peft_name)

# トークナイズの動作確認
# input_idsの最後にEOS「0」が追加されてることを確認
print(tokenize("hi there", tokenizer))

# データセットの確認
print(data["train"][5])

# プロンプトテンプレートの確認
print(generate_prompt(data["train"][5]))

def run(prompt):
    import torch
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # モデルの準備
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
    )

    # トークンナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # LoRAモデルの準備
    model = PeftModel.from_pretrained(
        model, 
        peft_name, 
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
    def generate(instruction,input=None,maxTokens=256):
        # 推論
        prompt = generate_prompt({'instruction':instruction,'input':input})
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
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
                print(decoded[sentinelLoc+len(sentinel):])
            else:
                print('Warning: Expected prompt template to be emitted.  Ignoring output.')
        else:
            print('Warning: no <eos> detected ignoring output')

    generate(prompt)

if __name__ == "__main__":
    # 引数のパーサを作成
    import argparse
    parser = argparse.ArgumentParser(description='Train or run a fine-tuned GPT-2 model.')
    parser.add_argument('-t', '--train', action='store_true', help='train the model by LoRA.')
    parser.add_argument('-r', '--run', action='store_true', help='Run the model.')
    args = parser.parse_args()

    if args.run:
        prompt = input("プロンプト：")
        if not prompt:
            prompt = "まどか☆マギカで一番かわいいのは？"
        run(prompt)
