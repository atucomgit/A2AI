# 参考：https://note.com/npaka/n/n932b4c0a2230
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model 
import transformers
from datasets import load_dataset
from accelerate import Accelerator
accelerator = Accelerator()

BASE_MODEL = "cyberagent/open-calm-7b"
LoRATUNED_MODEL_PATH = "./finetuned-LoRA/practice"
# DATA_SETS = "Abirate/english_quotes"
DATA_SETS = "kunishou/databricks-dolly-15k-ja"

def download_model():
    """モデルの読み込み"""
    print(f"**** Modelをダウンロード: {BASE_MODEL}")

    # 8bitモデルとしてダウンロード
    # Macだとllm_int8_enable_fp32_cpu_offloadの設定が必要で、その際は、offload_folderの設定（出力先なので任意のdirでOK）も必要
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        device_map="auto",
        # llm_int8_enable_fp32_cpu_offload=True,
        offload_folder="offload"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # #
    # # 8bitモデルに後処理を適用
    # # 
    # print("**** 8bitモデルに後処理を適用.")
    # for param in model.parameters():
    #     param.requires_grad = False  # モデルをフリーズ
    #     if param.ndim == 1:
    #         # 安定のためにレイヤーノルムをfp32にキャスト
    #         param.data = param.data.to(torch.float32)

    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()

    # class CastOutputToFloat(nn.Sequential):
    #     def forward(self, x): return super().forward(x).to(torch.float32)
    # model.lm_head = CastOutputToFloat(model.lm_head)

    return model, tokenizer

#
# PeftModelを読み込む
#
def train_by_rola(model, tokenizer):
    """LoRAでトレーニング"""
    print("**** LoRAでトレーニング開始.")
    print(model.config.vocab_size)
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    #
    # 学習の実行
    # 以下は、偉人の名言集のサンプル。
    # その他、データセットは以下のサイトでよくまとまっている。
    # https://note.com/npaka/n/n686d987adfb1
    # ・databricks-dolly-15k-jaなんかが日本語で使えるかも
    # data = load_dataset(DATA_SETS)
    data = load_dataset("./data_sets/language-modeling/")
    print("=================")
    print(data)
    print("=================")
    data = data.filter(lambda example: len(example['text']) > 0)  # 空白だと落ちる
    data = data.filter(lambda example: len(example['text']) < 256)  # 大きい入力だとembedで落ちる
    data = data.map(lambda samples: tokenizer(samples['text']), batched=True)

    trainer = transformers.Trainer(
        model=model, 
        train_dataset=data['train'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4, 
            gradient_accumulation_steps=4,
            warmup_steps=100, 
            max_steps=200, 
            optim="adamw_torch",
            learning_rate=2e-4, 
            # fp16=True,
            logging_steps=1, 
            output_dir=LoRATUNED_MODEL_PATH,
            # use_mps_device=True
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    model.config.use_cache = False  # 警告を黙らせます。 推論のために再度有効にしてください。
    trainer.train()
    model.config.use_cache = True

    trainer.model.save_pretrained(LoRATUNED_MODEL_PATH)
    print("**** トレーニング完了.")

def run(prompt):
    """推論の実行"""
    model_path = LoRATUNED_MODEL_PATH
    # model = AutoModelForCausalLM.from_pretrained(model_path, from_tf=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)

    # batch = tokenizer(prompt, return_tensors='pt')

    # with torch.cuda.amp.autocast():
    #     output_tokens = model.generate(**batch, max_new_tokens=50)

    # print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # fine_tuned_model = AutoModelForCausalLM.from_pretrained(LoRATUNED_MODEL_PATH, from_tf=True)
    fine_tuned_model = AutoModelForCausalLM.from_pretrained(LoRATUNED_MODEL_PATH)

    input = tokenizer.encode(prompt, return_tensors="pt")
    output = fine_tuned_model.generate(input, do_sample=True, max_length=300, num_return_sequences=1)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
    result_text = '\n'.join(decoded_output)
    result_text = result_text.replace(" ", "")
    print(result_text)

if __name__ == "__main__":
    # 引数のパーサを作成
    parser = argparse.ArgumentParser(description='Train or run a fine-tuned GPT-2 model.')
    parser.add_argument('-d', '--download_model', action='store_true', help='download the model. ')
    parser.add_argument('-t', '--train', action='store_true', help='train the model by LoRA.')
    parser.add_argument('-r', '--run', action='store_true', help='Run the model.')
    args = parser.parse_args()

    if args.train:
        model, tokenizer = download_model()
        train_by_rola(model, tokenizer)
    elif args.run:
        prompt = input("プロンプト：")
        if not prompt:
            prompt = "Two things are infinite: "
        run(prompt)
        