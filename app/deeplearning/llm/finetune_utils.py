import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from safetensors.torch import save_file
from peft import PeftModel

def save_safetensors(base_model, peft_model):

    base_safetensors_name = "../../../../stable-diffusion-webui/models/Stable-diffusion/" + base_model.split("/")[1] + ".safetensors"
    lora_safetensors_name = "../../../../stable-diffusion-webui/models/Stable-diffusion/" + base_model.split("/")[1] + "-lora.safetensors"

    # Instantiate the model
    model = AutoModelForCausalLM.from_pretrained(base_model)

    # Extract tensors from the model
    tensors = {name: param.clone().detach() for name, param in model.state_dict().items()}

    # Save tensors using safetensors
    save_file(tensors, base_safetensors_name)

    # Instantiate the peft model
    model = PeftModel.from_pretrained(model, peft_model)

    # Extract tensors from the peft model
    tensors = {name: param.clone().detach() for name, param in model.state_dict().items()}

    # Save tensors using safetensors
    save_file(tensors, lora_safetensors_name)

def load_safetensors_and_save_as_model(base_model):
    # Instantiate the model
    model = AutoModelForCausalLM.from_pretrained(base_model)

    # Load tensors from the safetensors file
    merged_safetensors_name = "../../../../stable-diffusion-webui/models/Stable-diffusion/" + base_model.split("/")[1] + "-merged.safetensors"

    with safe_open(merged_safetensors_name, framework="pt", device="mps") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    # Apply tensors to the model
    model.load_state_dict(tensors)

    # 保存先の定義
    save_dir = "./finetuned/lora-merged/" + base_model.split("/")[1]

    # save merged model
    model.save_pretrained(save_dir)

    # トークナイザーの保存
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    # 引数のパーサを作成
    parser = argparse.ArgumentParser(description='Train or run a fine-tuned GPT-2 model.')
    parser.add_argument('-ss', '--save_safetensors', action='store_true', help='save_safetensors')
    parser.add_argument('-ls', '--load_safetensors_and_save_as_model', action='store_true', help='load_safetensors_and_save_as_model')
    args = parser.parse_args()

    # 以下は例です。変更する場合は、base_modelを書き換えてください。
    base_model = "cyberagent/open-calm-medium"
    peft_model = "finetuned/lora/" + base_model.split("/")[1]

    if args.save_safetensors:
        save_safetensors(base_model, peft_model)
    elif args.load_safetensors_and_save_as_model:
        load_safetensors_and_save_as_model(base_model)
