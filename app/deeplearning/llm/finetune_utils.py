import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from safetensors.torch import save_file
from peft import PeftModel


BASE_MODEL = "cyberagent/open-calm-medium"
SAFETENSORS_NAME = "safetensors/open-calm-medium.safetensors"

PEFT_MODEL = "finetuned/lora-calm-medium"
PEFT_SAFETENSORS_NAME = "safetensors/open-calm-medium-lora.safetensors"

LOAD_SAFETENSORS_NAME = "safetensors/open-calm-medium-merged.safetensors"
SAVE_MERGED_MODEL = "./finetuned/base_and_lora_merged/open-calm-medium/"

def save_safetensors():

    # Instantiate the model
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # Extract tensors from the model
    tensors = {name: param.clone().detach() for name, param in model.state_dict().items()}

    # Save tensors using safetensors
    save_file(tensors, SAFETENSORS_NAME)

    if PEFT_MODEL:
        # Instantiate the peft model
        model = PeftModel.from_pretrained(model, PEFT_MODEL)

        # Extract tensors from the model
        tensors = {name: param.clone().detach() for name, param in model.state_dict().items()}

        # Save tensors using safetensors
        save_file(tensors, PEFT_SAFETENSORS_NAME)

def load_safetensors_and_save_as_model():
    # Instantiate the model
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # Load tensors from the safetensors file
    with safe_open(LOAD_SAFETENSORS_NAME, framework="pt", device="mps") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    # Apply tensors to the model
    model.load_state_dict(tensors)

    # save merged model
    model.save_pretrained(SAVE_MERGED_MODEL)

    # トークナイザーの保存
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(SAVE_MERGED_MODEL)

if __name__ == "__main__":
    # 引数のパーサを作成
    parser = argparse.ArgumentParser(description='Train or run a fine-tuned GPT-2 model.')
    parser.add_argument('-ss', '--save_safetensors', action='store_true', help='save_safetensors')
    parser.add_argument('-ls', '--load_safetensors_and_save_as_model', action='store_true', help='load_safetensors_and_save_as_model')
    args = parser.parse_args()

    if args.save_safetensors:
        save_safetensors()
    elif args.load_safetensors_and_save_as_model:
        load_safetensors_and_save_as_model()
