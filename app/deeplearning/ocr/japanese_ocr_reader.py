# 参考
# https://huggingface.co/spaces/Detomo/Japanese_OCR

import numpy as np
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor
import re
import jaconv
from PIL import Image

# load model
model_path = "./japanese_ocr_model/"
processor = ViTImageProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path)

def infer(image):
    """画像から文字を抽出"""
    image = image.convert('L').convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values
    output_ids = model.generate(pixel_values, max_new_tokens=300)[0]
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    text = post_process(text)
    return text

def post_process(text):
    """文字を成型"""
    text = ''.join(text.split())
    text = text.replace('…', '...')
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
    text = jaconv.h2z(text, ascii=True, digit=True)
    return text

if __name__ == "__main__":
    # 画像ファイルのパス
    image_path = "./image/screen_shot_jp.png"

    # 画像を開く
    image = Image.open(image_path)

    # 画像読み取り
    text = infer(image)

    # 読み取った文字列を成型
    text = post_process(text)
    print(f"---- 読み取り結果 ----\n{text}")
