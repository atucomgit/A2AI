from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# 画像ファイルのパス
image_path = "./image/screen_shot.png"

# 画像を開く
image = Image.open(image_path).convert("RGB")

# OCRプロセッサとモデルを読み込む
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')

# 画像をテンソルに変換
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# テキストを生成
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)