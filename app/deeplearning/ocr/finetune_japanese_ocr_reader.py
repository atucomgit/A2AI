import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

# Fine-tuning用のデータセットクラスを作成
class OCRDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained("./japanese_ocr_model")
        self.feature_extractor = ViTImageProcessor.from_pretrained("./japanese_ocr_model")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        features = self.feature_extractor(image, return_tensors="pt")
        label = self.labels[index]
        encoded_inputs = self.tokenizer(label, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = encoded_inputs.input_ids.squeeze()
        attention_mask = encoded_inputs.attention_mask.squeeze()
        return {
            "pixel_values": features["pixel_values"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

# Fine-tuningのためのデータセットを準備
image_paths = ["./image/train.png"]  # 画像データのリスト
labels = ["HOGE"]  # 上記画像に対応する正解のテキストデータのリスト
dataset = OCRDataset(image_paths, labels)

# モデルの準備
model = VisionEncoderDecoderModel.from_pretrained("./japanese_ocr_model")

# データローダーの設定
batch_size = 8
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ファインチューニングの設定
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model.to(device)

# ファインチューニングの実行
## 40回くらいepochsを回すと、今回のサンプルだといい感じに読み取ってくれた
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].squeeze(1).to(device)
        outputs = model(pixel_values=pixel_values, decoder_input_ids=input_ids, decoder_attention_mask=attention_mask, labels=input_ids)

        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

# ファインチューニングしたモデルを保存
## preprocessor_config.jsonは自動出力されないため、手動で元々のモデルから作成する
model.save_pretrained("./japanese_ocr_model_finetuned")
dataset.tokenizer.save_pretrained("./japanese_ocr_model_finetuned")
