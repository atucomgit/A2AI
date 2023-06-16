#
# 生のOpenAI提供のライブラリを使ったembeddingのサンプルコード
#
import os
import openai
import argparse
import tiktoken
from openai.embeddings_utils import cosine_similarity
import pandas as pd
import numpy as np

### Fist-generation models
# MODEL_NAME = "ada"
# MODEL_NAME = "babbage"
# MODEL_NAME = "curie"
# MODEL_NAME = "davinci"

### Second-generation models
MODEL_NAME = "text-embedding-ada-002"

# 1000トークンあたりのエンべディングコスト（ドル）
COST = {
    "text-embedding-ada-002": 0.0001,
    "ada": 0.0040,
    "babbage": 0.0050,
    "curie": 0.0200,
    "davinci": 0.2000
}

# APIキーを設定
openai.api_key = os.environ["OPENAI_API_KEY"]

# データセットのディレクトリ
data_dir = "./data_sets/"

# ストレージディレクトリ
storage_dir = f"./storage/{MODEL_NAME}/"

# ディレクトリが存在しない場合は作成
os.makedirs(storage_dir, exist_ok=True)

from decimal import Decimal

def num_tokens_from_string(string: str, encoding_name: str):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))

    price = COST[MODEL_NAME] * num_tokens / 1000
    print(f"エンべディングするトークン数:{num_tokens} 所要金額：${'{:.7f}'.format(Decimal(price))}")

def get_embedding(text, model):
   """OpenAIの指定モデルにてエンベッドしたいデータをベクター化する処理"""
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def create_embedding_data():
    """エンベッドするデータを生成しcsvファイルに保存"""
    # データセットのディレクトリにあるすべての.txtファイルを処理
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            with open(os.path.join(data_dir, filename), "r") as file:
                text = file.read()

            # トークン数をデバッグ出力
            num_tokens_from_string(text, "cl100k_base")

            df = pd.read_csv(os.path.join(data_dir, filename))
            df = df[["Essay"]]
            df = df.dropna()
            df["combined"] = (
                "Title: " + df.Essay.str.strip()
            )
            df.head(2)

            df['embedding'] = df.combined.apply(lambda x: get_embedding(x, model=MODEL_NAME))
            df.to_csv(os.path.join(storage_dir, filename), index=False)

    print("Embedding process is complete!")

def search_similarity(prompt, n=3, pprint=True):
    """エンべディングしたベクターデータから、promptで指定されたデータに最も近い文字列を探して取得する処理"""

    for filename in os.listdir(storage_dir):
        df = pd.read_csv(os.path.join(storage_dir, filename))
        df['embedding'] = df.embedding.apply(eval).apply(np.array)

        query = get_embedding(prompt, model=MODEL_NAME)

        df['similarities'] = df.embedding.apply(lambda x: cosine_similarity(x, query))
        res = df.sort_values('similarities', ascending=False).head(n)
        print(res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--create_index", action="store_true", help="インデックスの作成")
    args = parser.parse_args()

    if args.create_index:
        create_embedding_data()
    else:
        input = input("サーチ文字列を入力してください：")
        search_similarity(input)