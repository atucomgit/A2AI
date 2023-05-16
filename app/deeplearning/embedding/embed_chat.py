# 以下のソースはllama_index 0.6.6を使用
import logging
import sys
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, LLMPredictor, PromptHelper, ServiceContext
from llama_index.data_structs.node import Node
from langchain.chat_models import ChatOpenAI

import argparse
import os

def create_new_index(target_dir, save_dir):
    """
    adaを利用する場合以下のコストがかかる
        $0.0004  /  1,000 トークン
        $0.004   / 10,000
        $0.04    /100,000
        
        ※試したところ、20万トークンなので、8セントほど。
    """
    # ログレベルの設定
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)

    # インデックスの作成
    print("インデックス作成中")
    documents = SimpleDirectoryReader(target_dir).load_data()
    # print(documents)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=create_service_context())

    # インデックスの保存（デフォルトで、ローカルの./storageに入る）
    print("indexを保存")
    index.storage_context.persist(save_dir)

def recursive_create_new_index(directory_path, save_dir):

    # ログレベルの設定
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)

    # ディレクトリ内のプログラム情報を収集
    documents = []
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r") as file:
                    code = file.read()
                node = Node(doc_id=file_path, text=code)
                documents.append(node)
                print(f"Embed対象: {file_path}")

    # ドキュメントをインデックスに追加
    index = GPTVectorStoreIndex.from_documents(documents, service_context=create_service_context())

    # インデックスの保存（デフォルトで、ローカルの./storageに入る）
    print("indexを保存")
    index.storage_context.persist(save_dir)

def create_service_context():
    # max_tokensの設定
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0.7, 
            model_name="ada", 
            max_tokens=4096
        )
    )
    prompt_helper = PromptHelper(
        max_input_size = 4096,
        num_output = 1024,
        max_chunk_overlap = 20,
        chunk_size_limit = 600
    )
    return ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

def chat(save_dir, prompt):

    # ログレベルの設定
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING, force=True)

    # インデックスの読み込み
    storage_context = StorageContext.from_defaults(persist_dir=save_dir)
    index = load_index_from_storage(storage_context)

    # クエリエンジンの作成
    query_engine = index.as_query_engine()

    # 質問応答
    if not prompt:
        prompt = input("質問を入力してください：")
    print(f'回答：{query_engine.query(prompt)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--create_index", action="store_true", help="インデックスの作成")
    parser.add_argument("-rc", "--recursive_create_index", action="store_true", help="インデックスの作成")
    args = parser.parse_args()

    if args.create_index:
        create_new_index("../../utils", "./storage")
        chat("./storage", None)
    elif args.recursive_create_index:
        recursive_create_new_index("../../utils", "./storage")
        chat("./storage", None)
    else:
        chat("./storage", None)
