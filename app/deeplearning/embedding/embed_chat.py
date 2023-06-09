# 以下のソースはllama_index 0.6.6を使用
import logging
import sys
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, LLMPredictor, PromptHelper, ServiceContext
from llama_index.data_structs.node import Node
# from langchain.chat_models import ChatOpenAI <-うまくllama_indexで判定が効かないので、一旦コメントアウト
from langchain import OpenAI

import argparse
import os

sys.path.append("../../utils")
import speech_input


def create_new_index(target_dir, model):
    """
    adaを利用する場合以下のコストがかかる
        $0.0004  /  1,000 トークン
        $0.004   / 10,000
        $0.04    /100,000
        
        ※試したところ、20万トークンなので、8セントほど。
    """
    # ログレベルの設定
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

    # インデックスの作成
    print("インデックス作成中")
    documents = SimpleDirectoryReader(target_dir).load_data()
    # print(documents)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=create_service_context(model))

    # インデックスの保存（デフォルトで、ローカルの./storageに入る）
    print("indexを保存")
    index.storage_context.persist(f"./storage/{model}")

def recursive_create_new_index(directory_path, save_dir):
    """
    任意のディレクトリ配下のディレクトリを再起的に全て確認す、全てのdatasetをembedするメソッド
    
    ソースコードを全てエンベディングするために利用することを想定
    """

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
    index = GPTVectorStoreIndex.from_documents(documents, service_context=create_service_context(model))

    # インデックスの保存（デフォルトで、ローカルの./storageに入る）
    print("indexを保存")
    index.storage_context.persist(f"./storage/{model}")

def create_service_context(model):
    # max_tokensの設定の苦労日記
    ## 下記、max_tokensは効かない。無理やり効かせるためには、下記OpenAIクラスのmax_tokensを直接修正する必要あり。
    ## なお、以下の引数がどのような分岐で扱われているかは、LLMPredictor._get_llm_metadata()のソースを確認するとわかりやすい。
    chat_open_ai = OpenAI(
            temperature=0.7,
            model_name=model,
            max_tokens=3000
        )

    llm_predictor = LLMPredictor(
        llm=chat_open_ai
    )
    prompt_helper = PromptHelper(
        max_input_size=4096,
        num_output=1024,
        max_chunk_overlap=20,
        chunk_size_limit=600
    )
    return ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper
    )

def chat(model, engine):

    # ログレベルの設定
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING, force=True)

    save_dir = "./storage/" + model

    # インデックスの読み込み
    storage_context = StorageContext.from_defaults(persist_dir=save_dir)
    index = load_index_from_storage(storage_context)

    # クエリエンジンの作成
    if engine == "chat":
        while True:
            # prompt = speech_input.get_speech_input("チャット受付中：")
            prompt = input("質問を入力してください：")
            query_engine = index.as_chat_engine()
            print(f'回答：{query_engine.chat(prompt)}')
    elif engine == "query":
        prompt = input("質問を入力してください：")
        query_engine = index.as_query_engine()
        print(f'回答：{query_engine.query(prompt)}')

if __name__ == "__main__":
    """引数を何も指定しない場合、エンベッドしたデータに対するChatを起動できます"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--create_index", action="store_true", help="インデックスの作成")
    parser.add_argument("-rc", "--recursive_create_index", action="store_true", help="インデックスの作成")
    args = parser.parse_args()

    ### Fist-generation models
    # model = "ada"
    # model = "babbage"
    # model = "curie"
    # model = "davinci"

    ### Second-generation models
    model = "text-embedding-ada-002"

    engine = "chat"  # chat / query

    if args.create_index:
        # 特定のディレクトリ配下のソースファイルを覚え込ませるサンプル
        create_new_index("./data_sets", model)
        chat(model, engine)
    elif args.recursive_create_index:
        # 特定のディレクトリ配下のソースファイル（さらにその下以降のディレクトリも再起的に全て）を覚え込ませるサンプル
        recursive_create_new_index("../../utils", model)
        chat(model, engine)
    else:
        chat(model, engine)
