# https://openai.com/blog/function-calling-and-other-api-updates?fbclid=IwAR0vm92datK2bxNVJTVixvp3HegkAG1K_Yk5wYvbUoHmuef2p3u45MNb77s

import os
import openai
import requests

def run_chat_gpt(model, prompt):

    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # ChatGPTにリクエストを送信する関数
    print(f"model: {model}")
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "functions": [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        },
        json=data
    )

    # 以下、choices内に呼び出されたfunctionの構成情報が返却されるので、
    # 開発者がその情報を読み解いて、独自にfunction処理を構築する。
    print(response.json()['choices'])

    # -> [{'index': 0, 'message': {'role': 'assistant', 'content': None, 'function_call': {'name': 'get_current_weather', 'arguments': '{\n  "location": "Boston, MA"\n}'}}, 'finish_reason': 'function_call'}]

if __name__ == "__main__":
    # model = "gpt-4-0613"
    # model = "gpt-4-32k-0613"  # これはまだ使えない
    model = "gpt-3.5-turbo-0613"
    # model = "gpt-3.5-turbo-16k"
    
    # descriptionに合致するプロンプトの場合、functioncallとして処理される。
    prompt = "こんにちは"  # こちらだと、普通の回答が帰ってくる
    prompt = "今日の神戸の天気は？摂氏で教えて"  # こちらだとfunctioncallとして扱われる

    run_chat_gpt(model, prompt)

# 実行結果（choicesの中身）

# "こんにちは"
# [{'index': 0, 'message': {'role': 'assistant', 'content': 'こんにちは！どのようにお手伝いしましょうか？'}, 'finish_reason': 'stop'}]

# "今日の神戸の天気は？"
# [{'index': 0, 'message': {'role': 'assistant', 'content': None, 'function_call': {'name': 'get_current_weather', 'arguments': '{\n"location": "神戸"\n}'}}, 'finish_reason': 'function_call'}]

# "今日の神戸の天気は？摂氏で教えて"
# [{'index': 0, 'message': {'role': 'assistant', 'content': None, 'function_call': {'name': 'get_current_weather', 'arguments': '{\n  "location": "Kobe, Japan",\n  "unit": "celsius"\n}'}}, 'finish_reason': 'function_call'}]