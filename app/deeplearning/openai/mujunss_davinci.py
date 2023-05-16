import openai
import sys

def generate_completion(model_id, prompt):
    response = openai.Completion.create(
        model=model_id,
        prompt=prompt,
        max_tokens=300
    )
    completion = response.choices[0].text.strip()
    return completion

if __name__ == "__main__":
    # ファインチューニングしたモデルのIDを指定してください
    model_id = "davinci:ft-personal-2023-05-12-00-49-15"

    # コマンドライン引数からテキストのプロンプトを取得
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "こんにちは"

    # 応答の生成と結果の表示
    completion = generate_completion(model_id, prompt)
    print(completion)
