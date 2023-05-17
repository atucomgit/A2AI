
import os, requests

openai_api_key = os.environ.get("OPENAI_API_KEY")
model = "gpt-4"  # 解禁されていなければ、gpt-3.5-turboで差し替えてください。

def read_file(file_path):
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            file_text = f.read()
        return file_text
    else:
        return None

def run_gpt(template, data, prompt):

    print("---- ChatGPTにて報告書作成 ----")
    print(f"prompt:{prompt}")
    print("------------------------------")
    print("処理中...")
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"以下のテンプレートに従い、報告書を完成させてください。{template}"},
            {"role": "system", "content": f"報告書の元となるデータは以下のとおりです。{data}"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1024,
        "n": 1,
        "stop": None,
        "temperature": 0.7,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        },
        json=data
    )

    choice = response.json()['choices'][0]
    report = choice['message']['content'].strip()
    print(f"GPT作成結果:{report}")
    return report

def make_document(template_path, data_path, prompt, output_path):
    template = read_file(template_path)
    data = read_file(data_path)
    making_text = read_file(output_path)

    # 再作成をする場合は作成ずみを考慮するようにプロンプトを調整
    if making_text:
      prompt = f"{prompt}：書きかけの文章があるので、こちらに追記してさい。指示以外の部分は修正しないでください。{making_text}"

    # TODO: ここに危険ワードマスク処理を入れる

    # GPTにレポート作成を依頼
    text = run_gpt(template, data, prompt)

    # TODO: ここにマスクワードを戻す処理を入れる

    # 作成したレポートを保存
    with open(output_path, "w", encoding='utf-8') as outfile:
        outfile.write(text)

if __name__ == "__main__":
    template_path = "./templates/週次報告書.txt"
    data_path = "./data_sets/週次作業状況.txt"
    output_path = "./artifacts/週次報告書.txt"

    prompt = input("作成するドキュメントに対する指示をお願いします：")
    if not prompt:
      prompt = "レポートを作成して。"
    make_document(template_path, data_path, prompt, output_path)