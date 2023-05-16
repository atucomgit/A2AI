import argparse
from yt_dlp import YoutubeDL
import subprocess
import os
import time
import datetime
import openai
import requests
import sys
sys.path.append("../deeplearning/embedding")
import embed_chat

input_file = "../../youtube_audio/audio.m4a"
output_file = "../../youtube_audio/audio.wav"
output_dir = "../../youtube_audio/transcripts"
divide_output_dir = "../../youtube_audio/transcripts/divide"
embed_index = "../../youtube_audio/index"

# 要約したいYouTubeのURL（v=以降をコピペしてね）
youtube_url = "5MK_4zuoZyE"

def download_youtube():
    with YoutubeDL({'overwrites':True, 'format':'bestaudio[ext=m4a]', 'outtmpl':input_file}) as ydl:
        ydl.download(f"https://www.youtube.com/watch?v={youtube_url}")

    # ffmpegコマンドを生成する
    ffmpeg_cmd = f"ffmpeg -y -i {input_file} -acodec pcm_s16le -ac 1 -ar 16000 {output_file}"

    # コマンドを実行する
    os.system(ffmpeg_cmd)

def wav2text():
    # Whisperでテキスト化
    start_time = time.time()
    print(f"Whisperでテキスト化を開始: at {datetime.datetime.now()}")
    whisper_command = f"../../../whisper.cpp/main -m ../../../whisper.cpp/models/ggml-medium.bin -f {output_file} -l ja"
    output = subprocess.run(whisper_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Extract transcript from the output
    transcript = ""
    for line in output.stdout.split("\n"):
        if "-->" in line:
            transcript += line.split("]")[-1].strip() + " "

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Whisper処理時間: {elapsed_time:.3f}秒")

    # Save transcript to a file
    with open("../../youtube_audio/transcripts/audio.txt", "w") as f:
        f.write(transcript)

    return transcript

def divide_transcript():
    # Delete existing transcript files
    if not os.path.exists(divide_output_dir):
        os.makedirs(divide_output_dir)
    existing_files = os.listdir(divide_output_dir)
    for filename in existing_files:
        if filename != '.gitkeep':
            filepath = os.path.join(divide_output_dir, filename)
            os.remove(filepath)

    # Read transcript from file
    with open("../../youtube_audio/transcripts/audio.txt", "r") as f:
        transcript = f.read()

    # Divide transcript into chunks of 3000 characters
    chunk_size = 1900
    transcript_chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]

    # Write each chunk to a separate file
    for i, chunk in enumerate(transcript_chunks):
        filename = f"audio{i+1}.txt"
        filepath = os.path.join(divide_output_dir, filename)
        with open(filepath, "w") as f:
            f.write(chunk)

    return len(transcript_chunks)

def summarize_text(super_summarize):
    start_time = time.time()
    print(f"ChatGPTで要約を開始: at {datetime.datetime.now()}")

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    # Read transcript files
    transcript_files = os.listdir(divide_output_dir)

    print("**** 要約 ****")
    summary = ""
    for filename in transcript_files:
        filepath = os.path.join(divide_output_dir, filename)
        with open(filepath, "r") as f:
            text = f.read()

            # ChatGPTにリクエストを送信する関数
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "・から始まる箇条書きで要約してください。"},
                    {"role": "user", "content": f"以下を箇条書きで、できるだけ少ない文字で簡潔に要約してください。{text}"},
                ],
                "max_tokens": 2000,
                "n": 1,
                "stop": None,
                "temperature": 0.7,
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai.api_key}"
                },
                json=data
            )

            choice = response.json()['choices'][0]
            gpt_response = choice['message']['content'].strip()

            print(f"{gpt_response}")

            summary += gpt_response

    if super_summarize:
        # ChatGPTにリクエストを送信する関数
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "あなたは有能な編集者です。情報を視聴者にわかりやすく説明することを心がけてください。"},
                {"role": "user", "content": f"以下を、カテゴリ別に分類して、さらに要約を整理してさい。{summary}"},
            ],
            "max_tokens": 4000,
            "n": 1,
            "stop": None,
            "temperature": 0.7,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai.api_key}"
            },
            json=data
        )  
        choice = response.json()['choices'][0]
        final_summary = choice['message']['content'].strip()
        print("---- 超要約結果 ---")
        print(final_summary)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"要約時間: {elapsed_time:.3f}秒")

def summarise_text_by_embed():
    """embedの技術を使って全部入りのaudio.txtを一発でembedする場合"""
    embed_chat.create_new_index(output_dir, embed_index)

def chat_by_embed(prompt):
    """"embedしたデータでチャット（エンジンはadaを利用するので精度高いChatは不可）"""
    embed_chat.chat(embed_index, prompt)

def main():
    parser = argparse.ArgumentParser(description="Download audio from YouTube, convert to WAV, and generate text transcripts.")
    parser.add_argument("-y", "--download", action="store_true", help="download audio from YouTube")
    parser.add_argument("-t", "--wav2text", action="store_true", help="convert WAV audio to text using Whisper")
    parser.add_argument("-d", "--divide_transcript", action="store_true", help="divide transcript into 3000-character chunks and write to separate files")
    parser.add_argument("-s", "--summarize_text", action="store_true", help="summarize text using OpenAI's GPT-3.5")
    parser.add_argument("-p", "--super_summarize", action="store_true", help="perform super summarization using OpenAI's GPT-4")
    parser.add_argument("-e", "--embed_summarize", action="store_true", help="perform embed summarization using OpenAI Ada")
    parser.add_argument("-ec", "--chat_by_embed", action="store_true", help="perform embed summarization using OpenAI Ada")
    args = parser.parse_args()

    # 引数未指定なら全工程を一気に実施
    if not any(vars(args).values()):
        # No arguments specified, run all functions
        download_youtube()
        transcript = wav2text()
        num_files = divide_transcript()
        print(f"{num_files} files generated.")
        summarize_text(True)
    else:
        if args.download:
            download_youtube()
        if args.wav2text:
            transcript = wav2text()
        if args.divide_transcript:
            num_files = divide_transcript()
            print(f"{num_files} files generated.")
        if args.summarize_text:
            summarize_text(args.super_summarize)
        if args.super_summarize:
            summarize_text(args.super_summarize)
        if args.embed_summarize:
            download_youtube()
            wav2text()
            summarise_text_by_embed()
            chat_by_embed("内容を箇条書きで整理してください。")
        if args.chat_by_embed:
            chat_by_embed(None)

if __name__ == "__main__":
    main()