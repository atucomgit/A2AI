import speech_recognition as sr
import threading

def get_speech_input(prompt):
    r = sr.Recognizer()
    stop_event = threading.Event()
    recognized_texts = []  # 認識したテキストを保存するリスト

    def callback(recognizer, audio):
        nonlocal recognized_texts
        try:
            text = recognizer.recognize_google(audio, language="ja-JP")
            if not stop_event.is_set():
                recognized_texts.append(text)  # テキストをリストに追加
                print(text)
        except sr.UnknownValueError:
            if not stop_event.is_set():
                # print("音声が認識できませんでした。")
                pass # 上記はウザいので出力しない

    print(f"{prompt}")

    # バッファリングのための一時的なリスナー
    with sr.Microphone() as temp_source:
        r.adjust_for_ambient_noise(temp_source)

    # 音声認識を別スレッドで実行
    recognizer_thread = threading.Thread(target=r.listen_in_background, args=(sr.Microphone(), callback))
    recognizer_thread.start()

    # Enterキーが押されるまで待機
    input("（発言が終了したらEnterキーを押してください...）\n")
    stop_event.set()

    # 音声認識を停止
    recognizer_thread.join()

    # 認識したテキストを連結して返却
    result_text = " ".join(recognized_texts)
    return result_text

if __name__ == "__main__":
    text = get_speech_input("話してください...")
    print("全ての認識結果:", text)
