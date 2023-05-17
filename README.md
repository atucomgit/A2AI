# 環境構築方法
git clone [URL]
pip install -r requirements.txt


# ディレクトリ構成



# アプリ別の解説

## app/srcgen
### spqa_framework.py
#### 起動方法
cd app/srcgen
python spqa_framework.py

代表的な引数
引数なし | definitionに従ってソース生成。すでに出力先にソースがある場合はスキップ
 -f | 出力先にソースがあっても強制的に自動生成
 -r | 生成対象をフィルタ。ディレクトリ名や生成するファイル名の一部を記載
 -i | 再帰生成モード。継続して残りのコードを生成し続けるかを問い合わせるモードとして起動

### refactor_framework.py
#### 起動方法
cd app/srcgen
python refactor_framework.py

## app/utils
### youtube_audio
#### 起動方法
cd app/utils
python youtube_audio.py

代表的な引数
引数なし | YouTubeをダウンロードし、ChatGPTに要約を依頼。全自動。その後のチャットは不可
 -e | YouTube動画をダウンロードし、そのデータをエンベッド
 -ec | エンベッドした内容に対してチャット

#### 注意事項
A2AIと同列に、whisper.cppが必要です。

ディレクトリ構成例：
A2AI
whisper.cpp

whisper.cppのインストール方法は以下のサイトを参考にしてください。
https://zenn.dev/shu223/articles/whisper-coreml?fbclid=IwAR20EpbYK1bbP12hiNHiVMUdhAcKzbhbYg473IdQo0cVdmO9TgrQoY_Qw28

## app/deeplearning
### embedding
#### 起動方法
cd app/srcgen
python refactor_framework.py
