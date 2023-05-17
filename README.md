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

#### 代表的な引数
引数なし | definitionに従ってソース生成。すでに出力先にソースがある場合はスキップ
 -f | 出力先にソースがあっても強制的に自動生成
 -r | 生成対象をフィルタ。ディレクトリ名や生成するファイル名の一部を記載
 -i | 再帰生成モード。継続して残りのコードを生成し続けるかを問い合わせるモードとして起動

#### 特記事項
- A2AI/definitionディレクトリ配下に好きなディレクトリを掘り、ソース生成定義ファイルを作成してください
- 上記のディレクトリ構成の通りに、A2AI/app配下にディレクトリを作成し、ソースを生成します
- 作成されるソースファイル名は、定義ファイルの.mdを取り除いたものとなります。
　例）sample.py.md -> sample.py
- プログラム言語の種類は特に限定がないです。定義ファイルに色々書いてGPTさんに指示を出してください
- 定義ファイルはこちらを参考にしてください。A2AI/definition/utils/db_utils_sample.py.md
- 色々試した結果、Referenceブロックをしっかり書くのが最強でした
- トークン数の制約から、以下の挙動があります。参考にしてください
  - gpt-3.5-turboは長いプログラムを作るのに適している。ただし、難しいのは苦手。
  - gpt-4は高度なプログラムを作るのに適している。ただし、長いのは苦手。

### refactor_framework.py
#### 起動方法
cd app/srcgen
python refactor_framework.py

#### 注意事項
一旦、app/utils配下のソースをリファクタするようにしています。
変更したい場合はrefactor_framework.pyを修正してください。

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
※もしかしたらmacじゃないと動かないかも。

ディレクトリ構成例：
A2AI
whisper.cpp

whisper.cppのインストール方法は以下のサイトを参考にしてください。
https://zenn.dev/shu223/articles/whisper-coreml?fbclid=IwAR20EpbYK1bbP12hiNHiVMUdhAcKzbhbYg473IdQo0cVdmO9TgrQoY_Qw28

## app/deeplearning
### embedding
#### 起動方法
cd app/deeplearning/embedding
python embed_chat.py

#### 代表的な引数
引数なし | エンベッド済みのデータに対してチャットを実施
 -c | 指定したディレクトリ内のデータをエンベッド
 -rc | 指定したディレクトリディレクトリ配下の全てのディレクトリに格納されているデータをエンベッド

#### 注意事項
- 音声ファイルは英語じゃないと落ちるので、テキストファイルを読み込ませるのが良いです
- データは平文でずらずら書くのが適しています。意味のある単位にしたい場合は１行１文章にすると良いです
- エンベッドしたデータはstorageディレクトリに格納されます
- 利用されるGPTモデルはadaです（なので、精度高くないです）
- サンプルでは、-cをすると、app/utils配下のソースコードをエンベッドするようにパス設定しています。エンベッド後に、どんな機能を備えているの？とか、チャットしてみてください。

### openai
GPT3シリーズをファインチューニングして、独自のModelを作成する方法をコード化したものです。
お金が異常にかかるので、コードを参照するレベルに留めていただくのが良いです。
作成したModelはOpenAI社のクラウド上に格納され、呼び出すごとに課金されます。
データを準備する方法や詳細は、同ディレクトリに格納されている「チューニング＆モデル生成手順.txt」を参照してください。

### tensorflow
TransformerベースのLLMをローカルでファインチューニングするサンプルです。

#### 起動方法
cd app/deeplearning/tensorflow 
python finetune_rinna.py -r

#### 代表的な引数
-t | data_sets配下のデータを利用してトレーニング
-r | トレーニングしたModelに対してChat

#### 注意事項
A2AIと同列に、transformersのプログラム群が必要です。

ディレクトリ構成例：
A2AI
transformers

transformersのインストールコマンドは以下。
git clone https://github.com/tak6uch1/cuda-tensorflow.git

動かす場合は、同ディレクトリに格納してある「環境設定コマンド.txt」内記載のインストールを実施してください。
おそらく、mac（且つ、M1/M2チップ搭載機）じゃないと動かないかも？