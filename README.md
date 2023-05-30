# 環境構築方法
```
brew install portaudio
brew install mecab
brew install mecab-ipadic
pip install -r requirements.txt
```

- portaudioは、音声入力をするために必要です。youtube2textや、embed_chatなどで利用しています。
- mecabは、japanese_ocr_reader.pyを起動するために必要です。
- もしjapanese_ocr_reader.pyに興味がない場合は、requirements.txtを修正して必要なものだけpipしてください。

# ディレクトリ構成
```
A2AI
  /app
    /deeplearning : ディープラーニング系のプログラムを格納
    /docgen : 報告書（等）作成ツール
    /srcgen : プログラム自動生成系の機能を格納
    /utils : いろんなツール群
  /definition : spqa_framework.pyが参照するプログラム自動生成定義ファイルを配置
  /youtube_audio : youtube_audio.pyが利用する動画格納ディレクトリ
```

# 全体的な注意事項
- OpenAI社のAPI KEYを多用しているので、APIキーを発行し、システムの環境変数に設定してください。
```
export OPENAI_API_KEY=自分のキー
```

- 作者はM2MAXのMacで開発しているため、その他の環境では、プログラムが簡単には動かないかもしれません。

# アプリ別の解説

# app/srcgen
## ・spqa_framework.py
SQPAの概念の元、プログラムを自動生成するフレームワークです。
### 起動方法
```
cd app/srcgen
python spqa_framework.py
```

### 代表的な引数
| コマンド | 説明                                                 |
|----------|------------------------------------------------------|
| 引数なし | definitionに従ってソース生成。すでに出力先にソースがある場合はスキップ |
| -f       | 出力先にソースがあっても強制的に自動生成                 |
| -r       | 生成対象をフィルタ。ディレクトリ名や生成するファイル名の一部を記載 |
| -i       | 再帰生成モード。継続して残りのコードを生成し続けるかを問い合わせるモードとして起動 |


### 特記事項
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
- 直近ではトークン数が制約になるので、ソースコメントは書かせないほうが良い
- 定義が多くなると無視される定義が出てくるので、簡潔に書いたほうが良い

## ・refactor_framework.py
ソースコードを自動でリファクタリングするフレームワークです。

### 起動方法
```
cd app/srcgen
python refactor_framework.py
```

### 注意事項
- 一旦、app/utils配下のソースをリファクタするようにしています。
- 変更したい場合はrefactor_framework.pyを修正してください。

# app/docgen
## report_generator.py
報告書（等）作成ツール

### 起動方法
```
cd app/docgen
python report_generator.py
```

### 引数
- 特に無し。

### 特記事項
- 完成系の構成をテンプレートとして定義し、templatesディレクトリに格納してください
- data_setsに報告書のベースとなる情報（文章）をdata_setsディレクトリに格納してください。散文でも大丈夫です
- 作成された文書はartifactsディレクトリに格納されます。連続して起動すると、前回作成した文書を更新することができます
  - 使用例）
  - 1回目：空プロンプトで最初の生成を実施
  - 2回目：artifactsに生成された成果物を確認し、修正点をプロンプトで指示（例：トピックに曜日が抜けているので追加して）
  - 3回目：artifactsに生成された成果物を確認し、さらに修正した点をプロンプトで指示（例：AI分析は、いい感じに補強しといて。⚪︎⚪︎みたいなことをかっこよくやりたい）
- 上記のような感じで納得がいくまで修正を指示してみてください。
  - 例えばこんなプロンプトもアリです：
  - 今後のAI活用計画、正直なところよくわからないので、いい感じに補強して。できれば具体的な技術名称などもかっこよく含めてくれると助かる。
  - 「B社の微妙な反応」という文章が稚拙なので、もっとビジネスとしてビシッとかっこ良い文章に直して。


# app/utils
## youtube_audio
YouTubeの動画を要約するアプリです。

### 起動方法
```
cd app/utils
python youtube_audio.py
```

### 代表的な引数
| コマンド   | 説明                                                                  |
|------------|-----------------------------------------------------------------------|
| 引数なし   | YouTubeをダウンロードし、ChatGPTに要約を依頼。全自動。その後のチャットは不可 |
| -e         | YouTube動画をダウンロードし、そのデータをエンベッド                      |
| -ec        | エンベッドした内容に対してチャット                                       |

### 注意事項
- A2AIと同列に、whisper.cppが必要です。
- ※もしかしたらmacじゃないと動かないかも。

ディレクトリ構成例：
```
A2AI
whisper.cpp
```

whisper.cppのインストール方法は以下のサイトを参考にしてください。
https://zenn.dev/shu223/articles/whisper-coreml?fbclid=IwAR20EpbYK1bbP12hiNHiVMUdhAcKzbhbYg473IdQo0cVdmO9TgrQoY_Qw28

# app/deeplearning
## embedding
エンベッド技術を用いて、GPTモデルに対して手持ちのデータを学習させた後にチャットするサンプルです。

### 起動方法
```
cd app/deeplearning/embedding
python embed_chat.py
```

### 代表的な引数
| コマンド   | 説明                                                         |
|------------|--------------------------------------------------------------|
| 引数なし   | エンベッド済みのデータに対してチャットを実施                   |
| -c         | 指定したディレクトリ内のデータをエンベッド                     |
| -rc        | 指定したディレクトリディレクトリ配下の全てのディレクトリに格納されているデータをエンベッド |

### 注意事項
- 音声ファイルは英語じゃないと落ちるので、テキストファイルを読み込ませるのが良いです
- データは平文でずらずら書くのが適しています。意味のある単位にしたい場合は１行１文章にすると良いです
- エンベッドしたデータはstorageディレクトリに格納されます
- 利用されるGPTモデルはadaです（なので、精度高くないです）
- サンプルでは、-cをすると、app/utils配下のソースコードをエンベッドするようにパス設定しています。エンベッド後に、どんな機能を備えているの？とか、チャットしてみてください。
- max_tokensの設定方法が不透明でバグっぽい動きをします。最終的に、langchain.OpenAIクラスを直接修正しないと、設定が効きませんでした。

## ocr
- ローカルPCでOCRリーダーを実装するサンプル

### 起動方法
```
# 英語のみ読み取れる機能を試す場合
cd app/deeplearning/ocr
python ocr_reader.py

# 日本語読み取り機能対応版を試す場合
cd app/deeplearning/ocr
python japanese_ocr_reader.py

# 日本語読み取り機能対応版で利用しているModel(pytorch_model.bin)をファインチューニングするサンプル
cd app/deeplearning/ocr
python finetune_japanese_ocr_reader.py
```

### 特記事項
- imageディレクトリに解析したい画像ファイルを格納し、ソースコードのimage_pathを修正し、起動してください。
- ocr_reader.pyは追加学習したいところ。ワーニングを消したい。
- japanese_ocr_reader.pyを利用する場合は、pytorch_model.binを入手し、japanese_ocr_modeディレクトリに格納してください。
  - https://huggingface.co/spaces/Detomo/Japanese_OCR/tree/main/model
- japanese_ocr_readerで確認しましたが、対応する画像のサイズは150×150ピクセルまででした。（Model的には224×224まで対応するが、150を超えると著しく変な読み取りになる）
- ファインチューニングのサンプル（finetune_japanese_ocr_reader.py）では、train.pngを"HOGE"と読ませる訓練をしています。

## openai
- GPT3シリーズをファインチューニングして、独自のModelを作成する方法をコード化したものです。
- お金が異常にかかるので、コードを参照するレベルに留めていただくのが良いです。
- 作成したModelはOpenAI社のクラウド上に格納され、呼び出すごとに課金されます。
- データを準備する方法や詳細は、同ディレクトリに格納されている「チューニング＆モデル生成手順.txt」を参照してください。

## llm


### finetune.py
- モデルをファインチューニングするプログラムです。
- モデル自体をフルファインチューニング、LoRAチューニングなどを-t 引数の後にオプションを示すことでトレーニングを切り替えることができます。
- モデルを起動して文書生成を行わせる場合も、-r引数の後にオプションを示すことで対象のモデルを切り替えることができます。
- LoRAでチューニングした場合は、自動的にAutomatic1111にsafetensors形式に変換したモデルを転送することが可能です。マージしたい場合は転送してください。
- LoRAのファインチューニングについて、以下に参考文献を示します。
- 参考文献は[コチラ](https://note.com/npaka/n/na5b8e6f749ce)です。（必見）
- ほぼ参考文献通りのソースですが、MacBookで動くようにコード修正しています。
- 利用しているデータセットの中身を確認した場合は[コチラ](https://raw.githubusercontent.com/kunishou/databricks-dolly-15k-ja/main/databricks-dolly-15k-ja.json)をクリック。

### 起動方法
```
cd app/deeplearning/llm 
python finetune.py 引数
```

### 代表的な引数
| コマンド   | 説明                                                         |
|------------|--------------------------------------------------------------|
| -t         | トレーニングを実施                     |
| -r         | モデルを起動                              |
| -f         | ファインチューニングオプション                              |
| -l         | LoRAオプション                              |
| -m         | マージオプション                              |

### 起動例：対象のモデルをフルファインチューニングする場合
```
python finetune.py -t
```

### 起動例：対象のモデルをLoRAファインチューニングする場合
```
python finetune.py -t -l
```

### 起動例：Automatic1111でマージしたモデルをfinetuned-lora-mergedにModelとして保存する場合
```
python finetune.py -m
※ Automatic1111に、[ベースのモデル名]-merged.safetensorsというファイルを配置してから実行
```

### 起動例：ベースのモデルに対して文書生成させる場合
```
python finetune.py -r
```

### 起動例：フルファインチューニングしたモデルに対して文書生成させる場合
```
python finetune.py -r -f
```

### 起動例：LoRAファインチューニングしたモデルに対して文書生成させる場合
```
python finetune.py -r -l
```

### 起動例：ベースとLoRAでマージしたモデルに対して文書生成させる場合
```
python finetune.py -r -m
```

### 注意事項
```
LoRAする上で、AppleSillicon(M1/M2)を使わせるためには、device_mapでautoを指定するのは厳禁。
（RuntimeError: Placeholder storage has not been allocated on MPS device!　エラーが起きる）

# 8bit量子化してModelを取得する方法のMac版は以下の通り。
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        llm_int8_enable_fp32_cpu_offload=True,
    )

# ネットでよく見つかるcuda版は以下の通り。
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,  # Macだとこれ無理
        device_map="auto"   # device_map指定するとMacだと落ちる
    )
```

### utils_for_tuning.py
- 二つのモデルを一つにマージする際に必要となる機能を提供します
- LoRAしたモデルと元となったモデルを結合したりするのに使えます

### 起動方法
```
cd app/deeplearning/llm 
python utils_for_tuning.py -引数
```

### 代表的な引数
| コマンド   | 説明                                                         |
|------------|--------------------------------------------------------------|
| -ss         | モデルを.safetensors形式に変換します。ファイルはsafetensorsディレクトリに格納されます                     |
| -ls         | .savetensors形式のモデルを元のモデル（.bin, .h5）に戻します                              |

### モデルをマージする方法
- まずはAutomatic1111をインストールしてください。[Macでのインストール方法](https://gift-by-gifted.com/stablediffusion-3/)
- マージしたいモデルの.safetensorsファイルを用意します。コレはfinetune_utils.py -ssで作成できます
- 作成した.safetensorsファイルを、stable-diffusion-webui/models/Stable-diffusionディレクトリに格納
- Automatic1111を利用してモデルをマージします。[マージ方法](https://kurokumasoft.com/2023/04/01/howto-make-merge-models/)
- マージして作成された.safetensorsファイルをfinetune_utils.py -lsにてmodelに戻します。参照先ディレクトリは、ソースを直接修正してください
- ggml化、量子化する場合はまず、ggmlツールをインストールしてください。git clone https://github.com/ggerganov/ggml.git
- コマンドは、A2AI/app/deeplearning/llm/ggml/生成コマンド.txtに記載してあります。利用するモデルのベース（gpt-2, gpt-neoxなど）ごとにスクリプトが違うので注意です。

### 学習の評価
- tensorboardを利用します。
- 学習は行うとlogディレクトリにログを出力しているので、それをtensorboardに食わせるとグラフで確認することができます。

```
# 起動コマンド
tensorboard --logdir ./log
http://localhost:6006/
```

### 注意事項
- peft_llm.pyではうまくログが出力されます
- finetune_llm.pyはログが出ないので、後ほど直します。

### old_src_finetune_llm.py
- 勉強初期段階で作っていたコード。（今は、finetune.pyが高機能になったので、過去ログとして保持）
- TransformerベースのLLMをローカルでファインチューニングするサンプルです。
- トレーニング後の使い方としては、与えたプロンプトに続く文章を生成させることができます。
  - （チャットではないので注意してください）
- finetune_llm.pyの冒頭の、model_typeを修正すると、利用するモデルを切り替えることができます。
- finetune_llm.pyの冒頭の、EPOCHSを修正すると、学習サイクルを変更することができます。

### 起動方法
```
cd app/deeplearning/llm 
python finetune_llm.py -r
```

### 代表的な引数
| コマンド   | 説明                                                         |
|------------|--------------------------------------------------------------|
| -t         | data_sets配下のデータを利用してトレーニング                     |
| -r         | トレーニングしたModelに対してChat                              |

### 注意事項
```
A2AIと同列に、transformersのプログラム群が必要です。

ディレクトリ構成例：
A2AI
transformers

transformersのインストールコマンドは以下。
git clone https://github.com/tak6uch1/cuda-tensorflow.git

動かす場合は、同ディレクトリに格納してある「環境設定コマンド.txt」内記載のインストールを実施してください。
おそらく、mac（且つ、M1/M2チップ搭載機）じゃないと動かないかも？

「環境設定コマンド.txt」にも記載していますが、以下の環境変数設定も必要です。
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

ちなみに、finetune.pyにパイソンのコード版でのトレーニングを記載したので、後ほど差し替えて、transformersのgit-cloneは不要にする予定。
```
