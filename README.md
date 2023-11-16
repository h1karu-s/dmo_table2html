# demo table2html



## 利用方法
### 0. setup
Googleドライブ(https://drive.google.com/file/d/1wPMUuIVcPFf4qJCsL6l0FY44tMSdQOGl/view?usp=sharing) から学習済みパラメータを取得する.
以下のコマンドを実行
```
FILE_ID=1wPMUuIVcPFf4qJCsL6l0FY44tMSdQOGl
FILE_NAME=model.tar.gz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
```
取得したmode.tar.gzを以下のコマンドで展開する.
```
tar -zxvf model.tar.gz
```
対応するpaddle ocrを取得
※ホームページから環境に対応するものをダウンロードする(https://www.paddlepaddle.org.cn/en)
```
##cuda11.6の例
python -m pip install paddlepaddle-gpu==2.5.2.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

ライブラリーをインストール
 ※torchは対応するものをダウンロードする(https://pytorch.org/get-started/locally/)
```
###cuda11.6
pip install -r requirements.txt
```
### 1. 予測
--input_dirで指定したディレクトリー直下に画像を保存する. output_dirにはモデルの出力とOCRの結果が保存される. --model_pathで使用するモデルのパスを指定する.
```
python predict.py \
    --input_dir <image_dir>
    --output_dir <output_dir>
    --model_path <model path>
```
※予測したHTMLを表画像として確認したい場合, visualize_html.ipynbをご使用ください.


### model
model.tar.gzには, 4つのモデルを保存している.
* layoutlmv3-en_pubtabnet_200-400_paddle-ocr　\
  * pubtabnetの系列長が入力テキスト<200, 生成HTML<400となる表をpaddle ocrの出力を使って学習したLayoutLMv3-mbart
* layoutlmv3-en_fintabnet_512-1024_paddle-ocr \
  * fintabnetの系列長が入力テキスト<512, 生成HTML<1024となる表をpaddle ocrの出力を使って学習したLayoutLMv3-mbart
* swin-en_pubtabnet_200-400 \
  * pubtabnetの系列長が入力テキスト<200, 生成HTML<400となる表を使って学習したswin_transformer-mbart
* swin-en_fintabnet_512-1024 \
  * fintabnetの系列長が入力テキスト<512, 生成HTML<1024となる表を使って学習したswin_transformer-mbart

![モデル_表](https://github.com/h1karu-s/dmo_table2html/assets/65395880/80c162ba-74ba-4318-861b-f27ad4f79f6e)

