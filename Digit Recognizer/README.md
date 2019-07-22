概要
====

KaggleのCompetitions『[Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)』

そこで[最も人気のあるKernel](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)のコードにハイパーパラメータチューニングを施し、scoreを改善した。

score ： 0.99485 (before) ⇒ __0.99571__ (after) 

## 各ファイル説明

    before.py ： 参考にした元コード

    after.py ： 改善したコード

    tuning.py ： ハイパーパラメータチューニングを行うコード(自作)

    cnn_mnist_datagen.csv ： 最もスコアの良かったcsvファイル

    score.png ： パラメータ変更前、後のスコア比較

## 内容

最も評価数の多いKernelを見つけて試しに構築してみると、かなり精度の高い結果が得られた。

しかし全結合層、ドロップアウト層の値によっては更なる改善が見込めると予想した。

そこで元コードと同様のニューラルネットワークを構築し、

全結合層、ドロップアウト層に対して、 __hyperas__ ライブラリを用いることで

指定の値(範囲)から最適な値を割り出した。

結果としてより高いscoreを得ることができた。

## 環境

Tensorflow-gpu 1.14.0

CUDA 10.0

cuDNN 7.4.2
