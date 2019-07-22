概要
====

KaggleのCompetitions『Digit Recognizer』(https://www.kaggle.com/c/digit-recognizer)

そこで最も人気のあるKernels(https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)

のコードにハイパーパラメータチューニングを施し、scoreを改善した。

0.99485(before) ⇒ 0.99571(after) 

## 各ファイル説明

before.py　：　参考にした元コード

after.py　： 改善したコード

tuning.py　：　ハイパーパラメータチューニングを行うコード

cnn_mnist_datagen.csv　：　最もスコアの良かったcsvファイル

score.png　：　パラメータ変更前、後のスコア比較

## Description

元コードと同様のニューラルネットワークを構築し、全結合層、ドロップアウト層に対して
機械学習にはKerasを用いており、hyperasを用いてハイパーパラメータチューニングを行った。

## 環境
Tensorflow 1.14
CUDA 10.0
cuDNN 7.4.2
