# Faster R-CNN for Torchvision

![b1d968b9-563405f4](https://user-images.githubusercontent.com/63311737/165238247-60c27db4-b3d7-46a1-9e31-a8513c4e088f.jpg)


## 環境
```
python 3.6
CUDA Toolkit 11.1 Update 1
cuDNN v8.1.1 (Feburary 26th, 2021), for CUDA 11.0,11.1 and 11.2
```

## 環境構築

ターミナルに以下を入力
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```




## 準備

dataset_path.ymlにデータセット名,パス,dataset_typeを記入する

dataset_classes.ymlにdataset_typeに応じたクラス配列を記入する


## 学習
```
python main.py --dataset_name {trainデータセット名} --val_dataset_name　{valデータ名}
```


## 評価
```
python main.py --eval --dataset_name bdd_val --train_model_path {学習済みモデルが入っているフォルダまでのpath} --batchsize 1
```

## 推論
```
python main.py --test --train_model_path {学習済みモデルまでのpath} --img_path {推論したい画像フォルダのpath}　--dataset_name {学習に使ったデータセット名} --batchsize 1
```


## その他コマンド
```
--lr             :学習率 (default=0.001)
--epochs         :エポック数 (default=400)
--batchsize      :バッチサイズ (default=3)
--dataset_name   :用いるデータセット名(dataset_path.ymlに記述したデータ名)
--output_dir     :出力フォルダー(学習時は./log以降，推論時は./output以降のフォルダ)
```


