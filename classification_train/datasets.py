import os
import numpy as np
from PIL import Image
import cv2

import torch
import torch.utils.data as data

class ClassificationDataset(data.Dataset):
    """
    シングルラベル画像分類のDatasetクラス

    Attributes
    ----------
    list_file : string
        listファイル（画像ファイルのパスと対応するラベルの組が記載されたファイル）のパス
    transform : object
        前処理クラスのインスタンス
    phase : 'train' or 'val'
        学習か検証かを設定する。
    """
    def __init__(self, df, img_size, transform=None):
        self.transform = transform
        self.img_size = img_size
        self.data_paths = df["image_path"].values.tolist()
        self.labels = df["label"].values.tolist()
        self.image_ids = df["id"].values.tolist()
        # import pdb;pdb.set_trace()

        # root_path = os.path.dirname(list_file)
        # data_list = open(list_file, "r")
        # for line in data_list:
        #     img_path, label = line.split(' ')
        #     self.images.append(os.path.join(root_path, img_path))
        #     self.labels.append(int(label))

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.image_ids)
        
    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''
        # index番目の画像をロード
        img_path = os.path.join(self.data_paths[index], f'{self.image_ids[index]}.jpg')
        #img = Image.open(img_path).convert('RGB')  # [高さ][幅][色RGB]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # リサイズ
        img = cv2.resize(img, dsize=(self.img_size, self.img_size))

        # 画像の前処理を実施
        if self.transform is not None:
            img_transformed = self.transform(image=img)["image"]
            img_transformed = img_transformed.transpose(2,0,1)      # torch.Size([3, 224, 224])
        else:
            img_transformed = np.asarray(img)
        
        # ラベルを取得
        label = self.labels[index]
        
        return img_transformed, label

