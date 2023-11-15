import torch

import sys, os

import pandas as pd

import numpy as np

sys.path.append("../input/utils")
import get_KNN, eval_preds
from sklearn.feature_extraction.text import TfidfVectorizer

from PIL import Image

import albumentations

from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader

import math
import cv2
import timm
import os
import random
import gc

from sklearn.preprocessing import normalize

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from tqdm.notebook import tqdm


from custom_scheduler import ShopeeScheduler
from custom_activation import replace_activations, Mish
from custom_optimizer import Ranger


import warnings

warnings.filterwarnings("ignore")


class CFG:
    # data augmentation
    IMG_SIZE = 512
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    SEED = 2023

    # data split
    N_SPLITS = 5
    TEST_FOLD = 0
    VALID_FOLD = 1

    EPOCHS = 8
    BATCH_SIZE = 8

    NUM_WORKERS = 4
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CLASSES = 11014  #!!!!!!!!!!!!!
    SCALE = 30
    MARGIN = 0.6
    
    SCHEDULER_PARAMS = {
            "lr_start": 1e-5,
            "lr_max": 1e-5 * 32,
            "lr_min": 1e-6,
            "lr_ramp_ep": 5,
            "lr_sus_ep": 0,
            "lr_decay": 0.8,
        }
    

    MODEL_NAME = "eca_nfnet_l1"
    FC_DIM = 512
    MODEL_PATH = f"../input/shopee-models/{MODEL_NAME}_arc_face_epoch_{EPOCHS}_bs_{BATCH_SIZE}_margin_{MARGIN}.pt"
    FEAT_PATH = f"../input/shopee-embeddings/{MODEL_NAME}_arcface.npy"


def get_train_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE, always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            albumentations.Normalize(mean=CFG.MEAN, std=CFG.STD),
            ToTensorV2(p=1.0),
        ]
    )


def get_valid_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE, always_apply=True),
            albumentations.Normalize(mean=CFG.MEAN, std=CFG.STD),
            ToTensorV2(p=1.0),
        ]
    )


def get_test_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE, always_apply=True),
            albumentations.Normalize(mean=CFG.MEAN, std=CFG.STD),
            ToTensorV2(p=1.0),
        ]
    )


class ShopeeImageDataset(Dataset):
    """for training"""

    def __init__(self, df, transform=None, train=True):
        self.df = df
        self.img_path = df["image"].values
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        image = cv2.imread(self.img_path[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
            
        if self.train:
            label = self.df.label_group[index]
            label = torch.tensor(label).long()
            return {"image": image, "label": label}
        else:
            label = torch.tensor(1)
            return image, label

    def __len__(self):
        return len(self.df)


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=CFG.DEVICE)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output, nn.CrossEntropyLoss()(output, label)


class ShopeeModel(nn.Module):
    def __init__(
        self,
        n_classes=CFG.CLASSES,
        model_name=CFG.MODEL_NAME,
        fc_dim=CFG.FC_DIM,
        margin=CFG.MARGIN,
        scale=CFG.SCALE,
        use_fc=True,
        pretrained=True,
    ):
        super(ShopeeModel, self).__init__()
        print("Building Model Backbone for {} model".format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        if "efficientnet" in model_name:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif "resnet" in model_name:
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif "resnext" in model_name:
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif "nfnet" in model_name:
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=0.0)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.final = ArcMarginProduct(final_in_features, n_classes, s=scale, m=margin)
        self.training=True
        

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def set_training(self, training):
        self.training = training
        
    def forward(self, image, label):
        feature = self.extract_feat(image)
        logits = self.final(feature, label)
        return logits
        # if self.training:
        #     logits = self.final(feature, label)
        #     return logits
        # else:
        #     return feature

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc: # ???????
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)
        return x


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # set True to be faster


def train_fn(model, data_loader, optimizer, scheduler, i):
    model.train()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc="Epoch" + " [TRAIN] " + str(i + 1))

    for t, data in enumerate(tk):
        for k, v in data.items():
            data[k] = v.to(CFG.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()

        tk.set_postfix(
            {
                "loss": "%.6f" % float(fin_loss / (t + 1)),
                "LR": optimizer.param_groups[0]["lr"],
            }
        )

    if scheduler:
        scheduler.step()

    return fin_loss / len(data_loader)


def get_valid_embeddings(df, model):
    model.eval()

    image_dataset = ShopeeImageDataset(
        df, transform=get_valid_transforms(), train=False
    )
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=CFG.BATCH_SIZE,
        pin_memory=True,
        num_workers=CFG.NUM_WORKERS,
        drop_last=False,
    )

    embeds = []
    with torch.no_grad():
        for img, label in tqdm(image_loader):
            img = img.to(CFG.DEVICE)
            label = label.to(CFG.DEVICE)
            feat, _ = model(img, label)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)

    del model
    image_embeddings = np.concatenate(embeds)
    print(f"Our image embeddings shape is {image_embeddings.shape}")
    del embeds
    gc.collect()
    return image_embeddings


def run_training(train_df, valid_df, test_df, destination, threshold):
    # train_df, valid_df, test_df = read_dataset()
    train_dataset = ShopeeImageDataset(
        train_df, transform=get_train_transforms(), train=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE,
        pin_memory=True,
        num_workers=CFG.NUM_WORKERS,
        shuffle=True,
        drop_last=True,
    )

    print("You should set CFG.CLASSES = ", train_df["label_group"].nunique())

    model = ShopeeModel()
    model = replace_activations(model, torch.nn.SiLU, Mish())
    model.to(CFG.DEVICE)

    optimizer = Ranger(model.parameters(), lr = CFG.SCHEDULER_PARAMS['lr_start'])
    #optimizer = torch.optim.Adam(model.parameters(), lr = config.SCHEDULER_PARAMS['lr_start'])
    scheduler = ShopeeScheduler(optimizer,**CFG.SCHEDULER_PARAMS)
    
    best_valid_f1 = 0.0

    for i in range(CFG.EPOCHS):
        avg_loss_train = train_fn(model, train_dataloader, optimizer, scheduler, i)

        valid_embeddings = get_valid_embeddings(valid_df, model)
        # valid_df, valid_predictions = get_valid_neighbors(valid_df, valid_embeddings)
        
        # valid_df = cos_search.search(
        #     valid_df, valid_embeddings, destination, threshold
        # )
        valid_df = get_KNN.get_valid_neighbors(
            valid_df, valid_embeddings, destination, threshold=0.36
        )
        valid_f1 = valid_df.f1.mean()
        valid_recall = valid_df.recall.mean()
        valid_precision = valid_df.precision.mean()
        print(
            f"Valid f1 score = {valid_f1}, recall = {valid_recall}, precision = {valid_precision}"
        )

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            print("Valid f1 score improved, model saved")
            torch.save(model.state_dict(), CFG.MODEL_PATH)


def train(df, destination="oof_eca_nfnet_l1", threshold=0.93):
    seed_everything(CFG.SEED)

    gkf = GroupKFold(n_splits=CFG.N_SPLITS)
    df["fold"] = -1
    for i, (train_idx, valid_idx) in enumerate(
        gkf.split(X=df, groups=df["label_group"])
    ):
        df.loc[valid_idx, "fold"] = i

    labelencoder = LabelEncoder()
    df["label_group"] = labelencoder.fit_transform(df["label_group"])

    # train_df = df[df["fold"] != CFG.TEST_FOLD].reset_index(drop=True)
    # train_df = train_df[train_df["fold"] != CFG.VALID_FOLD].reset_index(drop=True)
    train_df = df
    valid_df = df[df["fold"] == CFG.VALID_FOLD].reset_index(drop=True)
    
    test_df=None
    # test_df = df[df["fold"] == CFG.TEST_FOLD].reset_index(drop=True)

    train_df["label_group"] = labelencoder.fit_transform(train_df["label_group"])

    # print(f"got embeddings! threshold={threshold}")
    # print("image embeddings shape", imagefeat.shape)

    # train = cos_search.search(train,imagefeat,destination="oof_tfidf",threshold=threshold)

    # print("CV score for eca_nfnet_l0_arcface_{threshold}=", train.f1.mean())

    # TODO 从上一次的模型继续训练，需要记录最好成绩
    run_training(train_df, valid_df, test_df, destination, threshold)

    # return df, df.f1.mean()

    
def get_test_embeddings(test_df):
    model = ShopeeModel()
    model.eval()
    model = replace_activations(model, torch.nn.SiLU, Mish()) ###
    model.load_state_dict(torch.load(CFG.MODEL_PATH))
    model = model.to(CFG.DEVICE)
    # model.set_training(False)

    image_dataset = ShopeeImageDataset(test_df,transform=get_test_transforms(),train=False)
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=CFG.BATCH_SIZE,
        pin_memory=True,
        num_workers = CFG.NUM_WORKERS,
        drop_last=False
    )

    embeds = []
    with torch.no_grad():
        for img,label in tqdm(image_loader): 
            img = img.cuda()
            label = label.cuda()
            feat,_ = model(img,label)
            # image_embeddings = feat.detach().cpu().numpy()
            image_embeddings=feat
            embeds.append(image_embeddings)
    
    del model
    # image_embeddings = np.concatenate(embeds)
    image_embeddings = torch.cat(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings

def eval(train, destination="oof_eca_nfnet_l1", threshold=0.36):
    imagefeat=get_test_embeddings(train)
    
    print(f"got embeddings! threshold={threshold}")
    print("image embeddings shape", imagefeat.shape)

    # text_embeddings = text_embeddings.cuda()

    train = get_KNN.get_valid_neighbors(
        train, imagefeat, destination=destination, threshold=threshold
    )

    # train = cos_search.search(
    #     train, imagefeat, destination=destination, threshold=threshold
    # )
    del imagefeat

    print(f"CV score for eca_nfnet_l0_arcface_{threshold} = ", train.f1.mean())

    return train

def show_mode():
    from torchsummary import summary
    model = ShopeeModel()
    model.eval()
    model = replace_activations(model, torch.nn.SiLU, Mish()) ###
    model.load_state_dict(torch.load(CFG.MODEL_PATH))
    model = model.to(CFG.DEVICE)
    
    # sumary(model,(BATCH_SIZE,IMG_SIZE,IMG_SIZE))