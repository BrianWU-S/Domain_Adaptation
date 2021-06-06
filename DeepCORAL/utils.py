import os
import torch
from torchvision import transforms, datasets
import torch.utils.data as Data
import pandas as pd


def data_preprocessing():
    # load the data
    sd_features_AA = pd.read_csv(r'D:\Google_Download\DS_Basics\Assignments\Assignment4\Dataset\Office-Home_resnet50\Art_Art.csv', header=None)
    sd_features_CC = pd.read_csv(r'D:\Google_Download\DS_Basics\Assignments\Assignment4\Dataset\Office-Home_resnet50\Clipart_Clipart.csv', header=None)
    sd_features_PP = pd.read_csv(r'D:\Google_Download\DS_Basics\Assignments\Assignment4\Dataset\Office-Home_resnet50\Product_Product.csv', header=None)
    td_features_AR = pd.read_csv(r'D:\Google_Download\DS_Basics\Assignments\Assignment4\Dataset\Office-Home_resnet50\Art_RealWorld.csv', header=None)
    td_features_CR = pd.read_csv(r'D:\Google_Download\DS_Basics\Assignments\Assignment4\Dataset\Office-Home_resnet50\Clipart_RealWorld.csv', header=None)
    td_features_PR = pd.read_csv(r'D:\Google_Download\DS_Basics\Assignments\Assignment4\Dataset\Office-Home_resnet50\Product_RealWorld.csv', header=None)
    
    sd_labels_AA = sd_features_AA.iloc[:, -1].astype(int)
    sd_labels_CC = sd_features_CC.iloc[:, -1].astype(int)
    sd_labels_PP = sd_features_PP.iloc[:, -1].astype(int)
    td_labels_AR = td_features_AR.iloc[:, -1].astype(int)
    td_labels_CR = td_features_CR.iloc[:, -1].astype(int)
    td_labels_PR = td_features_PR.iloc[:, -1].astype(int)
    
    sd_features_AA.drop(labels=2048, axis=1, inplace=True)
    sd_features_CC.drop(labels=2048, axis=1, inplace=True)
    sd_features_PP.drop(labels=2048, axis=1, inplace=True)
    td_features_AR.drop(labels=2048, axis=1, inplace=True)
    td_features_CR.drop(labels=2048, axis=1, inplace=True)
    td_features_PR.drop(labels=2048, axis=1, inplace=True)
    
    # convert to numpy
    sd_features_AA = sd_features_AA.to_numpy()
    sd_features_CC = sd_features_CC.to_numpy()
    sd_features_PP = sd_features_PP.to_numpy()
    td_features_AR = td_features_AR.to_numpy()
    td_features_CR = td_features_CR.to_numpy()
    td_features_PR = td_features_PR.to_numpy()
    sd_labels_AA = sd_labels_AA.to_numpy()
    sd_labels_CC = sd_labels_CC.to_numpy()
    sd_labels_PP = sd_labels_PP.to_numpy()
    td_labels_AR = td_labels_AR.to_numpy()
    td_labels_CR = td_labels_CR.to_numpy()
    td_labels_PR = td_labels_PR.to_numpy()
    
    return sd_features_AA, sd_features_CC, sd_features_PP, \
           td_features_AR, td_features_CR, td_features_PR, \
           sd_labels_AA, sd_labels_CC, sd_labels_PP, \
           td_labels_AR, td_labels_CR, td_labels_PR


def load_data(root_path, domain, batch_size, phase):
    """
    All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of
     shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a
     range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
    """
    transform_dict = {
        'src': transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ]),  # normalize for each channel
        'tar': transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])}
    data = datasets.ImageFolder(root=os.path.join(root_path, domain), transform=transform_dict[phase])
    data_loader = Data.DataLoader(data, batch_size=batch_size, shuffle=phase == 'src',
                                  drop_last=phase == 'tar', num_workers=4)
    return data_loader


def load_feature(Xs, Ys, Xt, Yt, batch_size):
    Xs = torch.from_numpy(Xs)
    Ys = torch.from_numpy(Ys)
    Xt = torch.from_numpy(Xt)
    Yt = torch.from_numpy(Yt)
    source_dataset = Data.TensorDataset(Xs.float(), Ys.float())
    target_dataset = Data.TensorDataset(Xt.float(), Yt.float())
    source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True)
    return source_loader, target_loader


