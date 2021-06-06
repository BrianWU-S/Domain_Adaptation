import pandas as pd
import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


tf.random.set_seed(2)


def set_GPU_Memory_Limit():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def data_preprocessing():
    # load the data
    sd_features_AA = pd.read_csv(
        r'D:\Google_Download\DS_Basics\Assignments\Assignment4\Dataset\Office-Home_resnet50\Art_Art.csv', header=None)
    sd_features_CC = pd.read_csv(
        r'D:\Google_Download\DS_Basics\Assignments\Assignment4\Dataset\Office-Home_resnet50\Clipart_Clipart.csv',
        header=None)
    sd_features_PP = pd.read_csv(
        r'D:\Google_Download\DS_Basics\Assignments\Assignment4\Dataset\Office-Home_resnet50\Product_Product.csv',
        header=None)
    td_features_AR = pd.read_csv(
        r'D:\Google_Download\DS_Basics\Assignments\Assignment4\Dataset\Office-Home_resnet50\Art_RealWorld.csv',
        header=None)
    td_features_CR = pd.read_csv(
        r'D:\Google_Download\DS_Basics\Assignments\Assignment4\Dataset\Office-Home_resnet50\Clipart_RealWorld.csv',
        header=None)
    td_features_PR = pd.read_csv(
        r'D:\Google_Download\DS_Basics\Assignments\Assignment4\Dataset\Office-Home_resnet50\Product_RealWorld.csv',
        header=None)
    
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


def batch_generator(data, batch_size):
    """Generate batches of data.

    Given a list of numpy data, it iterates over the list and returns batches of the same size
    This
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr


def load_data():
    features_mat = scipy.io.loadmat(
        r'D:\Google_Download\SP_ML_and_DataMining\Project\EEG_emotion_recognition/SEED/EEG_X.mat')
    features = features_mat.get('X')[0]
    labels_mat = scipy.io.loadmat(
        r'D:\Google_Download\SP_ML_and_DataMining\Project\EEG_emotion_recognition/SEED/EEG_Y.mat')
    labels = labels_mat.get('Y')[0]
    print("Loading Data:", np.shape(features), np.shape(labels))
    print("A sample of X has shape:", features[0].shape, "A sample of Y has shape:", labels[0].shape)
    return features, labels


def visualization(ys, yt, Xs_act, Xt_act, Xs_act_t, Xt_act_t):
    plt.scatter(Xs_act[:, 0], Xs_act[:, 1], c=[["lightgreen", "yellow", "blue"][k] for k in ys], alpha=0.4, s=10)
    plt.scatter(Xt_act[:, 0], Xt_act[:, 1], c=[["g", "red", "purple"][k] for k in yt], alpha=0.4, s=10)
    plt.title("Results--Without DANN")
    plt.show()

    plt.scatter(Xs_act_t[:, 0], Xs_act_t[:, 1], c=[["lightgreen", "yellow", "blue"][k] for k in ys], alpha=0.4, s=10)
    plt.scatter(Xt_act_t[:, 0], Xt_act_t[:, 1], c=[["g", "red", "purple"][k] for k in yt], alpha=0.4, s=10)
    plt.title("Results--With DANN")
    plt.show()


