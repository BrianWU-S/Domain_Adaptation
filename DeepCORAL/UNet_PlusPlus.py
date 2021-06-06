import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from torch import nn
from torch.nn import functional as F
import torch
import torchvision
from metrics import *
from plot import loss_plot
from plot import metrics_plot
from torchsummary import summary


EPOCHS = 100
BATCH_SIZE = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_transforms = transforms.Compose([
    transforms.ToTensor(),  # pixel value -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                         )  # pixel value ->[-1,1]
])
y_transforms = transforms.ToTensor()  # Just convert label to tensor


class IsbiCellDataset:
    '''
        The data set for DataLoader
    '''

    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'./isbi'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths, self.test_img_paths = None, None, None
        self.train_mask_paths, self.val_mask_paths, self.test_mask_paths = None, None, None
        # get image and label correspond to self.state
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        '''
            Get all the images and labels
        '''
        self.train_img_paths = glob(self.root + r'/train/images/*')
        self.train_mask_paths = glob(self.root + r'/train/label/*')
        self.val_img_paths = glob(self.root + r'/test/images/*')
        self.val_mask_paths = glob(self.root + r'/test/label/*')
        # Validation set is the same as test set
        self.test_img_paths, self.test_mask_paths = self.val_img_paths, self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        '''
            override [] operation
            read and convert image and label
        '''
        pic_path = self.pics[index]  # get certain image and label paths according to index
        mask_path = self.masks[index]
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
        # convert image and label to grayscale images then perform transformations
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def __len__(self):
        return len(self.pics)


class DoubleConv(nn.Module):
    '''
        A module that consists of two convolutional layers
        Will be used in U-Net++
    '''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class NestedUNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        '''
            Definition of blocks in U-Net++
            self.convi_j represents the j-th convolutional blocks in the top i-th pathway of the pyramid
            self.final_i represents the output of uppermost four convolutional blocks mentioned in deep supervision
            For the detailed illustration, please refer to our report
        '''
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]  # filter sizes

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)  # Up-sampling block using bilinear interpolation

        self.conv0_0 = DoubleConv(in_channel, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0])
        self.sigmoid = nn.Sigmoid()

        self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)

    def forward(self, input):
        '''
            The forward process
            Showing the network architecture
        '''
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))  # result of down-sampling
        # output of a block in the top skip pathway that concatenates output from conv0_0 and up-sampled output from conv1_0
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))  # conv1_0->conv2_0
        # conv1_0+conv2_0->conv1_1
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # conv0_0+conv0_1+conv1_1->conv0_2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))  # conv2_0->conv3_0
        # conv2_0+conv3_0->conv2_1
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # conv1_0+conv1_1+conv2_1->conv1_2
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # conv0_0+conv0_1+conv0_2+conv1_2->conv0_3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))  # conv3_0->conv4_0
        # conv3_0+conv4_0->conv3_1
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # conv2_0+conv2_1+conv3_1->conv2_2
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # conv1_0+conv1_1+conv1_2+conv2_2->conv1_3
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))  # conv0_0+conv0_1+conv0_2+conv0_3+conv1_3->conv0_4

        # final outputs
        output1 = self.final1(x0_1)
        output1 = self.sigmoid(output1)
        output2 = self.final2(x0_2)
        output2 = self.sigmoid(output2)
        output3 = self.final3(x0_3)
        output3 = self.sigmoid(output3)
        output4 = self.final4(x0_4)
        output4 = self.sigmoid(output4)

        return [output1, output2, output3, output4]  # deep supervision output


def getLog():
    '''
        log training & testing process
    '''
    filename = './log.log'
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    return logging


def getModel():
    '''
        Get U-Net++ model
    '''
    model = NestedUNet(3, 1).to(device)
    return model


def getDataset():
    '''
        Get training, validation and testing datasets in the form of torch.utils.data.DataLoader
    '''
    train_dataloaders, val_dataloaders, test_dataloaders = None, None, None
    train_dataset = IsbiCellDataset(
        r'train', transform=x_transforms, target_transform=y_transforms)
    train_dataloaders = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataset = IsbiCellDataset(
        r"val", transform=x_transforms, target_transform=y_transforms)
    val_dataloaders = DataLoader(val_dataset, batch_size=1)
    test_dataset = IsbiCellDataset(
        r"test", transform=x_transforms, target_transform=y_transforms)
    test_dataloaders = DataLoader(test_dataset, batch_size=1)
    return train_dataloaders, val_dataloaders, test_dataloaders


def val(model, best_iou, val_dataloaders):
    '''
        Validation process during training
    '''
    model = model.eval()  # set model to evaluation mode
    with torch.no_grad():
        i = 0
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)
        for x, _, pic, mask in val_dataloaders:  # for each image and label in the validation set
            x = x.to(device)
            y = model(x)
            # predict using U-Net++ and use the last output of deep supervision as the prediction
            img_y = torch.squeeze(y[-1]).cpu().numpy()

            # caculate hausdorff distance, iou and dice scores by comparing prediction and label
            hd_total += get_hd(mask[0], img_y)
            miou_total += get_iou(mask[0], img_y)
            dice_total += get_dice(mask[0], img_y)
            if i < num:
                i += 1

        # calculate average scores and log them in the log file
        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total/num
        print('Miou=%f,aver_hd=%f,aver_dice=%f' %
              (aver_iou, aver_hd, aver_dice))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f' %
                     (aver_iou, aver_hd, aver_dice))

        # save best model
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou, best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(
                aver_iou, best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            torch.save(model.state_dict(), r'./saved_model/UNETpp.pth')
        return best_iou, aver_iou, aver_dice, aver_hd


def train(model, criterion, optimizer, train_dataloader, val_dataloader, epochs, threshold):
    '''
        Training process
    '''
    best_iou, aver_iou, aver_dice, aver_hd = 0, 0, 0, 0
    num_epochs = epochs
    threshold = threshold
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []
    for epoch in range(num_epochs):
        model = model.train()  # set model to training mode
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y, _, mask in train_dataloader:  # for each images and labels in training set
            step += 1
            inputs = x.to(device)
            labels = y.to(device)

            optimizer.zero_grad()  # zero the parameter gradients
            outputs = model(inputs)  # get the four outputs of deep supervision
            loss = 0
            for output in outputs:  # implementing accuarate mode: average loss values of outputs of deep supervision as the final loss
                loss += criterion(output, labels)
            loss /= len(outputs)
            if threshold != None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()  # optimization step
                optimizer.step()
                epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) //
                                              train_dataloader.batch_size + 1, loss.item()))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) //
                                                     train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)

        # validate the model
        best_iou, aver_iou, aver_dice, aver_hd = val(
            model, best_iou, val_dataloader)
        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_plot(num_epochs, loss_list)

    # plot training process
    metrics_plot(num_epochs, 'iou&dice', iou_list, dice_list)
    metrics_plot(num_epochs, 'hd', hd_list)
    return model


def test(val_dataloaders, save_predict=False):
    '''
        Test process
    '''
    # load model parameters
    logging.info('final test........')
    if save_predict == True:
        dir = r'./saved_predict'
        if not os.path.exists(dir):
            os.makedirs(dir)
    model.load_state_dict(torch.load(
        r'./saved_model/UNETpp.pth', map_location='cpu'))
    model.eval()

    with torch.no_grad():
        i = 0
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)
        for pic, _, pic_path, mask_path in val_dataloaders:  # for each image and label in the validation set
            pic = pic.to(device)
            predict = model(pic)
            # predict using U-Net++ and use the last output of deep supervision as the prediction
            predict = torch.squeeze(predict[-1]).cpu().numpy()

            # caculate hausdorff distance, iou and dice scores by comparing prediction and label
            iou = get_iou(mask_path[0], predict)
            miou_total += iou
            hd_total += get_hd(mask_path[0], predict)
            dice = get_dice(mask_path[0], predict)
            dice_total += dice

            # plot the prediction label
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            plt.imshow(Image.open(pic_path[0]))
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
            plt.imshow(predict, cmap='Greys_r')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            plt.imshow(Image.open(mask_path[0]), cmap='Greys_r')
            if save_predict == True:
                plt.savefig(dir + '/' + mask_path[0].split('/')[-1])
                np.save(dir + '/' + mask_path[0].split('/')[-1], predict)
            print('iou={},dice={}'.format(iou, dice))
            if i < num:
                i += 1
        print('Miou=%f,aver_hd=%f,dv=%f' %
              (miou_total/num, hd_total/num, dice_total/num))
        logging.info('Miou=%f,aver_hd=%f,dv=%f' %
                     (miou_total/num, hd_total/num, dice_total/num))


if __name__ == '__main__':
    model=getModel()
    summary(model,input_size=(3,512,512))
    # logging = getLog()
    # print('**************************')
    # logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\n========' %
    #              ("UNet++", str(EPOCHS), "4"))
    # print('**************************')
    # model = getModel()
    # train_dataloaders, val_dataloaders, test_dataloaders = getDataset()
    # criterion = torch.nn.BCELoss()  # using BCELoss
    # optimizer = optim.Adam(model.parameters())  # using Adam
    # train(model, criterion, optimizer,
    #       train_dataloaders, val_dataloaders, EPOCHS, None)
    # test(test_dataloaders, save_predict=True)