import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio


def get_iou(mask_name, predict):
    '''
        Calculate IOU metric value between label and prediction
    '''
    # load label
    image_mask = cv2.imread(mask_name, 0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask, (576, 576))
    height = predict.shape[0]
    weight = predict.shape[1]

    # Covert pixel values of both prediction and label to 0/1
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:
                predict[row, col] = 0
            else:
                predict[row, col] = 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 125:
                image_mask[row, col] = 0
            else:
                image_mask[row, col] = 1
    predict = predict.astype(np.int16)

    # compute intersection areas between prediction and label
    interArea = np.multiply(predict, image_mask)
    tem = predict + image_mask
    unionArea = tem - interArea  # compute union areas between prediction and label
    inter = np.sum(interArea)
    union = np.sum(unionArea)
    iou_tem = inter / union  # IOU = intersection / union
    print('%s:iou=%f' % (mask_name, iou_tem))

    return iou_tem


def get_dice(mask_name, predict):
    '''
        Calculate DICE metric value between label and prediction
    '''
    # load label
    image_mask = cv2.imread(mask_name, 0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask, (576, 576))
    height = predict.shape[0]
    weight = predict.shape[1]

    # Covert pixel values of both prediction and label to 0/1
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:
                predict[row, col] = 0
            else:
                predict[row, col] = 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 125:
                image_mask[row, col] = 0
            else:
                image_mask[row, col] = 1
    predict = predict.astype(np.int16)
    # compute intersection areas between prediction and label
    intersection = (predict*image_mask).sum()
    # DICE = 2 * intersection / total
    dice = (2. * intersection) / (predict.sum()+image_mask.sum())
    return dice


def get_hd(mask_name, predict):
    '''
        Calculate hausdorff distance between label and prediction
    '''
    # load label
    image_mask = cv2.imread(mask_name, 0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask, (576, 576))

    # Covert pixel values of both prediction and label to 0/1
    height = predict.shape[0]
    weight = predict.shape[1]
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:
                predict[row, col] = 0
            else:
                predict[row, col] = 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 125:
                image_mask[row, col] = 0
            else:
                image_mask[row, col] = 1

    # compute hausdorff distance in both orders and output the larger one
    hd1 = directed_hausdorff(image_mask, predict)[0]
    hd2 = directed_hausdorff(predict, image_mask)[0]
    res = None
    if hd1 > hd2 or hd1 == hd2:
        res = hd1
        return res
    else:
        res = hd2
        return res


def show(predict):
    '''
        Show the predicted image
    '''
    height = predict.shape[0]
    weight = predict.shape[1]
    for row in range(height):
        for col in range(weight):
            predict[row, col] *= 255
    plt.imshow(predict)
    plt.show()