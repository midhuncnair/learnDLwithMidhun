#! /usr/bin/env python3
"""
"""


__all__ = [
    'plot_layer_outputs',
    'imshow',
    'apply_patch',
    'IoU_metric',
]
__license__ = "MIT"
__author__ = "Midhun C Nair"


import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.backend import epsilon
from matplotlib import pyplot as plt
from matplotlib import patches


LOCAL_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'images', 'classification'
)


def plot_layer_outputs(model, target_size, img=None):
    if img is None:
        img = os.path.join(
            LOCAL_DATA_PATH, 'flowers',
            'Train', 'roses', '145862135_ab710de93c_n.jpg'
        )
    layer_outputs = [layer.output for layer in model.layers[:]]
    vmodel = Model(inputs=model.input, outputs=layer_outputs)

    # load the image with the TARGET_SIZE
    img = load_img(img, target_size=target_size)
    x = img_to_array(img)
    x = x.reshape((1, *x.shape))
    # Rescale by 1/255
    x /= 255.0

    feature_maps = vmodel.predict(x)
    layer_names = [layer.name for layer in model.layers]
    for layer_name, feature_map in zip(layer_names, feature_maps):
        if len(feature_map.shape) == 4:
            # from summary we know the conv layers have feature shape len of 4.
            # shape => (1, height, width, features)
            size = feature_map.shape[1]
            n_features = feature_map.shape[-1]
            display_grid = np.zeros((size, size * n_features))

            for i in range(n_features):
                x = feature_map[0, :, :, i]
                # normalize the pixels
                x = (x - x.mean()) / x.std()  # z-score
                x = np.clip(x, 0, 255).astype('float')
                x = np.log(x / 255)  # .astype('uint8')  # scale and enhance low-details
                display_grid[:, i * size: (i + 1) * size] = x

            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            cmap = 'viridis'
#             cmap = 'gray'
            plt.imshow(display_grid, aspect='auto', cmap=cmap)


def imshow(img, inline=True):
    if inline is True:
        if img[img > 1].sum() > 0:
            img.astype('int')  # for float the the max value is 0
        cmap = None
        if len(img.shape) <= 2:
            cmap = 'gray'
        plt.imshow(img, cmap=cmap)
        plt.plot()
    else:
        cv2.imshow('img', img)
        while True:
            k = cv2.waitKey(1)
            if k == -1:  # if no key was pressed, -1 is returned
                continue
            else:
                break
        cv2.destroyWindow('img')
        cv2.destroyAllWindows()


def apply_patch(img, x0, y0, x1, y1):
    _, ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=2, edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)
    plt.show()


def custom_iou(y_act, y_pred):
    intersections = 0
    unions = 0

    gt = y_act  # ground truth; fancy name for actual
    pred = y_pred

    # Compute interection b/w pred and act.
    diff_width = np.minimum(
        gt[:, 0] + gt[:, 2],  # (x0 + height) -> x1 of act
        pred[:, 0] + pred[:, 2]  # (x0 + height) -> x1 of pred
    ) - np.maximum(
        gt[:, 0],  # x0 of act
        pred[:, 0]  # x0 of pred
    )
    diff_height = np.minimum(
        gt[:, 1] + gt[:, 3],  # (y0 + height) -> y1 of act
        pred[:, 1] + pred[:, 3]  # (y0 + height) -> y1 of pred
    ) - np.maximum(
        gt[:, 1],  # y0 of act
        pred[:, 1]  # y0 of pred
    )
    area_intersection = diff_width * diff_height  # this now is a 1d array

    # Compute union ==> area of pred + area of actual - area of intersection
    #    area of pred + area of actual ==> will have `area of intersection` 2 times;
    area_gt = gt[:, 2] * gt[:, 3]
    area_pred = pred[:, 2] * pred[:, 3]
    area_union = area_gt + area_pred - area_intersection  # this now is a 1d array

    # Compute area of intersection and area of union over the batch
    for j, _ in enumerate(area_union):
        if (  # basic sanity to make sure there is a valid intersection and union
            area_union[j] > 0  # this makes sure there is bounding box
            and area_intersection[j] > 0  # this makes sure there is an intersection
            and area_union[j] >= area_intersection[j]  # this makes sure intersection and union are valid.
        ):
            # this will accumulate the 1d array as single value;
            # we cannot do np.sum because of the validation.
            intersections += area_intersection[j]
            unions += area_union[j]

    # Compute IOU. Use epsilon to prevent division by zero
    iou = np.round(
        intersections / (
            unions + epsilon()
        ),
        4  # upto 4 decimal points
    )
    # This must match the type used in py_func
    iou = iou.astype(np.float32)
    return iou


def IoU_metric(y_act, y_pred):
    iou = tf.py_function(
        custom_iou, (y_act, y_pred),
        Tout=tf.float32
    )
    return iou
