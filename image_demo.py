#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
from keras import backend as K
from imutils.object_detection import non_max_suppression
import argparse
import time

ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", type=str, default=i,
#                help="path to input image")
ap.add_argument("-east", "--east", type=str, default="frozen_east_text_detection.pb",
                help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=544,
                help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=544,
                help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        #return (intersect / (sum_area - intersect)) * 1.0
        return (1.0*intersect/S_rec2)

def text_detection(i):

    # load the input image and grab the image dimensions
    image = cv2.imread(i)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (args["width"], args["height"])
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    # print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(args["east"])

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    # print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    new_boxes = []
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        new_boxes.append([startX,startY,endX,endY])
    #
    #     # draw the bounding box on the image
    #     cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    #
    # # show the output image
    # cv2.imshow("Text Detection", orig)
    # cv2.waitKey(0)
    return new_boxes

def y_filter(original_size,bboxes):
    mask_1 = [box[5]==1.0 for box in bboxes]
    y_gap = original_size[0]*0.3
    mask_2 = [box[1]<y_gap for box in bboxes]
    mask=[mask_1[i] and mask_2[i] for i in range(0,len(bboxes))]
    bboxes = np.array(bboxes)
    bboxes = bboxes[mask]
    bboxes.tolist()
    return bboxes

def another_filter(original_size,bboxes):
    y_gap = original_size[0] * 0.3
    mask_2 = [box[1] < y_gap for box in bboxes]
    bboxes = np.array(bboxes)
    bboxes = bboxes[mask_2]
    bboxes.tolist()
    return bboxes

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
#pb_file         = "./yolov3_coco.pb"
pb_file = "./yolov3_web.pb"
#image_path      = "./docs/images/road.jpeg"
#image_path      = "./image_100/Adobe Inc.+2019-07-28-20`22`08/shot.png"
num_classes     = 2
input_size      = 544
graph           = tf.Graph()
config          = tf.ConfigProto()
config.gpu_options.allow_growth = True

path = "./data/dataset/test.txt"
f=open(path,"r")
image_path = []
lines = f.readlines()
for line in lines:
    line = line.strip("\n")
    line = line.split(" ")
    while (1):
        if line[0].find('.png') != -1:
            image_path.append(line[0])
            break
        else:
            line[0] = line[0] + " " + line[1]
            line.pop(1)

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
with tf.Session(graph=graph, config = config) as sess:
    for i in image_path:
        original_image = cv2.imread(i)

        #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]
        image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]
        #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.1)
        bboxes = utils.nms(bboxes, 0.2, method='nms')

        bboxes = y_filter(original_image_size,bboxes)
        bboxes_2 = text_detection(i)
        bboxes_2 = another_filter(original_image_size,bboxes_2)

        for m in range(0,len(bboxes)):
            for n in range(0,len(bboxes_2)):
                iou = compute_iou(bboxes[m][0:4],bboxes_2[n])
                if(iou>0.7):
                    bboxes[m][0:4] = bboxes_2[n]
        image = utils.draw_bbox(original_image, bboxes)
        #image = Image.fromarray(image)
        store_path = "./data/detection_ocr/"+i.split("/")[-2]+".png"
        cv2.imwrite(store_path,image)
        #image.show()
        #tf.reset_default_graph()
        #K.clear_session()