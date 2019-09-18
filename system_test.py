import os
import cv2
import os
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
from imutils.object_detection import non_max_suppression
import argparse
import time
from scipy.optimize import linear_sum_assignment
import math
import image_perfect_match as ipm


def test_system():
    image_path = "./gui/imgs/shot.png"
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    pb_file = "./yolov3_web.pb"
    num_classes = 2
    input_size = 544
    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
    with tf.Session(graph=graph, config=config) as sess:
        try:
            original_image = cv2.imread(image_path)
            ocr_boxes, yolo_boxes = predict_logo_boxes(image_path, input_size, num_classes, original_image,
                                                            return_tensors, sess)

            # TODO retrieve images
            image_list = write_cropped_images(image_path, ocr_boxes, original_image, yolo_boxes)

            # TODO connect to Yuxuan


        except Exception as e:
            print(e)


def write_cropped_images(image_path, ocr_boxes, original_image, yolo_boxes):
    te = image_path.split("/")[-1]
    te = te.split(".png")[0]
    file_path = "./tmp_box/" + te
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    for k in range(0, len(yolo_boxes)):
        crop_img = original_image[int(yolo_boxes[k][1]):int(yolo_boxes[k][3]),
                   int(yolo_boxes[k][0]):int(yolo_boxes[k][2])]
        cv2.imwrite(file_path + "//" + str(k) + ".png", crop_img)
        crop_img_2 = original_image[int(ocr_boxes[k][1]):int(ocr_boxes[k][3]),
                     int(ocr_boxes[k][0]):int(ocr_boxes[k][2])]
        cv2.imwrite(file_path + "//" + str(k) + "_ocr.png", crop_img_2)


def predict_logo_boxes(image_path, input_size, num_classes, original_image, return_tensors, sess):
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
        feed_dict={return_tensors[0]: image_data})
    pred_yolo_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                     np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                     np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
    post_yolo_boxes = utils.postprocess_boxes(pred_yolo_bbox, original_image_size, input_size, 0.1)
    post_yolo_boxes = utils.nms(post_yolo_boxes, 0.2, method='nms')
    post_yolo_boxes = ipm.logo_filter(post_yolo_boxes)
    ocr_boxes = ipm.text_detection(image_path)

    # cost_match_matrix refers to the distance pairs between the yolo boxes and ocr boxes
    cost_match_matrix = ipm.cost_calculating(post_yolo_boxes, ocr_boxes)

    row_ind, col_ind = linear_sum_assignment(cost_match_matrix)
    ocr_boxes = [ocr_boxes[t] for t in col_ind]
    yolo_boxes = [post_yolo_boxes[t] for t in row_ind]

    distance = cost_match_matrix[row_ind, col_ind]
    mask = [d < 100 for d in distance]

    yolo_boxes = np.array(yolo_boxes)
    ocr_boxes = np.array(ocr_boxes)

    yolo_boxes = yolo_boxes[mask]
    yolo_boxes = yolo_boxes.tolist()

    ocr_boxes = ocr_boxes[mask]
    ocr_boxes = ocr_boxes.tolist()

    return ocr_boxes, post_yolo_boxes


if __name__ == "__main__":
    test_system()
    print("Everything passed")
