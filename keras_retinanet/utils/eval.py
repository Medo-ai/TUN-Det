"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations,draw_caption

import keras
import numpy as np
import os
import time

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

from .WBF import weighted_boxes_fusion

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def pre_process(boxes,scores,labels, scale,img_shape, score_threshold,max_detections):
        
    # correct boxes for image scale
    boxes /= scale
    boxes /= [np.array([img_shape[1],img_shape[0],img_shape[1],img_shape[0]])]
    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those scores
    scores = scores[0][indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)[:max_detections]

    # select detections
    image_boxes      = boxes[0, indices[scores_sort], :]
    image_scores     = scores[scores_sort]
    image_labels     = labels[0, indices[scores_sort]]
    return image_boxes, image_scores, image_labels

def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]
    all_inferences = [None for i in range(generator.size())]


    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)
        img_shape= raw_image.shape[0:2]

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))
        
        boxes_list=[]
        scores_list=[]
        labels_list=[]
        scale = np.array([scale[1],scale[0],scale[1],scale[0]])
        scale = np.reshape(scale,(1,4))
        # run network
        start = time.time()
        predictions = model.predict_on_batch(np.expand_dims(image, axis=0))
        inference_time = time.time() - start

        boxes0, scores0, labels0 = predictions[0], predictions[1], predictions[2]
        boxes1, scores1, labels1 = predictions[3], predictions[4], predictions[5]
        boxes2, scores2, labels2 = predictions[6], predictions[7], predictions[8]
        

        image_boxes, image_scores, image_labels =pre_process(boxes0,scores0,labels0, scale, img_shape, score_threshold,max_detections)
        boxes_list.append(image_boxes)
        labels_list.append(image_labels)
        scores_list.append(image_scores)

        image_boxes, image_scores, image_labels =pre_process(boxes1,scores1,labels1, scale, img_shape, score_threshold,max_detections)
        boxes_list.append(image_boxes)
        labels_list.append(image_labels)
        scores_list.append(image_scores)

        image_boxes, image_scores, image_labels =pre_process(boxes2,scores2,labels2, scale, img_shape, score_threshold,max_detections)
        boxes_list.append(image_boxes)
        labels_list.append(image_labels)
        scores_list.append(image_scores)

        boxes, scores, labels = weighted_boxes_fusion(boxes_list,scores_list,labels_list,iou_thr=0.3)
        boxes *= [np.array([raw_image.shape[1],raw_image.shape[0],raw_image.shape[1],raw_image.shape[0]])]
        
        indices = np.where(scores > 0.05)[0]
        scores = scores[indices]
        scores_sort = np.argsort(-scores)[:max_detections]
        image_boxes      = boxes[indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[indices[scores_sort]]

        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name, score_threshold=score_threshold)

            cv2.imwrite(os.path.join(save_path,generator.image_path(i).split('/')[-1]),raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        all_inferences[i] = inference_time

    return all_detections, all_inferences


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, all_inferences = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    average_precisions = {}

    F1_score = {}


    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label].astype('double')
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            debug=cv2.imread(generator.image_path(i))
            for ann in annotations:
                bbox = ann[0:4].astype(int)
                cv2.rectangle(debug,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2, cv2.LINE_AA)

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]


                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    
                    
                # bbox = d[0:4].astype(int)
                # cv2.rectangle(debug,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2, cv2.LINE_AA)
                # caption = "{:.3f}".format(d[4])
                # draw_caption(debug, bbox, caption)
                # cv2.imwrite('/debug/'+generator.image_path(i).split('/')[-3]+'_'+generator.image_path(i).split('/')[-2]+'_'+generator.image_path(i).split('/')[-1][:-4]+'.png',debug)
                

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # F1 =(2*precision*recall)/(precision+recall)
        # F1_scores[label]=F1[-1],num_annotations
        # print('f1',F1[-1])
        pr_score = 0.2
        conf=scores[indices]
        re= np.interp(-pr_score, -conf, recall)
        pr= np.interp(-pr_score, -conf, precision)
        f1 = (2 * pr * re) / (pr + re + 1e-16)
        F1_score[label]=f1 #re
        # print('recall=',re)
        # print('precision=',pr)
        # print(np.mean(recall),np.std(recall))
        # print(np.mean(precision),np.std(precision))

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
        # print('average_precision',average_precision)
        print('average_recall',np.mean(recall))

    # inference time
    inference_time = np.sum(all_inferences) / generator.size()
    print('inference time',inference_time)

    return F1_score,average_precisions,inference_time
