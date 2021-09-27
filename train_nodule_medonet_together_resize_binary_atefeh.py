#!/usr/bin/env python

import os
import sys
import csv

import keras
import keras.preprocessing.image
import tensorflow as tf

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import layers  
from keras_retinanet import losses
from keras_retinanet import models
from keras_retinanet.callbacks import RedirectModel

from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.image import random_visual_effect_generator
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.utils.image import preprocess_image

from keras_retinanet.models.medonet_MH2 import medo_det_net,det_model_inference

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from keras_retinanet.callbacks.eval import Evaluate

from keras_retinanet.utils.eval import evaluate
import numpy as np

def val_eval(validation_generator,model):
    # iouv = list(np.linspace(0.5,0.95,10))  # iou vector for mAP@0.5:0.95
    # iouv = list(np.linspace(0.35,0.70,8))
    iouv = [0.35,0.5,0.75]
    AP =[]
    f1=[]
    for thresh in iouv:
        F1_score,average_precisions,_ = evaluate(
            validation_generator,
            det_model_inference(model),
            iou_threshold=thresh,
            score_threshold=0.2,
            max_detections=100,
           
        )
        # compute per class average precision
        generator=validation_generator
        total_instances = []
        precisions = []
        F1s =[]
        
        for label, (average_precision, num_annotations) in average_precisions.items():
            
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
            F1s.append(F1_score[label])
      
        mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
        AP.append(mean_ap)

        mean_f1_cls =sum(F1s) / sum(x > 0 for x in total_instances)
        f1.append(mean_f1_cls)

    mean_AP=np.mean(AP)
    print('mAP: {:.4f}'.format(mean_AP))
    # print('f1:',f1,np.mean(f1))

def create_generators(bbx_csv_path_train, cls_csv_path_train, bbx_csv_path_valid, cls_csv_path_valid, preprocess_image):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : 1, #args.batch_size, # 1
        'config'           : '', #args.config, #
        'image_min_side'   : 512, #512,#800, #args.image_min_side, # 800
        'image_max_side'   : 512, #768,#1333, #args.image_max_side, # 1333
        'no_resize'        : False, #'store_true', #args.no_resize, #'store_true'
        'preprocess_image' : preprocess_image, #
    }

    # create random transform generator for augmenting training data
    if True:#args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
        visual_effect_generator = random_visual_effect_generator(
            contrast_range=(0.9, 1.1),
            brightness_range=(-.1, .1),
            hue_range=(-0.05, 0.05),
            saturation_range=(0.95, 1.05)
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)
        visual_effect_generator = None

    if os.path.exists(bbx_csv_path_train) and os.path.exists(cls_csv_path_train):
        train_generator = CSVGenerator(
            bbx_csv_path_train,
            cls_csv_path_train,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )
    else:
        # raise ValueError('Invalid data type received: {}'.format(args.dataset_type))
        raise ValueError('No ')

    if os.path.exists(bbx_csv_path_valid) and os.path.exists(cls_csv_path_valid):
        validation_generator = CSVGenerator(
            bbx_csv_path_valid,
            cls_csv_path_valid,
            shuffle_groups=False,
            **common_args
        )
    else:
        validation_generator = None
    # else:
        # raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


def collect_data_path(im_dirs, bbx_csvs, csv_name):
    print("generating the csv files...")
    im_bbx_path_data = []
    for i in range(len(bbx_csvs)):
        tmp_bbx_csv = bbx_csvs[i]
        with open(tmp_bbx_csv) as csvfile:
            csv_reader = csv.reader(csvfile)

            for row in csv_reader:
                if(i<len(im_dirs)):
                    row[0] = im_dirs[i]+row[0].split('/')[-1]
                row[1] = str(min(int(row[1]),int(row[3])))
                row[2] = str(min(int(row[2]),int(row[4])))
                row[3] = str(max(int(row[1]),int(row[3])))
                row[4] = str(max(int(row[2]),int(row[4])))
                if(row[1]==row[3] or row[2]==row[4]):
                    continue

                im_bbx_path_data.append(row)
        csvfile.close()

    tmp_name = csv_name
    with open(tmp_name,mode='w') as f:
        writer = csv.writer(f)
        for i in range(len(im_bbx_path_data)):
            writer.writerow(im_bbx_path_data[i])
    f.close()

    return tmp_name



def main():

    ## ------ load train data ------ ##
    bbx_csv_path_train = './generated_csv/train_binary.csv'

    # if(not os.path.exists(bbx_csv_path_train)):
    # bbx_csv_path_train = collect_data_path(train_im_dirs, train_bbx_csvs, bbx_csv_path_train)
    cls_csv_path_train = 'cls_binary.csv'


    ## ------ load validation data ------ ##
    bbx_csv_path_valid = './generated_csv/nodule_det_tmp_test_binary.csv'

    # if(not os.path.exists(bbx_csv_path_valid)):
    # bbx_csv_path_valid = collect_data_path(valid_im_dirs, valid_bbx_csvs, bbx_csv_path_valid)
    cls_csv_path_valid = 'cls_binary.csv'

    model_save_path = './saved_models/medo_det_paper_binary_512_512_4.0-test01_MH_coord_biFPN_cbam'
    # model_save_path = './saved_models/medo_det_paper_binary_512_512_4.0-test01_MH_coord_res_cbam'


    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    print("-->>bbx_csv_path_train:",bbx_csv_path_train)
    print("-->>cls_csv_path_train:",cls_csv_path_train)
    print("-->>bbx_csv_path_val:",bbx_csv_path_valid)
    print("-->>cls_csv_path_val:",cls_csv_path_valid)
    print("-->>model_save_path:",model_save_path)

    # create the generators
    train_generator, validation_generator = create_generators(
        bbx_csv_path_train,
        cls_csv_path_train,
        bbx_csv_path_valid,
        cls_csv_path_valid,
        preprocess_image)

    print("---->train generator size:",train_generator.size())
    print("---->validation generator size:",validation_generator.size())

    # build the model for training
    model = medo_det_net(num_classes=train_generator.num_classes(), inputs=None, num_anchors=None, lr=1e-5, is_training=True,config='')
    #model.summary()

    # model.load_weights(os.path.join(model_save_path,'medonet_ens196_loss_3.2686_valloss_4.4180_mAP_0.7682_nodule.h5'))#coord_res_cbam
    model.load_weights(os.path.join(model_save_path,'medonet_ens202_loss_3.2642_valloss_4.4649_mAP_0.7586_nodule.h5'))#coord_biFPN_cbam

    # create callbacks
    callbacks = []
    if validation_generator:
        evaluation = Evaluate(#validation_generator,tensorboard=None,weighted_average=False)
                            validation_generator,
                            iou_threshold=0.35,
                            score_threshold=0.2,
                            max_detections=100,
                            save_path=None,
                            tensorboard=None,
                            weighted_average=False,
                            verbose=1)
        callbacks.append(evaluation)


    checkpoint = ModelCheckpoint(
        os.path.join(model_save_path,'medonet_ens{epoch:02d}_loss_{loss:.4f}_valloss_{val_loss:.4f}_mAP_{mAP:.4f}_nodule.h5'),
        
        # monitor='losses',
        verbose=1,
        save_best_only=False,
        save_weights_only=True
        # monitor="mAP",
        # mode='max'
    )
    checkpoint = RedirectModel(checkpoint, model)
    callbacks.append(checkpoint)

    # start validation
    # val_eval(validation_generator,model)
    
    # start training
    print('start training...')
    return model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.size(),#1000,#args.steps,
        epochs=10000,#args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=1,#args.workers,
        use_multiprocessing=False,#args.multiprocessing,
        max_queue_size=10,#args.max_queue_size,
        validation_data=validation_generator,
        initial_epoch=89#args.initial_epoch
    )

if __name__ == '__main__':
    main()
