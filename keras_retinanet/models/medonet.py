import keras
from .. import initializers
from .. import losses
from .. import layers
from ..utils.anchors import AnchorParameters

import keras_resnet
import keras_resnet.models

import tensorflow as tf
from keras.layers import Layer, Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, AveragePooling2D, Add, UpSampling2D, Activation

### ==================================================================###
### -------------------- Define classification head ------------------###

def default_classification_model(
    num_classes,
    num_anchors,
    is_training=True,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    """ Creates the default classification submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    
    outputs = rsu_7(outputs,mid_channels=64,out_channels=pyramid_feature_size,is_training=is_training)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

### -------------------- Define regression head ------------------###

def default_regression_model(num_values, num_anchors, is_training=True, 
    pyramid_feature_size=256, 
    regression_feature_size=256, 
    name='regression_submodel'):
    """ Creates the default regression submodel.

    Args
        num_values              : Number of values to regress.
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """

    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs

    outputs = rsu_7(outputs,mid_channels=64,out_channels=pyramid_feature_size,is_training=is_training)

    outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

### -------------------- Define regression and classification submodels ------------------###

def default_submodels(num_classes, num_anchors, is_training=True):
    """ Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    return [
        ('regression', default_regression_model(4, num_anchors, is_training=is_training)),
        ('classification', default_classification_model(num_classes, num_anchors, is_training=is_training))
    ]
### -------------------- Define regression and classification heads ------------------###


def __build_model_pyramid(name, model, features):
    """ Applies a single submodel to each FPN level.

    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.

    Returns
        A tensor containing the response from the submodel on the FPN features.
    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """ Applies all submodels to each FPN level.

    Args
        models   : List of submodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]

### ================================================================================



def rsu_7(inputs,mid_channels=16,out_channels=64,is_training=True):
    # map_size_list = []

    # input conv
    layer_in = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(inputs)
    #------------------ENCODER----------------#

    #encoder stage 1
    hx1 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(layer_in)
    # map_size_list.append(tf.shape(hx1)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx1)
    #encoder stage 2
    hx2 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx2)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx2)
    #encoder stage 3
    hx3 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx3)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx3)
    #encoder stage 4
    hx4 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx4)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx4)
    #encoder stage 5
    hx5 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx5)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx5)
    #encoder stage 6
    hx6 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx6)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx6)
    #encoder stage 7
    hx7 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    #------------------DECODER----------------#

    #decoder stage 7
    hx7d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx7)
    # hx7dup = MedoResize2D(out_size=map_size_list[5],data_format='channels_last',interpolation='bilinear')(hx7d)
    hx7dup = layers.UpsampleLike(name='hx7d_upsampled_rsu7')([hx7d, hx6])
    #decoder stage 6
    hx6d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx7dup,hx6],axis=3))
    # hx6dup = MedoResize2D(out_size=map_size_list[4],data_format='channels_last',interpolation='bilinear')(hx6d)
    hx6dup = layers.UpsampleLike(name='hx6d_upsampled_rsu7')([hx6d, hx5])
    #decoder stage 5
    hx5d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx6dup,hx5],axis=3))
    # hx5dup = MedoResize2D(out_size=map_size_list[3],data_format='channels_last',interpolation='bilinear')(hx5d)
    hx5dup = layers.UpsampleLike(name='hx5d_upsampled_rsu7')([hx5d, hx4])
    #decoder stage 4
    hx4d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx5dup,hx4],axis=3))
    # hx4dup = MedoResize2D(out_size=map_size_list[2],data_format='channels_last',interpolation='bilinear')(hx4d)
    hx4dup = layers.UpsampleLike(name='hx4d_upsampled_rsu7')([hx4d, hx3])
    #decoder stage 3
    hx3d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx4dup,hx3],axis=3))
    # hx3dup = MedoResize2D(out_size=map_size_list[1],data_format='channels_last',interpolation='bilinear')(hx3d)
    hx3dup = layers.UpsampleLike(name='hx3d_upsampled_rsu7')([hx3d, hx2])
    #decoder stage 2
    hx2d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx3dup,hx2],axis=3))
    # hx2dup = MedoResize2D(out_size=map_size_list[0],data_format='channels_last',interpolation='bilinear')(hx2d)
    hx2dup = layers.UpsampleLike(name='hx2d_upsampled_rsu7')([hx2d, hx1])
    #decoder stage 1
    hx1d = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx2dup,hx1],axis=3))

    return Add()([layer_in, hx1d])

def rsu_6(inputs,mid_channels=16,out_channels=64,is_training=True):

    # map_size_list = []

    # input conv
    layer_in = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(inputs)
    #------------------ENCODER----------------#

    #encoder stage 1
    hx1 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(layer_in)
    # map_size_list.append(tf.shape(hx1)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx1)
    #encoder stage 2
    hx2 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx2)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx2)
    #encoder stage 3
    hx3 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx3)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx3)
    #encoder stage 4
    hx4 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx4)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx4)
    #encoder stage 5
    hx5 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx5)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx5)
    #encoder stage 6
    hx6 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    #------------------DECODER----------------#

    #decoder stage 6
    hx6d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx6)
    # hx6dup = MedoResize2D(out_size=map_size_list[4],data_format='channels_last',interpolation='bilinear')(hx6d)
    hx6dup = layers.UpsampleLike()([hx6d, hx5])
    #decoder stage 5
    hx5d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx6dup,hx5],axis=3))
    # hx5dup = MedoResize2D(out_size=map_size_list[3],data_format='channels_last',interpolation='bilinear')(hx5d)
    hx5dup = layers.UpsampleLike()([hx5d, hx4])
    #decoder stage 4
    hx4d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx5dup,hx4],axis=3))
    # hx4dup = MedoResize2D(out_size=map_size_list[2],data_format='channels_last',interpolation='bilinear')(hx4d)
    hx4dup = layers.UpsampleLike()([hx4d, hx3])
    #decoder stage 3
    hx3d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx4dup,hx3],axis=3))
    # hx3dup = MedoResize2D(out_size=map_size_list[1],data_format='channels_last',interpolation='bilinear')(hx3d)
    hx3dup = layers.UpsampleLike()([hx3d, hx2])
    #decoder stage 2
    hx2d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx3dup,hx2],axis=3))
    # hx2dup = MedoResize2D(out_size=map_size_list[0],data_format='channels_last',interpolation='bilinear')(hx2d)
    hx2dup = layers.UpsampleLike()([hx2d, hx1])
    #decoder stage 1
    hx1d = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx2dup,hx1],axis=3))

    return Add()([layer_in, hx1d])

def rsu_5(inputs,mid_channels=16,out_channels=64,is_training=True):

    # map_size_list = []
    # input conv
    layer_in = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(inputs)
    #------------------ENCODER----------------#

    #encoder stage 1
    hx1 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(layer_in)
    # map_size_list.append(tf.shape(hx1)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx1)
    #encoder stage 2
    hx2 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx2)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx2)
    #encoder stage 3
    hx3 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx3)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx3)
    #encoder stage 4
    hx4 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx4)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx4)
    #encoder stage 5
    hx5 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    #------------------DECODER----------------#

    #decoder stage 5
    hx5d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx5)
    # hx5dup = MedoResize2D(out_size=map_size_list[3],data_format='channels_last',interpolation='bilinear')(hx5d)
    hx5dup = layers.UpsampleLike()([hx5d, hx4])
    #decoder stage 4
    hx4d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx5dup,hx4],axis=3))
    # hx4dup = MedoResize2D(out_size=map_size_list[2],data_format='channels_last',interpolation='bilinear')(hx4d)
    hx4dup = layers.UpsampleLike()([hx4d, hx3])
    #decoder stage 3
    hx3d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx4dup,hx3],axis=3))
    # hx3dup = MedoResize2D(out_size=map_size_list[1],data_format='channels_last',interpolation='bilinear')(hx3d)
    hx3dup = layers.UpsampleLike()([hx3d, hx2])
    #decoder stage 2
    hx2d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx3dup,hx2],axis=3))
    # hx2dup = MedoResize2D(out_size=map_size_list[0],data_format='channels_last',interpolation='bilinear')(hx2d)
    hx2dup = layers.UpsampleLike()([hx2d, hx1])
    #decoder stage 1
    hx1d = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx2dup,hx1],axis=3))

    return Add()([layer_in, hx1d])

def rsu_4(inputs,mid_channels=16,out_channels=64,is_training=True):

    # map_size_list = []
    # input conv
    layer_in = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(inputs)
    #------------------ENCODER----------------#

    #encoder stage 1
    hx1 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(layer_in)
    # map_size_list.append(tf.shape(hx1)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx1)
    #encoder stage 2
    hx2 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx2)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx2)
    #encoder stage 3
    hx3 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    # map_size_list.append(tf.shape(hx3)[1:3])
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx3)
    #encoder stage 4
    hx4 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    #------------------DECODER----------------#

    #decoder stage 4
    hx4d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx4)
    # hx4dup = MedoResize2D(out_size=map_size_list[2],data_format='channels_last',interpolation='bilinear')(hx4d)
    hx4dup = layers.UpsampleLike()([hx4d, hx3])
    #decoder stage 3
    hx3d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx4dup,hx3],axis=3))
    # hx3dup = MedoResize2D(out_size=map_size_list[1],data_format='channels_last',interpolation='bilinear')(hx3d)
    hx3dup = layers.UpsampleLike()([hx3d, hx2])
    #decoder stage 2
    hx2d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx3dup,hx2],axis=3))
    # hx2dup = MedoResize2D(out_size=map_size_list[0],data_format='channels_last',interpolation='bilinear')(hx2d)
    hx2dup = layers.UpsampleLike()([hx2d, hx1])
    #decoder stage 1
    hx1d = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx2dup,hx1],axis=3))

    return Add()([layer_in, hx1d])


def medo_backbone(inputs, is_training=True):

    # inputs = Input(shape=(None, None, 1), name='net_input') # one channel 1/1
    C1 = Conv2D(64, (7, 7), strides=(2,2), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(inputs) #1/2
    hx = MaxPooling2D(pool_size=[2,2],strides=[2,2],padding='same')(C1) # 1/4

    # map_size_list = []

    #encoder stage 1
    C2 = rsu_7(hx,mid_channels=32,out_channels=128,is_training=is_training) #1/4
    # map_size_list.append(tf.shape(hx1)[1:3])
    hx = MaxPooling2D(pool_size=[2,2],strides=[2,2],padding='same')(C2)
    #encoder stage 2
    C3 = rsu_6(hx,mid_channels=64,out_channels=256,is_training=is_training) #1/8
    # map_size_list.append(tf.shape(hx2)[1:3])
    hx = MaxPooling2D(pool_size=[2,2],strides=[2,2],padding='same')(C3)
    #encoder stage 3
    C4 = rsu_5(hx,mid_channels=128,out_channels=512,is_training=is_training) #1/16
    # map_size_list.append(tf.shape(hx3)[1:3])
    hx = MaxPooling2D(pool_size=[2,2],strides=[2,2],padding='same')(C4)
    #encoder stage 4
    C5 = rsu_4(hx,mid_channels=256,out_channels=1024,is_training=is_training) #1/32

    return [C3,C4,C5]

def medo_det_net(num_classes, inputs=None, num_anchors=None, submodels=None, lr=1e-5, is_training=True, **kwargs):
    """ Constructs a medonet model using a resnet backbone.

    Args
        num_classes: Number of classes to predict.
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).

    Returns
        Medonet model with a medonet backbone.
    """
    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = Input(shape=(3, None, None))
        else:
            inputs = Input(shape=(None, None, 3))

    # create anchors
    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors, is_training=is_training)

    # get features from backbone net
    C3, C4, C5 = medo_backbone(inputs,is_training=is_training)

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = __create_pyramid_features(C3, C4, C5, 256)

    # for all pyramid levels, run available submodels
    pyramids = __build_pyramid(submodels, features)
    model = keras.models.Model(inputs=inputs, outputs=pyramids, name='medo_det_net')

    # compile model
    model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
    )

    return model

#--------------------------------------------------------------------------

def __build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)

## function for generate the bboxes and regression results
def det_model_inference(
    model                 = None,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'medo-det-net-infer',
    anchor_params         = None,
    **kwargs
):
    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # # create medonet model for inference

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    anchors  = __build_anchors(anchor_params, features)

    # we expect the anchors, regression and classification values as first output
    regression     = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])


    detections = layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        score_threshold       = 0.2,
        nms_threshold         = 0.3,
        name                  = 'filtered_detections'
    )([boxes, classification] + other)

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)
