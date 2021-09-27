import keras
from .. import initializers
from .. import losses
from .. import layers
from ..utils.anchors import AnchorParameters

import keras_resnet
import keras_resnet.models
from tensorflow.keras import backend as K

import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Lambda, Concatenate, Dense, multiply
from keras.layers import  Layer, Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, AveragePooling2D, Add, UpSampling2D, Activation


### -------------------- Define CoordConv classification head ------------------###

def default_classification_model_coord(
    num_classes,
    num_anchors,
    is_training=True,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel_coord'
):
    """ Creates the default classification submodel for CoordConv.

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

    outputs = rsu_7_coord(outputs,mid_channels=64,out_channels=pyramid_feature_size,is_training=is_training)
 

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

### -------------------- Define CoordConv regression head ------------------###

def default_regression_model_coord(num_values, num_anchors, is_training=True, 
    pyramid_feature_size=256, 
    regression_feature_size=256, 
    name='regression_submodel_coord'):
    """ Creates the default regression submodel for CoordConv.

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

    outputs = rsu_7_coord(outputs,mid_channels=64,out_channels=pyramid_feature_size,is_training=is_training)
   

    outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

### -------------------- Define BiFPN classification head ------------------###

def default_classification_model_biFPN(
    num_classes,
    num_anchors,
    is_training=True,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel_bifpn'
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
 
    outputs = rsu_7_biFPN(outputs,mid_channels=64,out_channels=pyramid_feature_size,is_training=is_training)



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

### -------------------- Define BiFPN regression head ------------------###

def default_regression_model_biFPN(num_values, num_anchors, is_training=True, 
    pyramid_feature_size=256, 
    regression_feature_size=256, 
    name='regression_submodel_bifpn'):
    """ Creates the default regression submodel for BiFPN.

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

    outputs = rsu_7_biFPN(outputs,mid_channels=64,out_channels=pyramid_feature_size,is_training=is_training)

    outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

### -------------------- Define CBAM classification head ------------------###

def default_classification_model_cbam(
    num_classes,
    num_anchors,
    is_training=True,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel_cbam'
):
    """ Creates the default classification submodel for CBAM.

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
    outputs= cbam_block(outputs)


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

### -------------------- Define CBAM regression head ------------------###

def default_regression_model_cbam(num_values, num_anchors, is_training=True, 
    pyramid_feature_size=256, 
    regression_feature_size=256, 
    name='regression_submodel_cbam'):
    """ Creates the default regression submodel for CBAM.

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
    outputs= cbam_block(outputs)

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
        ('regression_coord', default_regression_model_coord(4, num_anchors, is_training=is_training)),
        ('classification_coord', default_classification_model_coord(num_classes, num_anchors, is_training=is_training)),
        ('regression_bifpn', default_regression_model_biFPN(4, num_anchors, is_training=is_training)),
        ('classification_bifpn', default_classification_model_biFPN(num_classes, num_anchors, is_training=is_training)),
        ('regression_cbam', default_regression_model_cbam(4, num_anchors, is_training=is_training)),
        ('classification_cbam', default_classification_model_cbam(num_classes, num_anchors, is_training=is_training))
    ]



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

def coord(im):
    shp = tf.cast(tf.shape(im)[1:3],dtype=tf.float32)

    # each element in 'row' are the column number of the pixel
    # row = tf.linspace(tf.cast(0,dtype=tf.float32),tf.cast(tf.math.subtract(shp[1],1.0),dtype=tf.float32),tf.cast(shp[1],dtype=tf.int32))
    row = tf.linspace(tf.cast(-0.5,dtype=tf.float32),tf.cast(0.5,dtype=tf.float32),tf.cast(shp[1],dtype=tf.int32))
    row = tf.reshape(row,[1,shp[1]])
    row = tf.tile(row,[shp[0],1])
    row = tf.expand_dims(row,axis=2)
    row = tf.expand_dims(row,axis=0)
    row = tf.tile(row,[tf.shape(im)[0],1,1,1])

    # each element in 'col' are the row number of the pixel
    # col = tf.linspace(tf.cast(0,dtype=tf.float32),tf.cast(tf.math.subtract(shp[0],1.0),dtype=tf.float32),tf.cast(shp[0],dtype=tf.int32))
    col = tf.linspace(tf.cast(-0.5,dtype=tf.float32),tf.cast(0.5,dtype=tf.float32),tf.cast(shp[0],dtype=tf.int32))
    col = tf.reshape(col,[shp[0],1])
    col = tf.tile(col,[1,shp[1]])
    col = tf.expand_dims(col,axis=2)
    col = tf.expand_dims(col,axis=0)
    col = tf.tile(col,[tf.shape(im)[0],1,1,1])

    ###Adding conv for coordinate channel layers-Dec 16/2020
    coord = tf.concat([im,row,col],axis=3)
    return coord
def coord_layer(im):
    return Lambda(lambda x: coord(x))(im)


def rsu_7_coord(inputs,mid_channels=16,out_channels=64,is_training=True):

    # input conv
    inputs_coord =coord_layer(inputs)
    layer_in = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(inputs_coord)
    #------------------ENCODER----------------#

    #encoder stage 1
    layer_in_coord = coord_layer(layer_in)
    hx1 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(layer_in_coord)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx1)
    #encoder stage 2
    hx = coord_layer(hx)
    hx2 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx2)
    #encoder stage 3
    hx = coord_layer(hx)
    hx3 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx3)
    #encoder stage 4
    hx = coord_layer(hx)
    hx4 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx4)
    #encoder stage 5
    hx = coord_layer(hx)
    hx5 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx5)
    
    #encoder stage 6
    hx = coord_layer(hx)
    hx6 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx6)
    #encoder stage 7
    hx = coord_layer(hx)
    hx7 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    #------------------DECODER----------------#

    #decoder stage 7
    hx7 = coord_layer(hx7)
    hx7d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx7)
    
    hx7dup = layers.UpsampleLike(name='hx7d_upsampled_rsu7_c')([hx7d, hx6])
    #decoder stage 6
    hxcat = coord_layer(concatenate([hx7dup,hx6],axis=3))
    hx6d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hxcat)

    hx6dup = layers.UpsampleLike(name='hx6d_upsampled_rsu7_c')([hx6d, hx5])
    #decoder stage 5
    hxcat = coord_layer(concatenate([hx6dup,hx5],axis=3))
    hx5d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hxcat)

    hx5dup = layers.UpsampleLike(name='hx5d_upsampled_rsu7_c')([hx5d, hx4])
    #decoder stage 4
    hxcat = coord_layer(concatenate([hx5dup,hx4],axis=3))
    hx4d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hxcat)

    hx4dup = layers.UpsampleLike(name='hx4d_upsampled_rsu7_c')([hx4d, hx3])
    #decoder stage 3
    hxcat = coord_layer(concatenate([hx4dup,hx3],axis=3))
    hx3d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hxcat)

    hx3dup = layers.UpsampleLike(name='hx3d_upsampled_rsu7_c')([hx3d, hx2])
    #decoder stage 2
    hxcat = coord_layer(concatenate([hx3dup,hx2],axis=3))
    hx2d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hxcat)

    hx2dup = layers.UpsampleLike(name='hx2d_upsampled_rsu7_c')([hx2d, hx1])
    #decoder stage 1
    hxcat = coord_layer(concatenate([hx2dup,hx1],axis=3))
    hx1d = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hxcat)

    return Add()([layer_in, hx1d])


###================================================================================
def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat) 
    assert cbam_feature._keras_shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])


### ================================================================================
def rsu_7_biFPN(inputs,mid_channels=16,out_channels=64,is_training=True):

    # input conv
    layer_in = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(inputs)
    #------------------ENCODER----------------#

    #encoder stage 1
    hx1 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(layer_in)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx1)
    #encoder stage 2
    hx2 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx2)
    #encoder stage 3
    hx3 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx3)
    #encoder stage 4
    hx4 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx4)
    #encoder stage 5
    hx5 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx5)
    #encoder stage 6
    hx6 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx6)
    #encoder stage 7
    hx7 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    #------------------BIFPN LAYER-----------------#
    bhx7up = layers.UpsampleLike(name='hx7_biFPN_up')([hx7, hx6])
    bhx6 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([bhx7up,hx6],axis=3))
    bhx6up = layers.UpsampleLike(name='hx6_biFPN_up')([bhx6, hx5])
    #decoder stage 5
    bhx5 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([bhx6up,hx5],axis=3))
    bhx5up = layers.UpsampleLike(name='hx5_biFPN_up')([bhx5, hx4])
    #decoder stage 4
    bhx4 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([bhx5up,hx4],axis=3))
    bhx4up = layers.UpsampleLike(name='hx4_biFPN_up')([bhx4, hx3])
    #decoder stage 3
    bhx3 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([bhx4up,hx3],axis=3))
    
    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(bhx3)
    bhx4cat = Conv2D(mid_channels, (1, 1), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([bhx4,hx],axis=3))
    bhx4d = Add()([bhx4cat,hx4])
    bhx4d =  Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(bhx4d)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(bhx4d)
    bhx5cat = Conv2D(mid_channels, (1, 1), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([bhx5,hx],axis=3))
    bhx5d = Add()([bhx5cat,hx5])
    bhx5d =  Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(bhx5d)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(bhx5d)
    bhx5cat = Conv2D(mid_channels, (1, 1), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([bhx6,hx],axis=3))
    bhx6d = Add()([bhx5cat,hx6])
    bhx6d =  Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(bhx6d)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(bhx6d)
    bhx7d =  Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx7,hx],axis=3))
    #------------------DECODER----------------#

    #decoder stage 7
    hx7dup = layers.UpsampleLike(name='bhx7d_upsampled_rsu7')([bhx7d, bhx6d])
    #decoder stage 6
    hx6d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx7dup,bhx6d],axis=3))

    hx6dup = layers.UpsampleLike(name='bhx6d_upsampled_rsu7')([hx6d, bhx5d])
    #decoder stage 5
    hx5d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx6dup,bhx5d],axis=3))

    hx5dup = layers.UpsampleLike(name='bhx5d_upsampled_rsu7')([hx5d, bhx4d])
    #decoder stage 4
    hx4d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx5dup,bhx4d],axis=3))

    hx4dup = layers.UpsampleLike(name='bhx4d_upsampled_rsu7')([hx4d, bhx3])
    #decoder stage 3
    hx3d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx4dup,bhx3],axis=3))

    hx3dup = layers.UpsampleLike(name='bhx3d_upsampled_rsu7')([hx3d, hx2])
    #decoder stage 2
    hx2d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx3dup,hx2],axis=3))

    hx2dup = layers.UpsampleLike(name='bhx2d_upsampled_rsu7')([hx2d, hx1])
    #decoder stage 1
    hx1d = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx2dup,hx1],axis=3))

    return Add()([layer_in, hx1d])

###================================================================================= ensemble 


def rsu_7(inputs,mid_channels=16,out_channels=64,is_training=True):

    # input conv
    layer_in = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(inputs)
    #------------------ENCODER----------------#

    #encoder stage 1
    hx1 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(layer_in)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx1)
    #encoder stage 2
    hx2 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx2)
    #encoder stage 3
    hx3 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx3)
    #encoder stage 4
    hx4 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx4)
    #encoder stage 5
    hx5 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx5)
    #encoder stage 6
    hx6 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx6)
    #encoder stage 7
    hx7 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    #------------------DECODER----------------#

    #decoder stage 7
    hx7d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx7)

    hx7dup = layers.UpsampleLike(name='hx7d_upsampled_rsu7')([hx7d, hx6])
    #decoder stage 6
    hx6d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx7dup,hx6],axis=3))

    hx6dup = layers.UpsampleLike(name='hx6d_upsampled_rsu7')([hx6d, hx5])
    #decoder stage 5
    hx5d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx6dup,hx5],axis=3))

    hx5dup = layers.UpsampleLike(name='hx5d_upsampled_rsu7')([hx5d, hx4])
    #decoder stage 4
    hx4d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx5dup,hx4],axis=3))

    hx4dup = layers.UpsampleLike(name='hx4d_upsampled_rsu7')([hx4d, hx3])
    #decoder stage 3
    hx3d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx4dup,hx3],axis=3))

    hx3dup = layers.UpsampleLike(name='hx3d_upsampled_rsu7')([hx3d, hx2])
    #decoder stage 2
    hx2d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx3dup,hx2],axis=3))

    hx2dup = layers.UpsampleLike(name='hx2d_upsampled_rsu7')([hx2d, hx1])
    #decoder stage 1
    hx1d = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx2dup,hx1],axis=3))

    return Add()([layer_in, hx1d])

def rsu_6(inputs,mid_channels=16,out_channels=64,is_training=True):

    # input conv
    layer_in = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(inputs)
    #------------------ENCODER----------------#

    #encoder stage 1
    hx1 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(layer_in)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx1)
    #encoder stage 2
    hx2 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx2)
    #encoder stage 3
    hx3 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx3)
    #encoder stage 4
    hx4 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx4)
    #encoder stage 5
    hx5 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx5)
    #encoder stage 6
    hx6 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    #------------------DECODER----------------#

    #decoder stage 6
    hx6d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx6)

    hx6dup = layers.UpsampleLike()([hx6d, hx5])
    #decoder stage 5
    hx5d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx6dup,hx5],axis=3))

    hx5dup = layers.UpsampleLike()([hx5d, hx4])
    #decoder stage 4
    hx4d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx5dup,hx4],axis=3))

    hx4dup = layers.UpsampleLike()([hx4d, hx3])
    #decoder stage 3
    hx3d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx4dup,hx3],axis=3))

    hx3dup = layers.UpsampleLike()([hx3d, hx2])
    #decoder stage 2
    hx2d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx3dup,hx2],axis=3))

    hx2dup = layers.UpsampleLike()([hx2d, hx1])
    #decoder stage 1
    hx1d = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx2dup,hx1],axis=3))

    return Add()([layer_in, hx1d])

def rsu_5(inputs,mid_channels=16,out_channels=64,is_training=True):

    # input conv
    layer_in = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(inputs)
    #------------------ENCODER----------------#

    #encoder stage 1
    hx1 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(layer_in)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx1)
    #encoder stage 2
    hx2 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx2)
    #encoder stage 3
    hx3 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx3)
    #encoder stage 4
    hx4 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx4)
    #encoder stage 5
    hx5 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    #------------------DECODER----------------#

    #decoder stage 5
    hx5d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx5)

    hx5dup = layers.UpsampleLike()([hx5d, hx4])
    #decoder stage 4
    hx4d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx5dup,hx4],axis=3))

    hx4dup = layers.UpsampleLike()([hx4d, hx3])
    #decoder stage 3
    hx3d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx4dup,hx3],axis=3))

    hx3dup = layers.UpsampleLike()([hx3d, hx2])
    #decoder stage 2
    hx2d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx3dup,hx2],axis=3))

    hx2dup = layers.UpsampleLike()([hx2d, hx1])
    #decoder stage 1
    hx1d = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx2dup,hx1],axis=3))

    return Add()([layer_in, hx1d])

def rsu_4(inputs,mid_channels=16,out_channels=64,is_training=True):

    # input conv
    layer_in = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(inputs)
    #------------------ENCODER----------------#

    #encoder stage 1
    hx1 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(layer_in)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx1)
    #encoder stage 2
    hx2 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx2)
    #encoder stage 3
    hx3 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    hx = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last')(hx3)
    #encoder stage 4
    hx4 = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx)

    #------------------DECODER----------------#

    #decoder stage 4
    hx4d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(hx4)

    hx4dup = layers.UpsampleLike()([hx4d, hx3])
    #decoder stage 3
    hx3d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx4dup,hx3],axis=3))

    hx3dup = layers.UpsampleLike()([hx3d, hx2])
    #decoder stage 2
    hx2d = Conv2D(mid_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx3dup,hx2],axis=3))

    hx2dup = layers.UpsampleLike()([hx2d, hx1])
    #decoder stage 1
    hx1d = Conv2D(out_channels, (3, 3), strides=(1,1), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(concatenate([hx2dup,hx1],axis=3))

    return Add()([layer_in, hx1d])


def medo_backbone(inputs, is_training=True):

    # inputs = Input(shape=(None, None, 1), name='net_input') # one channel 1/1
    C1 = Conv2D(64, (7, 7), strides=(2,2), data_format='channels_last', dilation_rate=(1,1), activation='relu', padding='same')(inputs) #1/2
    hx = MaxPooling2D(pool_size=[2,2],strides=[2,2],padding='same')(C1) # 1/4


    #encoder stage 1
    C2 = rsu_7(hx,mid_channels=32,out_channels=128,is_training=is_training) #1/4

    hx = MaxPooling2D(pool_size=[2,2],strides=[2,2],padding='same')(C2)
    #encoder stage 2
    C3 = rsu_6(hx,mid_channels=64,out_channels=256,is_training=is_training) #1/8

    hx = MaxPooling2D(pool_size=[2,2],strides=[2,2],padding='same')(C3)
    #encoder stage 3
    C4 = rsu_5(hx,mid_channels=128,out_channels=512,is_training=is_training) #1/16

    hx = MaxPooling2D(pool_size=[2,2],strides=[2,2],padding='same')(C4)
    #encoder stage 4
    C5 = rsu_4(hx,mid_channels=256,out_channels=1024,is_training=is_training) #1/32

    return [C3,C4,C5]

def medo_det_net(num_classes, inputs=None, num_anchors=None, submodels=None, lr=1e-5, is_training=True, **kwargs):
    """ Constructs a medonet model using medo_backbone.

    Args
        num_classes: Number of classes to predict.
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).

    Returns
        Medonet with a medonet backbone.
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
            'regression_coord'    : losses.smooth_l1(),
            'classification_coord': losses.focal(),
            'regression_bifpn'    : losses.smooth_l1(),
            'classification_bifpn': losses.focal(),
            'regression_cbam'    : losses.smooth_l1(),
            'classification_cbam': losses.focal(),
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
    regression_coord     = model.outputs[0]
    classification_coord = model.outputs[1]
    regression_bifpn     = model.outputs[2]
    classification_bifpn = model.outputs[3]
    regression_cbam    = model.outputs[4]
    classification_cbam = model.outputs[5]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[6:]

    # apply predicted regression to anchors
    boxes_coord = layers.RegressBoxes(name='boxes0')([anchors, regression_coord])
    boxes_coord = layers.ClipBoxes(name='clipped_boxes0')([model.inputs[0], boxes_coord])

    boxes_bifpn = layers.RegressBoxes(name='boxes1')([anchors, regression_bifpn])
    boxes_bipfn = layers.ClipBoxes(name='clipped_boxes1')([model.inputs[0], boxes_bifpn])

    boxes_cbam = layers.RegressBoxes(name='boxes2')([anchors, regression_cbam])
    boxes_cbam = layers.ClipBoxes(name='clipped_boxes2')([model.inputs[0], boxes_cbam])


    detections0 = layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        score_threshold       = 0.2,
        nms_threshold         = 0.3,
        name                  = 'filtered_detections0'
    )([boxes_coord, classification_coord] + other)
    

    detections1 = layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        score_threshold       = 0.2,
        nms_threshold         = 0.3,
        name                  = 'filtered_detections1'
    )([boxes_bifpn, classification_bifpn] + other)


    detections2 = layers.FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        score_threshold       = 0.2,
        nms_threshold         = 0.3,
        name                  = 'filtered_detections2'
    )([boxes_cbam, classification_cbam] + other)
    # print(len(detections0),len(detections1),len(detections2))
    outputs = [detections0[0],detections0[1],detections0[2], detections1[0],detections1[1],detections1[2],detections2[0],detections2[1],detections2[2]]

    #------------------------------------------

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=outputs, name=name)
