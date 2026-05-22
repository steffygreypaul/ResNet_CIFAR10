from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def one_layer(input, num_filters, kernel_size=(3,3),strides=(1,1), BN=True, activation='relu' ):
  conv = keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', 
                             kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
  x = input
  x = conv(x)
  if BN:
    x = keras.layers.BatchNormalization()(x)
  if activation is not None:
    x = keras.layers.Activation(activation)(x)
  return x

#########################################################################################
def ResNet(input_shape, stages_residual_blocks=[3, 4, 6, 3], num_classes=10):
  num_filters = 64 #number of filters for the first set of residual blocks
  stage=0
  input = keras.Input(shape=input_shape)
  
  #the only difference between the network architecture in original paper(figure 3 in the paper) and my impelementation is that they used a stride of 2 in this step but I used one because we are training
  #the network on images of size 32x32 and shrinking the dimensions by 2 at the beginning will cause problems with dimentionality in later steps. The architecture in figure 3 of the paper is trained on 
  #ImageNet data set which has samples with larger dimentions.
  x= one_layer(input=input, num_filters=num_filters, kernel_size=(7,7),strides=(1,1))
  
  #next we implement 4 residual blocks and they have 3, 4, 6, 3 layers respectively.
  for num_blocks_in_stage in stages_residual_blocks:
    for block in range(num_blocks_in_stage):
        strides = (1,1)
        # first layer but not first stage
        if stage > 0 and block == 0:  
          strides = (2,2)  # downsample
        y = one_layer(input=x, num_filters=num_filters, strides=strides, activation='relu')
        y = one_layer(input=y,num_filters=num_filters, activation=None)
        # first layer but not first stage
        if stage > 0 and block == 0:  
          # managing the dimensions so that they are of the same dimension and can be added. This goal is achieved using a Convolution with 
          #kernel size of 1x1. 
          x = one_layer(input=x, num_filters=num_filters, kernel_size=(1,1), strides=strides, activation=None, BN=False)
        x = keras.layers.add([x, y])
        x = keras.layers.Activation('relu')(x)
    num_filters *= 2
    stage+=1
  #Next, there is a average pooling layer
  x = keras.layers.AveragePooling2D(pool_size=(2,2))(x)
  x = keras.layers.Flatten()(x)
  #Finally, we have a softmax layer with 10 neurons.
  output=keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
  #Creating the model
  model = keras.Model(inputs=input, outputs=output)
  return model

