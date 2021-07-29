import numpy as np
import tensorflow as tf
import tensorflow.python.keras.layers as layers

class Encoder_M(layers.Layer):
    """
    Encoder for morphology data. [histograms, soma_depth]
    
    Args:
        dropout_rate: dropout probability if training=True
        latent_dim: representation dimenionality
    """

    def __init__(self,
                 stddev=0.5,
                 latent_dim=3,
                 name='Encoder_M',
                 **kwargs):

        super(Encoder_M, self).__init__(name=name, **kwargs)
        self.addnoise = layers.GaussianNoise(stddev)
        self.conv1_ax = layers.Conv2D(filters=10,kernel_size=(4,3), strides=(4,1),padding='valid', activation='elu')
        self.conv1_de = layers.Conv2D(filters=10,kernel_size=(4,3), strides=(4,1),padding='valid', activation='elu')
        
        self.conv2_ax = layers.Conv2D(filters=10,kernel_size=(2,2),strides=(2,1),padding='valid',activation='elu')
        self.conv2_de = layers.Conv2D(filters=10,kernel_size=(2,2),strides=(2,1),padding='valid',activation='elu')

        self.flat = layers.Flatten()
        self.cat = layers.Concatenate()
        self.fc1 = layers.Dense(20, activation='elu')
        self.fc2 = layers.Dense(20, activation='elu')
        self.fc3 = layers.Dense(latent_dim, activation='linear')
        self.bn = layers.BatchNormalization(scale=False, center=False, epsilon=1e-10, momentum=0.95)
        return

    def call(self, inputs, training=True):
        #inputs: XM
        
        x = self.addnoise(inputs,training=training)
        ax,de = tf.split(x, 2, axis=-1)

        print('---')
        print(ax.shape)
        print(de.shape)

        ax = self.conv1_ax(ax)
        de = self.conv1_de(de)
        
        print('---')
        print(ax.shape)
        print(de.shape)

        ax = self.conv2_ax(ax)
        de = self.conv2_de(de)
        
        print('---')
        print(ax.shape)
        print(de.shape)

        x = self.cat(inputs=[self.flat(ax),self.flat(de)])
        
        print('---')
        print(x.shape)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.bn(x,training=training)
        
        print('---')
        print(x.shape)

        return x


class Decoder_M(layers.Layer):
    """
    Decoder for morphology data

    Args:
        output_dim: Should be same as input dim if using as an autoencoder
        intermediate_dim: Number of units in hidden keras.layers
        training: boolean value to indicate model operation mode
    """

    def __init__(self,
                 name='Decoder_M',
                 **kwargs):

        super(Decoder_M, self).__init__(name=name, **kwargs)
        self.fc1_dec = layers.Dense(20, activation='elu')
        self.fc2_dec = layers.Dense(20, activation='elu')
        self.fc3_dec = layers.Dense(300, activation='elu')
        self.reshape = layers.Reshape(target_shape=(15,1,10))

        self.convT1_ax=layers.Conv2DTranspose(filters=10, kernel_size=(2,2), strides=(2,2), padding='same',activation='elu')
        self.convT1_de=layers.Conv2DTranspose(filters=10, kernel_size=(2,2), strides=(2,2), padding='same',activation='elu')

        self.convT2_ax=layers.Conv2DTranspose(filters=1, kernel_size=(4,3), strides=(4,1), padding='valid',activation='elu')
        self.convT2_de=layers.Conv2DTranspose(filters=1, kernel_size=(4,3), strides=(4,1), padding='valid',activation='elu')

        self.crop = lambda x:x[:,2:-2,:,:]
        return

    def call(self, inputs, training=True):
        x = self.fc1_dec(inputs)
        x = self.fc2_dec(x)
        x = self.fc3_dec(x)
        
        # print('---')
        # print(x.shape)

        ax,de = tf.split(x,2,axis=-1)
        ax = self.reshape(ax)
        de = self.reshape(de)

        # print('reshape ---')
        # print(ax.shape)
        # print(de.shape)
        
        ax = self.convT1_ax(ax)
        de = self.convT1_de(de)
        
        # print('conv 1 result ---')
        # print(ax.shape)
        # print(de.shape)

        ax = self.convT2_ax(ax)
        de = self.convT2_de(de)

        # print('conv 2 result ---')
        # print(ax.shape)
        # print(de.shape)
        
        x = tf.concat([ax,de],axis=-1)
        # print('---')
        # print(x.shape)

        return x


class Model_M(tf.keras.Model):
    def __init__(self, 
               alpha_M=1.0,
               train_M=True,
               name='TME',
               **kwargs):

        super(Model_M, self).__init__(name=name, **kwargs)
        self.alpha_M=alpha_M
        self.encoder_M = Encoder_M()
        self.decoder_M = Decoder_M()
        self.train_M = train_M
        return


    def call(self, inputs):
        #inputs[0]: T,
        #inputs[1]: E
        #inputs[2]: Hist
        #inputs[3]: soma_density

        #M arm forward pass
        XM = tf.where(tf.math.is_nan(inputs[2]), x=0.0, y=inputs[2])  # Mask nans
        # Get mask to ignore error contribution
        maskM = tf.where(tf.math.is_nan(inputs[2]), x=0.0, y=1.0)
        zM = self.encoder_M(XM, training=self.train_M)
        XrM = self.decoder_M(zM, training=self.train_M)

        #Loss calculations
        mse_loss_M_ = tf.reduce_mean(tf.multiply(tf.math.squared_difference(XM, XrM), maskM))
        mse_loss_M = mse_loss_M_

        #Append to keras model losses for gradient calculations
        self.add_loss(self.alpha_M*mse_loss_M)

        #For logging only
        self.mse_loss_M = mse_loss_M_

        return zM,XrM
