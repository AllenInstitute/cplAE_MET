import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops


class Encoder_T(layers.Layer):
    """
    Encoder for transcriptomic data
    
    Args:
        dropout_rate: dropout probability if training=True
        latent_dim: dimensionality of representation
        intermediate_dim: number of units in hidden layers
    """

    def __init__(self,
                 dropout_rate=0.5,
                 latent_dim=3,
                 intermediate_dim=50,
                 name='Encoder_T',
                 **kwargs):

        super(Encoder_T, self).__init__(name=name, **kwargs)
        self.drp = layers.Dropout(rate=dropout_rate)
        self.fc0 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc0')
        self.fc1 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc1')
        self.fc2 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc2')
        self.fc3 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc3')
        self.fc4 = layers.Dense(latent_dim, use_bias=False, activation='linear', name=name+'fc4')
        self.bn = layers.BatchNormalization(scale=False, center=False, epsilon=1e-10, momentum=0.05, name=name+'BN')
        return

    def call(self, inputs, training=True):
        x = self.drp(inputs, training=training)
        x = self.fc0(x, training=training)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x, training=training)
        x = self.fc4(x, training=training)
        z = self.bn(x, training=training)
        return z


class Decoder_T(layers.Layer):
    """
    Decoder for transcriptomic data

    Args:
        output_dim: number of outputs
        intermediate_dim: number of units in hidden layers
    """

    def __init__(self,
                 output_dim,
                 intermediate_dim=50,
                 name='Decoder_T',
                 **kwargs):
        
        super(Decoder_T, self).__init__(name=name, **kwargs)
        self.fc0 = layers.Dense(intermediate_dim, activation='relu', name='fc0')
        self.fc1 = layers.Dense(intermediate_dim, activation='relu', name='fc1')
        self.fc2 = layers.Dense(intermediate_dim, activation='relu', name='fc2')
        self.fc3 = layers.Dense(intermediate_dim, activation='relu', name='fc3')
        self.Xout = layers.Dense(output_dim, activation='relu', name='Xout')
        return

    def call(self, inputs, training=True):
        x = self.fc0(inputs, training=training)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x, training=training)
        x = self.Xout(x)
        return x


class Encoder_E(layers.Layer):
    """
    Encoder for electrophysiology data
    
    Args:
        gaussian_noise_sd: std of gaussian noise injection if training=True
        dropout_rate: dropout probability if training=True
        latent_dim: representation dimenionality
        intermediate_dim: number of units in hidden layers
    """

    def __init__(self,
                 gaussian_noise_sd=0.05,
                 dropout_rate=0.1,
                 latent_dim=3,
                 intermediate_dim=40,
                 name='Encoder_E',
                 dtype=tf.float32,
                 **kwargs):
        
        super(Encoder_E, self).__init__(name=name, **kwargs)
        self.gnoise = WeightedGaussianNoise(stddev=gaussian_noise_sd)
        self.drp = layers.Dropout(rate=dropout_rate)
        self.fc0 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc0')
        self.fc1 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc1')
        self.fc2 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc2')
        self.fc3 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc3')
        self.fc4 = layers.Dense(latent_dim, use_bias=False, activation='linear', name=name+'fc4')
        self.bn = layers.BatchNormalization(scale=False, center=False, epsilon=1e-10, momentum=0.05, name=name+'BN')
        return

    def call(self, inputs, training=True):
        x = self.gnoise(inputs, training=training)
        x = self.drp(x, training=training)
        x = self.fc0(x, training=training)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x, training=training)
        x = self.fc4(x, training=training)
        z = self.bn(x, training=training)
        return z


class Decoder_E(layers.Layer):
    """
    Decoder for electrophysiology data.

    Args:
        output_dim: Should be same as input dim if using as an autoencoder
        intermediate_dim: Number of units in hidden layers
        training: boolean value to indicate model operation mode
    """

    def __init__(self,
                 output_dim,
                 intermediate_dim=40,
                 name='Decoder_E',
                 dtype=tf.float32,
                 **kwargs):

        super(Decoder_E, self).__init__(name=name, **kwargs)
        self.fc0 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc0')
        self.fc1 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc1')
        self.fc2 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc2')
        self.fc3 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc3')
        self.drp = layers.Dropout(rate=0.1)
        self.Xout = layers.Dense(output_dim, activation='linear', name=name+'Xout')
        return

    def call(self, inputs, training=True):
        x = self.fc0(inputs, training=training)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x, training=training)
        x = self.drp(x, training=training)
        x = self.Xout(x, training=training)
        return x


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
        x = self.addnoise(inputs,training=training)
        ax,de = tf.split(x, 2, axis=-1)
        ax = self.conv1_ax(ax)
        de = self.conv1_de(de)

        ax = self.conv2_ax(ax)
        de = self.conv2_de(de)

        x = self.cat(inputs=[self.flat(ax),self.flat(de)])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.bn(x,training=training)
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
        self.reshape = layers.Reshape(target_shape=(15, 1, 10))

        self.convT1_ax = layers.Conv2DTranspose(filters=10, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='elu')
        self.convT1_de = layers.Conv2DTranspose(filters=10, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='elu')

        self.convT2_ax = layers.Conv2DTranspose(filters=1, kernel_size=(4, 3), strides=(4, 1), padding='valid', activation='elu')
        self.convT2_de = layers.Conv2DTranspose(filters=1, kernel_size=(4, 3), strides=(4, 1), padding='valid', activation='elu')
        return

    def call(self, inputs, training=True):
        x = self.fc1_dec(inputs)
        x = self.fc2_dec(x)
        x = self.fc3_dec(x)

        ax, de = tf.split(x, 2, axis=-1)
        ax = self.reshape(ax)
        de = self.reshape(de)

        ax = self.convT1_ax(ax)
        de = self.convT1_de(de)

        ax = self.convT2_ax(ax)
        de = self.convT2_de(de)
        x = tf.concat([ax, de], axis=-1)
        return x


class Model_TE(tf.keras.Model):
    """
    Coupled autoencoder

    Args:
        T_output_dim: n(genes)
        E_output_dim: n(features)
        T_intermediate_dim: units in hidden layers of T autoencoder
        E_intermediate_dim: units in hidden layers of T autoencoder
        T_dropout: dropout probability for 
        E_gnoise_sd: gaussian noise std for E data
        E_dropout: dropout for E data
        latent_dim: dim for representations
        train_T: bool: set T encoder and decoder to training mode
        train_E: bool: set T encoder and decoder to training mode
        name: TE
    """

    def __init__(self,
               T_output_dim,
               E_output_dim,
               T_intermediate_dim=50,
               E_intermediate_dim=40,
               T_dropout=0.5,
               E_gnoise_sd=0.05,
               E_dropout=0.1,
               latent_dim=3,
               train_T=True,
               train_E=True,
               name='TE',
               **kwargs):
  
        super(Model_TE, self).__init__(name=name, **kwargs)
        self.encoder_T = Encoder_T(dropout_rate=T_dropout,
                                   latent_dim=latent_dim,
                                   intermediate_dim=T_intermediate_dim,
                                   name='Encoder_T')

        self.encoder_E = Encoder_E(gaussian_noise_sd=E_gnoise_sd,
                                   dropout_rate=E_dropout,
                                   latent_dim=latent_dim,
                                   intermediate_dim=E_intermediate_dim,
                                   name='Encoder_E')

        self.decoder_T = Decoder_T(output_dim=T_output_dim,
                                   intermediate_dim=T_intermediate_dim,
                                   name='Decoder_T')

        self.decoder_E = Decoder_E(output_dim=E_output_dim,
                                   intermediate_dim=E_intermediate_dim,
                                   name='Decoder_E')
        self.train_T = train_T
        self.train_E = train_E

    def call(self, inputs):
        #T arm
        zT = self.encoder_T(inputs[0],training=self.train_T)
        XrT = self.decoder_T(zT,training=self.train_T)
        

        #E arm
        zE = self.encoder_E(inputs[1],training=self.train_E)
        XrE = self.decoder_E(zE,training=self.train_E)
        return zT,zE,XrT,XrE


class WeightedGaussianNoise(layers.Layer):
    """Custom additive zero-centered Gaussian noise. Std is weighted.

    Args:
        stddev: Can be a scalar or vector
    call args:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
        training mode (adding noise) or in inference mode (doing nothing).
    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        Same shape as input.
    """

    def __init__(self, stddev, **kwargs):
        super(WeightedGaussianNoise, self).__init__(**kwargs)
        self.stddev = stddev
        return

    def call(self, inputs, training=None):
        def noised():
            return inputs + tf.random.normal(array_ops.shape(inputs),
                                             mean=0.0, stddev=self.stddev,
                                             dtype=inputs.dtype, seed=None)

        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(WeightedGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class Model_TE_aug_decoders(tf.keras.Model):
    """Coupled autoencoder model

    Args:
        T_dim: Number of genes in T data
        E_dim: Number of features in E data
        T_intermediate_dim: hidden layer dims for T model
        E_intermediate_dim: hidden layer dims for E model
        T_dropout: dropout for T data
        E_gnoise_sd: gaussian noise std for E data
        E_dropout: dropout for E data
        latent_dim: dim for representations
        train_T (bool): training/inference mode for E autoencoder
        train_E (bool): training/inference mode for E autoencoder
        augment_decoders (bool): augment decoder with cross modal representation if True
        name: TE
    """
    def __init__(self,
               T_dim,
               E_dim,
               T_intermediate_dim=50,
               E_intermediate_dim=40,
               alpha_T=1.0,
               alpha_E=1.0,
               lambda_TE=1.0,
               T_dropout=0.5,
               E_gauss_noise_wt = 1.0,
               E_gnoise_sd=0.05,
               E_dropout=0.1,
               latent_dim=3,
               train_T=True,
               train_E=True, 
               augment_decoders=True,
               name='TE',
               **kwargs):

        super(Model_TE_aug_decoders, self).__init__(name=name, **kwargs)
        self.T_dim = T_dim
        self.E_dim = E_dim

        self.alpha_T = tf.constant(alpha_T,dtype=tf.float32)
        self.alpha_E = tf.constant(alpha_E,dtype=tf.float32)
        self.lambda_TE = tf.constant(lambda_TE,dtype=tf.float32)

        E_gnoise_sd_weighted = E_gauss_noise_wt*E_gnoise_sd
        self.encoder_T = Encoder_T(dropout_rate=T_dropout,latent_dim=latent_dim, intermediate_dim=T_intermediate_dim, name='Encoder_T')
        self.encoder_E = Encoder_E(gaussian_noise_sd=E_gnoise_sd_weighted, dropout_rate=E_dropout, latent_dim=latent_dim, intermediate_dim=E_intermediate_dim, name='Encoder_E')
        
        self.decoder_T = Decoder_T(output_dim=T_dim, intermediate_dim=T_intermediate_dim, name='Decoder_T')
        self.decoder_E = Decoder_E(output_dim=E_dim, intermediate_dim=E_intermediate_dim, name='Decoder_E')

        self.train_T = train_T
        self.train_E = train_E
        self.augment_decoders = augment_decoders
        return


    def call(self, inputs):
        #T arm forward pass
        XT = inputs[0]
        zT = self.encoder_T(XT,training=self.train_T)
        XrT = self.decoder_T(zT,training=self.train_T)
        
        #E arm forward pass
        XE = tf.where(tf.math.is_nan(inputs[1]),x=0.0,y=inputs[1]) #Mask nans
        maskE = tf.where(tf.math.is_nan(inputs[1]),x=0.0,y=1.0)    #Get mask to ignore error contribution
        zE = self.encoder_E(XE,training=self.train_E)
        XrE = self.decoder_E(zE,training=self.train_E)

        #Loss calculations
        mse_loss_T = tf.reduce_mean(tf.math.squared_difference(XT, XrT))
        mse_loss_E = tf.reduce_mean(tf.multiply(tf.math.squared_difference(XE, XrE),maskE))
        cpl_loss_TE = min_var_loss(zT, zE)

        #Append to keras model losses for gradient calculations
        self.add_loss(self.alpha_T*mse_loss_T)
        self.add_loss(self.alpha_E*mse_loss_E)
        self.add_loss(self.lambda_TE*cpl_loss_TE)

        #Cross modal reconstructions - treat zE and zT as constants for this purpose
        mse_loss_T_aug = 0
        mse_loss_E_aug = 0
        if self.augment_decoders:
            XrT_aug = self.decoder_T(tf.stop_gradient(zE),training=self.train_T)
            XrE_aug = self.decoder_E(tf.stop_gradient(zT),training=self.train_E)
            mse_loss_T_aug = tf.reduce_mean(tf.math.squared_difference(XT, XrT_aug))
            mse_loss_E_aug = tf.reduce_mean(tf.multiply(tf.math.squared_difference(XE, XrE_aug),maskE))
            self.add_loss(self.alpha_T*mse_loss_T_aug)
            self.add_loss(self.alpha_E*mse_loss_E_aug)
        
        #For logging only
        self.mse_loss_T = mse_loss_T
        self.mse_loss_E = mse_loss_E
        self.mse_loss_TE = tf.reduce_mean(tf.math.squared_difference(zT, zE))
        self.mse_loss_T_aug = mse_loss_T_aug
        self.mse_loss_E_aug = mse_loss_E_aug
        return zT,zE,XrT,XrE


    def buildme(model):
        """
        Initialize the model using this if loading saved weights. 
        """
        x = tf.constant(np.random.rand(1, model.T_dim), dtype=tf.float32)
        y = tf.constant(np.random.rand(1, model.E_dim), dtype=tf.float32)
        model.train_E = False
        model.train_T = False
        _, _, _, _ = model((x, y))
        return model


def min_var_loss(zi, zj, Wij=None):
    """
    SVD is calculated over entire batch. MSE is calculated over only paired entries within batch
    
    Args:
        zi: i-th representation (batch_size x latent_dim)
        zj: j-th representation (batch_size x latent_dim)
        Wij: indicator vector (batch_size x latent_dim) (1 if samples are paired, 0 otherwise)
    """
    batch_size = tf.shape(zi)[0]
    if Wij is None:
        Wij_ = tf.ones([batch_size, ])
    else:
        Wij_ = tf.reshape(Wij, [batch_size, ])

    #Masking gets rid of unpaired entries
    zi_paired = tf.boolean_mask(zi, tf.math.greater(Wij_, 1e-2))
    zj_paired = tf.boolean_mask(zj, tf.math.greater(Wij_, 1e-2))
    Wij_paired = tf.boolean_mask(Wij_, tf.math.greater(Wij_, 1e-2))

    #SVD calculated over all entries in the batch
    vars_j_ = tf.square(tf.reduce_min(tf.linalg.svd(zj - tf.reduce_mean(zj, axis=0), compute_uv=False)))/tf.cast(batch_size - 1, tf.float32)
    vars_j  = tf.where(tf.math.is_nan(vars_j_), tf.cast(1e-2,dtype=tf.float32), vars_j_)

    vars_i_ = tf.square(tf.reduce_min(tf.linalg.svd(zi - tf.reduce_mean(zi, axis=0), compute_uv=False)))/tf.cast(batch_size - 1, tf.float32)
    vars_i  = tf.where(tf.math.is_nan(vars_i_), tf.cast(1e-2,dtype=tf.float32), vars_i_)

    #Wij_paired is the weight of matched pairs
    sqdist_paired = tf.multiply(tf.reduce_sum(tf.math.squared_difference(zi_paired, zj_paired),axis=1),Wij_paired)
    
    mean_sqdist = tf.reduce_sum(sqdist_paired,axis=None)/tf.reduce_sum(Wij_paired,axis=None)
    loss_ij = mean_sqdist/tf.maximum(tf.reduce_min([vars_i,vars_j], axis=None),tf.cast(1e-2,dtype=tf.float32))
    return loss_ij


def min_var_loss_asymmetric(zi, zj, Wij=None):
    """
    SVD is calculated over entire batch, only for zi. MSE is calculated over only paired entries within batch
    
    Args:
        zi: i-th representation (batch_size x latent_dim)
        zj: j-th representation (batch_size x latent_dim)
        Wij: indicator vector (batch_size x 1) (1 if samples are paired, 0 otherwise)
    """
    batch_size = tf.shape(zi)[0]
    if Wij is None:
        Wij_ = tf.ones([batch_size, ])
    else:
        Wij_ = tf.reshape(Wij, [batch_size, ])

    #Masking gets rid of unpaired entries
    zi_paired = tf.boolean_mask(zi, tf.math.greater(Wij_, 1e-2))
    zj_paired = tf.boolean_mask(zj, tf.math.greater(Wij_, 1e-2))
    Wij_paired = tf.boolean_mask(Wij_, tf.math.greater(Wij_, 1e-2))

    #SVD calculated over all entries in the batch
    #vars_j_ = tf.square(tf.reduce_min(tf.linalg.svd(zj - tf.reduce_mean(zj, axis=0), compute_uv=False)))/tf.cast(batch_size - 1, tf.float32)
    #vars_j  = tf.where(tf.math.is_nan(vars_j_), tf.cast(1e-2,dtype=tf.float32), vars_j_)

    vars_i_ = tf.square(tf.reduce_min(tf.linalg.svd(zi - tf.reduce_mean(zi, axis=0), compute_uv=False)))/tf.cast(batch_size - 1, tf.float32)
    vars_i  = tf.where(tf.math.is_nan(vars_i_), tf.cast(1e-2,dtype=tf.float32), vars_i_)
    vars_j = vars_i

    #Wij_paired is the weight of matched pairs
    sqdist_paired = tf.multiply(tf.reduce_sum(tf.math.squared_difference(zi_paired, zj_paired),axis=1),Wij_paired)
    mean_sqdist = tf.reduce_sum(sqdist_paired,axis=None)/tf.reduce_sum(Wij_paired,axis=None)
    loss_ij = mean_sqdist/tf.maximum(tf.reduce_min([vars_i,vars_j], axis=None),tf.cast(1e-2,dtype=tf.float32))
    return loss_ij


class Model_TME(tf.keras.Model):
    """Coupled autoencoder model

    Args:
        T_dim: Number of genes in T data
        E_dim: Number of features in E data
        T_intermediate_dim: hidden layer dims for T model
        E_intermediate_dim: hidden layer dims for E model
        T_dropout: dropout for T data
        E_gnoise_sd: gaussian noise std for E data
        E_dropout: dropout for E data
        latent_dim: dim for representations
        train_T (bool): training/inference mode for E autoencoder
        train_E (bool): training/inference mode for E autoencoder
        augment_decoders (bool): augment decoder with cross modal representation if True
        name: TE
    """
    def __init__(self,
               T_dim=1252,
               E_dim=68,
               M_dim=(120,4,2),
               T_intermediate_dim=50,
               E_intermediate_dim=40,
               alpha_T=1.0,
               alpha_E=1.0,
               alpha_M=1.0,
               lambda_TE=1.0,
               lambda_ME=0.5,
               lambda_TM=1.0,
               T_dropout=0.5,
               E_gauss_noise_wt = 1.0,
               E_gnoise_sd=0.05,
               E_dropout=0.1,
               M_gauss_noise_std=0.5,
               latent_dim=3,
               train_T=False,
               train_E=False, 
               train_M=False,
               augment_decoders=False,
               name='TME',
               **kwargs):

        super(Model_TME, self).__init__(name=name, **kwargs)
        self.T_dim = T_dim
        self.E_dim = E_dim
        self.M_dim = M_dim

        self.alpha_T = tf.constant(alpha_T,dtype=tf.float32)
        self.alpha_E = tf.constant(alpha_E,dtype=tf.float32)
        self.alpha_M = tf.constant(alpha_M,dtype=tf.float32)
        self.lambda_TE = tf.constant(lambda_TE,dtype=tf.float32)
        self.lambda_ME = tf.constant(lambda_ME,dtype=tf.float32)
        self.lambda_TM = tf.constant(lambda_TM,dtype=tf.float32)

        E_gnoise_sd_weighted = E_gauss_noise_wt*E_gnoise_sd
        self.encoder_T = Encoder_T(dropout_rate=T_dropout, latent_dim=latent_dim,intermediate_dim=T_intermediate_dim, name='Encoder_T')
        self.encoder_E = Encoder_E(gaussian_noise_sd=E_gnoise_sd_weighted, dropout_rate=E_dropout,latent_dim=latent_dim, intermediate_dim=E_intermediate_dim, name='Encoder_E')
        self.encoder_M = Encoder_M(stddev=M_gauss_noise_std, latent_dim=latent_dim, name='Encoder_M')
        
        self.decoder_T = Decoder_T(output_dim=T_dim, intermediate_dim=T_intermediate_dim, name='Decoder_T')
        self.decoder_E = Decoder_E(output_dim=E_dim, intermediate_dim=E_intermediate_dim, name='Decoder_E')
        self.decoder_M = Decoder_M(name='Decoder_M')

        self.train_T = train_T
        self.train_E = train_E
        self.train_M = train_M
        self.augment_decoders = augment_decoders
        return


    def call(self, inputs):
        #inputs[0]: T, 
        #inputs[1]: E 
        #inputs[2]: M

        #T arm forward pass
        XT = inputs[0]
        zT = self.encoder_T(XT, training=self.train_T)
        XrT = self.decoder_T(zT, training=self.train_T)

        #E arm forward pass
        XE = tf.where(tf.math.is_nan(inputs[1]), x=0.0, y=inputs[1])  # Mask nans
        # Get mask to ignore error contribution
        maskE = tf.where(tf.math.is_nan(inputs[1]), x=0.0, y=1.0)
        zE = self.encoder_E(XE, training=self.train_E)
        XrE = self.decoder_E(zE, training=self.train_E)

        #M arm forward pass
        XM = tf.where(tf.math.is_nan(inputs[2]),x=0.0,y=inputs[2]) #Mask nans
        maskM = tf.where(tf.math.is_nan(inputs[2]),x=0.0,y=1.0)    #Get mask to ignore error contribution
        zM = self.encoder_M(XM,training=self.train_M)
        XrM = self.decoder_M(zM,training=self.train_M)
        
        #Loss calculations
        mse_loss_T = tf.reduce_mean(tf.math.squared_difference(XT, XrT))
        mse_loss_E = tf.reduce_mean(tf.multiply(tf.math.squared_difference(XE, XrE), maskE))
        mse_loss_M = tf.reduce_mean(tf.multiply(tf.math.squared_difference(XM, XrM), maskM))
        
        keep_M = tf.where(tf.math.reduce_any(maskM > 0, axis=[1, 2, 3]),x=1.0,y=0.0)
        cpl_loss_TE = min_var_loss(zT, zE)
        #cpl_loss_ME = min_var_loss(zM, zE, Wij=keep_M)
        #cpl_loss_TM = min_var_loss(zT, zM, Wij=keep_M)
        cpl_loss_ME = min_var_loss_asymmetric(zi=zE, zj=zM, Wij=keep_M)
        cpl_loss_TM = min_var_loss_asymmetric(zi=zT, zj=zM, Wij=keep_M)

        #Append to keras model losses for gradient calculations
        self.add_loss(self.alpha_T*mse_loss_T)
        self.add_loss(self.alpha_E*mse_loss_E)
        self.add_loss(self.alpha_M*mse_loss_M)

        self.add_loss(self.lambda_TE*cpl_loss_TE)
        self.add_loss(self.lambda_ME*cpl_loss_ME)
        self.add_loss(self.lambda_TM*cpl_loss_TM)

        #Cross modal reconstructions - treat zE and zT as constants for this purpose
        mse_loss_T_aug = 0
        mse_loss_E_aug = 0
        mse_loss_M_aug = 0
        if self.augment_decoders:
            XrT_from_XE_aug = self.decoder_T(tf.stop_gradient(zE), training=self.train_T)
            XrT_from_XM_aug = self.decoder_T(tf.stop_gradient(zM), training=self.train_T)
            mse_loss_T_from_E_aug = tf.reduce_mean(tf.math.squared_difference(XT, XrT_from_XE_aug))
            mse_loss_T_from_M_aug = tf.reduce_mean(tf.math.squared_difference(XT, XrT_from_XM_aug))
            mse_loss_T_aug = mse_loss_T_from_E_aug + mse_loss_T_from_M_aug

            XrE_from_XT_aug = self.decoder_E(tf.stop_gradient(zT), training=self.train_E)
            XrE_from_XM_aug = self.decoder_E(tf.stop_gradient(zM), training=self.train_E)
            mse_loss_E_from_T_aug = tf.reduce_mean(tf.math.squared_difference(XE, XrE_from_XT_aug))
            mse_loss_E_from_M_aug = tf.reduce_mean(tf.math.squared_difference(XE, XrE_from_XM_aug))
            mse_loss_E_aug = mse_loss_E_from_T_aug + mse_loss_E_from_M_aug

            XrM_from_XT_aug = self.decoder_M(tf.stop_gradient(zT), training=self.train_M)
            XrM_from_XE_aug = self.decoder_M(tf.stop_gradient(zE), training=self.train_M)
            mse_loss_M_from_E_aug = tf.reduce_mean(tf.math.squared_difference(XM, XrM_from_XE_aug))
            mse_loss_M_from_T_aug = tf.reduce_mean(tf.math.squared_difference(XM, XrM_from_XT_aug))
            mse_loss_M_aug = mse_loss_M_from_E_aug + mse_loss_M_from_T_aug

            self.add_loss(self.alpha_T*(mse_loss_T_from_E_aug + mse_loss_T_from_M_aug))
            self.add_loss(self.alpha_E*(mse_loss_E_from_T_aug + mse_loss_E_from_M_aug))
            self.add_loss(self.alpha_M*(mse_loss_M_from_E_aug + mse_loss_M_from_T_aug))
        
        #For logging only
        self.mse_loss_T = mse_loss_T
        self.mse_loss_E = mse_loss_E
        self.mse_loss_M = mse_loss_M

        self.mse_loss_TE = tf.reduce_mean(tf.math.squared_difference(zT, zE))
        self.mse_loss_ME = tf.reduce_mean(tf.math.squared_difference(zM, zE))
        self.mse_loss_TM = tf.reduce_mean(tf.math.squared_difference(zT, zM))

        self.mse_loss_T_aug = mse_loss_T_aug
        self.mse_loss_E_aug = mse_loss_E_aug
        self.mse_loss_M_aug = mse_loss_M_aug
        return zT, zE, zM, XrT, XrE, XrM
