# Conv2d in karas vs torch

The input and output shape in keras are:

    input_shape = (N, H_in, W_in, C_in) 
    output_shape = (N, H_out, W_out, C_out)

The input and output shape in torch are:

    input_shape = (N, C_in, H_in, W_in) 
    output_shape = (N, C_out, H_out, W_out)

N is the batch size, H_in and W_in are the height and width of the input. C_in and C_out are the number of input and 
output channels.

given the input W_in and H_in, padding, strides, dilation and kernel_size, we can compute the W_out and H_out as 
the following:

    H_out = (H_in + 2 * padding[0] - dilation[0] * (kernel[0] -1) - 1) / stride[0] +1
    W_out = (W_in + 2 * padding[1] - dilation[1] * (kernel[1] -1) - 1) / stride[1] +1

padding = 'same' means a padding of (1, 1) and padding = 'valid' which is the default is a (0, 0). 

# Conv2D in keras:

    tf.keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
    )

Some examples:
The inputs are 28x28 RGB images with `channels_last` and the batch size is 4.  

    >>> input_shape = (4, 28, 28, 3)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Conv2D(
    ... 2, 3, activation='relu', input_shape=input_shape[1:])(x)
    >>> print(y.shape)
    (4, 26, 26, 2)


With `dilation_rate` as 2.  

    >>> input_shape = (4, 28, 28, 3)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Conv2D(
    ... 2, 3, activation='relu', dilation_rate=2, input_shape=input_shape[1:])(x)
    >>> print(y.shape)
    (4, 24, 24, 2)

 With `padding` as "same".

    >>> input_shape = (4, 28, 28, 3)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Conv2D(
    ... 2, 3, activation='relu', padding="same", input_shape=input_shape[1:])(x)
    >>> print(y.shape)
    (4, 28, 28, 2)

With extended batch shape [4, 7]:  

    >>> input_shape = (4, 7, 28, 28, 3)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Conv2D(
    ... 2, 3, activation='relu', input_shape=input_shape[2:])(x)
    >>> print(y.shape)
    (4, 7, 26, 26, 2)

# Conv2d in torch

    torch.nn.Conv2d(in_channels, 
                    out_channels, 
                    kernel_size, 
                    stride=1, 
                    padding=0, 
                    dilation=1, 
                    groups=1, 
                    bias=True, 
                    padding_mode='zeros', 
                    device=None, 
                    dtype=None)

Some examples:
    
    >>> input = torch.randn(20, 16, 50, 100)
    >>> # With square kernels and equal stride
    >>> m = nn.Conv2d(16, 33, 3, stride=2)
    >>> output = m(input)
    #output.shape = (20, 33, 24, 49)
    >>> # non-square kernels and unequal stride and with padding
    >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    #output.shape = (20, 33, 28, 100)
    >>> # non-square kernels and unequal stride and with padding and dilation
    >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
    #output.shape = (20, 33, 26, 100)


 