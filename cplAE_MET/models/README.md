![cplae](T_ME_coupled_AE_modules.png) 

#### bayesian_optimization_conv.py:
Training code with the option to run bayesian optimization.

#### model_classes.py:
Triple modalities coupled AE model

#### subnetworks_T.py:
T autoencoder arm which includes the following modules from the fig above:
```
- enc_xt_to_zt: T encoder module
- dec_zt_to_xt: T decoder module
```

#### subnetworks_E.py:
E autoencoder arm which consists of 4 module (see figure above): 
``` 
- enc_xe_to_ze_int: encoding module 1. Its weights will be copied and used for the ME arm as well. 
- enc_ze_int_to_ze: encoding module 2. Specific to the E autoencoder arm.
- dec_ze_to_ze_int: decoding module 1. Specific to the E autoencoder arm.
- dec_ze_int_to_xe: decoding module 2. Its weights will be copied and used for the ME arm as well.
```

#### subnetwork_M.py:
M autoencoder arm which consists of 4 modules (see figure above):
```
- enc_xm_to_zm_int: encoding module 1. Its weights will be copied and used for the ME arm as well.
- enc_zm_int_to_zm: encoding module 2. Specific to the M autoencoder arm.
- dec_zm_to_zm_int: decoding module 1. Specific to the M autoencoder arm.
- dec_zm_int_to_xm: decoding module 2. Its weights will be copied and used for the ME arm as well.
```

#### subnetwork_ME.py:
ME autoencoder arm which contains 2 specific modules of ME arm (see figure above):
```
- enc_zme_int_to_zme: specific encoding module of ME arm.
- dec_zme_to_zme_int: specific decoding module of ME arm.
``` 

#### subnetwork_M_PCs.py:
A modified version of subnetwork_M.py for when instead of arbor images, we use the PCs of the arbor images.

#### subnetwork_M_PCs_features.py:
A modified version of subnetwork_M_PCs.py for when not only we use the PCs of arbor images but also we use the m-features as well.

#### augmentation.py:
All the necessary augmentation functions to work with arbor density images. Such as expanding or shifting or blurring the images.
