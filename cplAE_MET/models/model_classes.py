# %%
from xml.sax import xmlreader
import torch
import torch.nn as nn
from cplAE_MET.models.subnetworks_M import AE_M, Dec_zm_int_to_xm, Enc_xm_to_zm_int
from cplAE_MET.models.subnetworks_E import AE_E, Dec_ze_int_to_xe, Enc_xe_to_ze_int
from cplAE_MET.models.subnetworks_T import AE_T
from cplAE_MET.models.subnetworks_ME import AE_ME_int
from torch_utils import min_var_loss

def get_output_dict(out_list, key_list):
    return dict(zip(key_list, out_list))

class Model_ME_T(nn.Module):
    """ME, T autoencoder
    """

    def __init__(self, model_config):

        super(Model_ME_T, self).__init__()
        
        self.ae_t = AE_T(config=model_config)
        self.ae_e = AE_E(config=model_config, gnoise_std=model_config['E']['gnoise_std'])
        self.ae_m = AE_M(config=model_config)
        self.ae_me = AE_ME_int(config=model_config)
        self.me_e_encoder = Enc_xe_to_ze_int(gnoise_std=model_config['E']['gnoise_std'],
                                             gnoise_std_frac=model_config['E']['gnoise_std_frac'],
                                             dropout_p=model_config['E']['dropout_p'])
        self.me_e_decoder = Dec_ze_int_to_xe()
        self.me_m_encoder = Enc_xm_to_zm_int()
        self.me_m_decoder = Dec_zm_int_to_xm()
        self.augment_decoders = model_config['augment_decoders']
        return
    

    def compute_rec_loss(self, x, xr, valid_x):
        return torch.mean(torch.masked_select(torch.square(x-xr), valid_x))
    
    def compute_cpl_loss(self, z1_paired, z2_paired):
        return min_var_loss(z1_paired, z2_paired)
    
    def forward(self, input):

        xm=input['xm']
        xsd=input['xsd']
        xe=input['xe']
        xt=input['xt']
        valid_xm=input['valid_xm']
        valid_xsd=input['valid_xsd']
        valid_xe=input['valid_xe']
        valid_xt=input['valid_xt']
        is_me_1d=torch.logical_and(input['is_m_1d'], input['is_e_1d'])
        is_met_1d=torch.logical_and(is_me_1d, input['is_t_1d'])

        # t arm
        zt, xrt = self.ae_t(xt)

        # e arm
        _, ze, _, xre = self.ae_e(xe)

        # m arm
        if self.training:
            xm = input['aug_xm']
            noisy_xsd=input['noisy_xsd']
            _, zm, _, xrm, xrsd = self.ae_m(xm, noisy_xsd)
        else:
            _, zm, _, xrm, xrsd = self.ae_m(xm, xsd)
        
        # me arm
        ze_int_enc_paired = self.me_e_encoder(xe)
        zm_int_enc_paired = self.me_m_encoder(xm, xsd)
        zme_paired = self.ae_me.enc_zme_int_to_zme(zm_int_enc_paired, ze_int_enc_paired)
        zm_int_dec_paired, ze_int_dec_paired = self.ae_me.dec_zme_to_zme_int(zme_paired)
        xre_me_paired = self.me_e_decoder(ze_int_dec_paired)
        xrm_me_paired, xrsd_me_paired = self.me_m_decoder(zm_int_dec_paired,
                                                          self.me_m_encoder.pool_0_ind,
                                                          self.me_m_encoder.pool_1_ind)

        # Loss calculations
        loss_dict={}
        loss_dict['rec_t'] = self.compute_rec_loss(xt, xrt, valid_xt)
        loss_dict['rec_e'] = self.compute_rec_loss(xe, xre, valid_xe)
        loss_dict['rec_m'] = self.compute_rec_loss(xm, xrm, valid_xm)
        loss_dict['rec_sd'] = self.compute_rec_loss(xsd, xrsd, valid_xsd)    
        loss_dict['rec_m_me'] = self.compute_rec_loss(xm[is_me_1d, ...], xrm_me_paired[is_me_1d, ...], valid_xm[is_me_1d, ...])
        loss_dict['rec_sd_me'] = self.compute_rec_loss(xsd[is_me_1d, ...], xrsd_me_paired[is_me_1d, ...], valid_xsd[is_me_1d, ...])
        loss_dict['rec_e_me'] = self.compute_rec_loss(xe[is_me_1d, ...], xre_me_paired[is_me_1d, ...], valid_xe[is_me_1d, ...])

        loss_dict['cpl_me_t'] = self.compute_cpl_loss(zme_paired[is_met_1d, ...], zt[is_met_1d, ...])
        loss_dict['cpl_me_m'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...].detach(), zm[is_me_1d, ...])
        loss_dict['cpl_me_e'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...].detach(), ze[is_me_1d, ...])

        ############################## get output dicts
        z_dict = get_output_dict([zm, ze, zt, zme_paired], 
                                 ["zm", "ze", "zt", "zme_paired"])

        xr_dict = get_output_dict([xrm, xrsd, xre, xrt, xrm_me_paired, xre_me_paired],
                                  ['xrm', 'xrsd', 'xre', 'xrt', 'xrm_me_paired', 'xre_me_paired'])

        return loss_dict, z_dict, xr_dict




class Model_M(nn.Module):
    """M autoencoder
    """

    def __init__(self, model_config):

        super(Model_M, self).__init__()
        self.ae_m = AE_M(config=model_config)
        return
    

    def compute_rec_loss(self, x, xr, valid_x):
        return torch.mean(torch.masked_select(torch.square(x-xr), valid_x))
    
    
    def forward(self, input):

        xm=input['xm']
        xsd=input['xsd']
        valid_xm=input['valid_xm']
        valid_xsd=input['valid_xsd']
    
        # m forward 
        if self.training:
            xm = input['aug_xm']
            noisy_xsd=input['noisy_xsd']
            _, zm, _, xrm, xrsd = self.ae_m(xm, noisy_xsd)
        else:
            _, zm, _, xrm, xrsd = self.ae_m(xm, xsd)
        

        # Loss calculations
        loss_dict={}
        loss_dict['rec_m'] = self.compute_rec_loss(xm, xrm, valid_xm)
        loss_dict['rec_sd'] = self.compute_rec_loss(xsd, xrsd, valid_xsd)    

        ############################## get output dicts
        z_dict = get_output_dict([zm], ["zm"])

        xr_dict = get_output_dict([xrm, xrsd], ['xrm', 'xrsd'])

        return loss_dict, z_dict, xr_dict