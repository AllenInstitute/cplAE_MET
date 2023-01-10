# %%
import torch
import torch.nn as nn
# from cplAE_MET.models.subnetwork_M_PCs import AE_M, Dec_zm_int_to_xm, Enc_xm_to_zm_int
# from cplAE_MET.models.subnetworks_M import AE_M, Dec_zm_int_to_xm, Enc_xm_to_zm_int
from cplAE_MET.models.subnetwork_M_PCs_features import AE_M, Dec_zm_int_to_xm, Enc_xm_to_zm_int
from cplAE_MET.models.subnetworks_E import AE_E, Dec_ze_int_to_xe, Enc_xe_to_ze_int
from cplAE_MET.models.subnetworks_T import AE_T
from cplAE_MET.models.subnetworks_ME import AE_ME_int
from cplAE_MET.models.torch_utils import min_var_loss

def get_output_dict(out_list, key_list):
    return dict(zip(key_list, out_list))

class Model_ME_T(nn.Module):
    """ME, T autoencoder
    """

    def __init__(self, model_config):

        super(Model_ME_T, self).__init__()
        
        self.ae_t = AE_T(config=model_config)
        self.ae_e = AE_E(config=model_config, gnoise_std=model_config['E']['gnoise_std'])
        self.ae_m = AE_M(config=model_config, gnoise_std=model_config['M']['gnoise_std'])
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
        is_te_1d=torch.logical_and(input['is_t_1d'], input['is_e_1d'])
        is_tm_1d=torch.logical_and(input['is_t_1d'], input['is_m_1d'])
        is_me_1d=torch.logical_and(input['is_m_1d'], input['is_e_1d'])
        is_met_1d=torch.logical_and(is_me_1d, input['is_t_1d'])

        # t arm
        zt, xrt = self.ae_t(xt)

        # e arm
        _, ze, _, xre = self.ae_e(xe)

        # m arm
        if self.training:
            noisy_aug_xm = input['noisy_aug_xm']
            xm = input['aug_xm']
            noisy_xsd=input['noisy_xsd']
            _, zm, _, xrm, xrsd = self.ae_m(noisy_aug_xm, noisy_xsd)
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

        loss_dict['cpl_me->t'] = self.compute_cpl_loss(zme_paired[is_met_1d, ...].detach(), zt[is_met_1d, ...])
        loss_dict['cpl_t->me'] = self.compute_cpl_loss(zme_paired[is_met_1d, ...], zt[is_met_1d, ...].detach())

        loss_dict['cpl_me->m'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...].detach(), zm[is_me_1d, ...])
        loss_dict['cpl_m->me'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...], zm[is_me_1d, ...].detach())

        loss_dict['cpl_me->e'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...].detach(), ze[is_me_1d, ...])
        loss_dict['cpl_e->me'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...], ze[is_me_1d, ...].detach())

        loss_dict['cpl_t->e'] = self.compute_cpl_loss(zt[is_te_1d, ...].detach(), ze[is_te_1d, ...])
        loss_dict['cpl_e->t'] = self.compute_cpl_loss(zt[is_te_1d, ...], ze[is_te_1d, ...].detach())

        loss_dict['cpl_t->m'] = self.compute_cpl_loss(zt[is_tm_1d, ...].detach(), zm[is_tm_1d, ...])
        loss_dict['cpl_m->t'] = self.compute_cpl_loss(zt[is_tm_1d, ...], zm[is_tm_1d, ...].detach())

        loss_dict['cpl_m->e'] = self.compute_cpl_loss(zm[is_me_1d, ...].detach(), ze[is_me_1d, ...])
        loss_dict['cpl_e->m'] = self.compute_cpl_loss(zm[is_me_1d, ...], ze[is_me_1d, ...].detach())


        # Augment decoders
        if (self.training and self.augment_decoders):
            # T and ME
            zm_int_dec_paired_from_zt, ze_int_dec_paired_from_zt = self.ae_me.dec_zme_to_zme_int(zt.detach())
            xre_me_paired_from_zt = self.me_e_decoder(ze_int_dec_paired_from_zt)
            xrm_me_paired_from_zt, xrsd_me_paired_from_zt = self.me_m_decoder(zm_int_dec_paired_from_zt,
                                                                          self.me_m_encoder.pool_0_ind,
                                                                          self.me_m_encoder.pool_1_ind)

            xrt_from_zme_paired = self.ae_t.dec_zt_to_xt(zme_paired.detach())

            # T and E
            ze_int_dec_from_zt = self.ae_e.dec_ze_to_ze_int(zt.detach())
            xre_from_zt = self.ae_e.dec_ze_int_to_xe(ze_int_dec_from_zt)

            # T and M
            zm_int_dec_from_zt = self.ae_m.dec_zm_to_zm_int(zt.detach())
            xrm_from_zt, xrsd_from_zt = self.ae_m.dec_zm_int_to_xm(zm_int_dec_from_zt, 
                                                                   self.ae_m.enc_xm_to_zm_int.pool_0_ind, 
                                                                   self.ae_m.enc_xm_to_zm_int.pool_1_ind)

            # M and ME
            zm_int_dec_from_zme_paired = self.ae_m.dec_zm_to_zm_int(zme_paired.detach())
            xrm_from_zme_paired, xrsd_from_zme_paired = self.ae_m.dec_zm_int_to_xm(zm_int_dec_from_zme_paired, 
                                                                                   self.ae_m.enc_xm_to_zm_int.pool_0_ind, 
                                                                                   self.ae_m.enc_xm_to_zm_int.pool_1_ind)

            # E and ME
            ze_int_dec_from_zme_paired = self.ae_e.dec_ze_to_ze_int(zme_paired.detach())
            xre_from_zme_paired = self.ae_e.dec_ze_int_to_xe(ze_int_dec_from_zme_paired)

            # E and M
            zm_int_dec_from_ze = self.ae_m.dec_zm_to_zm_int(ze.detach())
            xrm_from_ze, xrsd_from_ze = self.ae_m.dec_zm_int_to_xm(zm_int_dec_from_ze, 
                                                                   self.ae_m.enc_xm_to_zm_int.pool_0_ind, 
                                                                   self.ae_m.enc_xm_to_zm_int.pool_1_ind)


            # T and ME
            loss_dict['rec_e_me_paired_from_zt'] = self.compute_rec_loss(xe[is_met_1d, ...], xre_me_paired_from_zt[is_met_1d, ...], valid_xe[is_met_1d, ...])
            loss_dict['rec_m_me_paired_from_zt'] = self.compute_rec_loss(xm[is_met_1d, ...], xrm_me_paired_from_zt[is_met_1d, ...], valid_xm[is_met_1d, ...])
            loss_dict['rec_sd_me_paired_from_zt'] = self.compute_rec_loss(xsd[is_met_1d, ...], xrsd_me_paired_from_zt[is_met_1d, ...], valid_xsd[is_met_1d, ...])            
            loss_dict['rec_t_from_zme_paired'] = self.compute_rec_loss(xt[is_met_1d, ...], xrt_from_zme_paired[is_met_1d, ...], valid_xt[is_met_1d, ...])
            # T and E
            loss_dict['rec_e_from_zt'] = self.compute_rec_loss(xe[is_te_1d, ...], xre_from_zt[is_te_1d, ...], valid_xe[is_te_1d, ...])
            # T and M
            loss_dict['rec_m_from_zt'] = self.compute_rec_loss(xm[is_tm_1d, ...], xrm_from_zt[is_tm_1d, ...], valid_xm[is_tm_1d, ...])
            loss_dict['rec_sd_from_zt'] = self.compute_rec_loss(xsd[is_tm_1d, ...], xrsd_from_zt[is_tm_1d, ...], valid_xsd[is_tm_1d, ...])
            # M and ME
            loss_dict['rec_m_from_zme_paired'] = self.compute_rec_loss(xm[is_me_1d, ...], xrm_from_zme_paired[is_me_1d, ...], valid_xm[is_me_1d, ...])
            loss_dict['rec_sd_from_zme_paired'] = self.compute_rec_loss(xsd[is_me_1d, ...], xrsd_from_zme_paired[is_me_1d, ...], valid_xsd[is_me_1d, ...])
            # E and ME
            loss_dict['rec_e_from_zme_paired'] = self.compute_rec_loss(xe[is_me_1d, ...], xre_from_zme_paired[is_me_1d, ...], valid_xe[is_me_1d, ...])
            # M and E
            loss_dict['rec_m_from_ze'] = self.compute_rec_loss(xm[is_me_1d, ...], xrm_from_ze[is_me_1d, ...], valid_xm[is_me_1d, ...])
            loss_dict['rec_sd_from_ze'] = self.compute_rec_loss(xsd[is_me_1d, ...], xrsd_from_ze[is_me_1d, ...], valid_xsd[is_me_1d, ...])

        ############################## get output dicts
        z_dict = get_output_dict([zm, ze, zt, zme_paired], 
                                 ["zm", "ze", "zt", "zme_paired"])

        xr_dict = get_output_dict([xrm, xrsd, xre, xrt, xrm_me_paired, xre_me_paired],
                                  ['xrm', 'xrsd', 'xre', 'xrt', 'xrm_me_paired', 'xre_me_paired'])

        return loss_dict, z_dict, xr_dict


class Model_ME_T_v1(nn.Module):
    """ME, T autoencoder
    """

    def __init__(self, model_config):

        super(Model_ME_T_v1, self).__init__()
        
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
        is_te_1d=torch.logical_and(input['is_t_1d'], input['is_e_1d'])
        is_tm_1d=torch.logical_and(input['is_t_1d'], input['is_m_1d'])
        is_me_1d=torch.logical_and(input['is_m_1d'], input['is_e_1d'])
        is_met_1d=torch.logical_and(is_me_1d, input['is_t_1d'])

        # t arm
        zt, xrt = self.ae_t(xt)

        # e arm
        _, ze, _, xre = self.ae_e(xe)

        # m arm
        _, zm, _, xrm, xrsd = self.ae_m(xm, xsd)
        
        # me arm
        ze_int_enc_paired = self.me_e_encoder(xe)
        zm_int_enc_paired = self.me_m_encoder(xm, xsd)
        zme_paired = self.ae_me.enc_zme_int_to_zme(zm_int_enc_paired, ze_int_enc_paired)
        zm_int_dec_paired, ze_int_dec_paired = self.ae_me.dec_zme_to_zme_int(zme_paired)
        xre_me_paired = self.me_e_decoder(ze_int_dec_paired)
        xrm_me_paired, xrsd_me_paired = self.me_m_decoder(zm_int_dec_paired)

        # Loss calculations
        loss_dict={}
        loss_dict['rec_t'] = self.compute_rec_loss(xt, xrt, valid_xt)
        loss_dict['rec_e'] = self.compute_rec_loss(xe, xre, valid_xe)
        loss_dict['rec_m'] = self.compute_rec_loss(xm, xrm, valid_xm)
        loss_dict['rec_sd'] = self.compute_rec_loss(xsd, xrsd, valid_xsd)    
        loss_dict['rec_m_me'] = self.compute_rec_loss(xm[is_me_1d, ...], xrm_me_paired[is_me_1d, ...], valid_xm[is_me_1d, ...])
        loss_dict['rec_sd_me'] = self.compute_rec_loss(xsd[is_me_1d, ...], xrsd_me_paired[is_me_1d, ...], valid_xsd[is_me_1d, ...])
        loss_dict['rec_e_me'] = self.compute_rec_loss(xe[is_me_1d, ...], xre_me_paired[is_me_1d, ...], valid_xe[is_me_1d, ...])

        loss_dict['cpl_me->t'] = self.compute_cpl_loss(zme_paired[is_met_1d, ...].detach(), zt[is_met_1d, ...])
        loss_dict['cpl_t->me'] = self.compute_cpl_loss(zme_paired[is_met_1d, ...], zt[is_met_1d, ...].detach())

        loss_dict['cpl_me->m'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...].detach(), zm[is_me_1d, ...])
        loss_dict['cpl_m->me'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...], zm[is_me_1d, ...].detach())

        loss_dict['cpl_me->e'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...].detach(), ze[is_me_1d, ...])
        loss_dict['cpl_e->me'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...], ze[is_me_1d, ...].detach())

        loss_dict['cpl_t->e'] = self.compute_cpl_loss(zt[is_te_1d, ...].detach(), ze[is_te_1d, ...])
        loss_dict['cpl_e->t'] = self.compute_cpl_loss(zt[is_te_1d, ...], ze[is_te_1d, ...].detach())

        loss_dict['cpl_t->m'] = self.compute_cpl_loss(zt[is_tm_1d, ...].detach(), zm[is_tm_1d, ...])
        loss_dict['cpl_m->t'] = self.compute_cpl_loss(zt[is_tm_1d, ...], zm[is_tm_1d, ...].detach())

        loss_dict['cpl_m->e'] = self.compute_cpl_loss(zm[is_me_1d, ...].detach(), ze[is_me_1d, ...])
        loss_dict['cpl_e->m'] = self.compute_cpl_loss(zm[is_me_1d, ...], ze[is_me_1d, ...].detach())


        # Augment decoders
        if (self.training and self.augment_decoders):
            # T and ME
            zm_int_dec_paired_from_zt, ze_int_dec_paired_from_zt = self.ae_me.dec_zme_to_zme_int(zt.detach())
            xre_me_paired_from_zt = self.me_e_decoder(ze_int_dec_paired_from_zt)
            xrm_me_paired_from_zt, xrsd_me_paired_from_zt = self.me_m_decoder(zm_int_dec_paired_from_zt)

            xrt_from_zme_paired = self.ae_t.dec_zt_to_xt(zme_paired.detach())

            # T and E
            ze_int_dec_from_zt = self.ae_e.dec_ze_to_ze_int(zt.detach())
            xre_from_zt = self.ae_e.dec_ze_int_to_xe(ze_int_dec_from_zt)

            # T and M
            zm_int_dec_from_zt = self.ae_m.dec_zm_to_zm_int(zt.detach())
            xrm_from_zt, xrsd_from_zt = self.ae_m.dec_zm_int_to_xm(zm_int_dec_from_zt)

            # M and ME
            zm_int_dec_from_zme_paired = self.ae_m.dec_zm_to_zm_int(zme_paired.detach())
            xrm_from_zme_paired, xrsd_from_zme_paired = self.ae_m.dec_zm_int_to_xm(zm_int_dec_from_zme_paired)

            # E and ME
            ze_int_dec_from_zme_paired = self.ae_e.dec_ze_to_ze_int(zme_paired.detach())
            xre_from_zme_paired = self.ae_e.dec_ze_int_to_xe(ze_int_dec_from_zme_paired)

            # E and M
            zm_int_dec_from_ze = self.ae_m.dec_zm_to_zm_int(ze.detach())
            xrm_from_ze, xrsd_from_ze = self.ae_m.dec_zm_int_to_xm(zm_int_dec_from_ze)


            # T and ME
            loss_dict['rec_e_me_paired_from_zt'] = self.compute_rec_loss(xe[is_met_1d, ...], xre_me_paired_from_zt[is_met_1d, ...], valid_xe[is_met_1d, ...])
            loss_dict['rec_m_me_paired_from_zt'] = self.compute_rec_loss(xm[is_met_1d, ...], xrm_me_paired_from_zt[is_met_1d, ...], valid_xm[is_met_1d, ...])
            loss_dict['rec_sd_me_paired_from_zt'] = self.compute_rec_loss(xsd[is_met_1d, ...], xrsd_me_paired_from_zt[is_met_1d, ...], valid_xsd[is_met_1d, ...])            
            loss_dict['rec_t_from_zme_paired'] = self.compute_rec_loss(xt[is_met_1d, ...], xrt_from_zme_paired[is_met_1d, ...], valid_xt[is_met_1d, ...])
            # T and E
            loss_dict['rec_e_from_zt'] = self.compute_rec_loss(xe[is_te_1d, ...], xre_from_zt[is_te_1d, ...], valid_xe[is_te_1d, ...])
            # T and M
            loss_dict['rec_m_from_zt'] = self.compute_rec_loss(xm[is_tm_1d, ...], xrm_from_zt[is_tm_1d, ...], valid_xm[is_tm_1d, ...])
            loss_dict['rec_sd_from_zt'] = self.compute_rec_loss(xsd[is_tm_1d, ...], xrsd_from_zt[is_tm_1d, ...], valid_xsd[is_tm_1d, ...])
            # M and ME
            loss_dict['rec_m_from_zme_paired'] = self.compute_rec_loss(xm[is_me_1d, ...], xrm_from_zme_paired[is_me_1d, ...], valid_xm[is_me_1d, ...])
            loss_dict['rec_sd_from_zme_paired'] = self.compute_rec_loss(xsd[is_me_1d, ...], xrsd_from_zme_paired[is_me_1d, ...], valid_xsd[is_me_1d, ...])
            # E and ME
            loss_dict['rec_e_from_zme_paired'] = self.compute_rec_loss(xe[is_me_1d, ...], xre_from_zme_paired[is_me_1d, ...], valid_xe[is_me_1d, ...])
            # M and E
            loss_dict['rec_m_from_ze'] = self.compute_rec_loss(xm[is_me_1d, ...], xrm_from_ze[is_me_1d, ...], valid_xm[is_me_1d, ...])
            loss_dict['rec_sd_from_ze'] = self.compute_rec_loss(xsd[is_me_1d, ...], xrsd_from_ze[is_me_1d, ...], valid_xsd[is_me_1d, ...])

        ############################## get output dicts
        z_dict = get_output_dict([zm, ze, zt, zme_paired], 
                                 ["zm", "ze", "zt", "zme_paired"])

        xr_dict = get_output_dict([xrm, xrsd, xre, xrt, xrm_me_paired, xre_me_paired],
                                  ['xrm', 'xrsd', 'xre', 'xrt', 'xrm_me_paired', 'xre_me_paired'])

        return loss_dict, z_dict, xr_dict




class Model_ME_T_v2(nn.Module):
    """ME, T autoencoder
    """

    def __init__(self, model_config):

        super(Model_ME_T_v2, self).__init__()
        
        self.ae_t = AE_T(config=model_config)
        self.ae_e = AE_E(config=model_config, gnoise_std=model_config['E']['gnoise_std'])
        self.ae_m = AE_M(config=model_config, gnoise_std=model_config['M']['gnoise_std'])
        self.ae_me = AE_ME_int(config=model_config)
        self.me_e_encoder = Enc_xe_to_ze_int(gnoise_std=model_config['E']['gnoise_std'],
                                             gnoise_std_frac=model_config['E']['gnoise_std_frac'],
                                             dropout_p=model_config['E']['dropout_p'])
        self.me_e_decoder = Dec_ze_int_to_xe()
        self.me_m_encoder = Enc_xm_to_zm_int(gnoise_std=model_config['M']['gnoise_std'],
                                             gnoise_std_frac=model_config['M']['gnoise_std_frac'],
                                             dropout_p=model_config['M']['dropout_p'])
        self.me_m_decoder = Dec_zm_int_to_xm()
        # self.augment_decoders = model_config['augment_decoders']
        return
    

    def compute_rec_loss(self, x, xr, valid_x):
        return torch.mean(torch.masked_select(torch.square(x-xr), valid_x))
    
    def compute_cpl_loss(self, z1_paired, z2_paired):
        return min_var_loss(z1_paired, z2_paired)
    
    def forward(self, input):

        xm=input['xm']
        xe=input['xe']
        xt=input['xt']
        valid_xm=input['valid_xm']
        valid_xe=input['valid_xe']
        valid_xt=input['valid_xt']
        is_te_1d=torch.logical_and(input['is_t_1d'], input['is_e_1d'])
        is_tm_1d=torch.logical_and(input['is_t_1d'], input['is_m_1d'])
        is_me_1d=torch.logical_and(input['is_m_1d'], input['is_e_1d'])
        is_met_1d=torch.logical_and(is_me_1d, input['is_t_1d'])

        # t arm
        zt, xrt = self.ae_t(xt)

        # e arm
        _, ze, _, xre = self.ae_e(xe)

        # m arm
        _, zm, _, xrm = self.ae_m(xm)
        
        # me arm
        ze_int_enc_paired = self.me_e_encoder(xe)
        zm_int_enc_paired = self.me_m_encoder(xm)
        zme_paired = self.ae_me.enc_zme_int_to_zme(zm_int_enc_paired, ze_int_enc_paired)
        zm_int_dec_paired, ze_int_dec_paired = self.ae_me.dec_zme_to_zme_int(zme_paired)
        xre_me_paired = self.me_e_decoder(ze_int_dec_paired)
        xrm_me_paired = self.me_m_decoder(zm_int_dec_paired)

        # Loss calculations
        loss_dict={}
        loss_dict['rec_t'] = self.compute_rec_loss(xt, xrt, valid_xt)
        loss_dict['rec_e'] = self.compute_rec_loss(xe, xre, valid_xe)
        loss_dict['rec_m'] = self.compute_rec_loss(xm, xrm, valid_xm)
        loss_dict['rec_m_me'] = self.compute_rec_loss(xm[is_me_1d, ...], xrm_me_paired[is_me_1d, ...], valid_xm[is_me_1d, ...])
        loss_dict['rec_e_me'] = self.compute_rec_loss(xe[is_me_1d, ...], xre_me_paired[is_me_1d, ...], valid_xe[is_me_1d, ...])

        loss_dict['cpl_me->t'] = self.compute_cpl_loss(zme_paired[is_met_1d, ...].detach(), zt[is_met_1d, ...])
        loss_dict['cpl_t->me'] = self.compute_cpl_loss(zme_paired[is_met_1d, ...], zt[is_met_1d, ...].detach())

        loss_dict['cpl_me->m'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...].detach(), zm[is_me_1d, ...])
        loss_dict['cpl_m->me'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...], zm[is_me_1d, ...].detach())

        loss_dict['cpl_me->e'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...].detach(), ze[is_me_1d, ...])
        loss_dict['cpl_e->me'] = self.compute_cpl_loss(zme_paired[is_me_1d, ...], ze[is_me_1d, ...].detach())

        loss_dict['cpl_t->e'] = self.compute_cpl_loss(zt[is_te_1d, ...].detach(), ze[is_te_1d, ...])
        loss_dict['cpl_e->t'] = self.compute_cpl_loss(zt[is_te_1d, ...], ze[is_te_1d, ...].detach())

        loss_dict['cpl_t->m'] = self.compute_cpl_loss(zt[is_tm_1d, ...].detach(), zm[is_tm_1d, ...])
        loss_dict['cpl_m->t'] = self.compute_cpl_loss(zt[is_tm_1d, ...], zm[is_tm_1d, ...].detach())

        loss_dict['cpl_m->e'] = self.compute_cpl_loss(zm[is_me_1d, ...].detach(), ze[is_me_1d, ...])
        loss_dict['cpl_e->m'] = self.compute_cpl_loss(zm[is_me_1d, ...], ze[is_me_1d, ...].detach())


        # # Augment decoders
        # if (self.training and self.augment_decoders):
        #     # T and ME
        #     zm_int_dec_paired_from_zt, ze_int_dec_paired_from_zt = self.ae_me.dec_zme_to_zme_int(zt.detach())
        #     xre_me_paired_from_zt = self.me_e_decoder(ze_int_dec_paired_from_zt)
        #     xrm_me_paired_from_zt = self.me_m_decoder(zm_int_dec_paired_from_zt)

        #     xrt_from_zme_paired = self.ae_t.dec_zt_to_xt(zme_paired.detach())

        #     # T and E
        #     ze_int_dec_from_zt = self.ae_e.dec_ze_to_ze_int(zt.detach())
        #     xre_from_zt = self.ae_e.dec_ze_int_to_xe(ze_int_dec_from_zt)

        #     # T and M
        #     zm_int_dec_from_zt = self.ae_m.dec_zm_to_zm_int(zt.detach())
        #     xrm_from_zt = self.ae_m.dec_zm_int_to_xm(zm_int_dec_from_zt)

        #     # M and ME
        #     zm_int_dec_from_zme_paired = self.ae_m.dec_zm_to_zm_int(zme_paired.detach())
        #     xrm_from_zme_paired = self.ae_m.dec_zm_int_to_xm(zm_int_dec_from_zme_paired)

        #     # E and ME
        #     ze_int_dec_from_zme_paired = self.ae_e.dec_ze_to_ze_int(zme_paired.detach())
        #     xre_from_zme_paired = self.ae_e.dec_ze_int_to_xe(ze_int_dec_from_zme_paired)

        #     # E and M
        #     zm_int_dec_from_ze = self.ae_m.dec_zm_to_zm_int(ze.detach())
        #     xrm_from_ze = self.ae_m.dec_zm_int_to_xm(zm_int_dec_from_ze)


        #     # T and ME
        #     loss_dict['rec_e_me_paired_from_zt'] = self.compute_rec_loss(xe[is_met_1d, ...], xre_me_paired_from_zt[is_met_1d, ...], valid_xe[is_met_1d, ...])
        #     loss_dict['rec_m_me_paired_from_zt'] = self.compute_rec_loss(xm[is_met_1d, ...], xrm_me_paired_from_zt[is_met_1d, ...], valid_xm[is_met_1d, ...])
        #     loss_dict['rec_t_from_zme_paired'] = self.compute_rec_loss(xt[is_met_1d, ...], xrt_from_zme_paired[is_met_1d, ...], valid_xt[is_met_1d, ...])
        #     # T and E
        #     loss_dict['rec_e_from_zt'] = self.compute_rec_loss(xe[is_te_1d, ...], xre_from_zt[is_te_1d, ...], valid_xe[is_te_1d, ...])
        #     # T and M
        #     loss_dict['rec_m_from_zt'] = self.compute_rec_loss(xm[is_tm_1d, ...], xrm_from_zt[is_tm_1d, ...], valid_xm[is_tm_1d, ...])
        #     # M and ME
        #     loss_dict['rec_m_from_zme_paired'] = self.compute_rec_loss(xm[is_me_1d, ...], xrm_from_zme_paired[is_me_1d, ...], valid_xm[is_me_1d, ...])
        #     # E and ME
        #     loss_dict['rec_e_from_zme_paired'] = self.compute_rec_loss(xe[is_me_1d, ...], xre_from_zme_paired[is_me_1d, ...], valid_xe[is_me_1d, ...])
        #     # M and E
        #     loss_dict['rec_m_from_ze'] = self.compute_rec_loss(xm[is_me_1d, ...], xrm_from_ze[is_me_1d, ...], valid_xm[is_me_1d, ...])

        ############################## get output dicts
        z_dict = get_output_dict([zm, ze, zt, zme_paired], 
                                 ["zm", "ze", "zt", "zme_paired"])

        xr_dict = get_output_dict([xrm, xre, xrt, xrm_me_paired, xre_me_paired],
                                  ['xrm', 'xre', 'xrt', 'xrm_me_paired', 'xre_me_paired'])

        return loss_dict, z_dict, xr_dict