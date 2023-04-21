import torch
import torch.nn as nn
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
        self.me_m_encoder = Enc_xm_to_zm_int(gnoise_std=model_config['M']['gnoise_std'],
                                             gnoise_std_frac=model_config['M']['gnoise_std_frac'],
                                             dropout_p=model_config['M']['dropout_p'])
        self.me_m_decoder = Dec_zm_int_to_xm()
        self.variational = model_config['variational']
        return
    

    def compute_rec_loss(self, x, xr, valid_x):
        return torch.mean(torch.masked_select(torch.square(x-xr), valid_x))
    
    def compute_cpl_loss(self, z1_paired, z2_paired):
        return min_var_loss(z1_paired, z2_paired)

    def compute_KLD_loss(self, mu, log_sigma, mask):    
        mu = mu[mask] 
        log_sigma = log_sigma[mask]  
        return (-0.5 * torch.mean(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=0)).sum()
    
    def compute_BCE_loss(self, x, target):
        loss = nn.BCELoss()
        x_bin = 0. * x
        x_bin[x.abs()>0.001] = 1.
        target_bin = 0. * target
        target_bin[target.abs()>0.001] = 1.
        return loss(x_bin, target_bin)

    
    def forward(self, input):

        xm=input['xm']
        xe=input['xe']
        xt=input['xt']
        valid_xm=input['valid_xm']
        valid_xe=input['valid_xe']
        valid_xt=input['valid_xt']
        is_t_1d = input['is_t_1d']
        is_e_1d = input['is_e_1d']
        is_m_1d = input['is_m_1d']
        is_te_1d=torch.logical_and(input['is_t_1d'], input['is_e_1d'])
        is_tm_1d=torch.logical_and(input['is_t_1d'], input['is_m_1d'])
        is_me_1d=torch.logical_and(input['is_m_1d'], input['is_e_1d'])
        is_met_1d=torch.logical_and(is_me_1d, input['is_t_1d'])
        

        # t arm
        zt, xrt, mu_t, log_sigma_t = self.ae_t(xt)

        # e arm
        _, ze, _, xre, mu_e, log_sigma_e = self.ae_e(xe)

        # m arm
        _, zm, _, xrm, mu_m, log_sigma_m = self.ae_m(xm)
        
        # me arm
        ze_int_enc_paired = self.me_e_encoder(xe)
        zm_int_enc_paired = self.me_m_encoder(xm)

        if self.variational:
            mu_me, sigma_me = self.ae_me.enc_zme_int_to_zme(zm_int_enc_paired, ze_int_enc_paired)
            log_sigma_me = (sigma_me + 1e-6).log()
            zme_paired = self.ae_me.dec_zme_to_zme_int.reparametrize(mu_me, sigma_me)
        else:
            zme_paired = self.ae_me.enc_zme_int_to_zme(zm_int_enc_paired, ze_int_enc_paired)
            mu_me = []
            log_sigma_me = []

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


        # loss_dict['BCELoss_m'] = self.compute_BCE_loss(xrm[is_m_1d][valid_xm[is_m_1d]], xm[is_m_1d][valid_xm[is_m_1d]])
        # loss_dict['BCELoss_me_m'] = self.compute_BCE_loss(xrm[is_me_1d][valid_xm[is_me_1d]], xm[is_me_1d][valid_xm[is_me_1d]])
        
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

        if self.variational:
            loss_dict['KLD_t'] = self.compute_KLD_loss(mu_t, log_sigma_t, is_t_1d)
            loss_dict['KLD_e'] = self.compute_KLD_loss(mu_e, log_sigma_e, is_e_1d)
            loss_dict['KLD_m'] = self.compute_KLD_loss(mu_m, log_sigma_m, is_m_1d)
            loss_dict['KLD_me_paired'] = self.compute_KLD_loss(mu_me, log_sigma_me, is_me_1d)


        ############################## get output dicts
        z_dict = get_output_dict([zm, ze, zt, zme_paired], 
                                 ["zm", "ze", "zt", "zme_paired"])

        xr_dict = get_output_dict([xrm, xre, xrt, xrm_me_paired, xre_me_paired],
                                  ['xrm', 'xre', 'xrt', 'xrm_me_paired', 'xre_me_paired'])

        return loss_dict, z_dict, xr_dict