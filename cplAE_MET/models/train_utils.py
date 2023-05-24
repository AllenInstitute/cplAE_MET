import torch
import numpy as np
from cplAE_MET.utils.utils import savepkl
from cplAE_MET.models.torch_utils import tonumpy
from cplAE_MET.utils.utils import rm_emp_end_str



def init_losses(loss_dict):
    '''
    Initialize the loss dict to zero.
    Args:
        loss_dict: loss dictionary 
    '''
    t_loss = {}
    for k in loss_dict.keys():
        t_loss[k] = 0.
    return t_loss

def calculate_arbor_densities_from_nmfs_v1(rec_nmf, input_datamat):
    '''Reconstruct the arbor densities from the recnstructed nmfs
    Args:
        rec_nmf: reconstructed nmfs 
        input_datamat: input mat file which is loaded
    '''
    min_lim = 0
    rec_channel = {}
    for channel in ['ax', 'de', 'api', 'bas']:
        comp_name = "M_nmf_components_" + channel
        total_var_name = "M_nmf_total_vars_" + channel
        col_limit = (min_lim , min_lim + input_datamat[comp_name].shape[0])
        rec_channel[channel] = (np.dot(rec_nmf[:, col_limit[0]:col_limit[1]] * input_datamat[total_var_name], input_datamat[comp_name])).reshape(-1, 120, 4)
        min_lim = col_limit[1]
    
    return np.stack((rec_channel['ax'], rec_channel['de'], rec_channel['api'], rec_channel['bas'] ), axis=3)


def calculate_arbor_densities_from_nmfs(rec_nmf, input_datamat):
    '''Reconstruct the arbor densities from the recnstructed nmfs
    Args:
        rec_nmf: reconstructed nmfs 
        input_datamat: input mat file which is loaded
    '''
    min_lim = 0
    rec_channel = {}
    # for channel in ['inh', 'exc']:
    for channel in ['exc']:

        comp_name = "M_nmf_components_" + channel
        total_var_name = "M_nmf_total_vars_" + channel
        col_limit = (min_lim , min_lim + input_datamat[comp_name].shape[0])
        rec_channel[channel] = (np.dot(rec_nmf[:, col_limit[0]:col_limit[1]] * input_datamat[total_var_name], input_datamat[comp_name]))
        min_lim = col_limit[1]
    
    # ax = rec_channel['inh'][:,0:480].reshape(-1, 120, 4)
    # de = rec_channel['inh'][:,480:].reshape(-1, 120, 4)
    api = rec_channel['exc'][:,0:480].reshape(-1, 120, 4)
    bas = rec_channel['exc'][:,480:].reshape(-1, 120, 4)

    # return np.stack((ax, de, api, bas ), axis=3)
    return np.stack((api, bas ), axis=3)


# def calculate_arbor_densities(xrm):
#     ax = xrm[:,0:480].reshape(-1, 120, 4)*27135.287580
#     de = xrm[:,  480 : 480*2].reshape(-1, 120, 4)*7683.354957
#     api = xrm[:, 480*2 : 480*3].reshape(-1, 120, 4)*6339.695906
#     bas = xrm[:, 480*3:].reshape(-1, 120, 4)*8628.218423
#     return np.stack((ax, de, api, bas ), axis=3) 


def save_results(model, dataloader, input_datamat, fname, train_ind, val_ind):
    '''
    Takes the model, run it in the evaluation mode to calculate the embeddings and reconstructions for printing out.
    Args:
        model: the model 
        dataloader: the dataloader
        input_datamat: the input mat file that is loaded
        fname: name of the output pkl file that the results going to be saved in
        train_ind: the indices of the tarin cells
        val_ind: the indices of the validation cells
    '''
    model.eval()

    for all_data in iter(dataloader):
        with torch.no_grad():
            _, z_dict, xr_dict = model(all_data)
    
    zm_int_from_zt = model.ae_m.dec_zm_to_zm_int(z_dict['zt'])
    # xrm_from_zt = model.ae_m.dec_zm_int_to_xm(zm_int_from_zt, model.me_m_encoder.pool_0_ind,
    #                                       self.me_m_encoder.pool_1_ind)
    
    zm_int_from_ze = model.ae_m.dec_zm_to_zm_int(z_dict['ze'])
    # xrm_from_ze = model.ae_m.dec_zm_int_to_xm(zm_int_from_ze)

    rec_arbor_density =  tonumpy(xr_dict['xrm'])
    rec_arbor_density_from_zt = tonumpy(xr_dict['xrm'])
    rec_arbor_density_from_ze = tonumpy(xr_dict['xrm'])
    # rec_arbor_density = calculate_arbor_densities_from_nmfs(rec_nmf = tonumpy(xr_dict['xrm']), input_datamat=input_datamat)
    # rec_arbor_density_from_zt = calculate_arbor_densities_from_nmfs(rec_nmf = tonumpy(xrm_from_zt), input_datamat=input_datamat)
    # rec_arbor_density_from_ze = calculate_arbor_densities_from_nmfs(rec_nmf = tonumpy(xrm_from_ze), input_datamat=input_datamat)

    # rec_arbor_density = calculate_arbor_densities(tonumpy(xr_dict['xrm']))
    # rec_arbor_density_from_zt = calculate_arbor_densities(tonumpy(xrm_from_zt))
    # rec_arbor_density_from_ze = calculate_arbor_densities(tonumpy(xrm_from_ze))

        
    savedict = {'XT': tonumpy(all_data['xt']),
                'XM': tonumpy(all_data['xm']),
                'XE': tonumpy(all_data['xe']),
                'XrT': tonumpy(xr_dict['xrt']),
                'XrE': tonumpy(xr_dict['xre']),
                'XrM': tonumpy(xr_dict['xrm']),
                'XrM_me_paired': tonumpy(xr_dict['xrm_me_paired']),
                'XrE_me_paired': tonumpy(xr_dict['xre_me_paired']),
                'rec_arbor_density': rec_arbor_density,
                'rec_arbor_density_from_zt': rec_arbor_density_from_zt,
                'rec_arbor_density_from_ze': rec_arbor_density_from_ze,
                'zm': tonumpy(z_dict['zm']),
                'ze': tonumpy(z_dict['ze']),
                'zt': tonumpy(z_dict['zt']),
                'zme_paired': tonumpy(z_dict['zme_paired']),
                'is_t_1d':tonumpy(all_data['is_t_1d']),
                'is_e_1d':tonumpy(all_data['is_e_1d']),
                'is_m_1d':tonumpy(all_data['is_m_1d']), 
                'cluster_id': input_datamat['cluster_id'],
                'gene_ids': input_datamat['gene_ids'],
                'e_features': input_datamat['E_features'],
                'specimen_id': rm_emp_end_str(input_datamat['specimen_id']),
                'cluster_label': rm_emp_end_str(input_datamat['cluster_label']),
                'merged_cluster_label_at40': rm_emp_end_str(input_datamat['merged_cluster_label_at40']),
                'merged_cluster_label_at50': rm_emp_end_str(input_datamat['merged_cluster_label_at50']),
                'merged_cluster_label_at60': rm_emp_end_str(input_datamat['merged_cluster_label_at60']),
                'cluster_color': rm_emp_end_str(input_datamat['cluster_color']),
                'platform': input_datamat['platform'],
                'class':  input_datamat['class'],
                'class_id': input_datamat['class_id'],
                'group': input_datamat['group'],
                'subgroup': input_datamat['subgroup'],
                'hist_ax_de_api_bas' : input_datamat['hist_ax_de_api_bas'],
                # 'M_nmf_total_vars_ax': input_datamat['M_nmf_total_vars_ax'],
                # 'M_nmf_total_vars_de': input_datamat['M_nmf_total_vars_de'],
                # 'M_nmf_total_vars_api': input_datamat['M_nmf_total_vars_api'],
                # 'M_nmf_total_vars_bas': input_datamat['M_nmf_total_vars_bas'],
                # 'M_nmf_components_ax': input_datamat['M_nmf_components_ax'],
                # 'M_nmf_components_de': input_datamat['M_nmf_components_de'],
                # 'M_nmf_components_api': input_datamat['M_nmf_components_api'],
                # 'M_nmf_components_bas': input_datamat['M_nmf_components_bas'],
                # 'M_nmf_total_vars_inh': input_datamat['M_nmf_total_vars_inh'],
                # 'M_nmf_total_vars_exc': input_datamat['M_nmf_total_vars_exc'],
                # 'M_nmf_components_inh': input_datamat['M_nmf_components_inh'],
                # 'M_nmf_components_exc': input_datamat['M_nmf_components_exc'],
                'train_ind': train_ind,
                'val_ind': val_ind}

    savepkl(savedict, fname)
    model.train()
    return


def optimizer_to(optim, device):
    '''function to send the optimizer to device, this is required only when loading an optimizer 
    from a previous model'''
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def Criterion(model_config, loss_dict):
    ''' Loss function for the autoencoder'''

    criterion = model_config['M']['alpha_M'] * loss_dict['rec_m']  + \
                model_config['E']['alpha_E'] * loss_dict['rec_e'] + \
                model_config['T']['alpha_T'] * loss_dict['rec_t'] + \
                model_config['ME']['alpha_ME'] * (loss_dict['rec_m_me'] + loss_dict['rec_e_me']) + \
                model_config['TE']['lambda_TE'] * model_config['TE']['lambda_tune_T_E'] * loss_dict['cpl_t->e'] + \
                model_config['TE']['lambda_TE'] * model_config['TE']['lambda_tune_E_T'] * loss_dict['cpl_e->t'] + \
                model_config['TM']['lambda_TM'] * model_config['TM']['lambda_tune_T_M'] * loss_dict['cpl_t->m'] + \
                model_config['TM']['lambda_TM'] * model_config['TM']['lambda_tune_M_T'] * loss_dict['cpl_m->t'] + \
                model_config['ME_T']['lambda_ME_T'] * model_config['ME_T']['lambda_tune_T_ME'] * loss_dict['cpl_t->me'] + \
                model_config['ME_T']['lambda_ME_T'] * model_config['ME_T']['lambda_tune_ME_T'] * loss_dict['cpl_me->t'] + \
                model_config['ME_M']['lambda_ME_M'] * model_config['ME_M']['lambda_tune_ME_M'] * loss_dict['cpl_me->m'] + \
                model_config['ME_E']['lambda_ME_E'] * model_config['ME_E']['lambda_tune_ME_E'] * loss_dict['cpl_me->e']
     
                # model_config['ME_E']['lambda_ME_E'] * model_config['ME_E']['lambda_tune_E_ME'] * loss_dict['cpl_e->me']
#                    model_config['ME_M']['lambda_ME_M'] * model_config['ME_M']['lambda_tune_M_ME'] * loss_dict['cpl_m->me'] + \
 
    #             # model_config['M']['alpha_M'] * loss_dict['BCELoss_m'] + \
                    # model_config['ME']['alpha_ME'] * loss_dict['BCELoss_me_m']


    # if model_config['variational']:
    #     criterion = criterion + \
    #                 model_config['KLD_beta'] * loss_dict['KLD_t'] + \
    #                 model_config['KLD_beta'] * loss_dict['KLD_e'] + \
    #                 model_config['KLD_beta'] * loss_dict['KLD_m'] + \
    #                 model_config['KLD_beta'] * loss_dict['KLD_me_paired']      
    return criterion