import umap
import numpy as np
import matplotlib.pyplot as plt

def get_T_ME_model_umap_relation_list(model_output):
    '''
    Takes the model of the T_ME coupled AE and create relation dictionary between the following
    embeddings consecutively: zt, ze, zme, zm
    The first relation is between zt and ze, next is between ze and zme and the last is between zme and zm
    '''

    relation_dict = {}
    # relation between zt and ze is computed by using the TE mask on the zt and TE mask on the ze data
    relation_dict['T_E'] = dict(zip(np.where(model_output['TE_T'])[0], np.where(model_output['TE_E'])[0]))
    # relation between ze and zme is computed by using the ME mask on the ze and zme data
    relation_dict['E_ME'] = dict(
        zip(np.where(model_output['ME_E'])[0], np.where(model_output['ME_tot'][model_output['ME_tot']])[0]))
    # relation between zme and zm is computed by using the zme and ME mask on zm data
    relation_dict['ME_M'] = dict(
        zip(np.where(model_output['ME_tot'][model_output['ME_tot']])[0], np.where(model_output['ME_M'])[0]))

    return [relation_dict['T_E'], relation_dict['E_ME'], relation_dict['ME_M']]


def get_T_ME_model_emb_list(model_output):
    '''
    Takes the model and put the embeddings of zt, ze, zme and zm in this order in a list
    '''
    return [model_output['zt'], model_output['ze'], model_output['zme'], model_output['zm'][model_output['MT_M']]]


def get_T_ME_model_umap_colors_list(model_output):
    '''
    Takes the model output and return the Ttype colors for zt, ze, zme and zm in this order
    '''
    zt_color = [mystring.replace(" ", "") for mystring in model_output['cluster_color'][model_output['T_tot']]]
    ze_color = [mystring.replace(" ", "") for mystring in model_output['cluster_color'][model_output['TE_tot']]]
    zme_color = [mystring.replace(" ", "") for mystring in model_output['cluster_color'][model_output['MET_tot']]]
    zm_color = [mystring.replace(" ", "") for mystring in model_output['cluster_color'][model_output['MT_tot']]]
    color_list = [zt_color, ze_color, zme_color, zm_color]

    return color_list


def plot_aligned_umap_T_ME(embs, relations, color_list, title_list, label_list):
    '''
    Takes embedding and the relation between them and plot zt, ze, zme, zm in this order.
    Args:
    -----
    embs: list of embeddings of zt, ze, zme, and zm in this order
    relations: list of relations between zt & ze, then ze & zme and finally zme & zm, in this order
    color_list: list of colors for each subplot zt, ze, zme and zm in this order
    title_list: list of plot title for each subplot zt, ze, zme and zm in this order
    label_list: list of labels for each subplot zt, ze, zme and zm in this order
    '''
    # align the embeddings using the relations
    aligned_mapper = umap.AlignedUMAP().fit(embs, relations=relations)

    # plot zt, ze, zme, zm in this order
    def axis_bounds(embedding):
        left, right = embedding.T[0].min(), embedding.T[0].max()
        bottom, top = embedding.T[1].min(), embedding.T[1].max()
        adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
        return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]

    fig, axs = plt.subplots(1, 4, figsize=(30, 8))
    ax_bound = axis_bounds(np.vstack(aligned_mapper.embeddings_))
    for i, ax in enumerate(axs.flatten()):
        ax.scatter(*aligned_mapper.embeddings_[i].T, s=4, c=color_list[i], cmap="Spectral", label=label_list[i])
        ax.set_title(title_list[i])
        ax.axis(ax_bound)
        ax.set(xticks=[], yticks=[])
        ax.legend(prop={'size': 20})
        plt.tight_layout()
    return