import numpy as np
import matplotlib.pyplot as plt
import cplAE_MET.utils.utils as ut
import cplAE_MET.utils.load_helpers as loader
import seaborn as sns

sns.set(style='white')


def contingency(a, b, unique_a, unique_b):
    """Populate contingency matrix. Rows and columns are not normalized in any way.
    
    Args:
        a (np.array): labels
        b (np.array): labels
        unique_a (np.array): unique list of labels. Can have more entries than np.unique(a)
        unique_b (np.array): unique list of labels. Can have more entries than np.unique(b)

    Returns:
        C (np.array): contingency matrix.
    """
    assert a.shape == b.shape
    C = np.zeros((np.size(unique_a), np.size(unique_b)))
    for i, la in enumerate(unique_a):
        for j, lb in enumerate(unique_b):
            C[i, j] = np.sum(np.logical_and(a == la, b == lb))
    return C


def matrix_scatterplot(M, xticklabels, yticklabels, xlabel='', ylabel='', fig_width=10, fig_height=14, scale_factor=10.0):
    """Plots a matrix with points as in a scatterplot. Area of points proportional to each matrix element. 
    Suitable to show sparse matrices.

    Args:
        M (np.array): a 2D array
        xticklabels: label list
        yticklabels: label list
        fig_width (int): matplotlib figure width
        fig_height (int): matplotlib figure height
        scale_factor (float): scales the points by this value. 
    """
    Mplot = M.copy()*scale_factor
    Mplot = np.flip(Mplot, axis=0)
    yticklabels.reverse()
    x = np.arange(0, M.shape[1], 1)
    y = np.arange(0, M.shape[0], 1)
    xx, yy = np.meshgrid(x, y)
    plt.figure(figsize=(fig_width, fig_height))
    plt.scatter(np.ravel(xx), np.ravel(yy), s=np.ravel(Mplot), c='dodgerblue')
    ax = plt.gca()
    ax.set_xlim(np.min(x)-0.5, np.max(x)+0.5)
    ax.set_ylim(np.min(y)-0.5, np.max(y)+0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_yticks(y)
    ax.set_yticklabels(yticklabels, rotation=0)
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.tick_params(color='None')
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    for tick in ax.get_yticklabels():
        #tick.set_fontname("DejaVu Sans Mono")
        tick.set_fontfamily('monospace')
        tick.set_fontsize(12)

    for tick in ax.get_xticklabels():
        tick.set_fontfamily('monospace')
        tick.set_fontsize(12)

    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
    plt.box(False)
    plt.tight_layout()
    return


def scatter3(X,col,xlims=(3,3),ylims=(3,3),zlims=(3,3),fig=None):
    sns.set_style("whitegrid")
    if fig is None:
        fig = plt.figure(figsize=(4,4))
        

    plt.ion()    
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X[:,0], X[:,1], X[:,2],s=1,alpha=1,c=col)

    ax.set_xticks([])
    ax.set_zticks([])
    ax.set_yticks([])
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    ax.set_xlim(xlims[0],xlims[1])
    ax.set_ylim(ylims[0],ylims[1])
    ax.set_zlim(zlims[0],zlims[1])

    plt.axis('on')
    ax.set_frame_on(False)
    plt.tight_layout()
    return ax,sc


def show_ax_de_maps(Left, Right=None):
    """plots axon and dendrite density maps for comparison. Right image can be left empty.

    Args:
        Left: Data image with shape (120, 4, 2)
        Right: Predicted image with shape (120, 4, 2)
    """
    if Right is None:
        Right = np.zeros_like(Left)

    plt.figure(figsize=(9, 12))

    plt.subplot(221)
    sns.heatmap(np.squeeze(Left[..., 0]), vmin=0, vmax=np.max(Left[..., 0]), cbar=False)
    plt.gca().set(**{'title': r'$X_m$', 'ylabel': 'Dendrite map', 'xticks': [], 'yticks': []})
    plt.grid(False)

    plt.subplot(222)
    sns.heatmap(np.squeeze(Right[..., 0]), vmin=0, vmax=np.max(Right[..., 1]), cbar=False)
    plt.gca().set(**{'title': r'$Xrm-from-zM$', 'xticks': [], 'yticks': []})
    plt.grid(False)

    plt.subplot(223)
    sns.heatmap(np.squeeze(Left[..., 1]), vmin=0, vmax=np.max(Left[..., 0]), cbar=False)
    plt.gca().set(**{'ylabel': 'Axon map', 'xticks': [], 'yticks': []})
    plt.grid(False)

    plt.subplot(224)
    sns.heatmap(np.squeeze(Right[..., 1]), vmin=0, vmax=np.max(Right[..., 1]), cbar=False)
    plt.gca().set(**{'xticks': [], 'yticks': []})
    plt.grid(False)
    return


def plot3D_embedding(emb_array, color_list, **kwargs):
    """Takes embedding nparray and list of colors for all the points and plot3d the embedding

    Args:
    emb_array: nparray of embedding points coordinates
    color_list: a list of colors for each embedding point
    figsize(optional):
    xlim(optional): a tuple of x limits for plotting
    ylim(optional): a tuple of y limits for plotting
    zlim(optional): a tuple of z limits for plotting
    annotation_list(optional): annotation list of embedding points

    Returns:
    ax

    """
    figsize = kwargs.get('figsize', (10, 10))
    pointsize = kwargs.get('pointsize', 5)
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    zlim = kwargs.get('zlim', None)
    annotation_list = kwargs.get('annotation_list', None)
    x = emb_array[:, 0]
    y = emb_array[:, 1]
    z = emb_array[:, 2]

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=color_list, s=pointsize)

    if annotation_list is not None:
        for i, txt in enumerate(annotation_list):
            ax.text(x[i], y[i], z[i], txt, size=10)

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if zlim:
        ax.set_zlim(zlim[0], zlim[1])

    return ax


def multiple_scatter_2D_plot(x, y, **kwargs):
    """
    Takes x and multiple y(in a dict format) and scatter plot them

    Args
    ----------
    x: a list
    y: a dict with keys and y as values

    Return
    ----------
    a scatter plot of all ys with respect to x
    """

    fig_size = kwargs.get('fig_size', (10, 10))
    point_size = kwargs.get('point_size', 20)

    y_axis_label = kwargs.get('y_axis_label', "Y")
    x_axis_label = kwargs.get('x_axis_label', "X")

    xtick_labels = kwargs.get('xtick_labels', None)
    ytick_labels = kwargs.get('ytick_labels', None)

    xtick_label_size = kwargs.get('xtick_label_size', 10)
    ytick_label_size = kwargs.get('ytick_label_size', 10)

    xticks_rotation = kwargs.get('xticks_rotation', 0)
    yticks_rotation = kwargs.get('yticks_rotation', 0)

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    for k, v in y.items():
        ax.scatter(x, v, label=k, s=point_size)

    if xtick_labels:
        ax.set_xticks([i for i in range(len(xtick_labels))])
        ax.set_xticklabels(xtick_labels,
                           rotation=xticks_rotation,
                           fontsize=xtick_label_size)
    if ytick_labels:
        ax.set_xticks([i for i in range(len(ytick_labels))])
        ax.set_xticklabels(ytick_labels,
                           rotation=yticks_rotation,
                           fontsize=ytick_label_size)

    ax.set_xlabel(x_axis_label, fontsize=12)
    ax.set_ylabel(y_axis_label, fontsize=12)

    plt.legend()

    return ax


def plot_3D_embedding_run_id(plot_key, package_dir, exp_name, model_id, alpha_T, alpha_E, alpha_M, alpha_sd,
                             lambda_T_EM, augment_decoders, E_noise, M_noise, dilate_M, latent_dim, batchsize, n_epochs,
                             run_iter, n_fold, xlim=None, ylim=None, zlim=None):
    '''Plots the 3d embedding given the parametrs of the model

    Args:
        plot_key: The key from the output dict that we want to plot
        package_dir: The directory where the package is installed, there is a data folder in which
                      the output and input data are saved
        exp_name: Experiment set
        model_id: Model-specific id
        alpha_T: T reconstruction loss weight
        alpha_E: E reconstruction loss weight
        alpha_M: M reconstruction loss weight
        alpha_sd: soma depth reconstruction loss weight
        lambda_T_EM: T - EM coupling loss weight
        augment_decoders: 0 or 1 - Train with cross modal reconstruction
        latent_dim: Number of latent dims
        batchsize: Batch size
        n_epochs: Number of epochs to train
        run_iter: Run-specific id
        n_fold: Fold id in the kfold cross validation training

    '''
    # Get fileid
    file_id = loader.get_fileid(model_id=model_id,
                                alpha_T=alpha_T,
                                alpha_E=alpha_E,
                                alpha_M=alpha_M,
                                alpha_sd=alpha_sd,
                                lambda_T_EM=lambda_T_EM,
                                augment_decoders=augment_decoders,
                                E_noise=E_noise,
                                M_noise=M_noise,
                                dilate_M=dilate_M,
                                latent_dim=latent_dim,
                                batchsize=batchsize,
                                n_epochs=n_epochs,
                                run_iter=run_iter,
                                n_fold=n_fold)

    # Get the path to the fileid
    pth = loader.get_io_path(package_dir,
                             exp_name=exp_name,
                             output_fileid=file_id)

    output = ut.loadpkl(pth["output_path"])

    # set the limits of the x,y and z axis for plotting
    lim1= np.amin(output[plot_key], axis=0)
    lim2 = np.amax(output[plot_key], axis=0)
    if not xlim:
        xlim = (lim1[0], lim2[0])
    if not ylim:
        ylim = (lim1[1], lim2[1])
    if not zlim:
        zlim = (lim1[2], lim2[2])

    plot3D_embedding(output[plot_key],
                     output['cluster_color'],
                     xlim=xlim,
                     ylim=ylim,
                     zlim=zlim)

    print("plotting ", plot_key, " from this file:", pth["output_path"])


def plot_multiple_dict(mydict, xlabel="X", ylabel="Y", x_label_rotation=90, order_of_x_values=None):
        """
        take a dictionary in which each value is a dictionary itself and plot each of these values
        Args:
        _____
        mydict: a dictinary that each value is a dictinary itself and we want to plot these dictinaries
        order_of_x_values: list, if given, it will plot with that order
        """
        plt.figure(figsize=(15, 7))
        for k, v in mydict.items():
            if order_of_x_values:
                xvals = order_of_x_values
                yvals = [v[i] for i in xvals]
            else:
                xvals = v.keys()
                yvals = [v[i] for i in xvals]
            plt.scatter([i for i in range(len(xvals))], yvals)
            plt.plot([v for v in yvals], label=k)
            plt.xticks([i for i in range(len(xvals))], xvals, rotation=x_label_rotation)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()

        plt.show()


def plot_multiple_embeddings(left, right, figsize, plot_dim, left_color=None, right_color=None, left_marker='o',
                             right_marker='x', left_label=None, right_label=None, xlim=None, ylim=None,
                             zlim=None, side_by_side=False, scatter_point_size=None):
    """
    plot right and left embeddings on one plot

    Args
    ----------
    left: np array of the left emb
    right: np array of the right emb
    figsize: figsize
    plot_dim: 2d or 3d plot
    left_color: color of left points
    right_color: color of right points
    left_marker: right_marker
    right_marker: left_marker
    left_label: label of the left plot
    right_label: label of the right plot
    xlim: tuple of x limits
    ylim: tuple of y limits
    side_by_side: if True, plot right and left in two separate plot side by side otherwise
    it will plot both in the same plot
    scatter_point_size: scatter point size
    """

    if scatter_point_size is None:
        scatter_point_size = 30

    fig = plt.figure(figsize=figsize)

    def set_axis_lim(plot_dim, xlim, ylim, zlim):

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        if plot_dim == 3:
            if zlim is not None:
                ax.set_zlim(zlim)

    if plot_dim == 3:
        if side_by_side:
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(left[:, 0], left[:, 1], left[:, 2],
                       c=left_color, s=scatter_point_size, marker=left_marker, label=left_label, alpha=1)
            set_axis_lim(plot_dim, xlim, ylim, zlim)

            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(right[:, 0], right[:, 1], right[:, 2],
                       c=right_color, s=scatter_point_size, marker=right_marker, label=right_label, alpha=1)
            set_axis_lim(plot_dim, xlim, ylim, zlim)

        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(left[:, 0], left[:, 1], left[:, 2],
                       c=left_color, s=scatter_point_size, marker=left_marker, label=left_label, alpha=1)
            ax.scatter(right[:, 0], right[:, 1], right[:, 2],
                       c=right_color, s=scatter_point_size, marker=right_marker, label=right_label, alpha=1)
            set_axis_lim(plot_dim, xlim, ylim, zlim)

    else:
        if side_by_side:
            ax = fig.add_subplot(1, 2, 1)
            ax.scatter(left[:, 0], left[:, 1],
                       c=left_color, s=scatter_point_size, marker=left_marker, label=left_label, alpha=1)
            set_axis_lim(plot_dim, xlim, ylim, zlim)

            ax = fig.add_subplot(1, 2, 2)
            ax.scatter(right[:, 0], right[:, 1],
                       c=right_color, s=scatter_point_size, marker=right_marker, label=right_label, alpha=1)
            set_axis_lim(plot_dim, xlim, ylim, zlim)

        else:
            ax = fig.add_subplot(111)
            ax.scatter(left[:, 0], left[:, 1],
                       c=left_color, s=scatter_point_size, marker=left_marker, label=left_label, alpha=1)
            ax.scatter(right[:, 0], right[:, 1],
                       c=right_color, s=scatter_point_size, marker=right_marker, label=right_label, alpha=1)
            set_axis_lim(plot_dim, xlim, ylim, zlim)

    plt.legend()
    plt.show()


