import numpy as np

def get_padding_up_and_down(soma_depth, im):
    soma_shift = np.round(60 - soma_depth).astype(int).squeeze()
    upper_edge = np.zeros(soma_shift.shape)
    lower_edge = np.zeros(soma_shift.shape)
    n_cells = im.shape[0]
    for c in range(n_cells):
        select = np.nonzero(im[c, 0, :, :, :])
        upper_edge[c] = np.min(select[0]).item()
        lower_edge[c] = np.max(select[0]).item()
    mask_to_move_up = soma_shift < 0
    mask_to_move_down = soma_shift > 0
    upper_pad = np.max(abs(soma_shift[mask_to_move_up]) - upper_edge[mask_to_move_up])
    lower_pad = np.min(120 - lower_edge[mask_to_move_down] - soma_shift[mask_to_move_down])
    pad_lower_and_upper = max(abs(upper_pad), abs(lower_pad))
    return (np.ceil(pad_lower_and_upper/10) * 10).astype(int)

def get_padded_im(im, pad):
    """Returns image padded in the H dimension. 

    Args:
        im (np.array): dimensions are (N,1,H,W,C)
        pad (int): scalar padding around H

    Returns:
        padded_im
    """
    assert np.ndim(im)==5, 'im.shape should have shape (N,1,H,W,C)' 
    padded_shape = np.array(im.shape)
    padded_shape[2] = padded_shape[2] + pad * 2
    padded_im = np.zeros(padded_shape).astype(float)
    padded_im[:, 0, pad:-pad, :, :] = im[:, 0, ...]
    return padded_im


def shift3d(im, shift=0, fill_value=0):
    """shifts image in the H dimension. im is a single image, and shift is an int.  

    Args:
        im (np.array): dimensions are (H,W,C)
        shift (int): shift amount
        fill_value (int, optional): [description]. Defaults to 0.

    Returns:
        im: shifted image
    """
    assert np.ndim(im) == 3, 'im.shape should have shape (H,W,C)'
    shifted_im = np.full_like(im, fill_value)
    if shift > 0:  # moving down
        shifted_im[shift:, :, :] = im[:-shift, :, :]
    elif shift < 0:  # moving up
        shifted_im[:shift, :, :] = im[-shift:, :, :]
    else:
        shifted_im = im
    return shifted_im


def get_soma_aligned_im(im, soma_H):
    """Removes soma depth from the arbor density H axis.

    Args:
        im ([type]): dims are (N,1,H+2*pad,W,C).
        soma_H (np.array): dims are (N,). Should already account for pad.

    Returns:
        shifted soma
    """
    assert np.ndim(im) == 5, 'im.shape should have shape (N,1,H,W,C)'
    shifted_im = np.empty_like(im)
    center = int(im.shape[2]/2)
    shift = np.nan_to_num((center - soma_H)).astype(int)

    for c in range(im.shape[0]):
        shifted_im[c, 0, ...] = shift3d(im=im[c, 0, ...], shift=shift[c])
    return shifted_im

def get_celltype_specific_shifts(ctype, dummy=True):
    #TODO: populate logic in case using cell type specific shifting augmentation.
    if dummy:
        shifts = np.zeros((ctype.size, 2), dtype=int)
    return shifts


def undone_radial_correction(image):
    '''
    Takes the image and undone the radial correction. The pixels along the W axis were divided by pi*(r2^2 - r1^2)
    Args:
        image:images with the shape of (120, 4)

    '''
    raw_image = np.empty_like(image)
    for c in range(image.shape[1]):
        r1 = c
        r2 = c + 1
        raw_image[:, c] = image[:, c] * (np.pi * (r2 ** 2 - r1 ** 2))
    return raw_image


def do_radial_correction(image):
    '''
    Takes the image and undone the radial correction. The pixels along the W axis where divided by pi*(r2^2 - r1^2)
    Args:
        image:images with the shape of (120, 4)

    '''
    raw_image = np.empty_like(image)
    for c in range(image.shape[1]):
        r1 = c
        r2 = c + 1
        raw_image[:, c] = image[:, c] / (np.pi * (r2 ** 2 - r1 ** 2))
    return raw_image

