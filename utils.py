import scipy.io as sio
import os
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from keras import backend as K
from pathlib import Path
from skimage import measure


def readmat(filename, var_name):
    img = sio.loadmat(filename)
    img = img.get(var_name)
    img = img.astype(np.float32)
    
    
    # unsqueeze for channel size of 1
    # return np.expand_dims(img, 0)
    return img

def ind2onehot(indimg):
    indimg = indimg.astype('int')
    classes = indimg.max() + 1
    Y = np.stack([indimg == i for i in range(classes)], axis=4)
    return Y

def load_dataset(root_dir, var_name='data', return_paths=False, return_idx=False, pad=14, addcoords=False, coords_only=False):
    """
    Args:
        root_dir (string): Directory with all the images.
    """
    # get all .mat files 
    paths = [img_path for img_path in sorted(os.listdir(root_dir)) if img_path[-4:] == '.mat']
    # read all .mat files
    data = [readmat(root_dir + img_path, var_name) for i, img_path in tqdm(enumerate(paths), total=len(paths))]
    data = np.stack(data)
    X, Y = data[:,:,:,:,0], data[:,:,:,:,1]
    
    X = np.expand_dims(X, -1)
    X = np.pad(X, pad_width=((0,0), (pad,pad), (0,0), (0, 0), (0, 0)))
    Y = np.pad(Y, pad_width=((0,0), (pad,pad), (0,0), (0, 0)))

    if addcoords: X = add_coords(X, Y, coords_only=coords_only)
    
    if not return_idx:
        Y = ind2onehot(Y)

    if return_paths:
        return X, Y, [path.split('/')[-1] for path in paths]
    else:
        return X, Y 

# Source: https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
def add_coords(X_all, Y_all, coords_only=False):
    coords = []
    dims = X_all.shape[1:-1]
    for i in range(X_all.shape[0]):
        X = X_all[i]
        Y = Y_all[i]
        kidneyidx = 2
        liveridx = 1

        top = np.argwhere(Y == liveridx).min(axis=0)[2]
        left = np.argwhere(Y != 0).min(axis=0)[1]
        back = np.argwhere(Y != 0).min(axis=0)[0]

        center = get_center(Y)
        o = np.round(center).astype('int')

        x = (np.arange(X.shape[0]) - center[0]) / (center[0] - back) 
        y = (np.arange(X.shape[1]) - center[1]) / (center[1] - left)
        z = (np.arange(X.shape[2]) - center[2]) / (center[2] - top)

        xx = np.ones((dims[0], dims[1], dims[2])) * x.reshape(dims[0], 1, 1)
        yy = np.ones((dims[0], dims[1], dims[2])) * y.reshape(1, dims[1], 1)
        zz = np.ones((dims[0], dims[1], dims[2])) * z.reshape(1, 1, dims[2])

        if coords_only:
            coords.append(np.stack([xx, yy, zz], axis=3))
        else:
            coords.append(np.stack([X[:,:,:,0], xx, yy, zz], axis=3))
    return np.stack(coords)

def get_center(y):
    mask = y == 2
    mask = measure.label(mask)
        
    if mask.max() != 1:
        c1 = np.argwhere(mask == 1).mean(axis=0)
        c2 = np.argwhere(mask == 2).mean(axis=0)
        c = (c1 + c2) / 2
    else:
        c = np.argwhere(mask == 1).mean(axis=0)    
        c[1] = 64.14460244803878

    return c

def dice_coef(y_true, y_pred, smooth=1, numpy=False):
    """
    Dice = (2*|X & Y|)/ (|X| + |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    if numpy:
        intersection = np.sum(np.abs(y_true * y_pred))
        return (2. * intersection + smooth) / (np.sum(np.square(y_true)) + np.sum(np.square(y_pred)) + smooth)
    else:
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def export_outs(X, Y, out, out_path, paths=None):
    # saving probs
    print('Exporting...')
    Path(out_path).mkdir(parents=True, exist_ok=True)
    for i in range(out.shape[0]):
        img = X[i, :, :, :, 0]
        prob = out[i]
        seg = np.argmax(prob, axis=3)
        grt = np.argmax(Y[i], axis=3)

        output = np.stack((img, seg, grt), axis=3)
        if paths == None:
            sio.savemat('{}out{}.mat'.format(out_path, i), {'data': output})
            sio.savemat('{}prob_out{}.mat'.format(out_path, i), {'data': prob})
        else:
            sio.savemat('{}{}'.format(out_path, paths[i]), {'data': output})
            sio.savemat('{}prob_{}'.format(out_path, paths[i]), {'data': prob})


# def probs_to_mask(probs):

if __name__ == '__main__':
    print('Utils work perfectly')