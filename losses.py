
import numpy as np
from tensorflow.keras import backend as K

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    # weights.
    weights = K.variable(weights)
    weights = K.reshape(weights, [1,1,1,1,6])
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        l = y_true * K.log(y_pred) * weights 
        l = -K.mean(K.flatten(l))
        return l
    
    return loss

def fl():
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    # weights.
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        l = y_true * K.log(y_pred) 
        l = -K.mean(K.flatten(l))
        return l
    
    return loss


def focal_loss(weights, gamma=2):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    # weights.
    weights = K.variable(weights)
    weights = K.reshape(weights, [1,1,1,1,6])
    def loss(y_true, y_pred):
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        l = y_true * K.log(y_pred) * weights * ((1 - y_pred) ** gamma)
        l = -K.mean(K.flatten(l))
        return l
    
    return loss

def npwcce(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    # print(weights)
    weights.reshape(1,1,1,1,6)

    def loss(y_true, y_pred):
        # clip to prevent NaN's and Inf's
        y_pred = np.clip(y_pred, np.finfo(np.float).eps, 1 - np.finfo(np.float).eps)
        # calc
        l = y_true * np.log(y_pred) * weights
        l = -np.sum(l.flatten())
        return l
    
    return loss
