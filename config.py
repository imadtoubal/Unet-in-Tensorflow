import datetime
from utils import dice_coef
from tensorflow.keras.optimizers import Adam

def get_config():
    return {
        # Network architecture
        "net": {
            "epochs": 1000,
            "verbose": 1,
        },
        "net_cmp": {
            "optimizer": Adam(learning_rate=3e-4), 
            "metrics": [dice_coef]
        },
        # Data paths
        "data": {
            "train_path": 'data/train/',
            "val_path": 'data/val/',
            "test_path": 'data/test/',
            "out_path": 'data/out/',
            "256_train_path": 'data_256_200_64/train_train/',
            "256_val_path": 'data_256_200_64/train_val/',
            "256_test_path": 'data_256_200_64/test/',
            "256_out_path": 'data_256_200_64/out/',
            
        },
        # For checkpoint saving, early stopping...
        "train": {
            "ckpt": {
                "ckpt_path": 'ckpt',
                "verbose": 1, 
                "save_best_only": True
            },
            "early_stopping": {
                "patience": 50, 
                "monitor": 'val_loss'
            }

        }
    }

