"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing

def create_config(config_file_env, config_file_exp, seed, num_clusters=None):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    config['seed'] = seed
    if num_clusters is not None:
        config['num_classes'] = num_clusters

    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    pretext_dir = os.path.join(base_dir, 'pretext')
    mkdir_if_missing(base_dir)
    mkdir_if_missing(pretext_dir)
    cfg['pretext_dir'] = pretext_dir
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, f'checkpoint_seed{seed}.pth.tar')
    cfg['pretext_model'] = os.path.join(pretext_dir, f'model_seed{seed}.pth.tar')
    cfg['pretext_features'] = os.path.join(pretext_dir, f'features_seed{seed}.npy')
    cfg['topk_neighbors_train_path'] = os.path.join(pretext_dir, f'topk-train-neighbors_seed{seed}.npy')
    cfg['topk_neighbors_val_path'] = os.path.join(pretext_dir, f'topk-val-neighbors_seed{seed}.npy')

    # If we perform clustering or self-labeling step we need additional paths.
    # We also include a run identifier to support multiple runs w/ same hyperparams.
    if cfg['setup'] in ['scan', 'selflabel']:
        base_dir = os.path.join(root_dir, cfg['train_db_name'])
        scan_dir = os.path.join(base_dir, 'scan')
        selflabel_dir = os.path.join(base_dir, 'selflabel') 
        mkdir_if_missing(base_dir)
        mkdir_if_missing(scan_dir)
        mkdir_if_missing(selflabel_dir)
        cfg['scan_dir'] = scan_dir
        cfg['scan_checkpoint'] = os.path.join(scan_dir, f'checkpoint_seed{seed}_clusters{num_clusters}.pth.tar')
        cfg['scan_model'] = os.path.join(scan_dir, f'model_seed{seed}_clusters{num_clusters}.pth.tar')
        cfg['scan_features'] = os.path.join(scan_dir, f'features_seed{seed}_clusters{num_clusters}.npy')
        cfg['selflabel_dir'] = selflabel_dir
        cfg['selflabel_checkpoint'] = os.path.join(selflabel_dir, f'checkpoint_seed{seed}_clusters{num_clusters}.pth.tar')
        cfg['selflabel_model'] = os.path.join(selflabel_dir, f'model_seed{seed}_clusters{num_clusters}.pth.tar')

    return cfg 
