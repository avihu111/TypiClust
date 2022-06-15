import os
import sys
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch


# local

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('../..'))

import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
from pycls.datasets.data import Data
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
from pycls.utils.meters import TestMeter

logger = lu.get_logger(__name__)

def argparser():
    parser = argparse.ArgumentParser(description='Passive Learning - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)

    return parser

def plot_arrays(x_vals, y_vals, x_name, y_name, dataset_name, out_dir, isDebug=False):
    # if not du.is_master_proc():
    #     return
    
    import matplotlib.pyplot as plt
    temp_name = "{}_vs_{}".format(x_name, y_name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title("Dataset: {}; {}".format(dataset_name, temp_name))
    plt.plot(x_vals, y_vals)

    if isDebug: print("plot_saved at : {}".format(os.path.join(out_dir, temp_name+'.png')))

    plt.savefig(os.path.join(out_dir, temp_name+".png"))
    plt.close()

def save_plot_values(temp_arrays, temp_names, out_dir, isParallel=True, saveInTextFormat=False, isDebug=True):

    """ Saves arrays provided in the list in npy format """
    # Return if not master process
    # if isParallel:
    #     if not du.is_master_proc():
    #         return

    for i in range(len(temp_arrays)):
        temp_arrays[i] = np.array(temp_arrays[i])
        temp_dir = out_dir
        # if cfg.TRAIN.TRANSFER_EXP:
        #     temp_dir += os.path.join("transfer_experiment",cfg.MODEL.TRANSFER_MODEL_TYPE+"_depth_"+str(cfg.MODEL.TRANSFER_MODEL_DEPTH))+"/"

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if saveInTextFormat:
            # if isDebug: print(f"Saving {temp_names[i]} at {temp_dir+temp_names[i]}.txt in text format!!")
            np.savetxt(temp_dir+'/'+temp_names[i]+".txt", temp_arrays[i], fmt="%d")
        else:
            # if isDebug: print(f"Saving {temp_names[i]} at {temp_dir+temp_names[i]}.npy in numpy format!!")
            np.save(temp_dir+'/'+temp_names[i]+".npy", temp_arrays[i])

def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
        (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or
        (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH
    )


def main(cfg):

    # Setting up GPU args
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': cfg.DATA_LOADER.NUM_WORKERS, 'pin_memory': cfg.DATA_LOADER.PIN_MEMORY} if use_cuda else {}

    # Using specific GPU
    # os.environ['NVIDIA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # print("Using GPU : {}.\n".format(cfg.GPU_ID))

    # Getting the output directory ready (default is "/output")
    cfg.OUT_DIR = os.path.join(os.path.abspath('../..'), cfg.OUT_DIR)
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)
    # Create "DATASET/MODEL TYPE" specific directory
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, cfg.MODEL.TYPE)
    if not os.path.exists(dataset_out_dir):
        os.makedirs(dataset_out_dir)
    # Creating the experiment directory inside the dataset specific directory 
    # all logs, labeled, unlabeled, validation sets are stroed here 
    # E.g., output/CIFAR10/resnet18/{timestamp or cfg.EXP_NAME based on arguments passed}
    if cfg.EXP_NAME == 'auto':
        now = datetime.now()
        exp_dir = f'{now.year}_{now.month}_{now.day}_{now.hour}{now.minute}{now.second}'
    else:
        exp_dir = cfg.EXP_NAME

    exp_dir = os.path.join(dataset_out_dir, exp_dir)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        print("Experiment Directory is {}.\n".format(exp_dir))
    else:
        print("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
    cfg.EXP_DIR = exp_dir

    # Save the config file in EXP_DIR
    dump_cfg(cfg)

    # Setup Logger
    lu.setup_logging(cfg)

    # Dataset preparing steps
    print("\n======== PREPARING TEST DATA ========\n")
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)
    
    print("\nDataset {} Loaded Sucessfully. Total Test Size: {}\n".format(cfg.DATASET.NAME, test_size))
    logger.info("Dataset {} Loaded Sucessfully. Total Test Size: {}\n".format(cfg.DATASET.NAME, test_size))

    # Preparing dataloaders for testing
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)
    
    print("======== TESTING ========\n")
    logger.info("======== TESTING ========\n")
    test_acc = test_model(test_loader, os.path.join(os.path.abspath('../..'), cfg.TEST.MODEL_PATH), cfg)
    print("Test Accuracy: {}.\n".format(round(test_acc, 4)))
    logger.info("Test Accuracy {}.\n".format(test_acc))

    print('Check the test accuracy inside {}/stdout.log'.format(cfg.EXP_DIR))

    print("================================\n\n")
    logger.info("================================\n\n")


def test_model(test_loader, checkpoint_file, cfg, cur_episode=0):

    test_meter = TestMeter(len(test_loader))

    model = model_builder.build_model(cfg)
    model = cu.load_checkpoint(checkpoint_file, model)
    
    test_err = test_epoch(test_loader, model, test_meter, cur_episode)
    test_acc = 100. - test_err

    return test_acc


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""

    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    misclassifications = 0.
    totalSamples = 0.

    for cur_iter, (inputs, labels) in enumerate(tqdm(test_loader, desc="Test Data")):
        with torch.no_grad():
            # Transfer the data to the current GPU device
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)
            # Compute the predictions
            preds = model(inputs)
            # Compute the errors
            top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
            # Combine the errors across the GPUs
            # if cfg.NUM_GPUS > 1:
            #     top1_err = du.scaled_all_reduce([top1_err])
            #     #as above returns a list
            #     top1_err = top1_err[0]
            # Copy the errors from GPU to CPU (sync point)
            top1_err = top1_err.item()
            # Multiply by Number of GPU's as top1_err is scaled by 1/Num_GPUs
            misclassifications += top1_err * inputs.size(0) * cfg.NUM_GPUS
            totalSamples += inputs.size(0)*cfg.NUM_GPUS
            test_meter.iter_toc()
            # Update and log stats
            test_meter.update_stats(
                top1_err=top1_err, mb_size=inputs.size(0) * cfg.NUM_GPUS
            )
            test_meter.log_iter_stats(cur_epoch, cur_iter)
            test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()

    return misclassifications/totalSamples


if __name__ == "__main__":
    cfg.merge_from_file(argparser().parse_args().cfg_file)
    main(cfg)
