import os
import sys
from datetime import datetime
import argparse
import numpy as np

import torch


# local

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('../..'))

import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
import pycls.core.losses as losses
import pycls.core.optimizer as optim
from pycls.datasets.data import Data
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.net as nu
from pycls.utils.meters import TestMeter
from pycls.utils.meters import TrainMeter
from pycls.utils.meters import ValMeter

logger = lu.get_logger(__name__)

plot_epoch_xvalues = []
plot_epoch_yvalues = []

plot_it_x_values = []
plot_it_y_values = []

def argparser():
    parser = argparse.ArgumentParser(description='Passive Learning - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    parser.add_argument('--exp-name', dest='exp_name', help='Experiment Name', required=True, type=str)
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

    # Auto assign a RNG_SEED when not supplied a value
    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)

    # Using specific GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
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
    print("\n======== PREPARING DATA AND MODEL ========\n")
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)
    
    print("\nDataset {} Loaded Sucessfully.\nTotal Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))
    logger.info("Dataset {} Loaded Sucessfully. Total Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))
    
    trainSet_path, valSet_path = data_obj.makeTVSets(val_split_ratio=cfg.DATASET.VAL_RATIO, data=train_data, \
                                seed_id=cfg.RNG_SEED, save_dir=cfg.EXP_DIR)

    trainSet, valSet = data_obj.loadTVPartitions(trainSetPath=trainSet_path, valSetPath=valSet_path)

    print("Data Partitioning Complete. \nTrain Set: {},  Validation Set: {}\n".format(len(trainSet), len(valSet)))
    logger.info("\nTrain Set: {},  Validation Set: {}\n".format(len(trainSet), len(valSet)))

    # Preparing dataloaders for initial training
    trainSet_loader = data_obj.getIndexesDataLoader(indexes=trainSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)

    # Initialize the model.  
    model = model_builder.build_model(cfg)
    print("model: {}\n".format(cfg.MODEL.TYPE))
    logger.info("model: {}\n".format(cfg.MODEL.TYPE))

    # Construct the optimizer
    optimizer = optim.construct_optimizer(cfg, model)
    print("optimizer: {}\n".format(optimizer))
    logger.info("optimizer: {}\n".format(optimizer))

    # This is to seamlessly use the code originally written for AL episodes 
    cfg.EPISODE_DIR = cfg.EXP_DIR

    # Train model
    print("======== TRAINING ========")
    logger.info("======== TRAINING ========")
    
    best_val_acc, best_val_epoch, checkpoint_file = train_model(trainSet_loader, valSet_loader, model, optimizer, cfg)

    print("Best Validation Accuracy: {}\nBest Epoch: {}\n".format(round(best_val_acc, 4), best_val_epoch))
    logger.info("Best Validation Accuracy: {}\tBest Epoch: {}\n".format(round(best_val_acc, 4), best_val_epoch))

    # Test best model checkpoint
    print("======== TESTING ========\n")
    logger.info("======== TESTING ========\n")
    test_acc = test_model(test_loader, checkpoint_file, cfg, cur_episode=0)
    print("Test Accuracy: {}.\n".format(round(test_acc, 4)))
    logger.info("Test Accuracy {}.\n".format(test_acc))

    print("================================\n\n")
    logger.info("================================\n\n")


def train_model(train_loader, val_loader, model, optimizer, cfg):
    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    start_epoch = 0
    loss_fun = losses.get_loss_fun()

    # Create meters
    train_meter = TrainMeter(len(train_loader))
    val_meter = ValMeter(len(val_loader))

    # Perform the training loop
    # print("Len(train_loader):{}".format(len(train_loader)))
    logger.info('Start epoch: {}'.format(start_epoch + 1))
    val_set_acc = 0.

    temp_best_val_acc = 0.
    temp_best_val_epoch = 0
    
    # Best checkpoint model and optimizer states
    best_model_state = None
    best_opt_state = None

    val_acc_epochs_x = []
    val_acc_epochs_y = []

    clf_train_iterations = cfg.OPTIM.MAX_EPOCH * int(len(train_loader)/cfg.TRAIN.BATCH_SIZE)
    clf_change_lr_iter = clf_train_iterations // 25
    clf_iter_count = 0

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_loss, clf_iter_count = train_epoch(train_loader, model, loss_fun, optimizer, train_meter, \
                                        cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_train_iterations)

        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            nu.compute_precise_bn_stats(model, train_loader)
        

        # Model evaluation
        if is_eval_epoch(cur_epoch):
            # Original code[PYCLS] passes on testLoader but we want to compute on val Set
            val_loader.dataset.no_aug = True
            val_set_err = test_epoch(val_loader, model, val_meter, cur_epoch)
            val_set_acc = 100. - val_set_err
            val_loader.dataset.no_aug = False
            
            if temp_best_val_acc < val_set_acc:
                temp_best_val_acc = val_set_acc
                temp_best_val_epoch = cur_epoch + 1

                # Save best model and optimizer state for checkpointing
                model.eval()
                
                best_model_state = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
                best_opt_state = optimizer.state_dict()

                model.train()

            # Since we start from 0 epoch
            val_acc_epochs_x.append(cur_epoch+1)
            val_acc_epochs_y.append(val_set_acc)

        plot_epoch_xvalues.append(cur_epoch+1)
        plot_epoch_yvalues.append(train_loss)

        save_plot_values([plot_epoch_xvalues, plot_epoch_yvalues, plot_it_x_values, plot_it_y_values, val_acc_epochs_x, val_acc_epochs_y],\
            ["plot_epoch_xvalues", "plot_epoch_yvalues", "plot_it_x_values", "plot_it_y_values","val_acc_epochs_x","val_acc_epochs_y"], out_dir=cfg.EPISODE_DIR, isDebug=False)
        logger.info("Successfully logged numpy arrays!!")

        # Plot arrays
        plot_arrays(x_vals=plot_epoch_xvalues, y_vals=plot_epoch_yvalues, \
        x_name="Epochs", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
        
        plot_arrays(x_vals=val_acc_epochs_x, y_vals=val_acc_epochs_y, \
        x_name="Epochs", y_name="Validation Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)

        save_plot_values([plot_epoch_xvalues, plot_epoch_yvalues, plot_it_x_values, plot_it_y_values, val_acc_epochs_x, val_acc_epochs_y], \
                ["plot_epoch_xvalues", "plot_epoch_yvalues", "plot_it_x_values", "plot_it_y_values","val_acc_epochs_x","val_acc_epochs_y"], out_dir=cfg.EPISODE_DIR)

        print('Training Epoch: {}/{}\tTrain Loss: {}\tVal Accuracy: {}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, round(train_loss, 4), round(val_set_acc, 4)))

    # Save the best model checkpoint (Episode level)
    checkpoint_file = cu.save_checkpoint(info="vlBest_acc_"+str(int(temp_best_val_acc)), \
        model_state=best_model_state, optimizer_state=best_opt_state, epoch=temp_best_val_epoch, cfg=cfg)

    print('\nWrote Best Model Checkpoint to: {}\n'.format(checkpoint_file.split('/')[-1]))
    logger.info('Wrote Best Model Checkpoint to: {}\n'.format(checkpoint_file))

    plot_arrays(x_vals=plot_epoch_xvalues, y_vals=plot_epoch_yvalues, \
        x_name="Epochs", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)

    plot_arrays(x_vals=plot_it_x_values, y_vals=plot_it_y_values, \
        x_name="Iterations", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
        
    plot_arrays(x_vals=val_acc_epochs_x, y_vals=val_acc_epochs_y, \
        x_name="Epochs", y_name="Validation Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)

    plot_epoch_xvalues = []
    plot_epoch_yvalues = []

    plot_it_x_values = []
    plot_it_y_values = []
    
    best_val_acc = temp_best_val_acc
    best_val_epoch = temp_best_val_epoch

    return best_val_acc, best_val_epoch, checkpoint_file


def test_model(test_loader, checkpoint_file, cfg, cur_episode):

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    test_meter = TestMeter(len(test_loader))

    model = model_builder.build_model(cfg)
    model = cu.load_checkpoint(checkpoint_file, model)
    
    test_err = test_epoch(test_loader, model, test_meter, cur_episode)
    test_acc = 100. - test_err

    return test_acc


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_max_iter):
    """Performs one epoch of training."""
    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    # Shuffle the data
    #loader.shuffle(train_loader, cur_epoch)
    if cfg.NUM_GPUS>1:  train_loader.sampler.set_epoch(cur_epoch)

    # Update the learning rate
    # Currently we only support LR schedules for only 'SGD' optimizer
    lr = optim.get_epoch_lr(cfg, cur_epoch)
    if cfg.OPTIM.TYPE == "sgd":
        optim.set_lr(optimizer, lr)

    if torch.cuda.is_available():
        model.cuda()

    # Enable training mode
    model.train()
    train_meter.iter_tic() #This basically notes the start time in timer class defined in utils/timer.py

    len_train_loader = len(train_loader)
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        #ensuring that inputs are floatTensor as model weights are
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parametersSWA
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs
        # if cfg.NUM_GPUS > 1:
        #     #Average error and losses across GPUs
        #     #Also this this calls wait method on reductions so we are ensured
        #     #to obtain synchronized results
        #     loss, top1_err = du.scaled_all_reduce(
        #         [loss, top1_err]
        #     )
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err = loss.item(), top1_err.item()
        # #Only master process writes the logs which are used for plotting
        # if du.is_master_proc():
        if cur_iter != 0 and cur_iter%19 == 0:
            #because cur_epoch starts with 0
            plot_it_x_values.append((cur_epoch)*len_train_loader + cur_iter)
            plot_it_y_values.append(loss)
            save_plot_values([plot_it_x_values, plot_it_y_values],["plot_it_x_values.npy", "plot_it_y_values.npy"], out_dir=cfg.EPISODE_DIR, isDebug=False)
            # print(plot_it_x_values)
            # print(plot_it_y_values)
            #Plot loss graphs
            plot_arrays(x_vals=plot_it_x_values, y_vals=plot_it_y_values, x_name="Iterations", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR,)
            print('Training Epoch: {}/{}\tIter: {}/{}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, cur_iter, len(train_loader)))

        #Compute the difference in time now from start time initialized just before this for loop.
        train_meter.iter_toc()
        train_meter.update_stats(top1_err=top1_err, loss=loss, \
            lr=lr, mb_size=inputs.size(0) * cfg.NUM_GPUS)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return loss, clf_iter_count


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    misclassifications = 0.
    totalSamples = 0.

    for cur_iter, (inputs, labels) in enumerate(test_loader):
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
    cfg.EXP_NAME = argparser().parse_args().exp_name
    main(cfg)
