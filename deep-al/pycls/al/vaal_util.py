# This file is directly taken from a code implementation shared with me by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

# code modified from VAAL codebase
 
import os
import torch

from pycls.models import vaal_model as vm
import pycls.utils.logging as lu
# import pycls.datasets.loader as imagenet_loader

logger = lu.get_logger(__name__)

bce_loss = torch.nn.BCELoss().cuda()

def data_parallel_wrapper(model, cur_device, cfg):
    model.cuda(cur_device)
    model = torch.nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    return model

def distributed_wrapper(cfg, model, cur_device):
    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)
    
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device
        )
    return model

def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label in dataloader:
                yield img, label
    else:
        while True:
            for img, _ in dataloader:
                yield img

def vae_loss( x, recon, mu, logvar, beta):
    mse_loss = torch.nn.MSELoss().cuda()
    recon = recon.cuda()
    x = x.cuda()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD

def train_vae_disc_epoch(cfg, vae_model, disc_model, optim_vae, optim_disc, lSetLoader, uSetLoader, cur_epoch, \
    n_lu, curr_vae_disc_iter, max_vae_disc_iters, change_lr_iter,isDistributed=False):
    
    if isDistributed:
        lSetLoader.sampler.set_epoch(cur_epoch)
        uSetLoader.sampler.set_epoch(cur_epoch)
    
    print('len(lSetLoader): {}'.format(len(lSetLoader)))
    print('len(uSetLoader): {}'.format(len(uSetLoader)))

    labeled_data = read_data(lSetLoader)
    unlabeled_data = read_data(uSetLoader, labels=False)

    vae_model.train()
    disc_model.train()

    temp_bs = int(cfg.VAAL.VAE_BS)
    train_iterations = int(n_lu/temp_bs)
    
    for temp_iter in range(train_iterations):

        if curr_vae_disc_iter !=0 and curr_vae_disc_iter%change_lr_iter==0:
            #print("Changing LR ---- ))__((---- ")
            for param in optim_vae.param_groups:
                param['lr'] = param['lr'] * 0.9
    
            for param in optim_disc.param_groups:
                param['lr'] = param['lr'] * 0.9

        curr_vae_disc_iter += 1 

        ## VAE Step
        disc_model.eval()
        vae_model.train()
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)

        labeled_imgs = labeled_imgs.type(torch.cuda.FloatTensor)
        unlabeled_imgs = unlabeled_imgs.type(torch.cuda.FloatTensor)
        
        labeled_imgs = labeled_imgs.cuda()
        unlabeled_imgs = unlabeled_imgs.cuda()

        recon, z, mu, logvar = vae_model(labeled_imgs)
        recon = recon.view((labeled_imgs.shape[0], labeled_imgs.shape[1], labeled_imgs.shape[2], labeled_imgs.shape[3]))
        unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, cfg.VAAL.BETA)
        unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae_model(unlabeled_imgs)
        unlab_recon = unlab_recon.view((unlabeled_imgs.shape[0], unlabeled_imgs.shape[1], unlabeled_imgs.shape[2], unlabeled_imgs.shape[3]))
        transductive_loss = vae_loss(unlabeled_imgs, unlab_recon, unlab_mu, unlab_logvar, cfg.VAAL.BETA)

        labeled_preds = disc_model(mu)
        unlabeled_preds = disc_model(unlab_mu)

        lab_real_preds = torch.ones(labeled_imgs.size(0),1).cuda()
        unlab_real_preds = torch.ones(unlabeled_imgs.size(0),1).cuda()
        dsc_loss = bce_loss(labeled_preds, lab_real_preds) + bce_loss(unlabeled_preds, unlab_real_preds)

        total_vae_loss = unsup_loss + transductive_loss + cfg.VAAL.ADVERSARY_PARAM * dsc_loss
        
        optim_vae.zero_grad()
        total_vae_loss.backward()
        optim_vae.step()

        ##DISC STEP
        vae_model.eval()
        disc_model.train()

        with torch.no_grad():
            _, _, mu, _ = vae_model(labeled_imgs)
            _, _, unlab_mu, _ = vae_model(unlabeled_imgs)
        
        labeled_preds = disc_model(mu)
        unlabeled_preds = disc_model(unlab_mu)
                
        lab_real_preds = torch.ones(labeled_imgs.size(0),1).cuda()
        unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0),1).cuda()

        dsc_loss = bce_loss(labeled_preds, lab_real_preds) + \
            bce_loss(unlabeled_preds, unlab_fake_preds)

        optim_disc.zero_grad()
        dsc_loss.backward()
        optim_disc.step()

    
        if temp_iter%100 == 0:
            print("Epoch[{}],Iteration [{}/{}], VAE Loss: {:.3f}, Disc Loss: {:.4f}"\
                .format(cur_epoch,temp_iter, train_iterations, total_vae_loss.item(), dsc_loss.item()))

    return vae_model, disc_model, optim_vae, optim_disc, curr_vae_disc_iter

def train_vae_disc(cfg, lSet, uSet, trainDataset, dataObj, debug=False):
    
    cur_device = torch.cuda.current_device()
    if cfg.DATASET.NAME == 'MNIST':
        vae_model = vm.VAE(cur_device, z_dim=cfg.VAAL.Z_DIM, nc=1)
        disc_model = vm.Discriminator(z_dim=cfg.VAAL.Z_DIM)    
    else:
        vae_model = vm.VAE(cur_device, z_dim=cfg.VAAL.Z_DIM)
        disc_model = vm.Discriminator(z_dim=cfg.VAAL.Z_DIM)


    # vae_model = data_parallel_wrapper(vae_model, cur_device, cfg)
    # disc_model = data_parallel_wrapper(disc_model, cur_device, cfg)

    # if cfg.TRAIN.DATASET == "IMAGENET":
    #     lSetLoader = imagenet_loader.construct_loader_no_aug(cfg, indices=lSet, isDistributed=False, isVaalSampling=True)
    #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg, indices=uSet, isDistributed=False, isVaalSampling=True)
    # else:
    lSetLoader = dataObj.getIndexesDataLoader(indexes=lSet, batch_size=int(cfg.VAAL.VAE_BS) \
        ,data=trainDataset)

    uSetLoader = dataObj.getIndexesDataLoader(indexes=uSet, batch_size=int(cfg.VAAL.VAE_BS) \
        ,data=trainDataset)

    print("Initializing VAE and discriminator")
    logger.info("Initializing VAE and discriminator")
    optim_vae = torch.optim.Adam(vae_model.parameters(), lr=cfg.VAAL.VAE_LR)
    print(f"VAE Optimizer ==> {optim_vae}")
    logger.info(f"VAE Optimizer ==> {optim_vae}")
    optim_disc = torch.optim.Adam(disc_model.parameters(), lr=cfg.VAAL.DISC_LR)
    print(f"Disc Optimizer ==> {optim_disc}")
    logger.info(f"Disc Optimizer ==> {optim_disc}")
    print("==================================")
    logger.info("==================================\n")

    n_lu_points = len(lSet)+len(uSet)
    max_vae_disc_iters = int(n_lu_points/cfg.VAAL.VAE_BS)*cfg.VAAL.VAE_EPOCHS
    change_lr_iter = max_vae_disc_iters // 25
    curr_vae_disc_iter = 0

    vae_model = vae_model.cuda()
    disc_model = disc_model.cuda()

    for epoch in range(cfg.VAAL.VAE_EPOCHS):
        vae_model, disc_model, optim_vae, optim_disc, curr_vae_disc_iter = train_vae_disc_epoch(cfg, vae_model, disc_model, optim_vae, \
            optim_disc, lSetLoader, uSetLoader, epoch, n_lu_points, curr_vae_disc_iter, max_vae_disc_iters, change_lr_iter)

    #Save vae and disc models
    vae_sd = vae_model.module.state_dict() if cfg.NUM_GPUS > 1 else vae_model.state_dict()
    disc_sd = disc_model.module.state_dict() if cfg.NUM_GPUS > 1 else disc_model.state_dict()
    # Record the state
    vae_checkpoint = {
        'epoch': cfg.VAAL.VAE_EPOCHS + 1,
        'model_state': vae_sd,
        'optimizer_state': optim_vae.state_dict(),
        'cfg': cfg.dump()
    }
    disc_checkpoint = {
        'epoch': cfg.VAAL.VAE_EPOCHS + 1,
        'model_state': disc_sd,
        'optimizer_state': optim_disc.state_dict(),
        'cfg': cfg.dump()
    }   
    # Write the checkpoint
    os.makedirs(cfg.EPISODE_DIR, exist_ok=True)
    vae_checkpoint_file = os.path.join(cfg.EPISODE_DIR, "vae.pyth")
    disc_checkpoint_file = os.path.join(cfg.EPISODE_DIR, "disc.pyth")
    torch.save(vae_checkpoint, vae_checkpoint_file)
    torch.save(disc_checkpoint, disc_checkpoint_file)

    if debug: print("Saved VAE model at {}".format(vae_checkpoint_file))
    if debug: print("Saved DISC model at {}".format(disc_checkpoint_file))

    return vae_model, disc_model