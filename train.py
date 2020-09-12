"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset, get_test_loaders
from models import create_model
from util.visualizer import Visualizer
from util.util import format_time
from util import util
import tensorboardX as tensorboard
#from torch.utils.tensorboard import SummaryWriter
from models import misc
import os
import numpy as np


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    test_loader_a, test_loader_b = get_test_loaders(opt) # get test loader by hard-coding options
    fix_a = iter(test_loader_a).next()['A']
    fix_b = iter(test_loader_b).next()['A']

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    train_writer = tensorboard.SummaryWriter(opt.expr_dir) # setup train writer
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    cur_iters = 0                # the total number of training iterations
    if opt.resume_epoch is not None:
        cur_iters = opt.resume_epoch * opt.iterations_per_epoch
    start_time = time.time()  # timer for entire epoch
    total_iters = (opt.n_epochs + opt.n_epochs_decay) * opt.iterations_per_epoch
    print('Start training...\n')
    for epoch in range(opt.epoch_count+1, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        for _ in range(opt.iterations_per_epoch//opt.batch_size):  # inner loop within one epoch
            data = dataset.next()
            cur_iters += opt.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            beta_as_list, beta_bs_list = model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if cur_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                t_comp = (time.time() - start_time)
                info  = 'cur_iters: [%d/%d]' % (cur_iters, total_iters)
                info += ' total_time: %-12s' % format_time(t_comp)
                info += ' irw_loss_a: %4.2f' % model.irw_loss_A.item()
                info += ' irw_loss_b: %4.2f' % model.irw_loss_B.item()
                print(info)

            if cur_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file

                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, True, model.get_current_betas())
                util.write_loss(cur_iters, model, train_writer, prefix='training')
                visualizer.print_current_losses(model.get_current_losses())
                print('beta_as: %s' % np.array2string(beta_as_list, precision=2, suppress_small=True))
                print('beta_bs: %s' % np.array2string(beta_bs_list, precision=2, suppress_small=True))
                # evaluation
                test_images = model.test(fix_a, fix_b)
                test_path = os.path.join(opt.expr_dir, 'test_image-%03d.png' % epoch)
                misc.save_image_grid(test_images, test_path)

        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            model.save_networks('latest')
            model.save_networks(epoch)
            print('Start evaluating the image translation performance...')
            cur_test_dir = os.path.join(opt.expr_dir, 'test_images/epoch_%03d' % epoch)
            fid_a2b, fid_b2a = misc.test_fid(test_loader_a, model.gen_a2b, test_loader_b, model.gen_b2a, cur_test_dir)
            train_writer.add_scalar('evaluation/fid_a2b', fid_a2b, cur_iters)
            train_writer.add_scalar('evaluation/fid_b2a', fid_b2a, cur_iters)
            print('FID_a2b: %4.2f, FID_b2a: %4.2f' % (fid_a2b, fid_b2a))