import os
from util import util
import torch
from metrics.fid import compute_fid
import numpy as np
import torch.nn.functional as F

def save_image_grid(tensor_images, path, image_size=128):
    images = []
    for image in tensor_images:
        images.append(util.to_data(F.interpolate(image, scale_factor=image_size/image.size(-1), recompute_scale_factor=True ) ))
    images = np.concatenate(images, 0)
    grid_size = [len(images), 1]
    util.save_image_grid(images, path, grid_size=grid_size)


def make_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def test_fid(a_loader, net_a2b, b_loader, net_b2a, image_dir):
    net_a2b.eval()
    net_b2a.eval()
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    fake_a_path = os.path.join(image_dir, 'fake_A')
    fake_b_path = os.path.join(image_dir, 'fake_B')
    real_a_path = os.path.join(image_dir, 'real_A')
    real_b_path = os.path.join(image_dir, 'real_B')
    make_dirs([fake_a_path, fake_b_path, real_a_path, real_b_path])

    for i, data in enumerate(a_loader):
        data = data['A'].cuda()
        with torch.no_grad():
            fake_b = net_a2b(data).detach()
        file_name = os.path.join(fake_b_path, 'test_fake_B_%06d.png'%i)
        util.save_image(util.tensor2im(fake_b), file_name)
        file_name = os.path.join(real_a_path, 'test_real_A_%06d.png'%i)
        util.save_image(util.tensor2im(data), file_name)

    for i, data in enumerate(b_loader):
        data = data['A'].cuda()
        with torch.no_grad():
            fake_a = net_b2a(data).detach()
        file_name = os.path.join(fake_a_path, 'test_fake_A_%06d.png'%i)
        util.save_image(util.tensor2im(fake_a), file_name)
        file_name = os.path.join(real_b_path, 'test_real_B_%06d.png'%i)
        util.save_image(util.tensor2im(data), file_name)

    fid_a2b = compute_fid(real_b_path, fake_b_path)
    fid_b2a = compute_fid(real_a_path, fake_a_path)
    net_a2b.train()
    net_b2a.train()
    return fid_a2b, fid_b2a