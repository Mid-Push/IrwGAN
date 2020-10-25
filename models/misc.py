import os
from util import util
import torch
from metrics.fid import calculate_fid_given_paths
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

def save_train_image_grid(tensor_images, betas, path, display_size=10, image_size=128):
    assert len(betas) == len(tensor_images)
    if display_size > len(tensor_images):
        display_size = len(tensor_images)
    images = []
    # row-wise
    for image in (tensor_images):
        image = image.unsqueeze(0)
        images.append(util.to_data(
                F.interpolate(image, scale_factor=image_size / image.size(-1), recompute_scale_factor=True)))
    grid_size = [display_size, len(images)//display_size]
    images = np.concatenate(images, 0)
    img = util.convert_to_pil_image(util.create_image_grid(images, grid_size))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/gnu-free/FreeSerif.ttf', 25)
    except:
        font = ImageFont.load_default()
    grid_w, grid_h = grid_size
    for idx in range(len(images)):
        x = (idx % grid_w) * image_size
        y = (idx // grid_w) * image_size
        name = '%4.3f' % betas[idx]
        color = 'rgb(255, 255, 255)'  # white color
        draw.text((x, y), name, fill=color, font=font)
    # save the edited image
    img.save(path)

def save_image_grid(tensor_image_list, path, display_size, image_size=128):
    images = []
    # row-wise
    """
    for i in range(display_size):
        for j in range(len(tensor_image_list)):
            image = tensor_image_list[j][i].unsqueeze(0)
            #if j == len(tensor_image_list)//2:
            # insert blank between a2b and b2a
            #    images.append(util.to_data(torch.ones([1,3,image_size,image_size])))
            images.append(util.to_data(F.interpolate(image, scale_factor=image_size/image.size(-1), recompute_scale_factor=True ) ))
    grid_size = [len(images)//display_size, display_size]
    """
    num_images = 0
    for i in range(len(tensor_image_list)):
        for j in range(display_size):
            image = tensor_image_list[i][j]
            images.append(util.to_data(F.interpolate(image, scale_factor=image_size/image.size(-1), recompute_scale_factor=True ) ))
            num_images += 1
    assert len(images) == num_images
    grid_size = [display_size, num_images//display_size]

    images = np.concatenate(images, 0)
    util.save_image_grid(images, path, grid_size=grid_size)




def make_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def test_fid(a_loader, net_a2b, b_loader, net_b2a, run_dir, opt):
    net_a2b.eval()
    net_b2a.eval()
    fake_a_path = os.path.join(run_dir, 'fakeA')
    fake_b_path = os.path.join(run_dir, 'fakeB')
    real_a_path = os.path.join(opt.dataroot, 'testA')
    real_b_path = os.path.join(opt.dataroot, 'testB')
    make_dirs([fake_a_path, fake_b_path])
    with torch.no_grad():
        for i, item in enumerate(a_loader):
            data = item['A'].cuda()
            path = item['A_paths'][0].split('/')[-1]
            with torch.no_grad():
                fake_b = net_a2b(data).detach()
            file_name = os.path.join(fake_b_path, path)
            util.save_image(util.tensor2im(fake_b), file_name)

        for i, item in enumerate(b_loader):
            data = item['A'].cuda()
            path = item['A_paths'][0].split('/')[-1]
            with torch.no_grad():
                fake_a = net_b2a(data).detach()
            file_name = os.path.join(fake_a_path, path)
            util.save_image(util.tensor2im(fake_a), file_name)

    fid_a2b = calculate_fid_given_paths([real_b_path, fake_b_path])
    fid_b2a = calculate_fid_given_paths([real_a_path, fake_a_path])
    net_a2b.train()
    net_b2a.train()
    return fid_a2b, fid_b2a