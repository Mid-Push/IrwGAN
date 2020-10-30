import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import argparse
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from data.image_folder import ImageFolder
from models.irw_gan_model import IRWGANModel
import torch
from util.util import to_data
from models import networks

parser = argparse.ArgumentParser()
parser.add_argument('--clean_dir', required=True,
                    help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--model_dir', type=str, required=True,
                    help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--noisy_dir', type=str, default=None,
                    help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--direction', type=str, default="A2B",
                    help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--beta_mode', type=str, default="AB",
                    help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='name of the experiment. It decides where to store samples and models')

args = parser.parse_args()

# load model
model_snapshot_path = os.path.join(args.model_dir, 'model', 'network-snapshot-latest.pth')
beta_net = networks.BetaNet(3, n_layers=4).cuda()
if args.direction == 'A2B':
    beta_net.load_state_dict(torch.load(model_snapshot_path)['beta_net_a'])
else:
    beta_net.load_state_dict(torch.load(model_snapshot_path)['beta_net_b'])
print('[*] loading finished')
# loaders
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


if args.noisy_dir is not None:
    clean_data_dir = args.clean_dir
    noisy_data_dir = args.noisy_dir
    clean = DataLoader(ImageFolder(clean_data_dir, test_transform), batch_size=1, shuffle=False)
    noisy = DataLoader(ImageFolder(noisy_data_dir, test_transform), batch_size=1, shuffle=False)
    clean_images = []
    noisy_images = []
    for i, data in enumerate(clean):
        clean_images.append(data.cuda())
    for i, data in enumerate(noisy):
        noisy_images.append(data.cuda())
    clean_images = torch.cat(clean_images, 0)
    print(clean_images.size())
    noisy_images = torch.cat(noisy_images, 0)
    print(noisy_images.size())
    images = torch.cat([clean_images, noisy_images], 0)
    print(images.size())
    labels = np.concatenate([np.ones([len(clean_images)]), np.zeros(len(noisy_images))],0)
    with torch.no_grad():
        beta_net.eval()
        betas = (beta_net(images))
        print(betas.size())
        betas = to_data(betas)
    print(args.clean_dir, args.noisy_dir)
    #print(betas)
    print('irw_gan...')
    print('precision : %.2f'%precision_score(labels.flatten(), (betas>args.threshold).astype(float).flatten()))
    print('recall : %.2f'%recall_score(labels.flatten(), (betas>args.threshold).astype(float).flatten()))
    print('accuracy : %.2f'%accuracy_score(labels.flatten(), (betas>args.threshold).astype(float).flatten()))
    print('baseline...')
    print('precision : %.2f'%precision_score(labels.flatten(), np.ones([len(labels)]).astype(float).flatten()))
    print('recall : %.2f'%recall_score(labels.flatten(), np.ones([len(labels)]).astype(float).flatten()))
    print('accuracy : %.2f'%accuracy_score(labels.flatten(), np.ones([len(labels)]).astype(float).flatten()))

# python test_beta.py --clean_dir=../datasets/cat2dog/testA --noisy_dir=../datasets/horse2zebra/testA
# --model_dir=results/horse-cat2dog-anime_lsgan_20_AB_gl_1.0_1.0_thr0.1














