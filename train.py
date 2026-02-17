import os
import random
import numpy as np

import torch
from torchvision.transforms import transforms

from dataset import Paired_Dataset
from model.generator import Generator
from model.discriminator import Discriminator

from utils.arguments import get_args
from utils.trainer import CycleGANTrainer

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

# parse command-line args (see `utils/arguments.py` for flags)
args, grouped_args = get_args()

# set seeds for reproducibility
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

# image preprocessing pipeline used by the Paired_Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# build training and evaluation paired datasets
trainset = Paired_Dataset(
    data_root=os.path.join(args.data_root, 'trainset'),
    vendor1=args.vendor1,
    vendor2=args.vendor2,
    transforms=transform,
    data_range=(args.lower_bound, args.upper_bound)
)

evalset = Paired_Dataset(
    data_root=os.path.join(args.data_root, 'evalset'),
    vendor1=args.vendor1,
    vendor2=args.vendor2,
    transforms=transform,
    data_range=(args.lower_bound, args.upper_bound)
)

# instantiate two generators and discriminators used in CycleGAN
G1, D1 = Generator(in_features=1, out_features=1, n_res_blocks=9), Discriminator(in_features=1)
G2, D2 = Generator(in_features=1, out_features=1, n_res_blocks=9), Discriminator(in_features=1)

# create trainer and start training loop
trainer = CycleGANTrainer(args, grouped_args, trainset, evalset, G1, D1, G2, D2, device)
trainer.train()
