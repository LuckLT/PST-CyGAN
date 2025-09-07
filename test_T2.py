#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from PSTmodels import Generator
from datasets_T2 import ImageDataset_test

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    # parser.add_argument('--cuda', type=int, default=1, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='T2_model/netG_A2B.pth', help='A2B generator checkpoint file')
    # parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')

    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc, mask_nc=1)
    # netG_B2A = Generator(opt.output_nc, opt.input_nc, mask_nc=1)

    if opt.cuda:
        netG_A2B.cuda()
        # netG_B2A.cuda()

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    # netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

    # Set model's test mode
    netG_A2B.eval()
    # netG_B2A.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    # input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    input_C = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    # Dataset loader
    transforms_ = [ transforms.ToTensor(),
                    # transforms.Resize(128),
                    # transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                    transforms.Normalize((0.5), (0.5)) ]
    dataloader = DataLoader(ImageDataset_test(opt.dataroot, transform=transforms_, mode='test'),
                            batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
    ###################################

    ###### Testing######

    # Create output dirs if they don't exist
    # if not os.path.exists('output/A'):
    #     os.makedirs('output/A')
    if not os.path.exists('output/B'):
        os.makedirs('output/B')

    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        # real_B = Variable(input_B.copy_(batch['B']))
        prostate = Variable(input_C.copy_(batch['Prostate_A']))

        # Generate output
        fake_B = 0.5*(netG_A2B(real_A, prostate).data + 1.0)
        # fake_A = 0.5*(netG_B2A(real_B, prostate).data + 1.0)

        # Get the original filenames
        A_paths = batch['A_paths']
        # B_paths = batch['B_paths']

        # # Save image files
        # save_image(fake_A, 'output/A/%04d.png' % (i+1))
        # save_image(fake_B, 'output/B/%04d.png' % (i+1))
        # Save image files with original filenames
        for j, (fake_B_img, A_path) in enumerate(zip(fake_B, A_paths)):
            # save_image(fake_A_img, os.path.join('output/A', os.path.basename(A_path)))
            save_image(fake_B_img, os.path.join('output/B', os.path.basename(A_path)))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

    sys.stdout.write('\n')
    ###################################
if __name__ == '__main__':
    test()
