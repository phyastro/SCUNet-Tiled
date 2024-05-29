import os.path
import logging
import argparse
import time
import math
import imagetiles

import numpy as np
from datetime import datetime
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util


'''
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)
by Kai Zhang (2021/05-2021/11)
'''


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='scunet_color_real_psnr', help='scunet_color_real_psnr, scunet_color_real_gan')
    parser.add_argument('--testset_name', type=str, default='real3', help='test set, bsd68 | set12')
    parser.add_argument('--show_img', type=bool, default=False, help='show the image')
    parser.add_argument('--model_zoo', type=str, default='model_zoo', help='path of model_zoo')
    parser.add_argument('--testsets', type=str, default='testsets', help='path of testing folder')
    parser.add_argument('--results', type=str, default='results', help='path of results')
    parser.add_argument('--rows', type=int, default=5, help='number of tiles in a column')
    parser.add_argument('--columns', type=int, default=5, help='number of tiles in a row')

    args = parser.parse_args()

    n_channels = 3

    result_name = args.testset_name + '_' + args.model_name     # fixed
    model_path = os.path.join(args.model_zoo, args.model_name+'.pth')

    # ----------------------------------------
    # L_path, E_path
    # ----------------------------------------
    L_path = os.path.join(args.testsets, args.testset_name) # L_path, for Low-quality images
    E_path = os.path.join(args.results, result_name)        # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    from models.network_scunet import SCUNet as net
    model = net(in_nc=n_channels,config=[4,4,4,4,4,4,4],dim=64)

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    logger.info('model_name:{}'.format(args.model_name))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))

        img_L = util.imread_uint(img, n_channels=n_channels)

        util.imshow(img_L) if args.show_img else None

        grid = (args.rows, args.columns)
        overlap = math.ceil(0.125 * ((img_L.shape[0] // grid[0]) + (img_L.shape[1] // grid[1]))) # overlap pixels
        splitter = imagetiles.SplitImage(img_L, grid, overlap) # Create SplitImage object
        tiles, tile_pad_height, tile_pad_width = splitter.split_image() # Split image into tiles
        print(f"height of the tile: {len(tiles[0])}")
        print(f"width of the tile: {len(tiles[0][0])}")
        print(f"total number of tiles: {len(tiles)}")

        start = time.perf_counter()
        for k in range(len(tiles)):
            input = util.uint2tensor4(tiles[k])
            input = input.to(device)

            output = model(input)

            tiles[k] = util.tensor2numpyuint(output)
            print(f"tile {k + 1} has been finished")
        end = time.perf_counter()

        combiner = imagetiles.CombineTiles(tiles, grid, overlap, tile_pad_height, tile_pad_width) # Create CombineTiles object
        final = combiner.combine_tiles() # Combine tiles into original image
        final = final[0:img_L.shape[0], 0:img_L.shape[1]]

        # ------------------------------------
        # save results
        # ------------------------------------
        util.imsave(final, os.path.join(E_path, img_name+ext))
        print(f"Denoised The Image In {end - start}s")

if __name__ == '__main__':

    main()
