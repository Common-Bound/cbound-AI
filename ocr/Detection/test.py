# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from . import craft_utils
from . import imgproc
from . import file_utils
import json
import zipfile

from .craft import CRAFT

from collections import OrderedDict
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"



class parser():
    def __init__(self):
        self.canvas_size=None
        self.cuda=None
        self.link_threshold=None
        self.low_text=None
        self.mag_ratio=None
        self.poly=None
        self.show_time=None
        self.test_folder=None
        self.text_threshold=None
        self.trained_model=None
        self.image=None
        
    def parse_args(self,
                canvas_size=1280,cuda = True,link_threshold=0.4,
                low_text=0.4,mag_ratio=1.5,poly=False,show_time=False,
                test_folder='test-folder',text_threshold=0.7,
                trained_model='craft_mlt_25k.pth',image=None):
        self.canvas_size=canvas_size
        self.cuda=cuda
        self.link_threshold=link_threshold
        self.low_text=low_text
        self.mag_ratio=mag_ratio
        self.poly=poly
        self.show_time=show_time
        self.test_folder=test_folder
        self.text_threshold=text_threshold
        self.trained_model=trained_model
        self.image=image

args=parser()

def copyStateDict(state_dict):

    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, _ = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def main(canvas_size=1280,cuda = False,link_threshold=0.4,
                low_text=0.4,mag_ratio=1.5,poly=False,show_time=False,
                test_folder=None,text_threshold=0.7,
                trained_model='craft_mlt_25k.pth',image=None):
    
    args.parse_args(canvas_size,cuda,link_threshold,
                low_text,mag_ratio,poly,show_time,
                test_folder,text_threshold,
                trained_model,image)

    """ For test images in a folder """

    result_folder = './result/'
    image_list = None
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    
    return [image_list, result_folder]

def eval(image_list=None, result_folder=None):
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    
    
    net.eval()

    t = time.time()
    roi_list = []

    if args.test_folder is None: 
        #print("164=========================================================")
        bboxes, polys, score_text = test_net(net, args.image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly)
    
        roi_list.append(polys)
    else : 
        image_list, _, _ = file_utils.get_files(args.test_folder)
        
        for k, image_path in enumerate(image_list):
            image = imgproc.loadImage(image_path)

            bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly)

            roi_list.append(polys)

    print("elapsed time : {}s".format(time.time() - t))
    return roi_list


if __name__ == '__main__':
    main()
