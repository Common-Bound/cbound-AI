from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from .util import *
import argparse
import os
import os.path as osp
from .darknet import Darknet
import pickle as pkl
import pandas as pd
import random


class parser():
    def __init__(self):
        self.images = None
        self.det = None
        self.bs = None
        self.confidence = None
        self.nms_thresh = None
        self.cfgfile = None
        self.weightsfile = None
        self.reso = None

    def arg_parse(self, images="object_images", det="det", bs=1, confidence=0.5, nms_thresh=0.4,
                  cfgfile="yolov3.cfg", weightsfile="yolov3.weights", reso="416"):
        self.images = images
        self.det = det
        self.bs = bs
        self.confidence = confidence
        self.nms_thresh = nms_thresh
        self.cfgfile = cfgfile
        self.weightsfile = weightsfile
        self.reso = reso


def yolo_object(images="object_images", det="det", bs=1, confidence=0.5, nms_thresh=0.4,
                cfgfile="yolov3.cfg", weightsfile="yolov3.weights", reso="416"):

    args = parser()
    args.arg_parse(images, det, bs, confidence,
                   nms_thresh, cfgfile, weightsfile, reso)

    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes("coco.names")

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    # Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img)
                  for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    loaded_ims = [cv2.imread(x) for x in imlist]

    im_batches = list(map(prep_image, loaded_ims, [
                      inp_dim for x in range(len(imlist))]))
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i*batch_size: min((i + 1)*batch_size,
                                                              len(im_batches))])) for i in range(num_batches)]

    write = 0

    object_names = []

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    for i, batch in enumerate(im_batches):
        # load the image
        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        prediction = write_results(
            prediction, confidence, num_classes, nms_conf=nms_thesh)

        if type(prediction) == int:
            continue

        # transform the atribute from index in batch to index in imlist
        prediction[:, 0] += i*batch_size

        if not write:  # If we have't initialised output
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(imlist[i*batch_size: min((i + 1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            object_names = objs
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))

        if CUDA:
            torch.cuda.synchronize()
    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

    scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)


    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2



    output[:,1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

    colors = pkl.load(open("pallete", "rb"))


    object_locations = []

    def write(x, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        color = random.choice(colors)
        label = "{0}".format(classes[cls])
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        object_locations.append({
            'shape_attributes': {
                    'x': int(c1[0]),
                    'y': int(c1[1]),
                    'width': int(c2[0] - c1[0]),
                    'height': int(c2[1] - c1[1]),
                    'size': int(c2[0] - c1[0]) * int(c2[1] - c1[1]),
                }
            })
        return img


    list(map(lambda x: write(x, loaded_ims), output))


    torch.cuda.empty_cache()

    return object_names, object_locations
