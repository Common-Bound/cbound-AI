import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from .utils import CTCLabelConverter, AttnLabelConverter
from .dataset import RawDataset, AlignCollate
from .model import Model
import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class parser():
    def __init__(self):
        self.FeatureExtraction = None
        self.PAD = None
        self.Prediction = None
        self.SequenceModeling = None
        self.Transformation = None
        self.batch_max_length = None
        self.batch_size = None
        self.character = None
        self.hidden_size = None
        self.image_folder = None
        self.imgH = None
        self.imgW = None
        self.input_channel = None
        self.num_fiducial = None
        self.output_channel = None
        self.rgb = None
        self.saved_model = None
        self.sensitive = None
        self.workers = None

    def parse_args(self,
                   FeatureExtraction='ResNet', PAD=False,
                   Prediction='Attn', SequenceModeling='BiLSTM', Transformation='TPS',
                   batch_max_length=25, batch_size=192,
                   character='0123456789abcdefghijklmnopqrstuvwxyz',
                   hidden_size=256, image_folder='recognition_image/', imgH=32, imgW=100,
                   input_channel=1, num_fiducial=20, output_channel=512, rgb=False,
                   saved_model='TPS-ResNet-BiLSTM-Attn-case-sensitive.pth', sensitive=True, workers=4):

        self.FeatureExtraction = FeatureExtraction
        self.PAD = PAD
        self.Prediction = Prediction
        self.SequenceModeling = SequenceModeling
        self.Transformation = Transformation
        self.batch_max_length = batch_max_length
        self.batch_size = batch_size
        self.character = character
        self.hidden_size = hidden_size
        self.image_folder = image_folder
        self.imgH = imgH
        self.imgW = imgW
        self.input_channel = input_channel
        self.num_fiducial = num_fiducial
        self.output_channel = output_channel
        self.rgb = rgb
        self.saved_model = saved_model
        self.sensitive = sensitive
        self.workers = workers


def demo(opt):
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    AlignCollate_demo = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    scores = list()
    pred_words = list()
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)

            length_for_pred = torch.IntTensor(
                [opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(
                batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred).log_softmax(2)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.permute(1, 0, 2).max(2)
                preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred,
                              is_train=False)  # can train

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            soft_max = torch.nn.Softmax(dim=2)
            preds = soft_max(preds)

            for i in range(len(preds)):
                score = 0
                cnt = 0
                for j, index in enumerate(preds_index[i]):
                    if index > 1:
                        score += preds[i][j][index]
                        cnt += 1
                if cnt > 0:
                    score /= cnt
                else:
                    score /= (cnt + 1)

                # print(score)
                scores.append(float(score))
                pred_words.append(preds_str[i][:preds_str[i].find('[s]')])

    return pred_words, scores


def recognition(image_folder):
    opt = parser()
    opt.parse_args(image_folder=image_folder)

    if opt.sensitive:
        # same with ASTER setting (use 94 char).
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True

    if str(device) == "cuda":
        opt.num_gpu = torch.cuda.device_count()

    texts, scores = demo(opt)

    # for i, text in enumerate(texts):
    #    text = text[:text.find('[s]')]
    #    texts[i] = text
    #    #print(text)

    return texts, scores
