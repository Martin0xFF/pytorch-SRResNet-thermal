'''
Modified by Martin F.
- remove dependency on matlab
- only get performance of model
'''
import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint/l2/model_epoch_75.pth", type=str, help="model path")
parser.add_argument("--dataset", default="test", type=str, help="dataset name, Default: Set5")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model)["model"]

image_list = glob.glob("./testsets/" + opt.dataset + "/*.*")

avg_psnr_predicted = 0.0
avg_psnr_bicubic = 0.0
avg_elapsed_time = 0.0

for image_name in tqdm(image_list):
    im_l = cv2.imread(image_name)[:,:,0,None]
    im_l = im_l.astype(float)
    im_input = im_l.astype(np.float32).transpose(2,0,1)
    im_input = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
    im_input = Variable(torch.from_numpy(im_input/255.).float())

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    HR_4x = model(im_input)
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time

    HR_4x = HR_4x.cpu()

    im_h = HR_4x.data[0].numpy().astype(np.float32)

    im_h = im_h*255.
    im_h = np.clip(im_h, 0., 255.)

    cv2.imwrite( image_name.replace(opt.dataset, 'out')[2:], im_h[0,:,:])


print("It takes average {}s for processing".format(avg_elapsed_time/len(image_list)))
