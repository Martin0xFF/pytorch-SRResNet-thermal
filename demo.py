'''
Modified by Martin F.
2022, 4, 20
- modified code to remove dependency on matlab
- produces output for each image in set
'''
import argparse, os
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torch.nn import MSELoss
import numpy as np
import time, math
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from main_srresnet import FlirSetFromFolder

parser = argparse.ArgumentParser(description="PyTorch SRResNet Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint/l2/model_epoch_75.pth", type=str, help="model path")
parser.add_argument("--save_root", default="checkpoint/l2/model_75", type=str, help="where to save images")

parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--dataset", default="./data/flir/video_thermal_test", type=str, help="dataset name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model)["model"]

save_root = opt.save_root
os.makedirs(save_root, exist_ok=True)
to = ToTensor()

fsdl = DataLoader(dataset=FlirSetFromFolder(opt.dataset), num_workers=1, batch_size=4, shuffle=False)

crit = MSELoss()
if cuda:
    crit = crit.cuda()

print("Dataset=",opt.dataset)
print("Scale=",opt.scale)

for ind, batch in tqdm(enumerate(fsdl)):
    torch.cuda.empty_cache()
    im_input, im_gt_t = batch

    im_gt = im_gt_t.numpy()[0,0,:,:]
    im_b = cv2.resize(im_input.numpy()[0,0,:,:], (640, 512))

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
        im_gt_t = im_gt_t.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    out = model(im_input)

    with torch.no_grad():
        loss = crit(out, im_gt_t)

    elapsed_time = time.time() - start_time

    out = out.cpu()

    im_h = out.data[0].numpy().astype(np.float32)
    im_h = im_h*255.
    im_h[im_h<0] = 0
    im_h[im_h>255.] = 255.
    im_h = im_h.transpose(1,2,0)

    im_gt = im_gt*255.
    im_gt[im_gt<0] = 0
    im_gt[im_gt>255.] = 255.

    im_b = im_b*255.
    im_b[im_b<0] = 0
    im_b[im_b>255.] = 255.


    img_size = im_gt.shape
    filename = os.path.join(save_root, f"img_{ind}")
    new_image = Image.new('L',(img_size[1]*3, img_size[0]), (250))
    new_image.paste(Image.fromarray(im_gt.astype(np.uint8), mode='L'), (2*img_size[1],0))
    new_image.paste(Image.fromarray(im_b.astype(np.uint8), mode='L'),(0,0))
    new_image.paste(Image.fromarray(im_h[:,:,0].astype(np.uint8), mode='L'),(img_size[1],0))
    new_image.save('{}_results_loss_{}.png'.format(filename, str(loss.item())))

    '''
    fig = plt.figure()
    ax = plt.subplot(1,3,1)
    ax.imshow(im_gt)
    ax.set_title("GT")

    ax = plt.subplot(1,3,2)
    ax.imshow(im_b)
    ax.set_title("Input(Bicubic)")

    ax = plt.subplot(1,3,3)
    ax.imshow(im_h.astype(np.uint8))
    ax.set_title("Output(SRResNet)")
    plt.show()
    '''
