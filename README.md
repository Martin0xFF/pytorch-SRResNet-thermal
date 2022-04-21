# SRResNet for Thermal Image Upsampling

## Premise
Thermal imaging cameras are expensive, particularly for larger resolution (typical ADS thermal cameras are 512x640). Using an upsampling method may allow for the use of low resolution images as input to produce a feasible larger resolution image. The use of SRResNet in this case would therefore act as a baseline utilized against generative methods.

## Modification
The SRResNet implementation was modified to accept (120, 160) images and directly produce (512, 640) images. This modification can be seen in the main_srresnt.py file.

## Usage

In order to run the model for training you will need to use a GPU with atleast 12 GB vram.

Create a symbolic link to the flir thermal image data set within the data folder called flir

```
sudo ln -s ${FLIR_DATASET_FOLDER} data/flir
```

To train the model use the following command:
```
python main_srresnet.py --cuda --batchSize 8 --threads 12 
```
Here we use cuda with an image batchsize of 8 and 12 threads for dataloading.

During training the model will create a val_log file and train_log file which will have the losses of the model.

## Evalutation

We can run the following command to run the model on a collection of images to upscale.
```
python demo.py --cuda --model checkpoint/latest.pth --save_root out --dataset data/flir/video_thermal_test
```

This will save the images to a folder called out where they can be viewed and compared to the original and ground truth images in video_thermal_test.

---

# PyTorch SRResNet
Implementation of Paper: "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"(https://arxiv.org/abs/1609.04802) in PyTorch

## Usage
### Training
```
usage: main_srresnet.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS]
                        [--lr LR] [--step STEP] [--cuda] [--resume RESUME]
                        [--start-epoch START_EPOCH] [--threads THREADS]
                        [--pretrained PRETRAINED] [--vgg_loss] [--gpus GPUS]

optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        training batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=1e-4
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=500
  --cuda                Use cuda?
  --resume RESUME       Path to checkpoint (default: none)
  --start-epoch START_EPOCH
                        Manual epoch number (useful on restarts)
  --threads THREADS     Number of threads for data loader to use, Default: 1
  --pretrained PRETRAINED
                        path to pretrained model (default: none)
  --vgg_loss            Use content loss?
  --gpus GPUS           gpu ids (default: 0)
```
An example of training usage is shown as follows:
```
python main_srresnet.py --cuda --vgg_loss --gpus 0
```

### demo
```
usage: demo.py [-h] [--cuda] [--model MODEL] [--image IMAGE]
               [--dataset DATASET] [--scale SCALE] [--gpus GPUS]

optional arguments:
  -h, --help         show this help message and exit
  --cuda             use cuda?
  --model MODEL      model path
  --image IMAGE      image name
  --dataset DATASET  dataset name
  --scale SCALE      scale factor, Default: 4
  --gpus GPUS        gpu ids (default: 0)
```
We convert Set5 test set images to mat format using Matlab, for simple image reading
An example of usage is shown as follows:
```
python demo.py --model model/model_srresnet.pth --dataset Set5 --image butterfly_GT --scale 4 --cuda
```

### Eval
```
usage: eval.py [-h] [--cuda] [--model MODEL] [--dataset DATASET]
               [--scale SCALE] [--gpus GPUS]

optional arguments:
  -h, --help         show this help message and exit
  --cuda             use cuda?
  --model MODEL      model path
  --dataset DATASET  dataset name, Default: Set5
  --scale SCALE      scale factor, Default: 4
  --gpus GPUS        gpu ids (default: 0)
```
We convert Set5 test set images to mat format using Matlab. Since PSNR is evaluated on only Y channel, we import matlab in python, and use rgb2ycbcr function for converting rgb image to ycbcr image. You will have to setup the matlab python interface so as to import matlab library. 
An example of usage is shown as follows:
```
python eval.py --model model/model_srresnet.pth --dataset Set5 --cuda
```

### Prepare Training dataset
  - Please refer [Code for Data Generation](https://github.com/twtygqyy/pytorch-SRResNet/tree/master/data) for creating training files.
  - Data augmentations including flipping, rotation, downsizing are adopted.


### Performance
  - We provide a pretrained model trained on [291](http://cv.snu.ac.kr/research/VDSR/train_data.zip) images with data augmentation
  - Instance Normalization is applied instead of Batch Normalization for better performance 
  - So far performance in PSNR is not as good as paper, any suggestion is welcome
  
| Dataset        | SRResNet Paper | SRResNet PyTorch|
| :-------------:|:--------------:|:---------------:|
| Set5           | 32.05          | **31.80**       |
| Set14          | 28.49          | **28.25**       |
| BSD100         | 27.58          | **27.51**       |

### Result
From left to right are ground truth, bicubic and SRResNet
<p>
  <img src='result/result.png' height='260' width='700'/>
</p>
