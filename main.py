from easydict import EasyDict as edict
from torch.optim import Adam
from torchvision.models.resnet import resnet18
from torch.nn import CrossEntropyLoss
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from torch.nn import Conv2d, Linear
import torch
import numpy as np
import argparse
import yaml

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def add_extra_tensor_dims(data, args):
    full_data = [data]

    if args.squares:
        squares = data**2
        full_data.append(squares)
    if args.hshift:
        data_right = torch.roll(data, shifts=1, dims=3 )
        data_left = torch.roll(data, shifts=-1, dims=3 )
        full_data.extend([data_right, data_left])
    if args.vshift:
        data_down = torch.roll(data, shifts=1, dims=2 )
        data_up = torch.roll(data, shifts=-1, dims=2 )
        full_data.extend([data_down, data_up])
    if len(full_data) == 1:
        return data
    else:
        return torch.cat(full_data,dim=1)

def set_random(args):
    torch.manual_seed(args.seed)                       
    torch.cuda.manual_seed(args.seed)       
    np.random.seed(args.seed)                           
    state_np = np.random.get_state()           
    #torch.set_deterministic(True)               
    #torch.backends.cudnn.benchmark = False  

def train(model, dl, args):
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    print("Init Training:")
    for n_batch, (img, label) in enumerate(dl):
        opt.zero_grad()
        bs = img.shape[0]
        label = label.to(args.device)
        img = add_extra_tensor_dims(img, args).to(args.device)
        output = model(img)
        loss = criterion(output, label)
        loss_m.update(loss, n=1)

        # Get acc:
        preds = output.argmax(dim=-1)
        correct = (preds == label).sum().item()
        acc_m.update(correct/bs)
        loss.backward()
        opt.step()
        print("\rBatch {}/{} Avg Loss: {:.2f} Loss: {:.2f} - Avg Acc: {:.2f}% Acc: {:.2f}%".format(n_batch+1,
                                                                 len(dl),
                                                                 loss_m.avg,
                                                                 loss_m.val,
                                                                 100*acc_m.avg,
                                                                 100*acc_m.val), end="")
    return loss_m.avg, acc_m.avg

def test(model, dl, args):
    model.eval()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    print("\nInit Testing:")
    with torch.no_grad():
        for n_batch, (img, label) in enumerate(dl):
            bs = img.shape[0]
            label = label.to(args.device)
            img = add_extra_tensor_dims(img, args).to(args.device)
            output = model(img)
            loss = criterion(output, label)
            loss_m.update(loss, n=1)

            # Get acc:
            preds = output.argmax(dim=-1)
            correct = (preds == label).sum().item()
            acc_m.update(correct/bs)
            print("\rBatch {}/{} Avg Loss: {:.2f} Loss: {:.2f} - Avg Acc: {:.2f}% Acc: {:.2f}%".format(n_batch+1,
                                                                    len(dl),
                                                                    loss_m.avg,
                                                                    loss_m.val,
                                                                    100*acc_m.avg,
                                                                    100*acc_m.val), end="")
    return loss_m.avg, acc_m.avg

# Args settings
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config',default='',type=str, required=True)

params = parser.parse_args()
f = open(params.config, "r")
args = yaml.load(f, Loader=yaml.FullLoader)['settings']
f.close()
args = edict(args)
# Env settings
args.device = "cuda" if torch.cuda.is_available() else "cpu"
# Data settings

t = Compose([Resize(256),
    CenterCrop(224), 
    ToTensor(), 
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ds_train = CIFAR10(root=".", train=True, download=True, transform=t)
ds_test = CIFAR10(root=".", train=False, download=True, transform=t)
train_dl = DataLoader(ds_train, batch_size=args.train_bs)
val_dl = DataLoader(ds_train, batch_size=args.val_bs)
in_channels = 3

if args.squares:
    in_channels+=3
if args.hshift:
    in_channels+=6
if args.vshift:
    in_channels+=6


# Initialize random seeds
set_random(args)
# Model Settings
model = resnet18()
# Change starting and classification layers
model.conv1 = Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = Linear(in_features=512, out_features=10, bias=True)
model = model.to(args.device)
opt = Adam(model.parameters(), lr=args.lr)
criterion = CrossEntropyLoss()

metrics = {'loss_train': [], 'loss_val': [], 'acc_train': [], 'acc_val': []}

for epoch in range(1, args.epochs+1):
    print("Epoch {}:".format(epoch))
    tr_loss, tr_acc = train(model, train_dl, args)
    tst_loss, tst_acc = test(model, val_dl, args)
    metrics['loss_train'].append(tr_loss)
    metrics['loss_val'].append(tr_acc)
    metrics['acc_train'].append(tst_loss)
    metrics['acc_val'].append(tst_acc)
    torch.save(metrics, params.config.split(".")[0] + "_results.pth")
