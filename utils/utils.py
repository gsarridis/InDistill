import torch
import copy
from tqdm import tqdm
import time
from torch.nn.utils import prune
import torch.nn as nn

def train_step(model, optimizer, loss_fn, device, loader, phase, epoch):
    losses = []
    num_correct = 0
    num_samples = 0
    pbar = tqdm(loader[phase], total=len(loader[phase]), position=0, leave=True, desc="Epoch {}".format(epoch))
    uncert = 0
    img_counter = 0
    # for data, targets, img_name in pbar:
    for data, targets in pbar:
        # Get data to cuda if possible
        data = data.to(device)
        
        targets = targets.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            out = model(data)
            _, preds = torch.max(out, 1)
            loss = loss_fn(out, targets)

        losses.append(loss.item())
                
        num_correct += torch.sum(preds == targets.data)
        num_samples += preds.size(0)


        # backward
        if phase == 'train':
            loss.backward()
            optimizer.step()

 


    avg_loss = sum(losses) / len(losses)
    acc = (num_correct/num_samples).item()

    print("Phase:{}\tLoss:{:.4f}\tAccuracy:{:.4f}".format(phase,avg_loss,acc))
    return avg_loss, acc, model


def train_model(model, criterion, optimizer, dataloaders, num_epochs=25, phases=['train', 'val'], device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_ep = 0
    for epoch in range(num_epochs):
        # print("Learning Rate: {:f}".format(optimizer.param_groups[0]['lr']))
        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            epoch_loss, epoch_acc, model = train_step( model=model, optimizer=optimizer, loss_fn=criterion, device=device, loader=dataloaders, phase=phase, epoch=epoch)
            # deep copy the model
            if (phase == 'val' or phase == 'test') and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_ep = epoch



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    

    # load best model weights
    if 'val' in phases or 'test' in phases:
        model.load_state_dict(best_model_wts)
    return model, best_ep


def prune_model_l1_structured(model, layers, proportion, kernel_sizes=[3,3,3]):
    c=0
    for child in model.children():
        if c in layers:
            layer = child
            prune.ln_structured(layer, 'weight', proportion, n=1, dim=0)
            prune.remove(layer, 'weight')
            # print(layer.weight[1,:,:,:])
            
            filtered_w, idxs =  remove_zero_channels_from_a_tensor(layer.weight)
            # fix conv layer
            conv_layer = list(model.children())[c]
            
            if c == 0:
                conv_layer = nn.Conv2d(filtered_w.shape[1], filtered_w.shape[0], kernel_size=kernel_sizes[0])
                conv_layer.weight = filtered_w
                model.conv1 = copy.deepcopy(conv_layer)
            elif c == 2:
                conv_layer = nn.Conv2d(filtered_w.shape[1], filtered_w.shape[0], kernel_size=kernel_sizes[1])
                conv_layer.weight = filtered_w
                model.conv2 = copy.deepcopy(conv_layer)
            elif c == 4:
                conv_layer = nn.Conv2d(filtered_w.shape[1], filtered_w.shape[0], kernel_size=kernel_sizes[2])
                conv_layer.weight = filtered_w
                model.conv3 = copy.deepcopy(conv_layer)

            # fix bn layer
            bn_layer = list(model.children())[c+1]
            nw = bn_layer.weight[idxs]
            bn_layer = nn.BatchNorm2d(len(idxs))
            bn_layer.weight = torch.nn.Parameter(nw)
            if c == 0:
                model.conv1_bn = copy.deepcopy(bn_layer)
            elif c == 2:
                model.conv2_bn = copy.deepcopy(bn_layer)
            elif c == 4:
                model.conv3_bn = copy.deepcopy(bn_layer)
        c += 1
    return model




def ensure_reproducability():
    # disable convolution benchmarking to avoid randomness
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #seed torch
    torch.manual_seed(0)
    # generator for dataloader
    g = torch.Generator()
    g.manual_seed(0)
    return g

def check_accuracy(loader, model, device):

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        # for  x, y, img_name in loader:
        for  x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += torch.sum(predictions == y.data)
            num_samples += predictions.size(0)
    model.train()
    return (num_correct/num_samples).item()

def count_non_zero_weights(model):
    non_zeros = 0
    for param in model.parameters():
        if param is not None:
            non_zeros += torch.count_nonzero(param)
    return non_zeros


def remove_zero_channels_from_a_tensor(x):
    slice_size = x.shape[0]
    ch_keep = []
    for s in range(slice_size):
        ch_sum = torch.sum(x[s,:,:,:])
        if ch_sum != 0:
            # print(ch_sum)
            ch_keep.append(s)
        # else: 
        #     print("zerooooooooo")
    x = x[ch_keep,:,:,:]
    # x = x.permute(1,0,2,3)
    
    return torch.nn.Parameter(x), ch_keep


def number_of_zero_channels(model, layers):
    zeros = 0
    c = 0
    for child in model.children():
        if c in layers:
            layer = child[-1].conv2
            slice_size = layer.weight.shape[1]
            for s in range(slice_size):
                ch_sum = torch.sum(layer.weight[s,:,:,:])
                if ch_sum == 0:
                    zeros += 1
        c += 1
    return zeros