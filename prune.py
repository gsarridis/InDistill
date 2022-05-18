import torch
from utils.utils import *
from utils.loaders import load_model
import argparse
from models.cnn32 import Cnn32

# from torchsummary import summary

def arg_parser():
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--proportion', type=float, default=0.50, help='The porpotion of channels you want to remove')
    parser.add_argument('--layers', nargs='+', type=int, default=[0,1,2], help='The layers you want to prune')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--outpath', type=str, default='./results/models/pruned')
    parser.add_argument('--modelpath', type=str, default='./results/models/auxiliary_')
    parser.add_argument('--device', type=str, default='cuda:0')
    # Parse the argument
    args = parser.parse_args()
    return args

def main():
    _ = ensure_reproducability()

    # Create the parser
    args = arg_parser()
    device = torch.device(args.device)
    

    layers = args.layers
    if args.dataset == 'cifar10':
        model = Cnn32(num_classes=10,input_channels=3)
        kernel_sizes = [3,3,3]
    else:
        sys.exit("Please use a valid dataset!")
    layers = [x*2 for x in layers]
    load_model(model, args.modelpath + args.dataset + ".pt")
    model.to(device)

    print (f'Arguements: {args}')

    # check pruned
    model = prune_model_l1_structured(model, layers, args.proportion, kernel_sizes)

    torch.save(model.state_dict(),  args.outpath+'_auxiliary_'+ args.dataset +'.pt')

if __name__ == '__main__':
    main()

