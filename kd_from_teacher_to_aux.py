from utils.loaders import load_model
from utils.loaders import cifar10_loader
from utils.retrieval_evaluation import evaluate_model_retrieval
import torch
import torch
import argparse
from models.resnet import ResNet18
from models.cnn32 import Cnn32
from methods.pkt import PKT_loss
import torch.optim as optim
from tqdm import tqdm


def distill(teacher_net, aux_net, t_layer, a_layer, loader, learning_rates, epochs, device):
    for lr, ep in zip(learning_rates, epochs):
        optimizer = optim.Adam(params=list(aux_net.parameters()), lr=lr)
        for epoch in range(ep):
            aux_net.train()
            teacher_net.eval()

            train_loss = 0
            for (inputs, targets) in tqdm(loader):

                optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)
                
                teacher_feats = teacher_net.get_features(inputs, layers=t_layer, pruned_model= None, pruned_layers=[-1])

                aux_feats = aux_net.get_features(inputs, layers=a_layer, pruned_layers=[-1])

                loss = 0
                for teacher_f, student_f in zip(teacher_feats, aux_feats):

                    loss = PKT_loss(teacher_f, student_f)
                loss.backward()
                optimizer.step()
                train_loss += loss.cpu().data.item()
            print(f'epoch: {epoch}, loss: {train_loss}')
    return aux_net

    


def arg_parser():
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--a_layer', nargs='+', type=int, default=[3], help='Auxiliary\'s layers to transfer')
    parser.add_argument('--t_layer', nargs='+', type=int, default=[3], help='Teacher\'s layers to transfer')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--model_savepath', type=str, default='./results/models/auxiliary_')
    parser.add_argument('--scores_savepath', type=str, default='./results/scores/auxiliary_')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--lr', nargs='+', type=float, required=False, default=[0.001], help='Learning rate value')
    parser.add_argument('--ep', nargs='+', type=int, default=[50], required=False, help='Number of epochs')
    # Parse the argument
    args = parser.parse_args()
    return args

def main():
    args = arg_parser()
    device = torch.device(args.device)
    torch.manual_seed(12345)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load models and data
    if args.dataset == 'cifar10':
        teacher_net = ResNet18(num_classes=10)
        aux_net = Cnn32(num_classes=10,input_channels=3)
        loader = cifar10_loader
    
    train_loader, _, _ = loader(batch=128)   
    load_model(aux_net,'./results/models/auxiliary_baseline_'+args.dataset+'.pt')
    aux_net.to(device)
    load_model(teacher_net,'./results/models/teacher_baseline_'+args.dataset+'.pt')
    teacher_net.to(device)    


    # run kd
    aux_net = distill(teacher_net,aux_net,args.t_layer, args.a_layer, train_loader, args.lr, args.ep, device)

    # save model
    torch.save(aux_net.state_dict(), args.model_savepath + args.dataset + ".pt")

    # Evaluate auxiliary model
    if args.dataset == 'cifar10':
        loader = cifar10_loader
    evaluate_model_retrieval(net=aux_net, path="", dataset_loader=loader,
                             result_path=args.scores_savepath + args.dataset + '_retrieval.pickle', layer=3)
    evaluate_model_retrieval(net=aux_net, path='', dataset_loader=loader,
                             result_path=args.scores_savepath + args.dataset + '_retrieval_e.pickle', layer=3, metric='l2')
if __name__ == '__main__':
    main()


