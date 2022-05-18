from utils.loaders import load_model
from utils.loaders import cifar10_loader
from utils.retrieval_evaluation import evaluate_model_retrieval
import torch
import torch
import argparse
from models.cnn32 import Cnn32
from models.cnn32_small import Cnn32_Small
from methods.pkt import PKT_loss
import torch.optim as optim
from tqdm import tqdm
from torch import nn


def distill(teacher_net, aux_net, pruned_net, t_layer, a_layer, loader, learning_rates, epochs, device, i_method, l_method, learning_scheme):
    first_run = True 

    for lr, ep in zip(learning_rates, epochs):

        
        for epoch in range(ep):
            optimizer = optim.Adam(params=list(aux_net.parameters()), lr=lr)
            aux_net.train()
            teacher_net.eval()

            train_loss = 0

            # define weights and loss functions
            a = 2
            if i_method == 'indistill' and l_method == 'pkt' and learning_scheme == 'cls':
                
                if first_run:
                    if epoch<a+1 :
                        weights = (0, 0, 0, 0, 1)
                        loss_fn = nn.MSELoss()
                    elif epoch<a+1 + a+2 :
                        weights = (0, 0, 0, 1, 0)
                        loss_fn = nn.MSELoss()
                    elif epoch<a+1 + a+2 + a+3 :
                        weights = (0, 0, 1, 0, 0)
                        loss_fn = nn.MSELoss()
                    else :
                        weights = (0, 1, 0, 0, 0)
                        loss_fn = PKT_loss

            for (inputs, targets) in tqdm(loader):

                optimizer.zero_grad()
                inputs, targets = inputs.to(device), targets.to(device)
                

                # for indistill + pkt + cls
                if i_method == 'indistill' and l_method == 'pkt' and learning_scheme == 'cls':
                    teacher_feats = teacher_net.get_features(inputs, layers=t_layer, pruned_model= pruned_net, pruned_layers=[2,1,0])
                    aux_feats = aux_net.get_features(inputs, layers=a_layer, pruned_layers=[-1])
                    loss = 0
                   
                    for counter, (teacher_f, student_f, w) in enumerate(zip(teacher_feats, aux_feats, weights)):
                        if w > 0:
                            loss = loss_fn(teacher_f, student_f)


                loss.backward()
                optimizer.step()
                train_loss += loss.cpu().data.item()


            print(f'epoch: {epoch}, loss: {train_loss}')
        first_run = False
    return aux_net


def arg_parser():
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--a_layer', nargs='+', type=int, default=[4,3,2,1,0], help='Auxiliary\'s layers to transfer')
    parser.add_argument('--t_layer', nargs='+', type=int, default=[4,3,2,1,0], help='Teacher\'s layers to transfer')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--model_savepath', type=str, default='./results/models/student_')
    parser.add_argument('--scores_savepath', type=str, default='./results/scores/student_')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--lr', nargs='+', type=float, required=False, default=[0.001,0.0001], help='Learning rate value')
    parser.add_argument('--ep', nargs='+', type=int, default=[60,10], required=False, help='Number of epochs')
    parser.add_argument('--inter_method', type=str, default='indistill', choices=['indistill'])
    parser.add_argument('--single_layer_method', type=str, default='pkt', choices=['pkt'])
    parser.add_argument('--scheme', type=str, default='cls', choices=['cls'])
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
    pruned_model = None
    if args.dataset == 'cifar10':
        
        student_net = Cnn32_Small(num_classes=10,input_channels=3)
        teacher_net = Cnn32(num_classes=10,input_channels=3)
        if args.inter_method == 'indistill':
            pruned_model = Cnn32(num_classes=10,input_channels=3, pruned=True)
        loader = cifar10_loader
    train_loader, _, _ = loader(batch=128)  
    if  args.dataset == 'fmnist' and args.inter_method == 'indistill':
        load_model(teacher_net,'./results/models/aux_'+args.dataset+'_mkt_base.pt')
    else:
        load_model(teacher_net,'./results/models/auxiliary_'+args.dataset+'.pt')
    #load_model(student_net,'./results/models/student_baseline_'+args.dataset+'.pt')
    student_net.to(device)
    teacher_net.to(device)    
    
    if not pruned_model == None:
        load_model(pruned_model,'./results/models/pruned_auxiliary_'+args.dataset+'.pt')
        pruned_model.to(device)

    # run kd
    student_net = distill(teacher_net, student_net, pruned_model, args.t_layer, args.a_layer, train_loader, args.lr, args.ep, device, args.inter_method, args.single_layer_method, args.scheme)

    # save model
    torch.save(student_net.state_dict(), args.model_savepath + args.dataset + '_' + args.inter_method + '_' + args.single_layer_method + '_' + args.scheme+ ".pt")

    # Evaluate student model
    if args.dataset == 'cifar10':
        loader = cifar10_loader
    
    evaluate_model_retrieval(net=student_net, path="", dataset_loader=loader,
                             result_path=args.scores_savepath + args.dataset + '_' + args.inter_method + '_' + args.single_layer_method+ '_' + args.scheme+ '_retrieval.pickle', layer=3)
    evaluate_model_retrieval(net=student_net, path='', dataset_loader=loader,
                             result_path=args.scores_savepath + args.dataset + '_' + args.inter_method + '_' + args.single_layer_method+ '_' + args.scheme+ '_retrieval_e.pickle', layer=3, metric='l2')
if __name__ == '__main__':
    main()


