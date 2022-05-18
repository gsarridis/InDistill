import argparse
import torch.nn as nn
import torch.optim as optim
from utils.retrieval_evaluation import evaluate_model_retrieval
from utils.loaders import cifar10_loader
import torch
from utils.utils import *
from models.cnn32 import Cnn32
from models.cnn32_small import Cnn32_Small
from models.resnet import ResNet18

def train(net, optimizer, loss_fn, loader, epochs, device):
    for epoch in range(epochs):
        net.train()

        train_loss = 0
        correct = 0
        total = 0

        for (inputs, targets) in tqdm(loader):
            inputs= inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
        acc = correct/total
        print(f"epoch: {epoch}, loss = {train_loss}, accuracy = {acc}")
    return net

def test(net, loss_fn, loader, device):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    for (inputs, targets) in tqdm(loader):
        inputs= inputs.to(device)
        targets = targets.to(device)

        
        outputs = net(inputs)
        loss = loss_fn(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
    acc = correct/total
    print(f"test loss = {test_loss}, accuracy = {acc}")

def train_model(net, train_loader, test_loader, learning_rates, epochs, save_path, device):

    loss_fn = nn.CrossEntropyLoss()

    for lr, ep in zip(learning_rates, epochs):
        print(f'current lr: {lr}')
        optimizer = optim.Adam(net.parameters(), lr=lr)
        net = train(net, optimizer, loss_fn, train_loader, epochs=ep, device=device)

    test(net, loss_fn, test_loader,device=device)
    torch.save(net.state_dict(), save_path)
    return net

def metric_learning_eval(net, dataset, filename):
    if dataset == 'cifar10':
        loader = cifar10_loader

    evaluate_model_retrieval(net=net, path='', dataset_loader=loader,
                             result_path='./results/scores/'+filename+'_baseline_'+dataset+'_retrieval.pickle', layer=3)

    evaluate_model_retrieval(net=net, path='', dataset_loader=loader,
                             result_path='./results/scores/'+filename+'_baseline_'+dataset+'_retrieval_e.pickle', layer=3, metric='l2')

def arg_parser():
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--lr', nargs='+', type=float, required=False, default=[0.001,0.0001], help='Learning rate value')
    parser.add_argument('--ep', nargs='+', type=int, default=[50,30], required=False, help='Number of epochs')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--device', type=str, default='cuda:0')

    # Parse the argument
    args = parser.parse_args()
    return args

def main():
    _ = ensure_reproducability()

    # Create the parser
    args = arg_parser()
    device = torch.device(args.device)

    # Load Dataset and models
    if args.dataset == 'cifar10':
        train_loader, test_loader, _ = cifar10_loader(batch=args.batch_size)

        teacher_net = ResNet18(num_classes=10)
        aux_net = Cnn32(num_classes=10, input_channels=3)
        student_net = Cnn32_Small(num_classes=10, input_channels=3)

    # transfer to device
    teacher_net = teacher_net.to(device)
    aux_net = aux_net.to(device)
    student_net = student_net.to(device)

    # train teacher
    print("Training teacher model...")
    teacher_net = train_model(teacher_net, train_loader= train_loader, test_loader= test_loader, learning_rates=args.lr, epochs=args.ep,
                        save_path='./results/models/teacher_baseline_'+args.dataset+'.pt', device=device)
    # train auxiliary
    print("Training auxiliary model...")
    aux_net = train_model(aux_net, train_loader= train_loader, test_loader= test_loader, learning_rates=args.lr, epochs=args.ep,
                        save_path='./results/models/auxiliary_baseline_'+args.dataset+'.pt', device=device)
    # train student
    print("Training student model...")
    student_net = train_model(student_net, train_loader= train_loader, test_loader= test_loader, learning_rates=args.lr, epochs=args.ep,
                        save_path='./results/models/student_baseline_'+args.dataset+'.pt', device=device)
    

    # metric learning evaluation
    metric_learning_eval(teacher_net, args.dataset, 'teacher')
    metric_learning_eval(aux_net, args.dataset, 'auxiliary')
    metric_learning_eval(student_net, args.dataset, 'student')



if __name__ == '__main__':
    main()
    
