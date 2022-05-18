from torchvision import transforms
import torch
import torchvision


def load_model(net, file):
    state_dict = torch.load(file)
    net.load_state_dict(state_dict)

       
def cifar10_loader(batch = 128,dict_form=False, size=32, path = './data'):
    # Data
    print('Loading data..')
    if size == 128:
        precrop = 160
        crop = 128
        transform_train = transforms.Compose([

        transforms.Resize((precrop, precrop)),
        transforms.RandomCrop((crop, crop)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([

            transforms.Resize((crop, crop)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif size == 32:
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        transform_train = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])



    trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch, shuffle=True, num_workers=8, pin_memory=True)

    train_data_original = torchvision.datasets.CIFAR10(root=path, train=True, transform=transform_test, download=True)
    train_loader_original = torch.utils.data.DataLoader(train_data_original, batch_size=batch,
                                                        shuffle=True, num_workers=8, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True)
    
    if dict_form:
        dataloaders = {
        'train': trainloader,
        'test': testloader
        }   
        dataset_sizes = {
        'train': len(trainset),
        'test': len(testset)
        }
        return dataloaders, dataset_sizes
    else:
        return trainloader, testloader, train_loader_original
