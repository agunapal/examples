from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn

from datasets import load_dataset

torch.set_float32_matmul_precision('high')
cudnn.benchmark=True



def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):

        data, target = batch["image"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data, target = data["image"], data["labels"]
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    torch.cuda.synchronize()
    end.record()
    return result, start.elapsed_time(end) / 1000

def collate_fn(batch):
   batch = list(filter(lambda x: x is not None, batch))
   return torch.utils.data.dataloader.default_collate(batch) 


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--torch-compile', action='store_true', default=False,
                        help='To enable torch.compile')
    parser.add_argument('--reduce-overhead', action='store_true', default=False,
                        help='To enable reduce-overhead mode in torch.compile')
    parser.add_argument('--max-autotune', action='store_true', default=False,
                        help='To enable max-autotune mode in torch.compile')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD Momentum (default: 0.9)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    def train_transform_func(examples):
        examples["image"] =  [train_transform(pil_img.convert("RGB")) for pil_img in examples["image"]]
        return examples
    def test_transform_func(examples):
        examples["image"] =  [val_transform(pil_img.convert("RGB")) for pil_img in examples["image"]]
        return examples
    ds = load_dataset("cats_vs_dogs", split="train")
    ds = ds.train_test_split(test_size=0.2, shuffle=True)
    train_ds, test_ds = ds["train"], ds["test"]
    train_ds.set_transform(train_transform_func)
    test_ds.set_transform(test_transform_func)
    train_loader = torch.utils.data.DataLoader(train_ds,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    model = model.to(device)
    if args.torch_compile:
        print("torch.compile() enabled")
        if args.reduce_overhead:
            print("Mode 'reduce-overhead' enabled")
            opt_model = torch.compile(model, mode="reduce-overhead")
        elif args.max_autotune:
            print("Mode 'max-autotune' enabled")
            opt_model = torch.compile(model, mode="max-autotune")
        else:
            opt_model = torch.compile(model)
    else:
        print("torch.compile() disabled")
        opt_model = model
    
    optimizer = optim.SGD(opt_model.parameters(), lr=args.lr, momentum=args.momentum)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, args.epochs + 1):
        print(f"Training Time: {timed(lambda: train(args, opt_model, device, train_loader, optimizer, criterion, epoch))[1]}")
        print(f"Evaluation Time: {timed(lambda: test(opt_model, device, test_loader, criterion))[1]}")
        scheduler.step()

    if args.save_model:
        torch.save(opt_model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
