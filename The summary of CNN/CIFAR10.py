import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from leNet5 import Lenet5
from torch import nn,optim
from  resnet import  ResNet18

def main():
    cifar_train =datasets.CIFAR10('cifar',True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)
    cifar_train = DataLoader(cifar_train,batch_size=32,shuffle=True)

    cifar_test =datasets.CIFAR10('cifar',False,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)
    cifar_test = DataLoader(cifar_test,batch_size=32,shuffle=True)

    #x,label = iter(cifar_train).next()
    #print('x:',x.shape,'label',label.shape)

    device = torch.device('cuda')

    model = ResNet18().to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    criteon = nn.CrossEntropyLoss().to(device)

    for epoch in range(1000):
        model.train()
        for batchidx ,(x,label) in enumerate(cifar_train):
            #[b,3,32,32]
            #[b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            #logits:[b,10]
            #label:[10]
            #loss: tensor scalar
            loss = criteon(logits,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #test
        print(epoch,loss.item())
        model.eval()
        total_correct = 0
        total_num = 0
        with torch.no_grad():

            for x,label in cifar_test:
                #[b,3,32,32]
                #[b]
                x, label = x.to(device), label.to(device)

                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred,label).float().sum()
                total_num += x.size(0)

            acc = total_correct/total_num
            print(epoch,acc)
if __name__=='__main__':
    main()