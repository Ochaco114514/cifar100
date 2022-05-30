import time
import torch,torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import GoogLenet
import vgg
import matplotlib.pyplot as plt
        
net=GoogLenet.GoogleNet()
print(net)
#加载数据集
apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
 
train_dataset = torchvision.datasets.CIFAR100(root='\cifar100', train=True, download=True,transform=apply_transform)
test_dataset = torchvision.datasets.CIFAR100(root='\cifar100', train=False, download=False,transform=apply_transform)
 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
 
#定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=5e-4)
schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
net = net.to(device)
 
#训练模型
print('training on: ',device)
def test(): 
    net.eval()
    acc = 0.0
    sum = 0.0
    loss_sum = 0
    with torch.no_grad():
        for batch, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = criterion(output, target)
            acc+=torch.sum(torch.argmax(output,dim=1)==target).item()
            sum+=len(target)
            loss_sum+=loss.item()
    print('test  acc: %.2f%%, loss: %.4f'%(100*acc/sum, loss_sum/(batch+1)))

    schedule.step(loss_sum/(batch+1))
    return 100.0*acc/sum
 
def train():
        net.train()
        acc = 0.0
        sum = 0.0
        loss_sum = 0
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            acc +=torch.sum(torch.argmax(output,dim=1)==target).item()
            sum+=len(target)
            loss_sum+=loss.item()
            
            if batch%200==0:
                print('\tbatch: %d, loss: %.4f'%(batch, loss.item()))
        print('train acc: %.2f%%, loss: %.4f'%(100*acc/sum, loss_sum/(batch+1)))
        return loss_sum/(batch+1)
        
temp=0
lr=0.001
x=[]
losss=[]
accuracy=[]
for epoch in range(50):
    t0=time.time()
    print('epoch: %d'%epoch)
    loss=train()
    acc=test()
    t1=time.time()
    print("耗时：{}s".format(t1-t0))
    #schedule.step()
    
    if lr!=optimizer.state_dict()['param_groups'][0]['lr']:
        lr=optimizer.state_dict()['param_groups'][0]['lr']
        print("lr now changes:{}".format(lr))
    
    if acc>=50 and acc>temp:
        torch.save(net,"/model.pt")
        temp=acc
    
    x.append(epoch+1)
    losss.append(loss)
    accuracy.append(acc)

print(temp)
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(x,losss)

plt.xlabel("epoch")
plt.ylabel("loss")

plt.figure(2)
plt.plot(x,accuracy)

plt.xlabel("epoch")
plt.ylabel("acc")

plt.show()
