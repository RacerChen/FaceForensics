from __future__ import print_function, division

import copy
import time

import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from classification.dataset.transform import xception_default_data_transforms
from torch import tensor, nn, optim
import torch

from models import return_pytorch04_xception
from xception import xception

train_dataset = torchvision.datasets.ImageFolder(root='/home/jc/Faceforensics_onServer/Final_Faceforensics++/train',
                                                 transform=xception_default_data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

valid_dataset = torchvision.datasets.ImageFolder(root='/home/jc/Faceforensics_onServer/Final_Faceforensics++/val',
                                                 transform=xception_default_data_transforms['val'])
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=0)

# test_dataset = torchvision.datasets.ImageFolder(root='/home/jc/Faceforensics_onServer/Final_Faceforensics++/test',
#                                                 transform=xception_default_data_transforms['test'])
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)

image_datasets = {
    'train': train_dataset,
    'val': valid_dataset,
    # 'test': test_dataset
}

dataloaders = {
    'train': train_loader,
    'val': valid_loader,
    # 'test': test_loader
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# fine-tune here! (ft: fine-tune)
model_ft = return_pytorch04_xception()

# debug for:
# size mismatch for block1.rep.0.pointwise.weight: copying a param with shape torch.Size([128, 64])
# from checkpoint, the shape in current model is torch.Size([128, 64, 1, 1]).

num_ftrs = model_ft.last_linear.in_features
# for param in model_ft.parameters():
#     param.requires_grad = False
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

# 只优化最后的分类层
optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=1e-2, momentum=0.9)
# 观察到所有参数都被优化
optimizer_whole = optim.SGD(model_ft.parameters(), lr=1e-2, momentum=0.9)

# 每7个周期，LR衰减0.1倍
exp_lr_scheduler_ft = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
exp_lr_scheduler_whole = lr_scheduler.StepLR(optimizer_whole, step_size=7, gamma=0.1)

num_epoches = 10

# for epoch in range(num_epoches):
#     for img, label in train_loader:
#         print(img)
#         print(img.shape)
#         print(label)


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0

        # 迭代数据.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 零参数梯度
            optimizer.zero_grad()
            # 前向
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                # 后向+仅在训练阶段进行优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)


        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
        # 深度复制mo
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    # train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler_ft, 3)
    # for param in model_ft.parameters():
    #     param.requires_grad = True
    torch.save(train_model(model_ft, criterion, optimizer_whole, exp_lr_scheduler_whole, 15).state_dict(),
               '/home/jc/Faceforensics_onServer/Model/Trained_model.pth')
