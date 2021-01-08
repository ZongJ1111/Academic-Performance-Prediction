# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import h5py
import torch.utils.data as Data
import numpy as np
import os
from TB-CNN model import TBCNN

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

EPOCH = 50
BATCH_SIZE = 16
LR = 1e-5
LR_DECAY_EPOCH = 20
LR_DECAY_RATE = 0.5


class Train_Dataset(Data.Dataset):
    p1 = 'train_score_couple_topk.hdf5'
    p2 = '12_365_18_tensor.hdf5'
    def __init__(self, key=1):
        self.f1 = h5py.File(self.p1,'r')
        f2 = h5py.File(self.p2,'r')
        self.key = key
        self.card = f2['card_data']
        self.list = self.f1['college'+str(key)]

    def __getitem__(self, item):
        id1, loc1, id2, loc2, target,NDCG_10 = self.list[item, :].reshape(-1)
        cardData1 = self.card[loc1]
        cardData2 = self.card[loc2]
        return cardData1, cardData2, target,NDCG_10

    def __len__(self):
        return len(self.list)

    def set_list(self, key):
        self.list = self.f1['college'+str(key)]

class Test_Dataset(Data.Dataset):
    p1 = 'test_score_couple_del1000.hdf5'
    p2 = '12_365_18_tensor.hdf5'
    def __init__(self, key=1):
        self.f1 = h5py.File(self.p1,'r')
        f2 = h5py.File(self.p2,'r')
        self.key = key
        self.card = f2['card_data']
        self.list = self.f1['college'+str(key)]

    def __getitem__(self, item):
        id1, loc1, id2, loc2, target = self.list[item, :].reshape(-1)
        cardData1 = self.card[loc1]
        cardData2 = self.card[loc2]
        return cardData1, cardData2, target

    def __len__(self):
        return len(self.list)

    def set_list(self, key):
        self.list = self.f1['college'+str(key)]


train_dataset_list = []
test_dataset_list = []
train_loader_list = []
test_loader_list = []
train_dataset_len = 0
test_dataset_len = 0
for i in range(1, 20):
    train_dataset = Train_Dataset(i)
    test_dataset=Test_Dataset(i)
    train_dataset_list.append(train_dataset)
    test_dataset_list.append(test_dataset)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_loader_list.append(train_loader)
    test_loader_list.append(test_loader)
    train_dataset_len += len(train_dataset)
    test_dataset_len += len(test_dataset)

model = TBCNN()
if torch.cuda.is_available():
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LR_DECAY_EPOCH, LR_DECAY_RATE)


for epoch in range(EPOCH):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)

    model.train()
    train_all_loss = 0
    train_all_acc = 0
    for col in range(1,20):
        running_loss = 0.0
        running_acc = 0.0
        accuracy = 0.0
        print('college',str(col))
        print('*' * 10)

        for step, data in enumerate(train_loader_list[col-1],1):

            cardData1, cardData2,target, NDCG_10 = data
            cardData1 = cardData1.type(torch.FloatTensor)
            cardData2 = cardData2.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)
            cardData1 = cardData1.cuda()
            cardData2 = cardData2.cuda()
            target = target.cuda()
            out1 = model(cardData1,col)
            out2 = model(cardData2,col)
            distance = out2-out1
            distance = distance.reshape(-1)

            delt_NDCG_10 = 10*NDCG_10
            delt_NDCG_10 = delt_NDCG_10.type(torch.FloatTensor)
            delt_NDCG_10 = delt_NDCG_10.cuda()

            loss = torch.mean(delt_NDCG_10 * torch.max(torch.Tensor([0]).cuda(), (target * distance) + 2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for i in range(distance.shape[0]):
                if distance[i] >= 0:
                    distance[i] = -1
                else:
                    distance[i] = 1

            running_loss += loss.item() * target.size(0)
            train_all_loss += loss.item() * target.size(0)
            num_correct = (distance == target).sum()
            accuracy += num_correct.item()
            running_acc = num_correct.item()
            train_all_acc += num_correct.item()
            if step % 100 == 0:
                print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch + 1, EPOCH, running_loss / (BATCH_SIZE * step),
                    running_acc / (BATCH_SIZE )))
        print('Finish college{} epoch{}, Loss: {:.6f}, Acc: {:.6f}'.format(
            col, epoch + 1, running_loss / (len(train_dataset_list[col-1])), accuracy / (len(
                train_dataset_list[col-1]))))

    print('Epoch{} Train Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, train_all_loss / train_dataset_len,
                                                          train_all_acc / train_dataset_len))

    # 测试
    model.eval()
    with torch.no_grad():
        all_acc = 0
        all_loss = 0
        for col2 in range(1, 20):
            eval_loss = 0
            eval_acc = 0
            for data in test_loader_list[col2 - 1]:
                cardData1, cardData2, target = data

                cardData1 = cardData1.type(torch.FloatTensor)
                cardData2 = cardData2.type(torch.FloatTensor)
                target = target.type(torch.FloatTensor)
                cardData1 = cardData1.cuda()
                cardData2 = cardData2.cuda()
                target = target.cuda()
                out1 = model(cardData1, col2)
                out2 = model(cardData2, col2)
                distance = out2 - out1
                distance = distance.reshape(-1)
                loss = torch.mean(torch.max(torch.Tensor([0]).cuda(), (target * distance) + 1))
                eval_loss += loss.item() * target.size(0)
                all_loss += loss.item() * target.size(0)
                for i in range(distance.shape[0]):
                    if distance[i] >= 0:
                        distance[i] = 1
                    else:
                        distance[i] = -1

                num_correct = (distance == target).sum()
                eval_acc += num_correct.item()
                all_acc += num_correct.item()

            print('Epoch{} college{} Test Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, col2, eval_loss / (len(
                test_dataset_list[col2 - 1])), eval_acc / (len(test_dataset_list[col2 - 1]))))
        print('Epoch{} Test Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, all_loss / test_dataset_len,
                                                              all_acc / test_dataset_len))


torch.save(model.state_dict(), 'TRIPLE_STREAM_ATTN_NDCG.pkl')


