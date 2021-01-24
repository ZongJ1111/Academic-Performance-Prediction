# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import h5py
import torch.utils.data as Data
import numpy as np
import os

out_num = 1
FC_NUM = 3057

class ATTN_SPA(nn.Module):
    def __init__(self, c,attn_map_num,map_num_2):
        super(ATTN_SPA, self).__init__()
        self.attn_map_num = attn_map_num
        self.map_num_2 = map_num_2
        self.avgpool_1 = nn.AdaptiveAvgPool2d((365,1))
        self.avgpool_2 = nn.AdaptiveAvgPool2d((1,18))
        self.fc = nn.Sequential(
            nn.Linear(attn_map_num, round(attn_map_num / 4)),
            nn.ReLU(),
            nn.Linear(round(attn_map_num / 4), attn_map_num),
            nn.Sigmoid(),
        )
        nn.init.xavier_normal_(self.fc.children().__next__().weight)

    def forward(self, input,path):
        if path == 1 and self.map_num_2 > 1:
            x = torch.mean(input,dim=1,keepdim=True)
            x = self.avgpool_1(x)
            x = x.view(x.size(0), self.attn_map_num)
            x = self.fc(x)
            x = x.view(x.size(0), 1, self.attn_map_num, 1)
            output = torch.mul(x.expand_as(input), input)
            return output

        elif path == 2 and self.map_num_2 > 1:
            x = torch.mean(input,dim=1,keepdim=True)
            x = self.avgpool_2(x)
            x = x.view(x.size(0), self.attn_map_num )
            x = self.fc(x)
            x = x.view(x.size(0), 1, 1,self.attn_map_num)
            output = torch.mul(x.expand_as(input), input)
            return output

class ATTN_CNAL(nn.Module):
    def __init__(self, c):
        super(ATTN_CNAL, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(c, round(c / 4)),
            nn.ReLU(),
            nn.Linear(round(c / 4), c),
            nn.Sigmoid(),
        )
        nn.init.xavier_normal_(self.fc.children().__next__().weight)

    def forward(self, input):

        x = self.GAP(input)
        x = x.view(x.size(0), x.size(1))
        x = self.fc(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        output = x * input
        return output

class TCNN(nn.Module):
    def __init__(self):
        super(TBCNN,self).__init__()

        # path 1  follow 18 cov
        self.conv_A1 = nn.Sequential(
            nn.Conv2d(12,64,(1,3),1,(0,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )#365*18*64
        self.attn_A1 = ATTN_SPA(64, 18, 365)

        self.conv_A2 = nn.Sequential(
            nn.Conv2d(64,128,(1,3),1,(0,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )#365*18*128
        self.attn_A2 = ATTN_SPA(128, 18, 365)

        self.conv_A3 = nn.Sequential(
            nn.Conv2d(128,256,(1,3),1,(0,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )#365*18*256
        self.attn_A3 = ATTN_SPA(256, 18, 365)

        self.conv_A4 = nn.Sequential(
            nn.Conv2d(256,3,(1,18),1,0),
            nn.ReLU(),
        )#365*1*3


        # path 2 follow 365 cov
        self.conv_B1 = nn.Sequential(
            nn.Conv2d(12,64,(3,1),1,(1,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )#365*18*64
        self.attn_B1 = ATTN_SPA(64, 365, 18)

        self.conv_B2 = nn.Sequential(
            nn.Conv2d(64,128,(3,1),1,(1,0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )#365*18*128
        self.attn_B2 = ATTN_SPA(128, 365, 18)

        self.conv_B3 = nn.Sequential(
            nn.Conv2d(128,256,(3,1),1,(1,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )#365*18*256
        self.attn_B3 = ATTN_SPA(256, 365, 18)

        self.conv_B4 = nn.Sequential(
            nn.Conv2d(256,55,(365,1),1,0),
            nn.ReLU(),
        )#1*18*55

        # path 3 follow 12 cov
        self.conv_C1 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=60,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=12
            ),
            nn.BatchNorm2d(60),
            nn.ReLU(),
        )#365*18*60
        self.attn_C1 = ATTN_CNAL(60)

        self.conv_C2 = nn.Sequential(
            nn.Conv2d(
                in_channels=60,
                out_channels=120,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=12
            ),
            nn.BatchNorm2d(120),
            nn.ReLU(),
        )#365*18*120
        self.attn_C2 = ATTN_CNAL(120)

        self.conv_C3 = nn.Sequential(
            nn.Conv2d(
                in_channels=120,
                out_channels=240,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=12
            ),
            nn.BatchNorm2d(240),
            nn.ReLU(),
        )#365*18*240
        self.attn_C3 = ATTN_CNAL(240)

        self.conv_C4 = nn.Sequential(
            nn.Conv2d(
                in_channels=240,
                out_channels=12,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=12
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((9,9))
        )#9*9*12

        self.fc1 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc6 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc8 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc9 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc10 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc11 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc12 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc13 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc14 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc15 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc16 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc17 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc18 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        self.fc19 = nn.Sequential(
            nn.Linear(FC_NUM, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_num)
        )
        nn.init.xavier_normal_(self.conv_A1.children().__next__().weight)
        nn.init.xavier_normal_(self.conv_A2.children().__next__().weight)
        nn.init.xavier_normal_(self.conv_A3.children().__next__().weight)
        nn.init.xavier_normal_(self.conv_A4.children().__next__().weight)
        nn.init.xavier_normal_(self.conv_B1.children().__next__().weight)
        nn.init.xavier_normal_(self.conv_B2.children().__next__().weight)
        nn.init.xavier_normal_(self.conv_B3.children().__next__().weight)
        nn.init.xavier_normal_(self.conv_B4.children().__next__().weight)
        nn.init.xavier_normal_(self.conv_C1.children().__next__().weight)
        nn.init.xavier_normal_(self.conv_C2.children().__next__().weight)
        nn.init.xavier_normal_(self.conv_C3.children().__next__().weight)
        nn.init.xavier_normal_(self.conv_C4.children().__next__().weight)
        nn.init.xavier_normal_(self.fc1.children().__next__().weight)
        nn.init.xavier_normal_(self.fc2.children().__next__().weight)
        nn.init.xavier_normal_(self.fc3.children().__next__().weight)
        nn.init.xavier_normal_(self.fc4.children().__next__().weight)
        nn.init.xavier_normal_(self.fc5.children().__next__().weight)
        nn.init.xavier_normal_(self.fc6.children().__next__().weight)
        nn.init.xavier_normal_(self.fc7.children().__next__().weight)
        nn.init.xavier_normal_(self.fc8.children().__next__().weight)
        nn.init.xavier_normal_(self.fc9.children().__next__().weight)
        nn.init.xavier_normal_(self.fc10.children().__next__().weight)
        nn.init.xavier_normal_(self.fc11.children().__next__().weight)
        nn.init.xavier_normal_(self.fc12.children().__next__().weight)
        nn.init.xavier_normal_(self.fc13.children().__next__().weight)
        nn.init.xavier_normal_(self.fc14.children().__next__().weight)
        nn.init.xavier_normal_(self.fc15.children().__next__().weight)
        nn.init.xavier_normal_(self.fc16.children().__next__().weight)
        nn.init.xavier_normal_(self.fc17.children().__next__().weight)
        nn.init.xavier_normal_(self.fc18.children().__next__().weight)
        nn.init.xavier_normal_(self.fc19.children().__next__().weight)


    def forward(self, input,col):

        out_1 = self.conv_A1(input)
        out_1 = self.attn_A1(out_1,2)

        out_2 = self.conv_B1(input)
        out_2 = self.attn_B1(out_2,1)

        out_3 = self.conv_C1(input)
        out_3 = self.attn_C1(out_3)

        out_1 = self.conv_A2(out_1)
        out_1 = self.attn_A2(out_1,2)

        out_2 = self.conv_B2(out_2)
        out_2 = self.attn_B2(out_2, 1)

        out_3 = self.conv_C2(out_3)
        out_3 = self.attn_C2(out_3)

        out_1 = self.conv_A3(out_1)
        out_1 = self.attn_A3(out_1, 2)

        out_2 = self.conv_B3(out_2)
        out_2 = self.attn_B3(out_2, 1)

        out_3 = self.conv_C3(out_3)
        out_3 = self.attn_C3(out_3)

        out_1 = self.conv_A4(out_1)

        out_2 = self.conv_B4(out_2)

        out_3 = self.conv_C4(out_3)

        out_1 = out_1.view(out_1.size(0), -1)
        out_2 = out_2.view(out_2.size(0), -1)
        out_3 = out_3.view(out_3.size(0), -1)
        out_together_1 = torch.cat((out_1, out_2), 1)
        out_together = torch.cat((out_together_1, out_3), 1)

        if col == 1:
            f_out = self.fc1(out_together)
        elif col == 2:
            f_out = self.fc2(out_together)
        elif col == 3:
            f_out = self.fc3(out_together)
        elif col == 4:
            f_out = self.fc4(out_together)
        elif col == 5:
            f_out = self.fc5(out_together)
        elif col == 6:
            f_out = self.fc6(out_together)
        elif col == 7:
            f_out = self.fc7(out_together)
        elif col == 8:
            f_out = self.fc8(out_together)
        elif col == 9:
            f_out = self.fc9(out_together)
        elif col == 10:
            f_out = self.fc10(out_together)
        elif col == 11:
            f_out = self.fc11(out_together)
        elif col == 12:
            f_out = self.fc12(out_together)
        elif col == 13:
            f_out = self.fc13(out_together)
        elif col == 14:
            f_out = self.fc14(out_together)
        elif col == 15:
            f_out = self.fc15(out_together)
        elif col == 16:
            f_out = self.fc16(out_together)
        elif col == 17:
            f_out = self.fc17(out_together)
        elif col == 18:
            f_out = self.fc18(out_together)
        elif col == 19:
            f_out = self.fc19(out_together)

        return f_out
