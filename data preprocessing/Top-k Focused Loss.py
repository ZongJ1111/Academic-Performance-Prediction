import numpy as np
import os
import h5py
import math

p1 = 'train_stuid_list_del1000.hdf5'
f1 = h5py.File(p1,'r')
p2 = 'train_score_couple_del1000.hdf5'
f2 = h5py.File(p2,'r')
p3 = 'train_stu_rank_list.hdf5'
f3 = h5py.File(p3,'r')
p4 = 'train_stu_rank_list_reverse.hdf5'
f4 = h5py.File(p4,'r')

f11 = h5py.File('train_score_couple_topk.hdf5', 'w')

for i in range(19):

    train_list = f1['college'+str(i+1)][:]
    train_couple = f2['college' + str(i + 1)]
    train_rank_list = f3['college' + str(i + 1)][:]
    train_rank_list_reverse = f4['college' + str(i + 1)][:]

    train_len = len(train_list)
    print(i + 1,train_len)
    if(train_len>200):
        train_top_num_10 = int(0.05*train_len)
    else:
        train_top_num_10 = 10

    RANK_10 = np.asfarray(range(train_len-train_top_num_10,train_len+1))

    RANK_10 = np.flipud(RANK_10)

    # train_rank_list_rev = np.flipud(train_rank_list)
    risk_list_10 = np.squeeze(train_rank_list_reverse[:train_top_num_10,:1])
    # print(risk_list_10)
    list_reverse = np.squeeze(train_rank_list_reverse[:,:1])
    val_reverse = np.flipud(np.asfarray(range(1,train_len+1)))

    min_loc = 0
    max_loc = train_len-1
    max_DCG = (max_loc-min_loc)*(1/math.log2(min_loc+2) - 1/math.log2(max_loc+2))

    risk_couple_10 = [[0 for i in range(6)] for i in range(len(train_couple))]

    for j in range(len(train_couple)):
        risk_couple_10[j][0] = train_couple[j][0]
        risk_couple_10[j][1] = train_couple[j][1]
        risk_couple_10[j][2] = train_couple[j][2]
        risk_couple_10[j][3] = train_couple[j][3]
        if train_couple[j][4] == 1:
            risk_couple_10[j][4] = -1
        else:
            risk_couple_10[j][4] = 1

        NDCG_10 = 0

        sort_rank_10 = np.asfarray(range(train_len-train_top_num_10,train_len+1))

        sort_rank_10 = np.flipud(sort_rank_10)

        stu_id1 = train_couple[j][0]
        stu_id2 = train_couple[j][2]

        #NDCG@10 -------------------------------------------------------------------------------------------
        # stu1 and stu2 both in top10
        if stu_id1 in risk_list_10 and stu_id2 in risk_list_10:
            loc_1 = np.squeeze(np.where(risk_list_10==stu_id1)[0])
            loc_2 = np.squeeze(np.where(risk_list_10==stu_id2)[0])
            Rel_1 = sort_rank_10[loc_1]
            Rel_2 = sort_rank_10[loc_2]

            # print(loc_1, loc_2,Rel_1,Rel_2)
            DCG_10 = (Rel_1-Rel_2)*(1/math.log2(loc_1+2) - 1/math.log2(loc_2+2))

            NDCG_10 = DCG_10/max_DCG
            # print('1-1',DCG_10, NDCG_10)

        # stu1 in top10
        elif stu_id1 in risk_list_10 and stu_id2 not in risk_list_10:
            loc_1 = np.squeeze(np.where(risk_list_10==stu_id1)[0])
            loc_2 = np.squeeze(np.where(list_reverse==stu_id2)[0])

            Rel_1 = sort_rank_10[loc_1]
            Rel_2 = val_reverse[loc_2]

            DCG_10 = (Rel_1-Rel_2)*(1/math.log2(loc_1+2) - 1/math.log2(loc_2+2))
            NDCG_10 = DCG_10/max_DCG
            print('1-2',DCG_10,NDCG_10)

        # stu2 in top10
        elif stu_id1 not in risk_list_10 and stu_id2 in risk_list_10:
            loc_1 = np.squeeze(np.where(list_reverse == stu_id1)[0])
            loc_2 = np.squeeze(np.where(risk_list_10==stu_id2)[0])
            Rel_1 = val_reverse[loc_1]
            Rel_2 = sort_rank_10[loc_2]
            # print(loc_1, loc_2, Rel_1, Rel_2)
            DCG_10 = (Rel_1-Rel_2)*(1/math.log2(loc_1+2) - 1/math.log2(loc_2+2))

            NDCG_20 = DCG_10/max_DCG
            print('1-3',DCG_10,NDCG_20)

        # stu1 and stu2 both not in top10
        elif stu_id1 not in risk_list_10 and stu_id2 not in risk_list_10:
            NDCG_10 = 0.01
            # print('1-4',NDCG_10)

        risk_couple_10[j][5] = NDCG_10

    risk_couple_10 = np.array(risk_couple_10)

    f11.create_dataset("college" + str(i + 1), data=risk_couple_10)

f11.close()





