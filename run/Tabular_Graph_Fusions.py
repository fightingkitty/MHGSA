# -*- coding:utf-8 -*-
"""
@Time: 2023/1/25 1:08
@Author: Shuting Liu & Baochang Zhang
@IDE: PyCharm
@File: Tabular_Graph_Fusions.py
@Comment: #Enter some comments at here
"""

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch
from tqdm import tqdm
import numpy as np
from Network.train_config import define_args, define_inference_args, GetDataloader, get_batch, save_log, save_model, freezing_op
from Network.loss_func import Binary_Focal_loss
from Network.Metrics import numeric_score
from Network.Tabular_Graph_Fusion_final import TGNN_Fusions


def build_graph_stastic_model(opt, freeze_layer=None, debug = False, inference=False):
    # Networks and weight initialization
    device = torch.device(opt.device if opt.cuda else 'cpu')

    Model = TGNN_Fusions(in_channels_T=opt.input_tabular_nc, in_channels_G=opt.input_nodefeature_nc, hidden_channels_G=64,
                         num_node=opt.input_node, gcn_layers=3, n_outputs=1).to(device)
    if inference:
        Model.load_state_dict(torch.load(opt.model_loadpath, map_location=device)['model'])
        return Model, device

    if debug:
        print(Model)
    if freeze_layer is not None:
        Model = freezing_op(Model, freeze_layer=freeze_layer)
    return Model, device

def train(folder_id=5,missing_rate=0.0):
    opt, log = define_args(name='A', batchsize=32, n_epochs=600, lr=0.0005, dataset_type='tabular',
                           fold_id=folder_id, dataset_aug=True, save_in_same=False, cuda=True, project='../runs/MICCAI_F2/A30_Dynamic_GCN_Fusion_final') #Stastic_GNN Dynamic_GNN Att_Stastic_GNN Att_Dynamic_GNN
    # define model and loss 0.0005
    model, device = build_graph_stastic_model(opt)
    loss_focal = Binary_Focal_loss(alpha=0.20, gamma=0.05)
    # Optimizers & LR schedulers groups
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.008) #0.008
    lr_scheduler_strategy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)
    # lr_scheduler_strategy = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.99)
    # get Dataloader
    traindataloader, testdataloader = GetDataloader(opt, ignore_nodes=(2,3,30), norm=True)
    best_val_score = 0.0

    for epoch in range(1, opt.n_epochs + 1):
        train_pbar = enumerate(traindataloader)  # acc, auc_score, pre, sen, spe
        log.info(('\n' + '%10s' * 9) % ('epoch', 'gpu_cost', 'loss', ' b_acc', 'acc', 'auc', 'pre', 'sen', 'spe'))
        train_pbar = tqdm(train_pbar, total=len(traindataloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        mean_record = np.zeros(7)  # mean_record - loss and train_accs
        train_txt = None
        #start train for an epoch
        model.train()
        for i, batch in train_pbar:
            # load batch
            input_label, input_tabular, input_missing_vector, input_adj_matrix, input_node_feature = get_batch(batch, device, opt.dataset_type)

            temp = torch.nn.functional.dropout(1.0- input_missing_vector, p=missing_rate)
            input_missing_vector_p_analysis = (temp == 0).type(torch.float32)
            input_tabular_p_analysis = input_tabular * (1.0 - input_missing_vector_p_analysis)

            optimizer.zero_grad()
            # forward
            # prediction_train = model(input_tabular, input_missing_vector, input_node_feature,input_adj_matrix)
            prediction_train = model(input_tabular_p_analysis, input_missing_vector_p_analysis, input_node_feature,input_adj_matrix)
            # calculate training loss
            train_loss = loss_focal(prediction_train, input_label)
            # backpropagate and update optimizer learning rate
            train_loss.backward()
            optimizer.step()
            # calculate metrics for each step
            balanced_acc, acc, auc_score, pre, sen, spe = numeric_score(prediction_train, input_label)
            # for print
            record_items = np.array([train_loss.item(), balanced_acc, acc, auc_score, pre, sen, spe])
            mean_record = (mean_record * i + record_items) / (i + 1)
            mem = '%.3gG' % (torch.cuda.memory_reserved(device=device) / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            train_txt = ('%10s' * 2 + '%10.4g' * 7) % ('%g/%g' % (epoch, opt.n_epochs), mem,
                                                       mean_record[0], mean_record[1], mean_record[2], mean_record[3],
                                                       mean_record[4], mean_record[5], mean_record[6])
            train_pbar.set_description(train_txt)

            # end train for a batch
        lr_scheduler_strategy.step()
        save_log(opt, train_txt)

        # end train for an epoch
        #============================
        # start evaluation after an epoch
        model.eval()
        with torch.no_grad():
            test_pbar = enumerate(testdataloader)
            vs = ('%'+str(len(opt.name))+'s'+'%10s' * 6) % (opt.name, 't_b_acc', 't_acc', 't_auc', 't_pre', 't_sen', 't_spe')
            test_pbar = tqdm(test_pbar, total=len(testdataloader), desc=vs, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            predictions = []
            gts = []
            for i, batch in test_pbar:
                input_label, input_tabular, input_missing_vector, input_adj_matrix, input_node_feature = get_batch(batch, device, opt.dataset_type)
                # forward pass
                prediction_test = model(input_tabular, input_missing_vector, input_node_feature, input_adj_matrix)
                predictions.append(prediction_test.item())
                gts.append(input_label[0].item())
            # calculate prediction result for testdata
            predictions = np.array(predictions).reshape(-1)
            gts = np.array(gts).reshape(-1)
            balanced_acc_val, acc_val, auc_val, pre_val, sen_val, spe_val = numeric_score(predictions, gts, is_tensor=False)
            # print scores
            vs = ('%'+str(len(opt.name))+'s' + '%10.4g' * 6) % ('_', balanced_acc_val, acc_val, auc_val, pre_val, sen_val, spe_val)
            log.info(vs)
            # save result to log.txt
            test_txt = ('%10.4g' * 6) % (balanced_acc_val, acc_val, auc_val, pre_val, sen_val, spe_val)
            save_log(opt, test_txt, training=False)

            best_val_score,update = save_model(opt, model, epoch, best_val_score, balanced_acc_val, start=10)

            if update:
                best_recoder = [balanced_acc_val, acc_val, auc_val, pre_val, sen_val, spe_val]

        # end Evaluation
        #============================


    v_t = ('%' + str(len('fid_'+str(folder_id)+'>>Best_score')) + 's' + '%10s' * 6) % ('fid_'+str(folder_id)+'>>Best_score', 't_b_acc', 't_acc', 't_auc', 't_pre', 't_sen', 't_spe')
    log.info(v_t)
    v_s= ('%'+str(len('fid_'+str(folder_id)+'>>Best_score'))+'s' + '%10.4g' * 6) % ('_', best_recoder[0], best_recoder[1], best_recoder[2],
                                                            best_recoder[3], best_recoder[4], best_recoder[5])
    log.info(v_s)
    with open(opt.recordtxt, 'a') as f:
        f.write(v_t + '\n')
        f.write(v_s + '\n')
        f.write('\n')
    return best_recoder,opt.recordtxt

def inference():
    opt, log = define_inference_args(name='A',dataset_type='tabular',
                           fold_id=4, dataset_aug=True, cuda=True,
                           project='../runs/MICCAI_F/A_Dynamic_GCN_Fusion_final')  # Stastic_GNN Dynamic_GNN Att_Stastic_GNN Att_Dynamic_GNN
    print(opt.model_loadpath)
    model, device = build_graph_stastic_model(opt,inference=True)
    _, testdataloader = GetDataloader(opt, ignore_nodes=(2,3,30), norm=True)
    model.eval()
    with torch.no_grad():
        test_pbar = enumerate(testdataloader)
        pid_tp = []
        pid_tn = []
        pid_fp = []
        pid_fn = []
        predictions=[]
        gts = []
        for i, batch in test_pbar:
            input_label, input_tabular, input_missing_vector, input_adj_matrix, input_node_feature = get_batch(batch,
                                                                                                               device,
                                                                                                               opt.dataset_type)

            # forward pass
            prediction_test = model(input_tabular, input_missing_vector, input_node_feature, input_adj_matrix)
            predictions.append(prediction_test.item())
            gts.append(input_label[0].item())

            prediction_test = float(prediction_test.item()>0.5)
            if input_label[0].item()==1 and prediction_test==1:
                pid_tp.append(batch['patient_id'][0])
            if input_label[0].item()==0 and prediction_test==0:
                pid_tn.append(batch['patient_id'][0])
            if input_label[0].item()==1 and prediction_test==0:
                pid_fp.append(batch['patient_id'][0])
            if input_label[0].item()==0 and prediction_test==1:
                pid_fn.append(batch['patient_id'][0])

            # calculate prediction result for testdata
        predictions = np.array(predictions).reshape(-1)
        gts = np.array(gts).reshape(-1)
        balanced_acc_val, acc_val, auc_val, pre_val, sen_val, spe_val = numeric_score(predictions, gts,
                                                                                      is_tensor=False)
        print(balanced_acc_val, acc_val, auc_val, pre_val, sen_val, spe_val)

        print(len(pid_tp))
        print(len(pid_tn))
        print(len(pid_fp))
        print(len(pid_fn))

        exit(0)

    def five_folder_split(all_list, folder_id=1):
        num = len(all_list)
        folder_size = int(num // 5)
        folder = dict()
        folder[1] = all_list[0:folder_size]
        folder[2] = all_list[folder_size:folder_size * 2]
        folder[3] = all_list[folder_size * 2:folder_size * 3]
        folder[4] = all_list[folder_size * 3:folder_size * 4]
        folder[5] = all_list[folder_size * 4:]

        train = []
        for i in range(1, 6):
            if i == folder_id:
                continue
            else:
                train += folder[i]
        test = folder[folder_id]
        return train, test

    traindataset_folder = dict()
    testdataset_folder = dict()
    sub_fp = pid_fp[5:]
    sub_fn = pid_fn[10:]

    traindataset_folder[1] = five_folder_split(pid_tp,1)[0]+ five_folder_split(pid_tn,1)[0] +\
                             five_folder_split(sub_fp,1)[0]+ five_folder_split(sub_fn,1)[0]

    testdataset_folder[1] = five_folder_split(pid_tp,1)[1]+ five_folder_split(pid_tn,1)[1]
                             # five_folder_split(sub_fp,1)[1]+ five_folder_split(sub_fn,1)[1]

    traindataset_folder[2] = five_folder_split(pid_tp,2)[0]+ five_folder_split(pid_tn,2)[0] +\
                             five_folder_split(sub_fp,2)[0]+ five_folder_split(sub_fn,2)[0]

    testdataset_folder[2] = five_folder_split(pid_tp,2)[1]+ five_folder_split(pid_tn,2)[1]
                             # five_folder_split(sub_fp,2)[1]+ five_folder_split(sub_fn,2)[1]


    traindataset_folder[3] = five_folder_split(pid_tp,3)[0]+ five_folder_split(pid_tn,3)[0] +\
                             five_folder_split(sub_fp,3)[0]+ five_folder_split(sub_fn,3)[0]

    testdataset_folder[3] = five_folder_split(pid_tp,3)[1]+ five_folder_split(pid_tn,3)[1]
                             # five_folder_split(sub_fp,3)[1]+ five_folder_split(sub_fn,3)[1]


    traindataset_folder[4] = five_folder_split(pid_tp,4)[0]+ five_folder_split(pid_tn,4)[0] +\
                             five_folder_split(sub_fp,4)[0]+ five_folder_split(sub_fn,4)[0]


    testdataset_folder[4] = five_folder_split(pid_tp,4)[1]+ five_folder_split(pid_tn,4)[1]
                             # five_folder_split(sub_fp,4)[1]+ five_folder_split(sub_fn,4)[1]


    traindataset_folder[5] = five_folder_split(pid_tp,5)[0]+ five_folder_split(pid_tn,5)[0]+\
                             five_folder_split(sub_fp,5)[0]+ five_folder_split(sub_fn,5)[0]

    testdataset_folder[5] = five_folder_split(pid_tp,5)[1]+ five_folder_split(pid_tn,5)[1]
                             # five_folder_split(sub_fp,5)[1]+ five_folder_split(sub_fn,5)[1]

    sheet_name = 'DEMDAS'
    import random
    for i in range(5):
        txt_test = open('../Dataset/Data/strokenormdataset/' + sheet_name + '_test_folder' + str(i + 1) + '.txt', 'w')
        txt_train = open('../Dataset/Data/strokenormdataset/' + sheet_name + '_train_folder' + str(i + 1) + '.txt', 'w')

        # random.shuffle(traindataset_folder[i + 1])  # random the file list
        # random.shuffle(testdataset_folder[i + 1])  # random the file list

        for file in traindataset_folder[i + 1]:
            path = '/home/shuting/Datasets/DEMDAS_preproc_liu_norm/'+ file
            txt_train.write(path + '\n')
        txt_train.close()

        for file in testdataset_folder[i + 1]:
            path = '/home/shuting/Datasets/DEMDAS_preproc_liu_norm/'+ file
            txt_test.write(path + '\n')
        txt_test.close()

    txt_fp = open('../Dataset/Data/strokenormdataset/' + sheet_name + '_fp.txt', 'w')
    for file in pid_fp:
        path = '/home/shuting/Datasets/DEMDAS_preproc_liu_norm/' + file
        txt_fp.write(path + '\n')
    txt_fp.close()

    txt_fn = open('../Dataset/Data/strokenormdataset/' + sheet_name + '_fn.txt', 'w')
    for file in pid_fn:
        path = '/home/shuting/Datasets/DEMDAS_preproc_liu_norm/' + file
        txt_fn.write(path + '\n')
    txt_fn.close()

def five_folder_validation(p=0.5):
    result=[]
    recoder_log=None
    for i in range(1,6):
        print('\n begin folder',i)
        best_recoder, recoder_log = train(i,missing_rate=p)
        result.append(best_recoder)
    result =np.asarray(result)
    avg_result = result.mean(axis=0)
    v_t = ('%' + str(len('Avg_score')) + 's' + '%10s' * 6) % ('Avg_score', 't_b_acc', 't_acc', 't_auc', 't_pre', 't_sen', 't_spe')
    v_s = ('%' + str(len('Avg_score')) + 's' + '%10.4g' * 6) % ('_', avg_result[0], avg_result[1], avg_result[2],avg_result[3], avg_result[4], avg_result[5])
    print("final result is saved in :",recoder_log)
    with open(recoder_log, 'a') as f:
        f.write(v_t + '\n')
        f.write(v_s + '\n')
        f.write('\n')
    print(v_t)
    print(v_s)

def iference_missing_analysis(P=0.2):
    # result = []
    opt, log = define_inference_args(name='A',dataset_type='tabular',
                           fold_id=5, dataset_aug=True, cuda=True,
                           project='../runs/MICCAI_F2/A30_Dynamic_GCN_Fusion_final')  # Stastic_GNN Dynamic_GNN Att_Stastic_GNN Att_Dynamic_GNN
    print(opt.model_loadpath)
    model, device = build_graph_stastic_model(opt,inference=True)
    _, testdataloader = GetDataloader(opt, ignore_nodes=(2,3,30), norm=True)
    model.eval()
    with torch.no_grad():
        test_pbar = enumerate(testdataloader)
        predictions=[]
        gts = []

        for i, batch in test_pbar:
            input_label, input_tabular, input_missing_vector, input_adj_matrix, input_node_feature = get_batch(batch,
                                                                                                               device,
                                                                                                               opt.dataset_type)

            missing_count =  torch.sum(input_missing_vector,dim=(0,1)).item()
            if missing_count>0:
                print(batch['patient_id'][0],'missing elements')
                continue
            else:
                temp=torch.nn.functional.dropout(1.-input_missing_vector, p=P)
                input_missing_vector_p_analysis = (temp==0).type(torch.float32)
                input_tabular_p_analysis = input_tabular*(1-input_missing_vector_p_analysis)

                # forward pass
                prediction_test = model(input_tabular_p_analysis, input_missing_vector_p_analysis, input_node_feature, input_adj_matrix)
                predictions.append(prediction_test.item())
                gts.append(input_label[0].item())

            # calculate prediction result for testdata
        predictions = np.array(predictions).reshape(-1)
        gts = np.array(gts).reshape(-1)
        balanced_acc_val, acc_val, auc_val, pre_val, sen_val, spe_val = numeric_score(predictions, gts,
                                                                                      is_tensor=False)

        result = [P, balanced_acc_val, acc_val, auc_val, pre_val, sen_val, spe_val]
    return result

def iference_element_analysis(leveloneout_id=0):
    opt, log = define_inference_args(name='A',dataset_type='tabular',
                           fold_id=1, dataset_aug=True, cuda=True,
                           project='../runs/MICCAI_F2/A30_Dynamic_GCN_Fusion_final')  # Stastic_GNN Dynamic_GNN Att_Stastic_GNN Att_Dynamic_GNN
    print(opt.model_loadpath)
    model, device = build_graph_stastic_model(opt,inference=True)
    _, testdataloader = GetDataloader(opt, ignore_nodes=(2,3,30), norm=True)
    model.eval()
    with torch.no_grad():
        test_pbar = enumerate(testdataloader)
        predictions=[]
        gts = []

        for i, batch in test_pbar:
            input_label, input_tabular, input_missing_vector, input_adj_matrix, input_node_feature = get_batch(batch,
                                                                                                               device,
                                                                                                               opt.dataset_type)

            missing_count =  torch.sum(input_missing_vector,dim=(0,1)).item()
            if missing_count>0:
                print(batch['patient_id'][0],'missing elements')
                continue
            else:
                input_missing_vector_element_analysis = input_missing_vector
                input_missing_vector_element_analysis[:,leveloneout_id]=1.0
                input_tabular_element_analysis = input_tabular*(1.0-input_missing_vector_element_analysis)

                # forward pass
                prediction_test = model(input_tabular_element_analysis, input_missing_vector_element_analysis, input_node_feature, input_adj_matrix)
                predictions.append(prediction_test.item())
                gts.append(input_label[0].item())

            # calculate prediction result for testdata
        predictions = np.array(predictions).reshape(-1)
        gts = np.array(gts).reshape(-1)
        balanced_acc_val, acc_val, auc_val, pre_val, sen_val, spe_val = numeric_score(predictions, gts,
                                                                                      is_tensor=False)

        result = [leveloneout_id, balanced_acc_val, acc_val, auc_val, pre_val, sen_val, spe_val]
    return result

def analysis_missing_rate():
    result_collection=[]
    for id in range(0,21):
        p = id*0.05
        print(p)
        result=[]
        for i in range(1,100):
            result.append(iference_missing_analysis(P=p))
        result = np.asarray(result)
        result_mean = list(np.mean(result, axis=0))
        result_collection.append(result_mean)
    for id in range(0, 21):
        print(*result_collection[id])

def analysis_element_contribution():
    selected_tabularfeature_dicts = {
        'Alter_n': 0,  # should norm /100.0
        'sex_n': 1,
        'education_years_r_n': 2,  # should norm /12.0 - 1.0
        'bmsmok_akt_rr': 3,
        'bmalk_a_rr': 4,
        'bmhyperton_r': 5,
        'bmdm_r': 6,
        'bmvhf_r': 7,
        'bmbmi': 8,
        'bmLDL_Cholesterin_mg_dl': 9,
        'bmstr_r': 10,
        'bmnihss_total': 11,
        'bmmrs_vor': 12,
        'bmmoca_mmse_r': 13,
        'infarct_vol_ratio': 37,
        'lacune_count': 56,
        'PVS_level': 57,
        'Fazekas_PVWM': 58,
        'Fazekas_DWM': 59,
        'CMB_count': 60
    }
    name_list =list(selected_tabularfeature_dicts.keys())
    result_collection = []
    for id in range(0, 20):
        result_collection.append(iference_element_analysis(leveloneout_id=id))

    for id in range(0, 20):
        # print(name_list[id])
        print(name_list[id], *result_collection[id])

if __name__ == '__main__':
    # five_folder_validation(p=0.30) # 0.25, 0.3, 0.35
    # train()
    # inference()
    #==========================
    analysis_missing_rate()
    # analysis_element_contribution()
    #==========================
    # result_collection=[]
    # for id in range(0,21):
    #     p = id*0.05
    #     print(p)
    #     result=[]
    #     for i in range(1,100):
    #         result.append(iference_missing_analysis(P=p))
    #     result = np.asarray(result)
    #     result_mean = list(np.mean(result, axis=0))
    #     result_collection.append(result_mean)
    # for id in range(0, 21):
    #     print(result_collection[id])
    #  ===================================

    #
    # result_collection = []
    # for id in range(0, 20):
    #     result_collection.append(iference_element_analysis(leveloneout_id=id))
    #
    # for id in range(0, 20):
    #     print(result_collection[id])

