# -*- coding:utf-8 -*-
"""
@Time: 2022/10/28 17:57
@Author: Shuting Liu & Baochang Zhang
@IDE: PyCharm
@File: train_config.py
@Comment: #Enter some comments at here
"""
import logging
import argparse
from .other_utils import *
from torch.utils.data import DataLoader
from Dataset.Loader import Stroke_Dataset
# from Dataset.Loader_old import Stroke_Dataset_old


def define_args(name=None, batchsize=None, lr=None, n_epochs=None, decay_epoch=None, fold_id=None,
                dataset_type=None, dataset_aug=None, save_in_same=None, cuda=None, project=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='Baseline', help='save to project_name')
    parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--decay_epoch', type=int, default=5, help='epoch to start linearly decaying the learning rate to 0')

    parser.add_argument('--project', type=str, default='../runs/train', help='save to project/name')
    parser.add_argument('--Dataset', type=str, default='DEMDAS', help='the dataset name')
    parser.add_argument('--fold_id', type=int, default=1, help='set the fold id')

    parser.add_argument('--input_volume_nc', type=int, default=3, help='number of channels of input volume data')
    parser.add_argument('--input_node', type=int, default=131, help='number of channels of input volume data')
    parser.add_argument('--input_nodefeature_nc', type=int, default=18, help='number of channels of input volume data')
    parser.add_argument('--input_tabular_nc', type=int, default=20, help='number of channels of input volume data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')

    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--save_in_same', type=bool, default=False, help='load model and continue trainning')
    parser.add_argument('--dataset_aug', type=bool, default=True, help='augment dataset')
    opt = parser.parse_args()

    if name is not None:
        opt.name = name
    if batchsize is not None:
        opt.batchSize = batchsize
    if lr is not None:
        opt.lr = lr
    if n_epochs is not None:
        opt.n_epochs = n_epochs
    if decay_epoch is not None:
        opt.decay_epoch = decay_epoch
    if save_in_same is not None:
        opt.save_in_same = save_in_same
    if dataset_type is not None:
        opt.dataset_type = dataset_type
    if dataset_aug is not None:
        opt.dataset_aug = dataset_aug
    if cuda is not None:
        opt.cuda = cuda
    if fold_id is not None:
        opt.fold_id = fold_id
    if project is not None:
        opt.project = project

    proj_name = opt.name+'_fold'+str(opt.fold_id)
    if opt.save_in_same:
        opt.save_root = path_str(join(opt.project, proj_name))
    else:
        opt.save_root = increment_path(path_str(join(opt.project, proj_name)), exist_ok=False)

    opt.model_savepath = path_str(join(opt.save_root, 'weight'))
    opt.model_resumepath = path_str(join(opt.save_root, 'temp'))
    opt.recordtxt = path_str(join(opt.save_root, 'log.txt'))

    if save_in_same and os.path.exists(opt.recordtxt):
        os.remove(opt.recordtxt)

    maybe_makedirs(opt.model_savepath)
    maybe_makedirs(opt.model_resumepath)

    # for print something
    log = logging.getLogger(__name__)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    log.info(opt)

    if torch.cuda.is_available() and not opt.cuda:
        log.warning("You have a CUDA device, so you should probably run with --cuda")
    return opt, log

def define_inference_args(name=None,  fold_id=None, dataset_type=None, dataset_aug=None, cuda=None, project=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='Baseline', help='save to project_name')
    parser.add_argument('--project', type=str, default='../runs/train', help='save to project/name')
    parser.add_argument('--Dataset', type=str, default='DEMDAS', help='the dataset name')
    parser.add_argument('--weight', type=str, default='best.pt', help='the dataset name')
    parser.add_argument('--fold_id', type=int, default=1, help='set the fold id')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')

    parser.add_argument('--input_volume_nc', type=int, default=3, help='number of channels of input volume data')
    parser.add_argument('--input_node', type=int, default=131, help='number of channels of input volume data')
    parser.add_argument('--input_nodefeature_nc', type=int, default=18, help='number of channels of input volume data')
    parser.add_argument('--input_tabular_nc', type=int, default=20, help='number of channels of input volume data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')

    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    if name is not None:
        opt.name = name
    if dataset_type is not None:
        opt.dataset_type = dataset_type
    if dataset_aug is not None:
        opt.dataset_aug = dataset_aug
    if cuda is not None:
        opt.cuda = cuda
    if fold_id is not None:
        opt.fold_id = fold_id
    if project is not None:
        opt.project = project

    proj_name = opt.name+'_fold'+str(opt.fold_id)

    opt.save_root = path_str(join(opt.project, proj_name))
    opt.model_loadpath = path_str(join(path_str(join(opt.save_root, 'weight')),opt.weight))

    if not os.path.exists(opt.model_loadpath):
        print('The load path of this model  is not exist')
        exit(0)

    # for print something
    log = logging.getLogger(__name__)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    log.info(opt)

    if torch.cuda.is_available() and not opt.cuda:
        log.warning("You have a CUDA device, so you should probably run with --cuda")
    return opt, log

def freezing_op(Model, freeze_layer=[]):
    for k, v in Model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze_layer):
            print('freezing %s' % k)
            v.requires_grad = False
    return Model


def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def GetDataloader(opt,ignore_nodes=None, norm=False):

    train_list = '/home/shuting/LProjects/Stroke_Radiomics/Dataset/Data/strokenormdataset/' + opt.Dataset + '_train_folder'+str(opt.fold_id)+'.txt'
    test_list = '/home/shuting/LProjects/Stroke_Radiomics/Dataset/Data/strokenormdataset/' + opt.Dataset + '_test_folder'+str(opt.fold_id)+'.txt'

    # train_list = '/u/home/lius/Project/Stroke_Radiomics/Dataset/Data/strokenormdataset/DEMDAS_train_old.txt'
    # test_list = '/u/home/lius/Project/Stroke_Radiomics/Dataset/Data/strokenormdataset/DEMDAS_test_old.txt'

    traindataloader = DataLoader(Stroke_Dataset(datasetpath_list=train_list, augment=opt.dataset_aug, is_shuffle=True,
                                                ignore_nodes=ignore_nodes, norm=norm, type=opt.dataset_type),
                                 batch_size=opt.batchSize, shuffle=True, num_workers=8, drop_last=True)
    testdataloader = DataLoader(Stroke_Dataset(datasetpath_list=test_list, augment=False, is_shuffle=False,
                                               ignore_nodes=ignore_nodes, norm=norm, type=opt.dataset_type),
                                batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    return traindataloader, testdataloader

'''
def GetDataloader_old(opt):

    # set Dataset
    # train_list = ['./Dataset/Data/DEDEMAS_train.txt', './Dataset/Data/DEMDAS_train.txt']
    # test_list = ['./Dataset/Data/DEDEMAS_test.txt', './Dataset/Data/DEMDAS_test.txt']

    train_list = ['/u/home/lius/Project/Stroke_Radiomics/Dataset/Data/DEMDAS_train.txt']
    test_list = ['/u/home/lius/Project/Stroke_Radiomics/Dataset/Data/DEMDAS_test.txt']

    traindataloader = DataLoader(Stroke_Dataset_old(datasetpath_list=train_list, augment=opt.dataset_aug, is_shuffle=True, type=opt.dataset_type),
                                 batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)

    testdataloader = DataLoader(Stroke_Dataset_old(datasetpath_list=test_list, augment=False, is_shuffle=False, type=opt.dataset_type),
                                  batch_size=1, shuffle=False, num_workers=opt.n_cpu, drop_last=False)

    return traindataloader, testdataloader
'''




def get_batch(batch, device, dataset_type):

    if dataset_type == 'tabular' or dataset_type == 'all':
        input_tabular = batch['clinical_Tabular'].to(device=device, dtype=torch.float32)
        input_missing_vector = batch['missing_vector'].to(device=device, dtype=torch.float32)
        input_adj_matrix = batch['adj_matrix'].to(device=device, dtype=torch.float32)
        input_node_feature = batch['nood_features'].to(device=device, dtype=torch.float32)
    else:
        input_tabular = None
        input_missing_vector = None
        input_adj_matrix = None
        input_node_feature = None

    if dataset_type == 'volume' or dataset_type == 'all':
        input_DTI_MD = batch['DTI_MD'].to(device=device, dtype=torch.float32)
        input_DTI_Trace = batch['DTI_Trace'].to(device=device, dtype=torch.float32)
        input_Flair = batch['Flair'].to(device=device, dtype=torch.float32)
        input_wmh_gt = batch['WMH_mask'].to(device=device, dtype=torch.float32)
        input_infarct_gt = batch['infarct_mask'].to(device=device, dtype=torch.float32)
    else:
        input_DTI_MD = None
        input_DTI_Trace = None
        input_Flair = None
        input_wmh_gt = None
        input_infarct_gt = None

    input_label = batch['binary_label'].to(device=device, dtype=torch.float32)

    if dataset_type == 'all':
        return input_label, input_tabular, input_missing_vector, input_adj_matrix, input_node_feature,\
               input_DTI_MD, input_DTI_Trace, input_Flair, input_wmh_gt, input_infarct_gt
    if dataset_type == 'tabular':
        return input_label, input_tabular, input_missing_vector, input_adj_matrix, input_node_feature
    if dataset_type == 'volume':
        return input_DTI_MD, input_DTI_Trace, input_Flair, input_wmh_gt, input_infarct_gt


def save_log(opt, tx, training=True):
    if training:
        with open(opt.recordtxt, 'a') as f:
            title = ('%10s' * 9) % ('Epoch', 'GPU_cost', 'Train_Los', ' Train_Bac', 'Train_Acc', 'Train_AUC', 'Train_Pre', 'Train_Sen', 'Train_Spe')
            f.write(title + '\n')
            f.write(tx + '\n')
    else:
        with open(opt.recordtxt, 'a') as f:
            title = ('%10s' * 6) % ('Test_BAcc', 'Test_Acc', 'Test_AUC', 'Test_pre', 'Test_sen', 'Test_spe')
            f.write(title + '\n')
            f.write(tx + '\n')
            f.write('\n')


def save_model(opt, Model, epoch, best_val_score, balanced_acc_val,start=10):
    Model_checkpoints = {"model": Model.state_dict()}
    torch.save(Model_checkpoints, os.path.join(opt.model_savepath, 'last.pt'))
    mark = False
    if best_val_score < balanced_acc_val and epoch >= start:
        best_val_score = balanced_acc_val
        mark = True
        Model_checkpoints = {"model": Model.state_dict()}
        torch.save(Model_checkpoints, os.path.join(opt.model_savepath, 'best.pt'))
    return best_val_score, mark



