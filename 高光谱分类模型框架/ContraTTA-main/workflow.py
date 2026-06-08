import os, sys, time, json
import numpy as np
import time
import utils
from utils import recorder

from data_provider.data_provider import HSIDataLoader 
from trainer import get_trainer 
import evaluation
from utils import check_convention, config_path_prefix
import argparse
# import plot
from datetime import datetime

import random
import torch

DEFAULT_RES_SAVE_PATH_PREFIX = "./res_base/my new"
# def set_seed(seed):
#     # 设置Python的随机种子
#     random.seed(seed)
    
#     # 设置NumPy的随机种子
#     np.random.seed(seed)
    
#     # 设置PyTorch的随机种子
#     torch.manual_seed(seed)  # CPU
#     torch.cuda.manual_seed(seed)  # 单GPU
#     # torch.cuda.manual_seed_all(seed)  # 多GPU
    
#     # 设置cudnn的相关配置
#     torch.backends.cudnn.deterministic = True  # 强制确定性操作
#     torch.backends.cudnn.benchmark = False 
def train_by_param(param,aug):
    #0. recorder reset防止污染数据
    # set_seed(42) #42
    recorder.reset()
    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    train_loader, unlabel_loader, test_loader, train_test_loader, all_loader,testset = dataloader.generate_torch_dataset()

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(train_loader, unlabel_loader, test_loader)
    # trainer.calc_cmd_statistics(train_loader)
    start_eval_time = time.time()
    # eval_res = trainer.final_eval(test_loader)  # change eval data
    eval_res = trainer.final_eval(train_test_loader,aug)  # change eval data
    end_eval_time = time.time()
    eval_time = end_eval_time - start_eval_time
    print("eval time is %s" % eval_time) 
    recorder.record_time(eval_time)
    # pred_matrix = dataloader.reconstruct_pred(pred_all)


    #3. record all information
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    # recorder.record_pred(pred_matrix) 

    recorder.to_file(param['path_res'])

    #4. plot
    rawdata, TR, TE = dataloader.get_data()
    # plot.plot_all(pred_matrix, TR, TE, param['path_pic'])
    # plot.plot_labels(TR, TE, param['path_pic'])
    # plot.plot_img(rawdata, param['data']['data_sign'], param['path_pic'])

    return recorder


include_path = [
    # 'indian_transformer.json',
    # 'paviaU_transformer.json',
    # 'WH_transformer.json',
    # 'LK_transformer.json',
    # 'WHLK_transformer_noise.json',
    # 'WHLK_ssftt_0.json',
    # 'WHLK_ssftt_1.json',
    # 'WHLK_ssftt_2.json',
    # 'WHLK_ssftt_3.json',
    # 'WHLK_ssftt_4.json',
    # 'WHLK_transformer_noise_COTTA.json',
    # 'WHLK_transformer_noise_DELTA.json',
    # 'WHLK_transformer_noise_0.json',
    # 'WHLK_transformer_noise_2.json',
    # 'WHLK_transformer_noise_3.json',
    # 'WHLK_transformer_noise_4.json',
    # 'WHLK_transformer_noise_1.json',
    # 'WHLK_transformer_noise_2.json',
    # 'WHLK_transformer_noise_3.json',
    # 'WHLK_transformer_noise_4.json',
    # 'salinas_transformer.jstent-paviaU_fog0.125_10_epochon',
    # 'WHLK_SSRN.json',
    # 'WHLK_CASST.json',
    # 'pavia_CASST.json',
    # 'pavia_massformer.json',
    # 'LK_HCA.json',
    # 'LK_CNN.json',
    # 'PU_CNN.json',

    # 'indian_transformer_noise.json',
    # 'indian_SSRN.json',
    # 'indian_ssftt.json',

    'pavia_transformer_noise.json',
    # 'pavia_transformer_noise_1.json',
    # 'pavia_transformer_CNN.json',
    # 'pavia_SSRN.json',
    # 'pavia_ssftt.json',

    # 'WH_transformer_noise.json',
    # 'WH_transformer_CNN.json',
    # 'WH_SSRN.json',
]


def run_all():
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    for name in include_path:
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
        uniq_name = param.get('uniq_name', name)
        path_model_save = "%s/%s" % (utils.model_save_path_prefix, uniq_name)
        param['path_model_save'] = path_model_save 
        print('start to train %s...' % uniq_name)

        now = datetime.now()
        time_stamp = now.strftime("%m%d%H%M")
        uniq_model_id = '%s_%s' % (uniq_name, time_stamp)
        path = '%s/%s' % (save_path_prefix, uniq_model_id) 
        path_pic = '%s/%s.png' % (save_path_prefix, uniq_model_id) 
        param['path_res'] = path
        param['path_pic'] = path_pic
        train_by_param(param)
        print('model eval done of %s...' % uniq_name)

def result_file_exists(prefix, file_name_part):
    ll = os.listdir(prefix)
    for l in ll:
        if file_name_part in l:
            return True
    return False


def run_one_multi_times(json_str, ori_uniq_name):
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    times = 1
    for i in range(times): 
        uniq_name = '%s_%s' % (ori_uniq_name, i)
        if result_file_exists(DEFAULT_RES_SAVE_PATH_PREFIX, uniq_name):
            print('%s has been run. skip...' % uniq_name)
            continue
        path = '%s/%s' % (save_path_prefix, uniq_name) 
        json_str['path_res'] = path
        print('start to train %s...' % uniq_name)
        train_by_param(json_str)
        print(json_str)
        print('model eval done of %s...' % uniq_name)


# noise_type_list_temp = ['salt_pepper','stripes','deadlines']
# noise_type_list_temp = ['thick_fog']
noise_type_list_temp = ['jpeg']

# noise_type_list_temp = ['jpeg', 'zmguass', 'additive', 'poisson', 'salt_pepper', 'stripes', 'deadlines', 'kernal', 'thick_fog'] 

noise_type_list_temp_clean = ['clean']

# noise_type_list_temp = ['salt_pepper', 'kernal', 'thick_fog']
# noise_type_list_temp = ['additive', 'salt_pepper', 'kernal', 'thick_fog']
# noise_type_list_temp = ['salt_pepper', 'kernal', 'thin_fog', 'thick_fog']
# noise_type_list_temp = [ 'kernal', 'thick_fog', 'poisson',  'stripes', 'deadlines']
# noise_type_list_temp = ['thick_fog']

def run_serving_mode(json_str, train_sign='test'): # serving_type = 'test' or 'tent'
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    uniq_name = json_str.get('uniq_name', "")
    model_name = json_str.get('model_name', "")
    if model_name == "":
        model_name = uniq_name
    path_model_save = "%s/%s" % (utils.model_save_path_prefix, model_name)
    json_str['path_model_save'] = path_model_save 
    print('start to train %s...' % uniq_name)
    if train_sign == 'train':
        for noise_type in noise_type_list_temp_clean:
            json_str['data']['noise_type'] = noise_type
            json_str['train_sign'] = train_sign 
            now = datetime.now()
            time_stamp = now.strftime("%m%d%H%M")
            uniq_model_id = '%s_%s_%s_%s' % (uniq_name, train_sign, noise_type, time_stamp)
            path = '%s/%s' % (save_path_prefix, uniq_model_id) 
            path_pic = '%s/%s.png' % (save_path_prefix, uniq_model_id) 
            json_str['path_res'] = path
            json_str['path_pic'] = path_pic
            train_by_param(json_str,noise_type)
            print('model eval done of %s...' % uniq_name)
    if train_sign == 'test' or train_sign == 'tent' or train_sign == 'ctent':
        for noise_type in noise_type_list_temp:
            for aug in [0]:
                json_str['data']['noise_type'] = noise_type
                json_str['train_sign'] = train_sign 
                now = datetime.now()
                time_stamp = now.strftime("%m%d%H%M")
                uniq_model_id = '%s_%s_%s_%s' % (uniq_name, train_sign, noise_type, time_stamp)
                path = '%s/%s' % (save_path_prefix, uniq_model_id) 
                path_pic = '%s/%s.png' % (save_path_prefix, uniq_model_id)
                json_str['path_res'] = path
                json_str['path_pic'] = path_pic
                train_by_param(json_str,aug)
                print('model eval done of %s...' % uniq_name)

def run_test_tent():
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    for name in include_path:
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
            # set used_pca = true
            # param['data']['use_saved_pca'] = True # ??????
            # for train_sign in ['test', 'tent']:
            for _ in range(1):
                for train_sign in ['ctent']:
                # for train_sign in ['test']:
                    #for _ in range(5):
                    run_serving_mode(param, train_sign)
     

if __name__ == "__main__":
    # run_all()
    run_test_tent() 
