import os, sys, time, json
import numpy as np
import time
import utils
from utils import recorder

from data_provider.data_provider import HSIDataLoader 
from trainer import get_trainer 
import evaluation
from utils import check_convention, config_path_prefix
from datetime import datetime
import random
import torch

DEFAULT_RES_SAVE_PATH_PREFIX = "./res_base/my new"


def train_by_param(param, aug):
    # 0. recorder reset
    recorder.reset()

    # 1. 数据生成
    dataloader = HSIDataLoader(param)
    train_loader, unlabel_loader, test_loader, train_test_loader, all_loader, testset = \
        dataloader.generate_torch_dataset()

    # 注入 all_loader 和 dataloader，供 Trainer 生成分类图
    param['_all_loader'] = all_loader
    param['_dataloader'] = dataloader

    # 2. 训练和测试
    trainer = get_trainer(param)
    trainer.train(train_loader, unlabel_loader, test_loader)

    start_eval_time = time.time()
    eval_res = trainer.final_eval(train_test_loader, aug)
    if param.get('_all_loader') and param.get('_dataloader'):
        trainer.save_classification_map(
            param['_all_loader'],
            param['_dataloader'],
            param['path_res']
    )
    end_eval_time = time.time()
    eval_time = end_eval_time - start_eval_time
    print("eval time is %s" % eval_time)
    recorder.record_time(eval_time)

    # 3. 记录所有信息
    recorder.record_param(param)
    recorder.record_eval(eval_res)

    # 清理不可序列化的字段
    param.pop("_push_event", None)
    param.pop("_job_id", None)
    param.pop("_all_loader", None)
    param.pop("_dataloader", None)

    recorder.to_file(param['path_res'])

    # 4. 生成分类图（npy + jpg）
    try:
        trainer.save_classification_map(
            all_loader, dataloader, param['path_res']
        )
    except Exception as e:
        print(f"[workflow] 分类图生成失败: {e}")

    return recorder


# ── 以下为命令行运行相关代码（保持不变）────────────────────────

include_path = [
    'pavia_transformer_noise.json',
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
        train_by_param(param, aug=0)
        print('model eval done of %s...' % uniq_name)


def result_file_exists(prefix, file_name_part):
    ll = os.listdir(prefix)
    for l in ll:
        if file_name_part in l:
            return True
    return False


noise_type_list_temp = ['jpeg']
noise_type_list_temp_clean = ['clean']


def run_serving_mode(json_str, train_sign='test'):
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    uniq_name = json_str.get('uniq_name', "")
    model_name = json_str.get('model_name', "") or uniq_name
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
            train_by_param(json_str, noise_type)
            print('model eval done of %s...' % uniq_name)
    if train_sign in ['test', 'tent', 'ctent']:
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
                train_by_param(json_str, aug)
                print('model eval done of %s...' % uniq_name)


def run_test_tent():
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    for name in include_path:
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
        for _ in range(1):
            for train_sign in ['ctent']:
                run_serving_mode(param, train_sign)


if __name__ == "__main__":
    run_test_tent()
