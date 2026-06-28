import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from operator import truediv
import time, json
import os, sys
import joblib


path_preprocess_model_save = './preprocess_model_save'
if os.path.exists(path_preprocess_model_save) == False:
    os.makedirs(path_preprocess_model_save)

""" Training dataset"""
class DataSetIter(torch.utils.data.Dataset):
    def __init__(self, _base_img, _base_labels, _index2pos, _margin, _patch_size, _append_dim, random_rotate=False) -> None:
        self.base_img = _base_img
        self.base_labels = _base_labels
        self.index2pos = _index2pos
        self.size = len(_index2pos)

        self.margin = _margin
        self.patch_size = _patch_size
        self.append_dim = _append_dim

        self.random_rotate = random_rotate
    
    def __getitem__(self, index):
        start_x, start_y = self.index2pos[index]
        patch = self.base_img[start_x:start_x+2*self.margin+1 , start_y:start_y+2*self.margin+1,:]
        if self.random_rotate:
            temp = patch
            for i in range(np.random.randint(0,4)):
                temp = np.transpose(temp, (1, 0, 2))
                temp = np.flipud(temp)
            patch = temp
        if self.append_dim:
            patch = np.expand_dims(patch, 0)
            patch = patch.transpose((0,3,1,2))
        else:
            patch = patch.transpose((2, 0, 1))
        label = self.base_labels[start_x, start_y] - 1
        return torch.FloatTensor(patch.copy()), torch.LongTensor(label.reshape(-1))[0], start_x, start_y

    def __len__(self):
        return self.size

    def dump(self, dump_path):
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        np.save("%s/base_img" % dump_path, self.base_img)
        np.save('%s/base_labels' % dump_path, self.base_labels)
        meta = {
            'index2pos': self.index2pos,
            'margin': self.margin,
            'patch_size': self.patch_size,
            'append_dim': self.append_dim
        }
        ss = json.dumps(meta)
        with open('%s/meta' % dump_path, 'w') as fout:
            fout.write(ss)
            fout.flush()

    @staticmethod 
    def load(dump_path):
        base_img = np.load("%s/base_img.npy" % dump_path)
        base_labels = np.load("%s/base_labels.npy" % dump_path)
        with open('%s/meta' % dump_path, 'r') as fin:
            js = json.loads(fin.read())
            index2pos_temp = js['index2pos']
            margin = js['margin']
            patch_size = js['patch_size']
            append_dim = js['append_dim']
            index2pos = {int(k):v for k,v in index2pos_temp.items()}
        return DataSetIter(base_img, base_labels, index2pos, margin, patch_size, append_dim)
    

class HSIDataLoader(object):
    def __init__(self, param) -> None:
        self.data_param = param['data']
        self.data_path_prefix = "./data"
        self.data = None
        self.labels = None
        self.TR = None
        self.TE = None

        self.data_path_prefix = self.data_param.get('data_path_prefix', './data')
        self.if_numpy = self.data_param.get('if_numpy', False)
        self.data_sign = self.data_param.get('data_sign', 'Indian')
        self.data_file = self.data_param.get('data_file', self.data_sign)
        self.patch_size = self.data_param.get('patch_size', 13)
        self.remove_zeros = self.data_param.get('remove_zeros', True)
        self.test_ratio = self.data_param.get('test_ratio', 0.9)
        self.batch_size = self.data_param.get('batch_size', 256)
        self.none_zero_num = self.data_param.get('none_zero_num', 0)
        self.spectracl_size = self.data_param.get("spectral_size", 0)
        self.append_dim = self.data_param.get("append_dim", False)
        self.use_norm = self.data_param.get("use_norm", True)
        self.norm_type = self.data_param.get("norm_type", 'max_min')

        self.random_rotate = self.data_param.get("random_rotate", False)
        self.noise_type = self.data_param.get('noise_type', 'clean')
        self.use_saved_pca = self.data_param.get('use_saved_pca', False)

        # ── 新增：前端上传的 split.mat 路径（优先级最高）────────────
        self.uploaded_file_path = self.data_param.get('uploaded_file_path', None)

    def load_raw_data(self):
        # ── 优先使用前端上传的 split.mat ──────────────────────────
        if self.uploaded_file_path and os.path.exists(self.uploaded_file_path):
            print(f"[data_provider] 使用上传文件: {self.uploaded_file_path}")
            try:
                all_data = sio.loadmat(self.uploaded_file_path)
            except Exception as e:
                raise ValueError(f"上传文件读取失败: {e}")

            # 兼容两种 key 格式
            # 格式1（标准 split.mat）: input / TR / TE
            # 格式2（旧格式）: data / TR / TE  或  HSI / TR / TE
            data = None
            for key in ['input', 'data', 'HSI']:
                if key in all_data:
                    data = all_data[key]
                    break
            if data is None:
                # 找第一个三维数组
                for k, v in all_data.items():
                    if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 3:
                        data = v
                        break

            if data is None:
                raise ValueError("上传文件中未找到 HSI 数据数组（需包含 input/data/HSI 字段）")

            TR = all_data.get('TR', None)
            TE = all_data.get('TE', None)

            if TR is None or TE is None:
                raise ValueError("上传文件中未找到 TR/TE 划分信息，请先执行数据划分")

            labels = TR + TE
            self.TR = TR
            self.TE = TE
            print(f"[data_provider] 上传数据加载成功: data={data.shape}, "
                  f"train={int((TR>0).sum())}, test={int((TE>0).sum())}")

            # 添加噪声（如果需要）
            if self.noise_type != 'clean':
                data_eval_path = '%s/noise_%s/%s.mat' % (
                    self.data_path_prefix, self.data_sign, self.noise_type)
                if os.path.exists(data_eval_path):
                    noise_data_all = sio.loadmat(data_eval_path)
                    data = noise_data_all['data']
                    print(f"[data_provider] 加载噪声数据: {self.noise_type}")

            return data, labels, TR, TE

        # ── 原有逻辑：从固定目录加载 ───────────────────────────────
        data, labels = None, None
        assert self.data_sign in ['Indian', 'Pavia', 'Houston', 'Salinas', 'Honghu', 'WH', 'Longkou'], \
            f"不支持的数据集: {self.data_sign}"
        data_path = '%s/%s/%s_split.mat' % (self.data_path_prefix, self.data_sign, self.data_file)
        all_data = sio.loadmat(data_path)
        data = all_data['input']
        TR = all_data['TR']
        TE = all_data['TE']
        labels = TR + TE

        if self.noise_type != 'clean':
            data_eval_path = '%s/noise_%s/%s.mat' % (
                self.data_path_prefix, self.data_sign, self.noise_type)
            noise_data_all = sio.loadmat(data_eval_path)
            noise_data = noise_data_all['data']
            noise_param = noise_data_all['params']
            print("load noise data %s_%s" % (noise_data_all['data_sign'], noise_data_all['noise_type']))
            print("load data %s" % (str(noise_param)))
            data = noise_data
            print("load raw data from %s" % data_eval_path)
        else:
            print("load raw data clean")
        return data, labels, TR, TE

    def get_data(self):
        return self.data, self.TR, self.TE

    def load_data(self):
        ori_data, labels, TR, TE = self.load_raw_data()
        return ori_data, labels, TR, TE

    def _padding(self, X, margin=2):
        w, h, c = X.shape
        new_x, new_h, new_c = w+margin*2, h+margin*2, c
        returnX = np.zeros((new_x, new_h, new_c))
        start_x, start_y = margin, margin
        returnX[start_x:start_x+w, start_y:start_y+h,:] = X
        return returnX
    
    def get_valid_num(self, y):
        tempy = y.reshape(-1)
        validy = tempy[tempy > 0]
        print('valid y shape is ', validy.shape)
        return validy.shape[0]

    def get_train_test_num(self, TR, TE):
        train_num, test_num = TR[TR>0].reshape(-1).size, TE[TE>0].reshape(-1).size
        print("train_num=%s, test_num=%s" % (train_num, test_num))
        return train_num, test_num

    def get_train_test_patches(self, X, y, TR, TE):
        h, w, c = X.shape
        windowSize = self.patch_size
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = self._padding(X, margin=margin)
        
        train_num, test_num = self.get_train_test_num(TR, TE)
        trainX_index2pos = {}
        testX_index2pos = {}
        train_testX_index2pos = {}
        all_index2pos = {}

        patchIndex = 0
        trainIndex = 0
        testIndex = 0
        train_test_Index = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                start_x, start_y = r-margin, c-margin
                tempy = y[start_x, start_y]
                temp_tr = TR[start_x, start_y] 
                temp_te = TE[start_x, start_y]
                if temp_tr > 0 and temp_te > 0:
                    print("here", temp_tr, temp_te, r, c)
                    raise Exception("data error, find sample in trainset as well as testset.")

                if temp_tr > 0:
                    trainX_index2pos[trainIndex] = [start_x, start_y]
                    train_testX_index2pos[train_test_Index] = [start_x, start_y]
                    trainIndex += 1
                    train_test_Index += 1
                elif temp_te > 0:
                    testX_index2pos[testIndex] = [start_x, start_y]
                    train_testX_index2pos[train_test_Index] = [start_x, start_y]
                    testIndex += 1
                    train_test_Index += 1
                all_index2pos[patchIndex] = [start_x, start_y]
                patchIndex = patchIndex + 1
        return zeroPaddedX, y, trainX_index2pos, testX_index2pos, train_testX_index2pos, all_index2pos, margin, self.patch_size 

    def applyPCA(self, X, numComponents=30):
        newX = np.reshape(X, (-1, X.shape[2]))
        path = '%s/%s_pca_%s.pkl' % (path_preprocess_model_save, self.data_file, numComponents)
        if self.use_saved_pca:
            pca = joblib.load(path)
        else:
            pca = PCA(n_components=numComponents, whiten=True)
            pca.fit(newX)
            joblib.dump(pca, path)
        newX = pca.transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        return newX

    def mean_var_norm(self, data):
        print("use mean_var norm...")
        h, w, c = data.shape
        data = data.reshape(h * w, c)
        data = StandardScaler().fit_transform(data)
        data = data.reshape(h, w, c)
        return data

    def data_preprocessing(self, data):
        if self.norm_type == 'max_min':
            norm_data = np.zeros(data.shape)
            for i in range(data.shape[2]):
                input_max = np.max(data[:,:,i])
                input_min = np.min(data[:,:,i])
                norm_data[:,:,i] = (data[:,:,i]-input_min)/(input_max-input_min)
        elif self.norm_type == 'mean_var':
            norm_data = self.mean_var_norm(data)
        else:
            norm_data = data 
        pca_num = self.data_param.get('pca', 0)
        if pca_num > 0:
            print('before pca')
            pca_data = self.applyPCA(norm_data, int(self.data_param['pca']))
            norm_data = pca_data
            print('after pca')
        if self.spectracl_size > 0:
            norm_data = norm_data[:,:,:self.spectracl_size]
        return norm_data

    def generate_numpy_dataset(self):
        self.data, self.labels, self.TR, self.TE = self.load_data()
        print('[load data done.] load data shape data=%s, label=%s' % (str(self.data.shape), str(self.labels.shape)))
        norm_data = self.data_preprocessing(self.data) 
        print('[data preprocessing done.] data shape data=%s, label=%s' % (str(norm_data.shape), str(self.labels.shape))) 
        h, w, c = norm_data.shape
        norm_data = norm_data.reshape((h*w,c))
        norm_label = self.labels.reshape((h*w))
        TR_reshape = self.TR.reshape((h*w))
        TE_reshape = self.TE.reshape((h*w))
        TrainX = norm_data[TR_reshape>0]
        TrainY = norm_label[TR_reshape>0]
        TestX = norm_data[TE_reshape>0]
        TestY = norm_label[TE_reshape>0]
        train_test_data = norm_data[norm_label>0]
        train_test_label = norm_label[norm_label>0]
        print('------[data] split data to train, test------')
        print("X_train shape : %s" % str(TrainX.shape))
        print("Y_train shape : %s" % str(TrainY.shape))
        print("X_test shape : %s" % str(TestX.shape))
        print("Y_test shape : %s" % str(TestY.shape))
        return TrainX, TrainY, TestX, TestY, norm_data

    def reconstruct_pred(self, y_pred):
        h, w = self.labels.shape
        return y_pred.reshape((h, w))

    def prepare_data(self):
        self.data, self.labels, self.TR, self.TE = self.load_data()
        print('[load data done.] load data shape data=%s, label=%s' % (str(self.data.shape), str(self.labels.shape)))
        norm_data = self.data_preprocessing(self.data) 
        print('[data preprocessing done.] data shape data=%s, label=%s' % (str(norm_data.shape), str(self.labels.shape)))

        base_img, labels, train_index2pos, test_index2pos, train_test_index2pos, all_index2pos, margin, patch_size \
              = self.get_train_test_patches(norm_data, self.labels, self.TR, self.TE)

        print('------[data] split data to train, test------')
        print("train len: %s" % len(train_index2pos))
        print("test len : %s" % len(test_index2pos))
        print('train_test len: %s' % len(train_test_index2pos))
        print("all len: %s" % len(all_index2pos))
        print("random rotate is %s" % self.random_rotate)

        trainset = DataSetIter(base_img, labels, train_index2pos, margin, patch_size, self.append_dim, random_rotate=self.random_rotate) 
        unlabelset = DataSetIter(base_img, labels, test_index2pos, margin, patch_size, self.append_dim, random_rotate=self.random_rotate)
        testset = DataSetIter(base_img, labels, test_index2pos, margin, patch_size, self.append_dim, random_rotate=self.random_rotate) 
        train_test_set = DataSetIter(base_img, labels, train_test_index2pos, margin, patch_size, self.append_dim, random_rotate=self.random_rotate)
        allset = DataSetIter(base_img, labels, all_index2pos, margin, patch_size, self.append_dim, random_rotate=self.random_rotate) 
        
        return trainset, unlabelset, testset, train_test_set, allset
 
    def generate_torch_dataset(self):
        if self.if_numpy:
            return self.generate_numpy_dataset()

        trainset, unlabelset, train_test_set, testset, allset = self.prepare_data()

        multi = self.data_param.get('unlabelled_multiple', 1)
        train_loader = torch.utils.data.DataLoader(
            dataset=trainset, batch_size=self.batch_size,
            shuffle=True, drop_last=False)
        unlabel_loader = torch.utils.data.DataLoader(
            dataset=unlabelset, batch_size=int(self.batch_size*multi),
            shuffle=False, num_workers=0, drop_last=False)
        test_loader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=self.batch_size,
            shuffle=False, num_workers=0, drop_last=False)
        train_test_loader = torch.utils.data.DataLoader(
            dataset=train_test_set, batch_size=self.batch_size,
            shuffle=False, num_workers=0, drop_last=False)
        all_loader = torch.utils.data.DataLoader(
            dataset=allset, batch_size=self.batch_size,
            shuffle=False, num_workers=0, drop_last=False)
        
        return train_loader, unlabel_loader, train_test_loader, test_loader, all_loader, testset


if __name__ == "__main__":
    dataloader = HSIDataLoader({"data": {
        "data_path_prefix": '../../data',
        "data_sign": "Indian",
        "data_file": "Indian_20",
        "use_dump": True
    }})
    train_loader, unlabel_loader, test_loader, train_test_loader, all_loader = dataloader.generate_torch_dataset()
