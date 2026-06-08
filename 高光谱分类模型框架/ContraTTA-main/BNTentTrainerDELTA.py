import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from models import transformer as transformer
from models import SSRN
from models import CNN
from models import HCANet
import utils
from evaluation import HSIEvaluation
from copy import deepcopy
import matplotlib.pyplot as plt
# from SimsiamTTA import AdaSimSiam
from augment import do_augment
import torch
# import torchvision.transforms.functional as F
import torch.nn.functional as F
from utils import device
from utils import recorder
# from AdaMoco import AdaMoCo

from codecs import BOM_BE
from copy import deepcopy
# from curses import noecho
from pickle import NEWOBJ_EX
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

# inspired by https://github.com/bethgelab/robustness/tree/aa0a6798fe3973bae5f47561721b59b39f126ab7


# def stripe_preprocess(inputs, method='median', kernel_size=5):
#     """
#     inputs: [B, C, H, W], normalized in [0,1]
#     method: 'median' or 'mean'
#     kernel_size: 3 or 5（通常为3即可）
#     """
#     x = inputs.clone()

#     if method == 'median':
#         # 使用中值滤波（仅在 CPU 上跑，模拟高保边缘）
#         x_out = torch.zeros_like(x)
#         for b in range(x.shape[0]):
#             for c in range(x.shape[1]):
#                 x_img = x[b, c].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
#                 # 使用 unfold 实现 median filter
#                 patches = x_img.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)  # [1,1,H-k+1,W-k+1,k,k]
#                 patches = patches.contiguous().view(1, 1, -1, kernel_size * kernel_size)
#                 median = patches.median(dim=-1)[0]
#                 median_img = median.view(x_img.size(2) - kernel_size + 1, x_img.size(3) - kernel_size + 1)
#                 # padding恢复
#                 pad = kernel_size // 2
#                 x_out[b, c] = F.pad(median_img, [pad, pad, pad, pad], mode='reflect')
#         return torch.clamp(x_out, 0, 1)

#     elif method == 'mean':
#         # 用空间均值滤波（grouped conv 实现）
#         padding = kernel_size // 2
#         weight = torch.ones((x.size(1), 1, kernel_size, kernel_size), device=x.device) / (kernel_size ** 2)
#         smoothed = F.conv2d(x, weight, padding=padding, groups=x.size(1))
#         return smoothed

#     else:
#         raise ValueError("method must be 'median' or 'mean'")
    
# def deadlines_preprocess(inputs, strength=0.5, std_thresh=1e-4):
#     """
#     自动检测并修复死波段（波段方差过小）
#     strength: 0 表示不处理；1 表示强修复
#     std_thresh: 方差小于该值将被视为 dead band
#     inputs: [B, C, H, W], normalized in [0,1]
#     """
#     if strength == 0:
#         return inputs

#     x = inputs.clone()
#     B, C, H, W = x.shape

#     # 计算每个 band 的方差（对 B,H,W 三个维度）
#     stds = x.view(B, C, -1).std(dim=2).mean(dim=0)  # shape: [C]

#     # 找出异常小的 band
#     missing_bands = (stds < std_thresh).nonzero(as_tuple=True)[0]

#     if len(missing_bands) == 0:
#         return x  # 无需处理

#     for b in missing_bands:
#         b = b.item()
#         if b == 0:
#             x[:, b] = x[:, b + 1]
#         elif b == C - 1:
#             x[:, b] = x[:, b - 1]
#         else:
#             x[:, b] = 0.5 * (x[:, b - 1] + x[:, b + 1])
#     return torch.clamp(x, 0, 1)

def find_bns(parent, prior):
    replace_mods = []
    if parent is None:
        return []
    for name, child in parent.named_children():
        if isinstance(child, nn.BatchNorm2d):
            module = TBR(child, prior).cuda()
            replace_mods.append((parent, name, module))
        else:
            replace_mods.extend(find_bns(child, prior))
    return replace_mods


class TBR(nn.Module):
    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1
        super().__init__()
        self.layer = layer
        self.layer.eval()
        self.prior = prior
        self.rmax = 3.0
        self.dmax = 5.0
        self.tracked_num = 0
        # self.running_mean = deepcopy(layer.running_mean)
        # self.running_std = deepcopy(torch.sqrt(layer.running_var) + 1e-5)
        self.running_mean = None
        self.running_std = None

    def forward(self, input):
        batch_mean = input.mean([0, 2, 3])
        batch_std = torch.sqrt(input.var([0, 2, 3], unbiased=False) + self.layer.eps)

        if self.running_mean is None:
            self.running_mean = batch_mean.detach().clone()
            self.running_std = batch_std.detach().clone()

        r = (batch_std.detach() / self.running_std) #.clamp_(1./self.rmax, self.rmax)
        d = ((batch_mean.detach() - self.running_mean) / self.running_std) #.clamp_(-self.dmax, self.dmax)
        
        input = (input - batch_mean[None,:,None,None]) / batch_std[None,:,None,None] * r[None,:,None,None] + d[None,:,None,None]
        # input = (input - self.running_mean[None,:,None,None]) / self.running_std[None,:,None,None]

        # if len(input)>=128:
        self.running_mean = self.prior * self.running_mean + (1. - self.prior) * batch_mean.detach()
        self.running_std = self.prior * self.running_std + (1. - self.prior) * batch_std.detach()
        # else:
        #     print('too small batch size, using last step model directly...')

        self.tracked_num+=1

        return input * self.layer.weight[None,:,None,None] + self.layer.bias[None,:,None,None]


class DELTA(nn.Module):
    def __init__(self, args, model_old):
        super().__init__()
        self.args = args

        self.model_old = model_old
        self.model_old.eval()
        self.model_old.requires_grad_(False)
        
        # proj_dim = 1024
        # feat_dim = 64
        # self.projector = nn.Sequential(
        #     nn.Linear(feat_dim, proj_dim, bias=False),
        #     nn.BatchNorm1d(proj_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(proj_dim, feat_dim, bias=False)
        # ).cuda()

        # # SimSiam predictor
        # self.predictor = nn.Sequential(
        #     nn.Linear(proj_dim, proj_dim//4, bias=False),
        #     nn.BatchNorm1d(proj_dim),
        #     nn.ReLU(),
        #     nn.Linear(proj_dim, proj_dim, bias=False)
        # ).cuda()
        proj_dim = 1024
        feat_dim = 64
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(proj_dim, feat_dim, bias=False)
        ).cuda()

        # SimSiam predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim//4, bias=False),
            nn.BatchNorm1d(proj_dim//4),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(proj_dim//4, proj_dim, bias=False)
        ).cuda()
        
        self.reset()


    def reset(self):
        self.model = deepcopy(self.model_old)

        
        replace_mods = find_bns(self.model, 0.999) #PU 0.999   LK 0.95
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        
        self.model.requires_grad_(False)
        params = []
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        p.requires_grad_(True)
                        params.append(p)
        
        simsiam_bn_params = []
        for m in list(self.projector.modules()) :
            if isinstance(m, nn.BatchNorm1d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        p.requires_grad_(True)
                        simsiam_bn_params.append(p)
        for m in self.predictor.modules():
            if isinstance(m, nn.BatchNorm1d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        p.requires_grad_(True)
                        simsiam_bn_params.append(p)

        
        
        self.optimizer = torch.optim.SGD(params, lr=0.00025) #0.00001目前最低 0.1低过头了 0.000001
       # self.optimizer = torch.optim.SGD([
       # {'params': params, 'lr': 0.00025}, #0.00025 0.001          0.00025 stripes
      #  {'params': simsiam_bn_params, 'lr': 0.25},
        # {'params': self.classifier_head.parameters(), 'lr': 0.25} # 对比学习部分更大学习率 0.01   0.0025 stripes
    #], momentum=0.9, weight_decay=0)

        self.qhat = torch.zeros(1, 9).cuda() + (1. / 9)

    def div(self,logits, epsilon=1e-8):
        probs = F.softmax(logits, dim=1)
        probs_mean = probs.mean(dim=0)
        loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

        return loss_div
    def forward(self, x,x2 = None,target = None,epoch = None):
        with torch.enable_grad():
            output = self.model(x)
            outputs  =  output[0]
            # print(outputs.shape)

            p = F.softmax(outputs, dim=1)
            p_max, pls = p.max(dim=1)
            logp = F.log_softmax(outputs, dim=1)

            ent_weight = torch.ones_like(pls)
            
            entropys = -(p * logp).sum(dim=1)
            
            # ent_weight = torch.exp(math.log(9) * 0.5 - entropys.clone().detach())
            # use_idx = (ent_weight>1.)
            use_idx = (entropys==entropys)
            confidence_threshold = 0.75
            conf_mask = p_max > confidence_threshold
       
            
            
            class_weight = 1. / self.qhat
            class_weight = class_weight / class_weight.sum()
            sample_weight = class_weight.gather(1, pls.view(1,-1)).squeeze()
            sample_weight = sample_weight / sample_weight.sum() * len(pls)
            ent_weight = ent_weight * sample_weight
            # print(entropys.shape)
            loss = (entropys * ent_weight)[use_idx].mean()
            # loss_ce = F.cross_entropy(outputs[conf_mask], pls[conf_mask])
            if x2 is not None:
                with torch.no_grad():
                    output2 = self.model(x2)
                    # output3 = self.model(x3)
                    # feat2 = output2[0]  # 提取第二视图的中间特征

                # SimSiam流
                z1 = self.projector(output[1])  # B x proj_dim
                # print(output[1].shape)
                # p3 = self.predictor(z3)

                z2 = self.projector(output2[1])
                # p2 = self.predictor(z2)

                # z1 = self.projector(output[1])
                # outputs_new = self.predictor(z1)
                
                # feat1 = output3[1][conf_mask]
                # feat2 = output2[1][conf_mask]
                simsiam_loss = -0.5 * (F.cosine_similarity(output[1], z2.detach(), dim=1).mean() +
                                       F.cosine_similarity(output[1], z1.detach(), dim=1).mean())
                # pseudo_ce_loss = F.cross_entropy(outputs[conf_mask], pls[conf_mask])

                # loss_simsiam = simsiam_loss
                # # loss_div = self.div(output2[1]) + self.div(output[1])
                loss =  loss + (simsiam_loss)*0.01  #0.01jpeg 0.001
                # print(loss)
                # print(loss_simsiam*0.01)
            if use_idx.sum()!=0:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            
            with torch.no_grad():
                self.qhat = 0.95 * self.qhat + (1. - 0.95) * F.softmax(outputs, dim=1).mean(dim=0, keepdim=True)
                
            return outputs, loss


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Compute the entropy of the softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def monitor_layernorm_parameters(model):
    """
    Monitor LayerNorm parameters (scale and shift) in a Transformer model.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            print(f"LayerNorm {name} - scale (weight): {module.weight}, shift (bias): {module.bias}")


# def collect_params(model):
#     """Collect the affine scale + shift parameters from batch norms.

#          Walk the model's modules and collect all batch normalization parameters.
#          Return the parameters and their names.

#          Note: other choices of parameterization are possible!
#          """
#     params = []
#     names = []
#     for nm, m in model.named_modules():
#         if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
#         # if isinstance(m, nn.BatchNorm2d):
#             # print(m)
#             for np, p in m.named_parameters():
#                 if np in ['weight', 'bias']:  # weight is scale, bias is shift
#                     params.append(p)
#                     names.append(f"{nm}.{np}")
        
#     print("params:-------------------")
#     print(names)
#     print("params:-------------------")
#     return params, names


# def configure_model(model_0):
#     """
#     Configure model for TENT: only update LayerNorm layers' scale (weight) and shift (bias).
#     """
#     # deepcopy to ensure original model is not modified
#     model = deepcopy(model_0)
#     model.requires_grad_(False)
#     # configure norm for tent updates: enable grad + force batch statisics
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
#         # if isinstance(m, nn.LayerNorm):
#             m.requires_grad_(True)
#             # force use of batch stats in train and eval modes
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#         if isinstance(m, nn.LayerNorm):
#             m.requires_grad_(True)

#     return model


class BaseTrainer(object):
    def __init__(self, params) -> None:
        self.tent_model = None
        self.tent_net = None
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.train_sign = self.params.get('train_sign', 'train') # train, test or tent
        self.path_model_save = self.params.get('path_model_save', '')
        self.model_loaded = False
        self.device = device 
        self.evalator = HSIEvaluation(param=params)
        self.class_num = self.params['data'].get('num_classes')
        self.aug=params.get("aug",None)
        self.net = None
        self.criterion = None
        self.optimizer = None
        self.clip = 15
        self.unlabel_loader=None

        # init model and check if use tent mode. if tent, configure tent model.
        self.real_init()

        # test configure
        if self.train_sign == 'test':
            self.load_model(self.path_model_save)
        # tent configure
        if self.train_sign == 'tent':
            self.load_model(self.path_model_save)
        if self.train_sign == 'ctent':
            self.load_model(self.path_model_save)
            # self.confiture_tent()

    def real_init(self):
        pass

    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)
       
    def train(self, train_loader, unlabel_loader=None, test_loader=None):
        # if tent, skip train...
        if self.train_sign in ['test', 'tent', 'ctent']:
            print("%s model skip train.." % (self.train_sign))
            return True
        epochs = self.params['train'].get('epochs', 100)
        total_loss = 0
        epoch_avg_loss = utils.AvgrageMeter()
        max_oa = 0
        for epoch in range(epochs):
            self.net.train()
            epoch_avg_loss.reset()
            for i, (data, target,_,_) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.net(data)
                loss = self.get_loss(outputs[1], target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                # batch stat
                total_loss += loss.item()
                epoch_avg_loss.update(loss.item(), data.shape[0])
            recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
            print('[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (epoch + 1,
                                                                             epoch_avg_loss.get_avg(), 
                                                                             total_loss / (epoch + 1),
                                                                             loss.item(), epoch_avg_loss.get_num()))
            # 一定epoch下进行一次eval
            if test_loader and (epoch+1) % 10 == 0:
                y_pred_test, y_test = self.test(test_loader)
                temp_res = self.evalator.eval(y_test, y_pred_test, self.class_num)
                max_oa = max(max_oa, temp_res['oa'])
                recorder.append_index_value("train_oa", epoch+1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch+1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch+1, temp_res['kappa'])
                recorder.append_index_value("max_oa", epoch+1, max_oa)
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (epoch+1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
        print('Finished Training')

        torch.save(self.net.state_dict(), self.path_model_save)
        print("model saved.")
        return True

    def final_eval(self, test_loader,aug):
        if self.train_sign == 'ctent':
            # y_pred_test1, y_test = self.test(test_loader)  # test first
            y_pred_test, y_test = self.test_ctent(test_loader,aug)  
        if self.train_sign == 'tent':
            y_pred_test, y_test = self.test_tent(test_loader) 
        else:
            y_pred_test, y_test = self.test(test_loader)  # test change
        # print(y_test)
        temp_res = self.evalator.eval(y_test, y_pred_test, self.class_num)
        return temp_res

    def get_logits(self, output):
        if type(output) == tuple:
            return output[1]
        return output

    def load_model(self, model_path):
        if self.path_model_save == "":
            raise ValueError("tent model need model path.")
        else:
            print("load model from model_path: %s" % self.path_model_save)
        self.net.load_state_dict(torch.load(model_path))
        self.model_loaded = True
        return self.net
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        """Compute the entropy of the softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)
    
    def confiture_tent(self): 
        print("start to confiture tent model...")
        assert self.model_loaded == True
        # self.tent_net = configure_model(self.net)  # self.net
        self.tent_net =  deepcopy(self.net)
       
        # optimizer = optim.SGD(params, lr=0.0001, momentum=0.9)
    #     self.tent_model = AdaSimSiam(self.tent_net, 64)
    #     backbone_params, extra_params = self.tent_model.get_params()
    #     param_groups = [
    #     {'params': backbone_params, 'lr': 0.0001},
    #     {'params': extra_params, 'lr': 0.001},
    # ]
    #     optimizer = torch.optim.SGD(param_groups)
        return True
    
    
    def test_ctent(self, test_loader,aug):
        """
            Test the model on the test set, applying TENT adaptation if tent_model is defined.
        """
        # with torch.no_grad():
        #     self.tent_net.train()  # 必须是train模式，BN才会统计
        #     for inputs, _, _, _ in test_loader:
        #         inputs = inputs.to(self.device)
        #         _ = self.tent_net(inputs)
        self.tent_model = DELTA(self.params, self.net).cuda()
        self.tent_model.reset()
        # for param in self.tent_model.parameters():
        #     param.requires_grad = False
        # self.tent_model.eval()
    #     backbone_params, extra_params = self.tent_model.get_params()
    #     param_groups = [
    #     {'params':backbone_params,'lr':0.0001},
    #     {'params': extra_params, 'lr': 0.0001},
    # ] 
        # params, param_names = collect_params(self.tent_net)
        # optimizer = optim.SGD(params, lr=0.0001, momentum=0.9)
        
        
        
        tent_epochs = 10
        epoch_avg_loss = utils.AvgrageMeter()
        total_loss = 0
  
        print("start to test in tent mode...")
        for epoch in range(tent_epochs):
            
            # model.eval()
            # self.tent_model.eval()
            epoch_avg_loss.reset()
            count = 0
            num = 0
            y_pred_test = 0
            y_test = 0
            x = []
            y = []
            
            for inputs, labels,x0,y0 in test_loader:
                
                # print(labels)
                # inputs = inputs.to(self.device)
                # for i in range(len(x0)):
                #     x.append(x0[i])
                #     y.append(y0[i])
                inputs = inputs.to(self.device)
                targets = labels.to(self.device)
                
                weak_data, strong_data = do_augment(self.aug,inputs, aug)
                # inputs = fog_preprocess(inputs, fog_strength=0.7 if noise_type in ['thin_fog', 'thick_fog'] else 0)
                # inputs = fog_preprocess(inputs, fog_strength=0.7)
                # inputs = deadlines_preprocess(inputs)
                # inputs = stripe_preprocess(inputs)

                if self.tent_model is not None:

                    logits, loss = self.tent_model (weak_data, strong_data ,targets,epoch)
                    # print(logits.shape)
                    # logits = outputs.argmax(dim=1)
                    # logits = outputs[-1]
                    # loss_con = -(criterion(outputs[0], outputs[3]).mean() + criterion(outputs[1], outputs[2]).mean()+2)
                    # loss_cls = self.get_loss(outputs[-1],target_list)
                    # # loss_div = self.div(outputs[-1]) + self.div(outputs[-2])
                    # # loss_div = self.div(logits_w) + self.div(logits_s)
                    # loss =  loss_cls 
                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()  # This is likely a tuple
                # else:
                #     # outputs = model(inputs)
                #     raise ValueError('tent_model should not be None.')
                
                if len(logits.shape) == 1:
                    continue
                if count % 10 == 0:
                    print("tent train epoch=%s, tent_loss=%s"  % (epoch, loss.detach().cpu().numpy()))
                # print(logits)
                # Get predictions (argmax over the softmax outputs)
                preds = np.argmax(logits.detach().cpu().numpy(), axis=1)


                
                # Concatenate predictions and ground truth labels
                if count == 0:
                    y_pred_test = preds
                    y_test = labels
                    count = count + 1
                else:
                    y_pred_test = np.concatenate((y_pred_test, preds))
                    y_test = np.concatenate((y_test, labels))

            if test_loader and (epoch + 1) % 1 == 0:
                # self.tent_model.eval()
                # for inputs, labels,x0,y0 in test_loader:
                #     inputs = inputs.to(self.device)
                #     left_data, right_data = do_augment(self.aug,inputs)

                #     logits2 = self.tent_model(inputs,left_data)[-2]
                # #     if len(logits.shape) == 1:
                # #         continue
                # outputs2 = np.argmax(logits2.detach().cpu().numpy(), axis=1)
                
                # if count == 0:
                #     y_pred_test = outputs2
                #     y_test = labels
                #     count = 1
                # else:
                #     y_pred_test = np.concatenate((y_pred_test, outputs2))
                #     y_test = np.concatenate((y_test, labels))
                # monitor_layernorm_parameters(self.tent_model)  # 监控 LayerNorm 参数的变化
                class_num = self.params['data'].get('num_classes')
                temp_res = self.evalator.eval(y_test, y_pred_test, self.class_num)
                recorder.append_index_value("train_oa", epoch + 1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch + 1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch + 1, temp_res['kappa'])
                
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (
                epoch + 1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
        # dat = np.zeros((550,400))
        # print(len(x))
        # for i in range(len(y_pred_test)):
        #     dat[x[i],y[i]] = y_pred_test[i]
        # # plt.imshow(dat)
        # # plt.show()
        # np.save(f'SSRN_Longkou_{noise_type}.npy',dat)
        return y_pred_test, y_test
    def test_tent(self, test_loader):
        """
            Test the model on the test set, applying TENT adaptation if tent_model is defined.
        """
        
        self.tent_model = DELTA(self.params, self.net).cuda()
        self.tent_model.reset()
        tent_epochs = 10
        epoch_avg_loss = utils.AvgrageMeter()
        total_loss = 0
        print("start to test in tent mode...")
        for epoch in range(tent_epochs):
            
            # model.eval()
            # self.tent_model.eval()
            epoch_avg_loss.reset()
            count = 0
            num = 0
            y_pred_test = 0
            y_test = 0
            x = []
            y = []
            for inputs, labels,x0,y0 in test_loader:
                inputs = inputs.to(self.device)
                targets = labels.to(self.device)
                # inputs = fog_preprocess(inputs, fog_strength=0.7 if noise_type in ['thin_fog', 'thick_fog'] else 0)
                
                # inputs = deadlines_preprocess(inputs)
                # inputs = stripe_preprocess(inputs)

                if self.tent_model is not None:

                    logits, loss = self.tent_model (inputs)
                    # print(logits.shape)
                    # logits = outputs.argmax(dim=1)
                    # logits = outputs[-1]
                    # loss_con = -(criterion(outputs[0], outputs[3]).mean() + criterion(outputs[1], outputs[2]).mean()+2)
                    # loss_cls = self.get_loss(outputs[-1],target_list)
                    # # loss_div = self.div(outputs[-1]) + self.div(outputs[-2])
                    # # loss_div = self.div(logits_w) + self.div(logits_s)
                    # loss =  loss_cls 
                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()  # This is likely a tuple
                # else:
                #     # outputs = model(inputs)
                #     raise ValueError('tent_model should not be None.')
                
                if len(logits.shape) == 1:
                    continue
                if count % 10 == 0:
                    print("tent train epoch=%s, tent_loss=%s"  % (epoch, loss.detach().cpu().numpy()))
                # print(logits)
                # Get predictions (argmax over the softmax outputs)
                preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

                # Concatenate predictions and ground truth labels
                if count == 0:
                    y_pred_test = preds
                    y_test = labels
                    count = count + 1
                else:
                    y_pred_test = np.concatenate((y_pred_test, preds))
                    y_test = np.concatenate((y_test, labels))

            if test_loader and (epoch + 1) % 1 == 0:
                # self.tent_model.eval()
                # for inputs, labels,x0,y0 in test_loader:
                #     inputs = inputs.to(self.device)
                #     left_data, right_data = do_augment(self.aug,inputs)

                #     logits2 = self.tent_model(inputs,left_data)[-2]
                # #     if len(logits.shape) == 1:
                # #         continue
                # outputs2 = np.argmax(logits2.detach().cpu().numpy(), axis=1)
                
                # if count == 0:
                #     y_pred_test = outputs2
                #     y_test = labels
                #     count = 1
                # else:
                #     y_pred_test = np.concatenate((y_pred_test, outputs2))
                #     y_test = np.concatenate((y_test, labels))
                # monitor_layernorm_parameters(self.tent_model)  # 监控 LayerNorm 参数的变化
                class_num = self.params['data'].get('num_classes')
                temp_res = self.evalator.eval(y_test, y_pred_test, self.class_num)
                recorder.append_index_value("train_oa", epoch + 1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch + 1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch + 1, temp_res['kappa'])
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (
                epoch + 1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
        # dat = np.zeros((550,400))
        # print(len(x))
        # for i in range(len(y_pred_test)):
        #     dat[x[i],y[i]] = y_pred_test[i]
        # # plt.imshow(dat)
        # # plt.show()
        # np.save(f'SSRN_Longkou_{noise_type}.npy',dat)
        return y_pred_test, y_test

    def test(self, test_loader):
        """
        provide test_loader, return test result(only net output)
        """
        model = self.net
        count = 0
        self.net.eval()
        y_pred_test = 0
        y_test = 0
        x = []
        y = []
        # for inputs, labels,x0,y0 in test_loader:
        #     inputs = inputs.to(self.device)
        #     for i in range(len(x0)):
        #         x.append(x0[i])
        #         y.append(y0[i])
        for inputs, labels,_,_ in test_loader:
            # print(labels)
            inputs = inputs.to(self.device)
            logits = model(inputs)[0]
            if len(logits.shape) == 1:
                continue
            outputs = np.argmax(logits.detach().cpu().numpy(), axis=1)
            
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
        # dat = np.zeros((610,340))
        # print(len(x))
        # for i in range(len(y_pred_test)):
        #     dat[x[i],y[i]] = y_pred_test[i]
        # # plt.imshow(dat)
        # # plt.show()
        # np.save(f'SSRN_Pavia_clean.npy',dat)
        return y_pred_test, y_test


class TransformerTrainer_DELTA(BaseTrainer):
    def __init__(self, params):
        super(TransformerTrainer_DELTA, self).__init__(params)
        self.lr = None
        self.weight_decay = None

    def real_init(self):
        # net
        self.net = transformer.TransFormerNet(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        """
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        """
        logits = outputs
        
        loss_main = nn.CrossEntropyLoss()(logits, target) 

        return loss_main   

class SSRNTrainer(BaseTrainer):
    def __init__(self, params):
        super(SSRNTrainer, self).__init__(params)
        self.lr = None
        self.weight_decay = None

    def real_init(self):
        # net
        self.net = SSRN.SSRN(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        """
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        """
        logits = outputs
        
        loss_main = nn.CrossEntropyLoss()(logits, target) 

        return loss_main   

class CNNTrainer(BaseTrainer):
    def __init__(self, params):
        super(CNNTrainer, self).__init__(params)
        self.lr = None
        self.weight_decay = None

    def real_init(self):
        # net
        self.net = CNN.CNNNet(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        """
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        """
        logits = outputs
        
        loss_main = nn.CrossEntropyLoss()(logits, target) 

        return loss_main   

class HCANetTrainer(BaseTrainer):
    def __init__(self, params):
        super(HCANetTrainer, self).__init__(params)
        self.lr = None
        self.weight_decay = None
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def real_init(self):
        # net
        self.net = HCANet.HCANet(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
    
    def get_loss(self, outputs, target):
        """
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        """
        logits = outputs
        
        loss_main = nn.CrossEntropyLoss()(logits, target) 

        return loss_main   


