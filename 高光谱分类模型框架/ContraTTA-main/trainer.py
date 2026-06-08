import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from models import transformer as transformer
from models import SSRN
import utils
from utils import recorder
from evaluation import HSIEvaluation
from copy import deepcopy
from utils import device
# from ContrastiveTrainer import CNNTrainer2
# from BNTentTrainer import SSRNTrainer, CNNTrainer
# from LNTentTrainer import TransformerTrainer, SSFTTNEWTrainer, SQSTrainer, TransformerOriTrainer,CNNTrainer_LN,HCANetTrainer_LN
# from LNTentTrainer_origin import CASSTTrainer, MASSTrainer
# from BNLNtentTrainer import TransformerTrainer as TransformerTrainerBNLN
# from LNMemoTrainer import TransformerTrainer_MEMO
from BNTentTrainerDELTA import TransformerTrainer_DELTA
# from BNTentTrainerCOTTA import TransformerTrainer_CFA
# from BNTentTrainer_origin import TransformerTrainer_BN
# from BNAdaptTrainer import TransformerTrainer_BNAdapt
# from BNCOTTATrainer import TransformerTrainer_COTTA



def get_trainer(params):
    trainer_type = params['net']['trainer']
    if trainer_type == "transformer_origin":
        return TransformerOriTrainer(params)
    if trainer_type == "transformer":
        return TransformerTrainer(params)
    if trainer_type == "SSRN":
        return SSRNTrainer(params)
    if trainer_type == "CNN":
        return CNNTrainer_LN(params)
    if trainer_type == "ssftt":
        return SSFTTNEWTrainer(params)
    if trainer_type == "sqs":
        return SQSTrainer(params)

    if trainer_type == "transformer_bnln":
        return TransformerTrainerBNLN(params)
    if trainer_type == "HCANet":
        return HCANetTrainer_LN(params)
    if trainer_type == "transformer_MEMO":
        return TransformerTrainer_MEMO(params)
    if trainer_type == "transformer_DELTA":
        return TransformerTrainer_DELTA(params)
    # if trainer_type == "transformer_CFA":
    #     return TransformerTrainer_CFA(params)
    if trainer_type == "transformer_BN":
        return TransformerTrainer_BN(params)
    if trainer_type == "BNAdapt":
        return TransformerTrainer_BNAdapt(params)
    if trainer_type == "COTTA":
        return TransformerTrainer_COTTA(params)
    if trainer_type == "casst":
        return CASSTTrainer(params)
    if trainer_type == "massformer":
        return MASSTrainer(params)
    

    assert Exception("Trainer not implemented!")

