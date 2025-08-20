""" Config to train the whole parkour oracle policy """
import numpy as np
from os import path as osp
from collections import OrderedDict
from datetime import datetime

from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO, Go2RoughCfgPPO

multi_process_ = True
logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
data_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "data")
# shared_path = "/mnt/rpl_project"            #${SHARED_PATH}  # Change this to your shared path if needed
# logs_root = osp.join(shared_path, "logs")  # shared path for NFS
# data_root = osp.join(shared_path, "data")  # shared path for NFS
# logs_root = osp.join("/mnt/rpl_project", "logs") # shared path for NFS

from legged_gym.envs.go2.go2_distill_config import Go2DistillCfg, Go2DistillCfgPPO

class Go2DistillAWBCCfg(Go2DistillCfg):
    class custom(Go2DistillCfg.custom):
        name = "distill_go2_awbc"

class Go2DistillAWBCCfgPPO(Go2DistillCfgPPO):
    class algorithm(Go2DistillCfgPPO.algorithm):
        distill_target = "l1"
        awbc_weighting = False
        awbc_weight_type = "percentile"
        awbc_percentile = 95
        awbc_weight_clip = 1.0
        # 其余参数继承