""" Config to train the whole parkour oracle policy """
import numpy as np
from os import path as osp
from collections import OrderedDict
from datetime import datetime
import os
from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO, Go2RoughCfgPPO


from legged_gym.envs.go2.go2_distill_config import Go2DistillCfg, Go2DistillCfgPPO


multi_process_ = True

# 模块级别的路径配置，供所有类使用
_shared_path_enabled = os.getenv("LEGGED_GYM_USE_SHARED_PATH", "0").lower() in ("1", "true", "yes")
_shared_root = os.getenv("LEGGED_GYM_SHARED_PATH") or os.getenv("LEGGED_GYM_NFS_PATH") or "/mnt/rpl_project"
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 模块级别的 logs_root 和 data_root
logs_root = os.getenv("LEGGED_GYM_LOGS_ROOT") or (
    os.path.join(_shared_root, "logs") if _shared_path_enabled else os.path.join(_repo_root, "logs")
)
data_root = os.getenv("LEGGED_GYM_DATA_ROOT") or (
    os.path.join(_shared_root, "data") if _shared_path_enabled else os.path.join(_repo_root, "data")
)

# NOTE: This config intentionally disables AW-BC (advantage-weighted BC)
class Go2DistillNoAWBCCfg(Go2DistillCfg):
    class custom(Go2DistillCfg.custom):
        name = "go2_distill_no_awbc"

        # 是否启用共享路径
        shared_path = _shared_path_enabled

        # 使用模块级别定义的路径
        logs_root = logs_root
        data_root = data_root

    class env( Go2DistillCfg.env ):
        num_envs = 256 

class Go2DistillNoAWBCCfgPPO(Go2DistillCfgPPO):
    class algorithm(Go2DistillCfgPPO.algorithm):
        distill_target = "l1"
        awbc_weighting = False
        awbc_weight_type = "percentile"
        awbc_percentile = 95
        awbc_weight_clip = 1.0
        # Plan A: 即便关闭 AW-BC，也可以开启审计导出以便对照（weights 会退化为全 1）
        awbc_audit = dict(
            enable=True,
            every=4000,
            per_iter=16,
            save_dir=os.path.join(Go2DistillCfg.custom.logs_root, "awbc_audit", "go2_distill_no_awbc"),
            # position_obs_key=[0, 3],
        )
        # 其余参数继承

        # teacher_ac_path = osp.join(logs_root, "field_go2",
        #     "May26_20-05-28_Go2_10skills_pEnergy2.e-07_pTorques-1.e-07_pLazyStop-3.e+00_pPenD5.e-02_penEasier200_penHarder100_leapHeight2.e-01_motorTorqueClip_fromMay26_18-40-14",
        #     "model_40000.pt"
        # )

        teacher_ac_path = osp.join(logs_root, "field_go2",
            "Aug19_18-16-38_Go2_9skills_pEnergy2.e-07_pTorques-1.e-07_pLazyStop-3.e+00_pPenD5.e-02_penEasier200_penHarder100_motorTorqueClip_fromAug19_10-32-13",
            "model_50000.pt"
        )

    class runner(Go2DistillCfgPPO.runner):
        policy_class_name = "EncoderStateAcRecurrent"
        algorithm_class_name = "EstimatorTPPO"
        experiment_name = "go2_distill_no_awbc"
        num_steps_per_env = 32

        resume = True
        # load_run = osp.join(logs_root, "field_go2",
        #     "May26_20-05-28_Go2_10skills_pEnergy2.e-07_pTorques-1.e-07_pLazyStop-3.e+00_pPenD5.e-02_penEasier200_penHarder100_leapHeight2.e-01_motorTorqueClip_fromMay26_18-40-14",
        # )

        load_run = osp.join(logs_root, "field_go2",
            "Aug19_18-16-38_Go2_9skills_pEnergy2.e-07_pTorques-1.e-07_pLazyStop-3.e+00_pPenD5.e-02_penEasier200_penHarder100_motorTorqueClip_fromAug19_10-32-13",
        )

        # Extend previous distillation
        # resume = True
        # load_run = osp.join(logs_root, "distill_go2",
        #     "Jul13_06-15-12_Go2_8skills_fromMay26_20-05-28",
        # )

        class pretrain_dataset(Go2DistillCfgPPO.runner.pretrain_dataset):
            data_dir = osp.join(Go2DistillCfg.custom.data_root, "go2_distill_no_awbc")
            dataset_loops = -1
            random_shuffle_traj_order = True
            keep_latest_n_trajs = 1500
            starting_frame_range = [0, 50]

        
        max_iterations = 30000
        log_interval = 100