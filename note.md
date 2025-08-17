python legged_gym/scripts/collect.py --headless --task go2_distill --log --load_run Jun27_18-12-12_Go2_10skills_fromMay26_20-05-28

heightfield_raw data shape: 1936 5520 border size: 200  (collect)
heightfield_raw data shape: 1936 5520 border size: 200
heightfield_raw data shape: 1168 528 border size: 200  (train)

---
server

sudo mkdir -p /mnt/rpl_project/logs
sudo mkdir -p /mnt/rpl_project/data

sudo mv /export/rpl_project/logs/* /mnt/rpl_project/logs/

sudo chown -R mscstudent:mscstudent /mnt/rpl_project
sudo chmod -R 777 /mnt/rpl_project

sudo gedit /etc/exports
/mnt/rpl_project 192.168.2.22(rw,sync,all_squash,anonuid=1001,anongid=1001,no_subtree_check)
/mnt/rpl_project 192.168.2.25(rw,sync,all_squash,anonuid=1001,anongid=1001,no_subtree_check)


sudo exportfs -ra
sudo systemctl restart nfs-kernel-server

---
client

---

```Bash
# 在客户端上运行

sudo umount /mnt/rpl_project_remote
sudo rmdir /mnt/rpl_project_remote

sudo mkdir -p /mnt/rpl_project

sudo mount 192.168.2.21:/mnt/rpl_project /mnt/rpl_project

sudo gedit /etc/fstab
192.168.2.21:/mnt/rpl_project /mnt/rpl_project nfs defaults 0 0


```
STEP 1
cd /home/jlcucumber/projects/isaac_gym_parkour
mamba activate mamba_env/isaacgym_hb/
cd robot_parkour_learning/legged_gym/
 
STEP 2
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH=/home/data/environments/isaac_gym_parkour/lib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


STEP 3
python legged_gym/scripts/train.py --headless --task go2_distill
python legged_gym/scripts/train.py --headless --task go2_field



python legged_gym/scripts/collect.py --headless --task go2_distill --log --load_run Jul08_18-44-07_Go2_8skills_fromMay26_20-05-28/
python legged_gym/scripts/collect.py --headless --task go2_distill --log --load_run Jul04_18-55-33_Go2_10skills_fromMay26_20-05-28/ --sim_device=cuda:0 --rl_device=cuda:0 --graphics_device_id=0
python legged_gym/scripts/train.py --headless --task=go2_distill

(
echo $LD_LIBRARY_PATH
/home/jlcucumber/miniforge3/envs/isaacgym_parkour_lhb/lib/libpython3.8.so.1.0
)

(
MESA_VK_DEVICE_SELECT=list vulkaninfo
error: XDG_RUNTIME_DIR not set in the environment.
selectable devices:
  GPU 0: 10de:24b0 "NVIDIA RTX A4000" discrete GPU 0000:00:00.0
  GPU 1: 10de:24b0 "NVIDIA RTX A4000" discrete GPU 0000:00:00.0

os.environ['MESA_VK_DEVICE_SELECT'] = '10de:24b0'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
)



---
```Bash
echo $CUDA_VISIBLE_DEVICES       # 应该=1
python - <<'PY'
import torch, os
print('CUDA_VISIBLE_DEVICES =', os.getenv('CUDA_VISIBLE_DEVICES'))
print('torch sees', torch.cuda.device_count(), 'GPU(s)')
print('current device index ->', torch.cuda.current_device())
PY
```

Jul04_19-28-58_ jump hurdle down tilted_ramp stairsup stairsdown slope wave_ blockLength2.4_teacherProb0.0_ randOrder_fric0.0-2.0 _aStd0.10 _Jul04_19-26-26


---
# Quick Start Train & Collect


tmux new -s train_session
conda activate isaac_gym_parkour
python legged_gym/scripts/train.py --task=go2_distill --headless


# Quick Delete Tmp Data
cd /mnt/rpl_project/data
rm -rf path


ssh -L 8081:ruddy-l.cs.ucl.ac.uk:8443 hongboli@knuckles.cs.ucl.ac.uk


# Quick Start 7+1 GPU training and collection

## Step 0:

remove data in remote servers:
rm -rf /mnt/rpl_projects/data/*

## Step 1: Launch Training on lab 4090 in Tmux

## Step 2: Launch Sync bash


---
# tensorboard
python legged_gym/scripts/play.py --task go2_distill --load_run /home/data/projects/robot_parkour_learning/legged_gym/logs/distill_go2/Jul13_06-15-12_Go2_8skills_fromMay26_20-05-28

tensorboard --logdir=/home/data/projects/robot_parkour_learning/legged_gym/logs/distill_go2/Jul13_06-15-12_Go2_8skills_fromMay26_20-05-28


# Launch IsaacGym On  my laptop

cd robot_parkour_learning/legged_gym