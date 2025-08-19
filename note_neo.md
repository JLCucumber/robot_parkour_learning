
# Quick Launch IsaacGym on UCL's GPU

## Step 1

tmux new -s collect bash
or
tmux attach -t collect 

nvidia-smi
cd /cs/student/projects2/rai/2024/hongboli
mamba activate /cs/student/projects2/rai/2024/hongboli/mamba_envs/isaacgym_parkour

// python test_gui.py

## Step 2

### Option 1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export LD_LIBRARY_PATH=/cs/student/projects2/rai/2024/hongboli/mamba_envs/isaacgym_parkour/lib

### Option 2
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH=/cs/student/projects2/rai/2024/hongboli/mamba_envs/isaacgym_parkour/lib


## Step 3
cd my_projects/robot_parkour_learning/legged_gym/

(1) Collect Trajectory
python legged_gym/scripts/collect.py --headless --task go2_distill --log --load_run Aug19_10-00-16_Go2_9skills_fromJul20_16-15-23/

(2) Train Go2 

(3) Train Go2 Field
python legged_gym/scripts/train.py --headless --task go2_field 



# Quick Start 7+1 GPU training and collection (Overall)

## Step 0:

remove data in remote servers:
rm -rf /mnt/rpl_projects/data/*


mkdir -p 


## Step 1: Launch Training on lab 4090 in Tmux

## Step 2: Launch Sync bash

---

# 