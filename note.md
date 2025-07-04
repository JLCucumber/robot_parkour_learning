python legged_gym/scripts/collect.py --headless --task go2_distill --log --load_run Jun27_18-12-12_Go2_10skills_fromMay26_20-05-28

heightfield_raw data shape: 1936 5520 border size: 200  (collect)
heightfield_raw data shape: 1936 5520 border size: 200
heightfield_raw data shape: 1744 4240 border size: 200
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

python legged_gym/scripts/collect.py --headless --task go2_distill --log --load_run Jul04_17-49-03_Go2_10skills_fromMay26_20-05-28/

python legged_gym/scripts/collect.py --headless --task go2_distill --log --load_run Jul04_18-55-33_Go2_10skills_fromMay26_20-05-28/


Jul04_19-28-58_ jump hurdle down tilted_ramp stairsup stairsdown slope wave_ blockLength2.4_teacherProb0.0_ randOrder_fric0.0-2.0 _aStd0.10 _Jul04_19-26-26