python legged_gym/scripts/collect.py --headless --task go2_distill --log --load_run Jun27_18-12-12_Go2_10skills_fromMay26_20-05-28

heightfield_raw data shape: 1936 5520 border size: 200  (collect)
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

sudo exportfs -ra
sudo systemctl restart nfs-kernel-server

---
client