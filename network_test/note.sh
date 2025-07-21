
rsync -avz -e "ssh -J hongboli@knuckles.cs.ucl.ac.uk" ~/trajectory_logs/ hongboli@trailbreaker.cs.ucl.ac.uk:/home/hongboli/sync_data_test/



# 文件同步（A 端拉 B 端数据）：
rsync -avz -e "ssh -J hongboli@knuckles.cs.ucl.ac.uk" \
  hongboli@trailbreaker.cs.ucl.ac.uk:/cs/student/projects2/rai/2024/hongboli/network_test/trajectory_logs/ \
  /home/mscstudent/hongbo_li/network_test/collect_process_1/


# enforced version
# 终端1