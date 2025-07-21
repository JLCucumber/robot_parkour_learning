import os
import pickle
import random

# Fake Data Generator for Testing
# generate 50 trajectories folders with 3 .pickle files each, save in this current folder path
def generate_fake_data(base_path, num_folders=50, num_files=3):
    for folder_idx in range(num_folders):
        folder_name = f"trajectory_{folder_idx}"
        folder_path = os.path.join(base_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        for file_idx in range(num_files):
            file_name = f"data_{file_idx}.pickle"
            file_path = os.path.join(folder_path, file_name)
            
            # Generate random fake data
            fake_data = {
                "position": [random.uniform(-100, 100) for _ in range(3)],
                "velocity": [random.uniform(-10, 10) for _ in range(3)],
                "timestamp": random.uniform(0, 1000)
            }
            
            # Save fake data to pickle file
            with open(file_path, "wb") as f:
                pickle.dump(fake_data, f)

# Generate fake data in the current folder
current_folder = os.path.dirname(os.path.abspath(__file__))
generate_fake_data(current_folder)


# 