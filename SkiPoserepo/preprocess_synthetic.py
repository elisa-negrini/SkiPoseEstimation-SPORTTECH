import os
import json


dataset = []


folder_path = "/home/federico.diprima/Synthetic_Ski_Jump"
dataset_path = "/home/federico.diprima/Synthetic_Ski_Jump/dataset"


for dir in os.listdir(dataset_path):
    dir_path = os.path.join(dataset_path,dir)

    sequences = [seq for seq in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path,seq))]

    for seq in sequences:

        with open(os.path.join(dir_path, seq, "step0.frame_data.json")) as f:
            data = json.load(f)

        keypoints = data["captures"][0]["annotations"][0]["values"][0]["keypoints"]
        
        pose = []
        for joint in keypoints:
            pose.append([joint["location"][0]/741.0, joint["location"][1]/431.0])

        # scambio tail and hip, annotation error
        pose[18], pose[21] = pose[21], pose[18]
        pose[14], pose[17] = pose[17], pose[14]

        dataset.append(pose)


dataset = [[1, pose] for pose in dataset]

with open(os.path.join(folder_path, "synthetic_processed.json"), 'w') as file_json:
    json.dump(dataset, file_json)


