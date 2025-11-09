import os
import json
import random
from utils import split_dict_dataset

annotation_path = "/home/federico.diprima/Ski_Jump_Annotated/annotations"

with open(os.path.join(annotation_path,"annotations_ski_jump.json")) as f:
   data = json.load(f)


# sorted dictionary
data = dict(sorted(data.items(), key=lambda item: int(item[0].split('.')[0])))


# Train (350) and Test split (65)
#test_dict, train_dict = split_dict_dataset(data,65)

train_ski_jump_list = []

# train set saved as list of list
for image, pose in data.items():
   train_ski_jump_list.append(pose)

# add a value to know where the sample it comes from (1 = ski jump, 0 = ski 2d)
train_ski_jump_list = [[1, pose] for pose in train_ski_jump_list]

with open(os.path.join(annotation_path,"train.json"), 'w') as f:
   json.dump(train_ski_jump_list, f)

print("Ski Jump train set: ", len(train_ski_jump_list))


# # test set saved as dictionary, need image names to plot results
# with open(os.path.join(annotation_path,"test.json"), 'w') as f:
#    json.dump(test_dict, f)

# print("Ski Jump test set: ", len(test_dict))


# Ski 2D pose Dataset
annotation_path = "/home/federico.diprima/Ski_2DPose_Dataset/annotations"

with open(os.path.join(annotation_path,"ski2dpose_processed.json")) as f:
   ski_2d_list = json.load(f)

ski_2d_list = [[0, pose] for pose in ski_2d_list]

with open(os.path.join(annotation_path,"train.json"), 'w') as f:
   json.dump(ski_2d_list, f)

print("Ski 2D dataset: ", len(ski_2d_list))

# Total dataset made of Ski 2D pose Dataset + train ski jump dataset(manually annotated)
total_ski_list = ski_2d_list + train_ski_jump_list

#print("Complete dataset: ", len(total_ski_list))

annotation_path = "/home/federico.diprima/Total_Ski_Dataset"

with open(os.path.join(annotation_path,"train.json"), 'w') as f:
   json.dump(total_ski_list, f)

# Final dataset made of Ski 2D pose Dataset + train ski jump dataset + synthetic ski jump dataset

annotation_path = "/home/federico.diprima/Synthetic_Ski_Jump"

with open(os.path.join(annotation_path, "synthetic_processed.json")) as f:
   synthetic = json.load(f)

synthetic = [[1, pose] for pose in synthetic]

# How many synthetic samples ?
synthetic = synthetic

print("Synthetic dataset: ", len(synthetic))

final = total_ski_list + synthetic

print("Complete dataset (with synthetic data): ", len(final))

annotation_path = "/home/federico.diprima/Final_Ski_Dataset"

with open(os.path.join(annotation_path, "train.json"), 'w') as f:
   json.dump(final, f)


