import os
import json


def dictionary_split(dictionary, start_index, end_index):
    """
    Extracts two dictionaries: one with elements between the specified indices and the other with the remaining elements.

    Args:
        dictionary (dict): The original dictionary.
        start_index (int): The starting index of extraction.
        end_index (int): The ending index of extraction (inclusive).

    Returns:
        dict, dict: Two dictionaries, one extracted and one remaining.
    """
    test_dictionary = {}
    remaining_dictionary = {}
    keys = list(dictionary.keys())

    for index, key in enumerate(keys):
        if start_index <= index <= end_index:
            test_dictionary[key] = dictionary[key]
        else:
            remaining_dictionary[key] = dictionary[key]

    return test_dictionary, remaining_dictionary

annotation_path = "/home/federico.diprima/Ski_Jump_Annotated/annotations"

with open(os.path.join(annotation_path,"annotations_ski_jump.json")) as f:
   data = json.load(f)


# sorted dictionary
data = dict(sorted(data.items(), key=lambda item: int(item[0].split('.')[0])))


# JUMP 1 DIRECTORY

jump_dir = os.path.join(annotation_path, "jump1")

if not os.path.exists(jump_dir):
    os.makedirs(jump_dir)
    
test_dict, train_dict = dictionary_split(data, 0, 63)

train_jump_list = []

# train set saved as list of list
for image, pose in train_dict.items():
   train_jump_list.append(pose)

# add a value to know where the sample it comes from (1 = ski jump, 0 = ski 2d)
train_jump_list = [[1, pose] for pose in train_jump_list]


with open(os.path.join(jump_dir,"train.json"), 'w') as f:
    json.dump(train_jump_list, f)

with open(os.path.join(jump_dir,"test.json"), 'w') as f:
    json.dump(test_dict, f)

# create BODY_25 set for testing
annotation_path = "/home/federico.diprima/Ski_Jump_BODY25_poses/annotations"

jump_dir = os.path.join(annotation_path, "jump1")

if not os.path.exists(jump_dir):
    os.makedirs(jump_dir)

with open(os.path.join(annotation_path, "BODY_25_ski_jump_annotated_processed.json")) as f:
   body_25_data = json.load(f)

keep_keys = test_dict.keys()

test_BODY_25 = {key: value for key, value in body_25_data.items() if key in keep_keys}

with open(os.path.join(jump_dir,"body_25_test.json"), 'w') as f:
   json.dump(test_BODY_25, f)


# JUMP 2 DIRECTORY
annotation_path = "/home/federico.diprima/Ski_Jump_Annotated/annotations"
jump_dir = os.path.join(annotation_path, "jump2")

if not os.path.exists(jump_dir):
    os.makedirs(jump_dir)
    
test_dict, train_dict = dictionary_split(data, 121, 163)

train_jump_list = []

# train set saved as list of list
for image, pose in train_dict.items():
   train_jump_list.append(pose)

# add a value to know where the sample it comes from (1 = ski jump, 0 = ski 2d)
train_jump_list = [[1, pose] for pose in train_jump_list]


with open(os.path.join(jump_dir,"train.json"), 'w') as f:
    json.dump(train_jump_list, f)

with open(os.path.join(jump_dir,"test.json"), 'w') as f:
    json.dump(test_dict, f)

# create BODY_25 set for testing
annotation_path = "/home/federico.diprima/Ski_Jump_BODY25_poses/annotations"

jump_dir = os.path.join(annotation_path, "jump2")

if not os.path.exists(jump_dir):
    os.makedirs(jump_dir)

with open(os.path.join(annotation_path, "BODY_25_ski_jump_annotated_processed.json")) as f:
   body_25_data = json.load(f)


keep_keys = test_dict.keys()

test_BODY_25 = {key: value for key, value in body_25_data.items() if key in keep_keys}

with open(os.path.join(jump_dir,"body_25_test.json"), 'w') as f:
   json.dump(test_BODY_25, f)


# JUMP 3 DIRECTORY
annotation_path = "/home/federico.diprima/Ski_Jump_Annotated/annotations"
jump_dir = os.path.join(annotation_path, "jump3")

if not os.path.exists(jump_dir):
    os.makedirs(jump_dir)
    
test_dict, train_dict = dictionary_split(data, 342, 414)
#test_dict, train_dict = dictionary_split(data, 195, 265)


train_jump_list = []

# train set saved as list of list
for image, pose in train_dict.items():
   train_jump_list.append(pose)

# add a value to know where the sample it comes from (1 = ski jump, 0 = ski 2d)
train_jump_list = [[1, pose] for pose in train_jump_list]

with open(os.path.join(jump_dir,"train.json"), 'w') as f:
    json.dump(train_jump_list, f)

with open(os.path.join(jump_dir,"test.json"), 'w') as f:
    json.dump(test_dict, f)

# create BODY_25 set for testing
annotation_path = "/home/federico.diprima/Ski_Jump_BODY25_poses/annotations"

jump_dir = os.path.join(annotation_path, "jump3")

if not os.path.exists(jump_dir):
    os.makedirs(jump_dir)

with open(os.path.join(annotation_path, "BODY_25_ski_jump_annotated_processed.json")) as f:
   body_25_data = json.load(f)

keep_keys = test_dict.keys()

test_BODY_25 = {key: value for key, value in body_25_data.items() if key in keep_keys}

with open(os.path.join(jump_dir,"body_25_test.json"), 'w') as f:
   json.dump(test_BODY_25, f)


# JUMP 4 DIRECTORY
annotation_path = "/home/federico.diprima/Ski_Jump_Annotated/annotations"
jump_dir = os.path.join(annotation_path, "jump4")

if not os.path.exists(jump_dir):
    os.makedirs(jump_dir)
    
test_dict, train_dict = dictionary_split(data, 273, 334)



train_jump_list = []

# train set saved as list of list
for image, pose in train_dict.items():
   train_jump_list.append(pose)

# add a value to know where the sample it comes from (1 = ski jump, 0 = ski 2d)
train_jump_list = [[1, pose] for pose in train_jump_list]

with open(os.path.join(jump_dir,"train.json"), 'w') as f:
    json.dump(train_jump_list, f)

with open(os.path.join(jump_dir,"test.json"), 'w') as f:
    json.dump(test_dict, f)

# create BODY_25 set for testing
annotation_path = "/home/federico.diprima/Ski_Jump_BODY25_poses/annotations"

jump_dir = os.path.join(annotation_path, "jump4")

if not os.path.exists(jump_dir):
    os.makedirs(jump_dir)

with open(os.path.join(annotation_path, "BODY_25_ski_jump_annotated_processed.json")) as f:
   body_25_data = json.load(f)

keep_keys = test_dict.keys()

test_BODY_25 = {key: value for key, value in body_25_data.items() if key in keep_keys}

with open(os.path.join(jump_dir, "body_25_test.json"), 'w') as f:
   json.dump(test_BODY_25, f)