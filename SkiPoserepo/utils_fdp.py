import numpy as np
from numpy.linalg import lstsq
import random

def theta_rotation(theta):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return rotation_matrix

# trasla la pelvis in (0,0)
def translation(v, type):
    if type == 'ski':
        root_joint = 22
    if type == 'body_25':
        root_joint = 8
    v = np.array(v)
    v_new = np.empty(shape=v.shape)
    for i in range(0,v.shape[1]):
        v_new[:,i,0] = np.array(v[:,i,0] - v[:,root_joint,0])
        v_new[:,i,1] = np.array(v[:,i,1] - v[:,root_joint,1])
    offset = np.empty(shape=(v.shape[0], v.shape[2]))
    offset[:, 0] = np.array(v[:,root_joint,0])
    offset[:, 1] = np.array(v[:,root_joint,1])

    return v_new, offset

def invert_translation(v, offset):
    v = np.array(v)
    offset=np.array(offset)
    v_new = np.empty(shape=v.shape)
    for i in range(0,v.shape[1]):
        v_new[:,i,0] = np.array(v[:,i,0] + offset[:,0])
        v_new[:,i,1] = np.array(v[:,i,1] + offset[:,1])
    return v_new
# normalizing function

def normalize_head(poses_2d, type):
    # center at root joint
    if type == "ski":
        root_joint=22
        p2d = poses_2d.reshape(-1, 2, 23)

        p2d -= p2d[:, :, [root_joint]]

        scale = np.linalg.norm(p2d[:, :, root_joint] - p2d[:, :, 0], axis=1, keepdims=True)
        p2ds = poses_2d / scale.mean()
        p2ds = p2ds * (1/10)
        p2ds = p2ds.reshape(-1,46)
    
    elif type == "body_25":
        root_joint=8
        p2d = poses_2d.reshape(-1, 2, 25)  #GIUSTO
        #p2d = poses_2d.reshape(-1, 2, 15)

        p2d -= p2d[:, :, [root_joint]]

        scale = np.linalg.norm(p2d[:, :, root_joint] - p2d[:, :, 0], axis=1, keepdims=True)

        p2ds = poses_2d / scale.mean()
        p2ds = p2ds * (1/10)

        p2ds = p2ds.reshape(-1,50) 
        #p2ds = p2ds.reshape(-1,30)
        
    return p2ds, scale.mean()

def invert_normalize_head(poses_2d, scale):
    p2ds = poses_2d * 10

    p2ds = p2ds * scale

    return p2ds

def align_ski(pose):
    # align right ski
    ski_r = pose[14:18]

    x_r = np.array([joint[0] for joint in ski_r])
    y_r = np.array([joint[1] for joint in ski_r])

    # regression line
    A = np.vstack([x_r, np.ones(len(x_r))]).T
    m, c = lstsq(A, y_r, rcond=None)[0]

    projection_x_r = (x_r + m * y_r - m * c) / (1 + m**2)
    projection_y_r = m*projection_x_r + c

    new_ski_r = [[x, y] for x, y in zip(projection_x_r, projection_y_r)]
    pose[14:18] = new_ski_r

    # align left ski
    ski_l = pose[18:22]
    
    x_l = np.array([joint[0] for joint in ski_l])
    y_l = np.array([joint[1] for joint in ski_l])

    # regression line
    A = np.vstack([x_l, np.ones(len(x_l))]).T
    m, c = lstsq(A, y_l, rcond=None)[0]

    projection_x_l = (x_l + m * y_l - m * c) / (1 + m**2)
    projection_y_l = m*projection_x_l + c

    new_ski_l = [[x, y] for x, y in zip(projection_x_l, projection_y_l)]

    pose[18:22] = new_ski_l

    return pose

def split_dict_dataset(dictionary, n1):
   keys = list(dictionary.keys())
   #random.shuffle(keys)
   train_dict = {key: dictionary[key] for key in keys[:n1]}
   test_dict = {key: dictionary[key] for key in keys[n1:]}
   return train_dict, test_dict