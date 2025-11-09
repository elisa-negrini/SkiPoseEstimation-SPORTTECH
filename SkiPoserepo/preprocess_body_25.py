import sys
import os
import json
import cv2
sys.path.append("..")
from utils import translation, normalize_head
from plot import plot
from einops import rearrange
import torch

body_25_poses_path = "/home/federico.diprima/Ski_Jump_BODY25_poses/annotations"

with open(os.path.join(body_25_poses_path, "BODY_25_ski_jump_annotated.json")) as f:
   data = json.load(f)

print(len(data))

# elimina pose vuote
data = {k: v for k, v in data.items() if v is not None}

# elimina confidence (posso tenerla per printare joint colorati in base confidence)
data = {
    key: [
        [lista_interna[:2] for lista_interna in lista_esterna]
        for lista_esterna in item
    ]
    for key, item in data.items()
}

# elimina multipose 
data = {
    key: item[0] for key, item in data.items()
}

print(len(data))

with open(os.path.join(body_25_poses_path, "BODY_25_ski_jump_annotated_processed.json"), 'w') as f:
    json.dump(data, f)


# data_list = []
# for key, item in data.items():
#     data_list.append(data[key])



# ora ho per ogni immagine una posa di 25 joints, devo capire come gestire quelli a 0, cioe quelli che non trova



# remove useless keypoints, feets and eyes, ears, forse faccio senza popparli via perche lo dovrebbe fre da solo dopo
# for key,_ in data.items():
#     person = data[key]
#     person.pop(24)
#     person.pop(23)
#     person.pop(22)
#     person.pop(21)
#     person.pop(20)
#     person.pop(19)
#     person.pop(18)
#     person.pop(17)
#     person.pop(16)
#     person.pop(15)

# se rimuovo questo ho una posa di 15 joints



# Crea un nuovo dizionario eliminando i valori [0.0, 0.0]
# data = {
#     key: [lista for lista in item if lista != [0.0, 0.0]]
#     for key, item in data.items()
# }



