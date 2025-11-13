import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from MIL import AttentionMIL

test_set = [str(i) for i in range(19, 25)]


def softmax(x):
    x = np.array(x, dtype=np.float64)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


#model parameters

input_dim = 2048
hidden_dim = 64

#
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "" #path

model = AttentionMIL(input_dim, hidden_dim)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

attentions = [] #list of n attentions, each element is a list of the attention of the various patches of a slide
with torch.no_grad():
    for slide_name in test_set:
        bags_npy   = np.load(f"bags/{slide_name}/bags.npy", allow_pickle=True)
        bags = [torch.from_numpy(np.array(bag, dtype=np.float32)) for bag in bags_npy]
        bag_values = []
        for bag in bags:
            bag = bag.to(device)
            _, attn = model(bag)
            bag_values.extend(attn.cpu().numpy().flatten())
        attentions.append(bag_values)

scores = []
for bag_values in attentions:
    scores.append(softmax(bag_values))


