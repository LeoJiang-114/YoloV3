import numpy as np
import torch
import torch.nn.functional as F

# a=torch.Tensor([[[[1,2],
#                 [3,4]]]])
# b=F.interpolate(a,scale_factor=2,mode="nearest")
# print(a.shape)
# print(b,b.shape)
# c=torch.Tensor([[[1,2],
#                 [3,4]]])
# d=F.interpolate(c,scale_factor=2,mode="nearest")
# print(c.shape)
# print(d,d.shape)

# a=np.array([[[[1,2],[3,4]]]])
# b=np.array([[[[1,2]]]])
# c=np.row_stack((a,b))
# print(c)

import torch.nn as nn
import torch
Leaky=nn.LeakyReLU(0.1)
x=torch.Tensor([-100])
y=Leaky(x)

a=torch.Tensor([0.9,0.8,0.7,0.1,0.2,0.3])
b=torch.Tensor([1,0,0,0,0,0])
softmax=nn.Softmax()
c=softmax(a)
d=softmax(b)
print(c,)
print(d)
def Loss_design(out, feature_map, x):
    out = out.permute(0, 2, 3, 1)
    out = torch.reshape(out, shape=(out.size(0), out.size(1), out.size(2), 3, -1))

    mask_have = feature_map[..., 0] > 0
    mask_none = feature_map[..., 0] == 0

    loss_Logit = nn.BCEWithLogitsLoss()
    loss_conf_h = loss_Logit(out[mask_have][..., 0], feature_map[mask_have][..., 0])
    loss_conf_n = loss_Logit(out[mask_none][..., 0], feature_map[mask_none][..., 0])
    loss_conf = loss_conf_h + (1 -x)* loss_conf_n

    loss_MSE = nn.MSELoss()
    loss_data_h = loss_MSE(out[mask_have][..., 1:5], feature_map[mask_have][..., 1:5])

    Softmax = nn.Softmax()
    loss_class_h = loss_MSE(Softmax(out[mask_have][..., 5:]), feature_map[mask_have][..., 5:])
    Loss = loss_conf + loss_data_h + loss_class_h

    return Loss