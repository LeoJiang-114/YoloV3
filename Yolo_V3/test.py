import math
import torch
import numpy as np

# x=1.5
# a,b=math.modf(x)
# print(a)
# print(b)

# a=torch.rand(2,13,13,42)
# print(a.size(3))
# a=torch.reshape(a,shape=(a.size(0),a.size(1),a.size(2),3,-1))
# print(a.shape)

output=torch.tensor([[[[[4, 3, 1, 4, 3],
           [1, 4, 3, 3, 2],
           [1, 2, 2, 4, 3]],
          [[1, 2, 4, 4, 3],
           [4, 3, 1, 4, 1],
           [4, 4, 4, 4, 4]],
          [[3, 3, 1, 2, 1],
           [4, 1, 4, 4, 4],
           [4, 1, 3, 3, 3]]],
         [[[2, 3, 4, 1, 2],
           [3, 4, 3, 4, 4],
           [2, 4, 1, 4, 2]],
          [[3, 1, 3, 3, 4],
           [1, 1, 2, 3, 3],
           [1, 2, 1, 1, 4]],
          [[2, 3, 2, 3, 2],
           [4, 1, 3, 4, 2],
           [4, 3, 4, 1, 2]]],
         [[[1, 4, 1, 2, 2],
           [2, 4, 1, 3, 1],
           [4, 1, 1, 2, 1]],
          [[3, 2, 4, 2, 2],
           [2, 1, 1, 4, 4],
           [4, 4, 2, 1, 2]],
          [[3, 1, 2, 4, 1],
           [4, 2, 2, 3, 3],
           [1, 1, 1, 2, 2]]]]])
print(output.shape)
mask = output[..., 0] >3
print("mask:",mask.shape, mask)
idxs = mask.nonzero()
print("idxes:",idxs.shape, idxs)
output_=output[mask]
print(output_.shape,output_)

# if idxs.shape[0]==0:
#     print("Ture")
# a=torch.Tensor([])
# b=torch.Tensor([1])
# c=torch.cat([a,b],dim=0)
# print(c)
# vecs = output[mask]
# print(vecs.shape)
# print(vecs[:,3])
# print(vecs[:,2])
# a=torch.stack([vecs[:,3],vecs[:,2]],dim=1)
# print(a)

# dict_label={"狗":0,"人":1,"羊驼":2,"小汽车":3,"自行车":4,"海豚":5,"松鼠":6,"马":7,"猫":8}
# dict_label_={}
# for i in dict_label:
#     dict_label_[dict_label[i]]=i
# print(dict_label_)

# a = torch.Tensor([[0., 1., 0., 0., 0.],
#                   [0., 1., 0., 0., 0.],
#                   [0., 1., 0., 0., 0.],
#                   [0., 0., 0., 0., 1.],
#                   [0., 0., 0., 0., 1.],
#                   [0., 0., 0., 0., 1.]])
# b=torch.argmax(a,dim=1)
# print(b)

# a=torch.Tensor([[1,2,3,4,5,6,7,8,9],
#                 [2,3,4,5,6,7,8,9,10]])
# b=a[:,5:]
# print(b)
# c=torch.argmax(b,dim=1)
# print(c)

from PIL import Image,ImageDraw
import PIL.ImageFont as Font

# word_type=Font.truetype((r"E:\jkl\simkai.ttf"),20)
# img_path=r"F:\Yolo_Datasets\pic_deal\12.jpg"
# img_open=Image.open(img_path)
# img_draw=ImageDraw.Draw(img_open)
# img_draw.text((50,50),text="海豚",fill=(0,255,255),font=word_type)
# img_open.show()

#softmax处理：
import torch.nn as nn

# a=np.array([0.3542, 0.5765, 0.5010, 0.2686, 0.0675, 0.8790, 0.9125, 0.6205, 0.4691,0.0044])
# np.set_printoptions(suppress=True)
# a=torch.Tensor(a)
# print(a)
# softmax=nn.Softmax()
# b=softmax(a)
# print(b)
# def S_softmax(x):
#     return torch.exp(10*x)/torch.sum(torch.exp(10*x))
# c=S_softmax(a)
# print(c)
#
# [0.3542,     0.5765,     0.5010,     0.2686,     0.0675,     0.8790,    0.9125,     0.6205,     0.4691,     0.0044]
# [9.3303e-06, 7.9574e-04, 1.7579e-04, 1.6842e-06, 3.0175e-08, 3.3748e-01,6.5952e-01, 1.9184e-03, 9.2877e-05, 8.5423e-09]

#字典：
# dict={1:[1],2:[2],3:[3],4:[4],5:[5],6:[6],7:[7],8:[8],9:[9],
#       11:[11],12:[12],13:[13],14:[14],15:[15],16:[16],17:[17],18:[18],19:[19]}
# list=[]
# list=[[0.6973293768545994, 13, 6, 3, 0, 1.0],
#        [0.32867132867132864, 13, 6, 3, 1, 1.0],
#        [0.08348821526669846, 13, 6, 3, 2, 1.0],
#        [0.48542335053703445, 13, 8, 4, 0, 4.0],
#        [0.4675237375010503, 13, 8, 4, 1, 4.0],
#        [0.17686968535666706, 13, 8, 4, 2, 4.0],
#        [0.2508891665865616, 13, 5, 8, 0, 3.0],
#        [0.7422858790733442, 13, 5, 8, 1, 3.0],
#        [0.3422095758153917, 13, 5, 8, 2, 3.0],
#        [0.1802600472813239, 26, 12, 7, 0, 1.0],
#        [0.274822695035461, 26, 12, 7, 1, 1.0],
#        [0.5899453754282011, 26, 12, 7, 2, 1.0],
#        [0.08508857581252616, 26, 17, 8, 0, 4.0],
#        [0.1297252057469661, 26, 17, 8, 1, 4.0],
#        [0.2841773576412334, 26, 17, 8, 2, 4.0],
#        [0.04397769874074786, 26, 11, 17, 0, 3.0],
#        [0.0670479669326156, 26, 11, 17, 1, 3.0],
#        [0.16872536768239932, 26, 11, 17, 2, 3.0],
#        [0.0128053585500394, 52, 24, 15, 0, 1.0],
#        [0.04728132387706856, 52, 24, 15, 1, 1.0],
#        [0.07476359338061465, 52, 24, 15, 2, 1.0],
#        [0.00604454363695541, 52, 35, 17, 0, 4.0],
#        [0.022318314967219977, 52, 35, 17, 1, 4.0],
#        [0.03529083554191659, 52, 35, 17, 2, 4.0],
#        [0.003124098817648755, 52, 22, 34, 0, 3.0],
#        [0.011535134095933866, 52, 22, 34, 1, 3.0],
#        [0.018239930789195426, 52, 22, 34, 2, 3.0]]
# # for i in dict.keys():
# #     list.extend(dict[i])
# # print(list)
# classify=3
# dict={}
# for i in range(len(list)//9):
#     list_iou=[]
#     start=i*3
#     for j in range(0+start, len(list), 3*classify):
#         list_iou.extend(list[j:j + 3])
#     dict[i]=list_iou
# print(dict)


#IOU_Softmax处理
import Yolo_V3.Yolo_Tools as tools

# a = np.array([[[[0.1, 105.49066, 241.82239, 138.03465, 195.94635, 8.],
#               [0.5, 189.62878, 203.15657, 129.86807, 82.68615, 5.],
#                [0.9, 189.62878, 203.15657, 129.86807, 82.68615, 5.]],
#
#               [[0.3, 73.774704, 191.63197, 115.89243, 81.98269, 5.],
#               [0.7, 328.54578, 145.71254, 143.17365, 165.46413, 7.],
#                [0.6, 189.62878, 203.15657, 129.86807, 82.68615, 5.]]]])
# # print(a.shape)
# iou_dic={0.1:[0,0,0],0.5:[0,0,1],0.9:[0,0,2],0.3:[0,1,0],0.7:[0,1,1],0.6:[0,1,2]}
# # print(a[0,1,1])
# iou_list = []
# for iou_ in iou_dic.keys():
#     iou_list.append(iou_)
# print(iou_list)
# iou_tensor = torch.Tensor(iou_list)
# iou_softmax = tools.Softmax_Deal(iou_tensor)
# print(iou_softmax)
# for i, iou_ in enumerate(iou_dic.keys()):
#     # print(i,iou_)
#     index = iou_dic[iou_]
#     #print(index)
#     a[int(index[0]), int(index[1]), index[2],0] = iou_softmax[i]
# print(a)

#独热编码softmax处理
# a = np.array([[0.1, 0.1, 0.8, 0.2, 0.15, 0.12],
#               [0.5, 0.1, 0.9, 0.2, 0.15, 0.12],
#                [0.9, 0.1, 0.7, 0.2, 0.15, 0.12],
#               [0.3, 0.1, 0.85, 0.2, 0.15, 0.1],
#               [0.7, 0.1, 0.86, 0.2, 0.15, 0.12],
#                [0.6, 0.1, 0.92, 0.2, 0.15, 0.12]])
# a=torch.Tensor(a)
# b=tools.Softmax_Deal(a[...,1:])
# print(b.size(0),b)
# for i in range(b.size(0)):
#     a[i,1:]=b[i]
# print(a)

#数据转图片
# import matplotlib.pyplot as plt
# import os
#
# img_path=r"F:\Yolo_Datasets\原图"
#
# plt.ion()
# for img_ in os.listdir(img_path):
#     img_open=Image.open(os.path.join(img_path,img_))
#     img_open.convert("RGB")
#     img_np = np.array(img_open)
#     img_tor = torch.Tensor(img_np)
#     # print(img_data)
#     img_back = np.array(img_tor, dtype=np.uint8)
#     img_back = Image.fromarray(img_back, "RGB")
#     img_draw = ImageDraw.Draw(img_back)
#
#     plt.clf()
#     plt.imshow(img_open)
#     plt.pause(0.5)
# plt.ioff()

#取放
# mask_have = feature_map[..., 0] > 0
# out_have = out[mask_have]
# feature_have = feature_map[mask_have]
#
# # print(out_have.shape,out_have)
# # b = tools.Softmax_Deal(out_have[..., 5:])
# # for i in range(b.size(0)):
# #     out_have[i, 5:] = b[i]
#
# mask_none = feature_map[..., 0] == 0
# out_none = out[mask_none]
# feature_none = feature_map[mask_none]