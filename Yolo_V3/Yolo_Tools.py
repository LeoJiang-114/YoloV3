import torch
import numpy as np
import os
from PIL import Image

dict_label={"狗":0,"人":1,"羊驼":2,"汽车":3,"自行车":4,"海豚":5,"松鼠":6,"马":7,"猫":8}
def Trans_img(img):
    img_data=np.array(img)
    img_data=np.transpose(img_data,(2,0,1))/255-0.5
    img_data=torch.Tensor(img_data)

    return img_data

def Archor():
    archor={
        13:[[116,90], [156,198],[373,326]],
        26:[[30,61],  [62,45],  [59,119]],
        52:[[10,13],  [16,30],  [33,23]]}

    return archor

def Archor_Area():
    Area = {
        13: [w * h for w, h in Archor()[13]],
        26: [w * h for w, h in Archor()[26]],
        52: [w * h for w, h in Archor()[52]]}

    return Area

def IOU_forlabel(box,W_H):
    box=list(box)
    W_H=list(W_H)
    cx=float(box[1])
    cy=float(box[2])
    box_w=int(box[3])
    box_h=int(box[4])
    archor_w=int(W_H[0])
    archor_h=int(W_H[1])

    b_x1 = int(cx - box_w / 2)
    b_y1 = int(cy - box_h / 2)
    b_x2 = int(cx + box_w / 2)
    b_y2 = int(cy + box_h / 2)

    archor_x1 = int(cx - archor_w / 2)
    archor_y1 = int(cy - archor_h / 2)
    archor_x2 = int(cx + archor_w / 2)
    archor_y2 = int(cy + archor_h / 2)

    x_max=max(b_x1,archor_x1)
    y_max=max(b_y1,archor_y1)
    x_min=min(b_x2,archor_x2)
    y_min=min(b_y2,archor_y2)

    inter=(x_min-x_max)*(y_min-y_max)
    iou=inter/(box_w*box_h+archor_w*archor_h-inter)
    return iou

def One_Hot(all_num,classify_num):
    zero=np.zeros(all_num)
    zero[classify_num]=1
    return zero

def IOU_forNMS(First_box,Backward_boxes,integrate=False):
    F_x1 = First_box[1] - First_box[3] / 2
    F_y1 = First_box[2] - First_box[4] / 2
    F_x2 = First_box[1] + First_box[3] / 2
    F_y2 = First_box[2] + First_box[4] / 2
    First_Area=(F_x2-F_x1)*(F_y2-F_y1)

    B_x1 = Backward_boxes[:,1] - Backward_boxes[:, 3] / 2
    B_y1 = Backward_boxes[:,2] - Backward_boxes[:, 4] / 2
    B_x2 = Backward_boxes[:,1] + Backward_boxes[:, 3] / 2
    B_y2 = Backward_boxes[:,2] + Backward_boxes[:, 4] / 2
    Backward_Area=(B_x2-B_x1)*(B_y2-B_y1)

    inter_x1 = np.maximum(F_x1, B_x1)
    inter_y1 = np.maximum(F_y1, B_y1)
    inter_x2 = np.minimum(F_x2, B_x2)
    inter_y2 = np.minimum(F_y2, B_y2)
    inter_Area=np.maximum(0,inter_x2-inter_x1)*np.maximum(0,inter_y2-inter_y1)

    if integrate:
        IOU_Result=np.true_divide(inter_Area,np.minimum(First_Area,Backward_Area))
    else:
        IOU_Result=np.true_divide(inter_Area,(First_Area+Backward_Area-inter_Area))
    return IOU_Result

def NMS(boxes,conf_judge,integrate=False):
    if boxes.shape[0] == 0:
        return np.array([])
    index=(-boxes[:,0]).argsort()
    _boxes=boxes[index]
    save_boxes=[]

    while _boxes.shape[0]>1:
        First_box=_boxes[0]
        # print(First_box.shape)
        Backward_boxes=_boxes[1:]
        # print(Backward_boxes.shape)
        save_boxes.append(First_box)
        IOU_Result=IOU_forNMS(First_box,Backward_boxes,integrate)
        index=np.where(IOU_Result<conf_judge)
        # print(index)
        _boxes=Backward_boxes[index]
        # print(_boxes.shape)
        # print(_boxes)
    if _boxes.shape[0]>0:
        save_boxes.append(_boxes[0])
    # print(save_boxes)
    save_boxes=np.stack(save_boxes)
    return torch.Tensor(save_boxes)

def Max_Deal(x):
    y=torch.argmax(x)
    x[y]=1
    x[0:y],x[y+1:]=0,0
    return x
    #return torch.exp(20*x)/torch.sum(torch.exp(20*x))

def IOU_Deal(iou_dic,boxes,feature_data):
    list_ = []
    for i in iou_dic.keys():
        list_.append(iou_dic[i])
    # print(list_)
    dict_ = {}
    for i in range(len(list_) // 9):
        list_iou = []
        start = i * 3
        for j in range(0 + start, len(list_), 3 * len(boxes)):
            list_iou.extend(list_[j:j + 3])
        # print(list_iou)
        dict_[i] = torch.Tensor(list_iou)
    # print(dict_)
    for i in dict_.keys():
        iou_softmax = Max_Deal(dict_[i][:, 0])
        # print(dict_[i][:,0])
        for j, iou in enumerate(iou_softmax):
            feature_data[int(dict_[i][j, 1])][int(dict_[i][j, 2]), int(dict_[i][j, 3]), int(dict_[i][j, 4]), 0] = float(iou)

    return feature_data


if __name__ == '__main__':
    # area=Archor_Area()
    # a,b,c=area
    # print(a,b,c)

    # box1=[  1., 127., 193.,  94., 108.]
    # W_H1=[10, 13]
    # iou=IOU_forlabel(box1,W_H1)
    # print(iou)

    # one_hot=One_Hot(9,5)
    # print(one_hot)

    # for feature_size,W_H in Archor().items():
    #      print(feature_size,W_H)

    # x=torch.randint(0,9,(9,))
    # print(x)
    y=Max_Deal(torch.Tensor([2, 7, 8, 1, 5, 2, 7, 6, 1]))
    print(y)