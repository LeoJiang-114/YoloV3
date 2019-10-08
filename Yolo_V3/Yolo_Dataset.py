import torch
import torch.utils.data as data
import numpy as np
import os
import Yolo_V3.Yolo_Tools as tools
import math
from PIL import Image,ImageDraw

class Y_Datasets(data.Dataset):
    def __init__(self,labels_path,img_path,all_num):
        super().__init__()
        self.img_path=img_path
        self.all_num=all_num
        self.datasets=[]
        with open(labels_path) as f:
            self.datasets.extend(f.readlines())


    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        feature_data={}
        line=self.datasets[0].split()
        img_name=line[0]
        line_data=[]
        img_open=Image.open(os.path.join(self.img_path,img_name))
        img_data=tools.Trans_img(img_open)

        for i in line[1:]:
            line_data.append(float(i))
        line_data=np.array(line_data)
        boxes=np.split(line_data,len(line_data)//5)

        iou_dic = {}
        for feature_size,W_H in tools.Archor().items():
            feature_data[feature_size]=np.zeros(shape=(feature_size,feature_size,3,5+self.all_num))
            # iou_dic[feature_size]=np.zeros(shape=(9,4))
            # print(W_H)
            for box in boxes:
                cx,cy=float(box[1]),float(box[2])
                cx_off,cx_index=math.modf(cx*feature_size/416)
                cy_off,cy_index=math.modf(cy*feature_size/416)
                w,h=int(box[3]),int(box[4])
                for i,archor_area in enumerate(tools.Archor_Area().items()):
                    iou=tools.IOU_forlabel(box,W_H[i])
                    # print(iou)
                    t_w = w / W_H[i][0]
                    t_h = h / W_H[i][1]
                    one_hot=tools.One_Hot(int(self.all_num),int(box[0]))
                    iou_dic[iou]=[iou,feature_size,int(cy_index),int(cx_index),i,box[0]]
                    #print(np.array([iou,cx_off, cy_off, np.log(t_w), np.log(t_h),*one_hot]))
                    feature_data[feature_size][int(cy_index),int(cx_index),i]=np.array(
                        [iou,cx_off, cy_off, np.log(t_w), np.log(t_h),*one_hot])
        # print(iou_dic)
        feature_data=tools.IOU_Deal(iou_dic,boxes,feature_data)

        return img_data,torch.Tensor(feature_data[13]),torch.Tensor(feature_data[26]),torch.Tensor(feature_data[52])

if __name__ == '__main__':
    label_path=r"F:\Datasets_Dispose\Yolo_Datasets\Labels_Yolo-1.txt"
    img_path=r"F:\Datasets_Dispose\Yolo_Datasets\pic_deal-1"
    data=Y_Datasets(label_path,img_path,9)
    img_data,feature_13,feature_26,feature_52=data[0]
    # print(img_data.shape)
    print("13",feature_13.shape)
    print("26", feature_26.shape)
    print("52", feature_52.shape)

    # out = feature_13.permute( 1, 2, 0)
    # out = torch.reshape(out, shape=(out.size(0), out.size(1), 3, -1))
    # print(out.shape,out)

    # mask_have = feature_13[..., 0] > 0
    # feature_have = feature_13[mask_have]
    # print( feature_have.shape,feature_have)
    # mask_have = feature_26[..., 0] > 0
    # feature_have = feature_26[mask_have]
    # print(feature_have.shape,feature_have)
    # mask_have = feature_52[..., 0] > 0
    # feature_have = feature_52[mask_have]
    # print(feature_have.shape,feature_have)
