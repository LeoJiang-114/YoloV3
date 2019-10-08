import os
import linecache
from PIL import Image,ImageDraw
import numpy as np

img_path=r"F:\Yolo_Datasets\pic_deal"
labelYolo_path=r"F:\Yolo_Datasets\Labels_Yolo.txt"
labelxy_path=r"F:\Yolo_Datasets\Labels_xy.txt"

pic_name=9

xy_labels=open(labelxy_path,"r").readlines()
Yolo_labels=open(labelYolo_path,"r").readlines()

img_open=Image.open(os.path.join(img_path,linecache.getline(labelxy_path,pic_name).split()[0]))
img_draw=ImageDraw.Draw(img_open)
xy_data=linecache.getline(labelxy_path,pic_name).split()[1:]
Yolo_data=linecache.getline(labelYolo_path,pic_name).split()[1:]
xy=[]
yolo=[]

for i in xy_data:
    xy.append(int(i))
for i in Yolo_data:
    yolo.append(float(i))
xy=np.array(xy)
xy=np.split(xy,len(xy)//5)
yolo=np.array(yolo)
yolo=np.split(yolo,len(yolo)//5)
print(xy)
print(yolo)
for data in xy:
    img_draw.rectangle((int(data[1]),int(data[2]),int(data[3]),int(data[4])),outline="red",width=1)
for data in yolo:
    img_draw.point((data[1], data[2]), fill="red")
img_open.show()
