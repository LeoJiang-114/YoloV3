import json
from PIL import Image,ImageDraw
import os
import linecache

json_path=r"F:\Yolo_Datasets\json_data"
labelYolo_path=r"F:\Yolo_Datasets\Labels_Yolo.txt"
labelxy_path=r"F:\Yolo_Datasets\Labels_xy.txt"
dict_label={"狗":0,"人":1,"羊驼":2,"小汽车":3,"自行车":4,"海豚":5,"松鼠":6,"马":7,"猫":8}
for json_file in os.listdir(json_path):
    with open(os.path.join(json_path,json_file), "r", encoding="UTF-8") as j_f:
        j_file = json.load(j_f)
        print(j_file)
    img_name = os.path.split(j_file['path'])[1]
    with open(labelxy_path, "a") as xy_f:
        xy_f.write("{0}".format(img_name))
    with open(labelYolo_path, "a") as Y_f:
        Y_f.write("{0}".format(img_name))
    # with open(label_path,"a") as l_f:
    #     l_file=json.load(l_f)
    # from json_data in os.listdir(json_path)
    for class_data in j_file['outputs']['object']:
        classify_name = class_data['name']
        classify_num = dict_label[classify_name]
        # print(verify_num)
        x1 = class_data['bndbox']['xmin']
        y1 = class_data['bndbox']['ymin']
        x2 = class_data['bndbox']['xmax']
        y2 = class_data['bndbox']['ymax']
        with open(labelxy_path, "a") as xy_f:
            xy_f.write(" {0} {1} {2} {3} {4}".format(str(classify_num), str(x1), str(y1), str(x2), str(y2)))

        cx = float((x2 - x1) / 2) + x1
        cy = float((y2 - y1) / 2) + y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        # data_Yolo.append([cx,cy,w,h])
        with open(labelYolo_path, "a") as Y_f:
            Y_f.write(" {0} {1} {2} {3} {4}".format(str(classify_num), str(cx), str(cy), str(w), str(h)))
    with open(labelxy_path, "a") as xy_f:
        xy_f.write("\n")
    with open(labelYolo_path, "a") as Y_f:
        Y_f.write("\n")
