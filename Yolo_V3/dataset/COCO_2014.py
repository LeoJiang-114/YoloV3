import os
import cv2

def coco_():
    img_path = r"F:\COCO\train2014"
    lable_path = r"F:\COCO\labels\train2014"
    save_lable = r"F:\COCO\labels\lable_integraty.txt"
    # print(w,h,c)
    for lable in os.listdir(lable_path):
        list = []
        #list.extend([lable.split(".txt")[0] + ".jpg"])
        with open(os.path.join(lable_path, lable), "r") as f:
            img_open = cv2.imread(os.path.join(img_path, lable.split(".txt")[0] + ".jpg"))
            h, w, c = img_open.shape
            for line in f.readlines():
                line = line.split()
                id = line[0]
                x_ = float(line[1]) * w
                y_ = float(line[2]) * h
                w_ = float(line[3]) * w
                h_ = float(line[4]) * h
                # print(x_, y_, w_, h_)
                #list.append([str(x_), str(y_), str(w_), str(h_)])
                list.append([x_, y_, w_, h_])
        # with open(os.path.join(save_lable), "a") as w_f:
        #     for x in list:
        #         w_f.write("{0} ".format(x))
        #     w_f.write("\n")
        print(len(list),list)
        for box in list:
            x_, y_, w_, h_ = box
            cv2.rectangle(img_open, (round(x_ - w_ / 2), round(y_ - h_ / 2)), (round(x_ + w_ / 2), round(y_ + h_ / 2)),
                          color=(0, 0, 255))
            cv2.circle(img_open, (round(x_), round(y_)), 1, color=(0, 0, 255))
        cv2.imshow("", img_open)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


import json

def make_lable():
    lable_re = r"F:\Datasets_Original\COCO_2014\annotations\train2014.txt"
    with open(r"F:\Datasets_Original\COCO_2014\annotations\instances_train2014.json", "r", encoding="UTF-8") as f:
        f_open = json.load(f)
        #print(f_open.keys())  # dict_keys(['info', 'images', 'licenses', 'categories', 'annotations'])
        #print(f_open['annotations'][0].keys())  # ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
        # {'segmentation': {'counts': 'omh51Y=0ng31PXL01O10iW10ThN1PPY2', 'size': [426, 640]}, 'area': 11.0, 'iscrowd': 0, 'image_id': 139, 'bbox': [444.0, 226.0, 20.0, 11.0], 'category_id': 105, 'id': 20000002}

    with open(r"F:\Datasets_Original\COCO_2014\annotations\train2014_annotations.txt", "w") as w_f:
        for line in f_open['annotations']:
            w_f.write("{0}\n".format(line))

    with open(r"F:\Datasets_Original\COCO_2014\annotations\train2014_categories.txt", "w") as w_f:
        for line in f_open['categories']:
            w_f.write("{0}\n".format(line))
    with open(lable_re, "w") as w_f:
        img_id = 0  # 000000000139
        for line in f_open['annotations']:
            if line["image_id"] == img_id:
                # print(12-len(str(line['category_id'])))
                w_f.write("{0} {1} {2} {3} {4} "
                          .format(line['category_id'], str(line['bbox'][0]), str(line['bbox'][1]), str(line['bbox'][2]),
                                  str(line['bbox'][3])))
            else:
                img_id = line["image_id"]
                w_f.write("\n")
                w_f.write("{0}.jpg ".format("0" * (12 - len(str(line["image_id"]))) + str(img_id)))
                w_f.write("{0} {1} {2} {3} {4} "
                          .format(line['category_id'], str(line['bbox'][0]), str(line['bbox'][1]), str(line['bbox'][2]),
                                  str(line['bbox'][3])))
#print(f_open)

import linecache
import cv2 as cv
import os
def Show():
    img_path = r"F:\Datasets_Original\COCO_2014\train2014"
    lable_re = r"F:\Datasets_Original\COCO_2014\annotations\train2014.txt"
    line = linecache.getline(lable_re, 13).split()
    print(line)
    img_open = cv.imread(os.path.join(img_path, line[0]))
    for i in range((len(line) - 1) // 5):
        box = line[i * 5 + 1:i * 5 + 6]
        id = box[0]
        x = float(box[1])
        y = float(box[2])
        w = float(box[3])
        h = float(box[4])
        print(box)
        cv.putText(img_open, id, (int(x), int(y) + 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                   color=(0, 0, 255), thickness=2)
        cv.rectangle(img_open, (int(x), int(y)), (int(x + w), int(y + h)), color=(0, 0, 255))
    cv.namedWindow("", 0)
    cv.imshow("", img_open)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    coco_()
    #make_lable()
    #Show()