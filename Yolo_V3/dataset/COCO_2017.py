import json

img_path=r"F:\Datasets_Original\COCO_2017\train2017"
lable_save=r"F:\Datasets_Original\COCO_2017\COCO_annotations2017\stuff_train2017.txt"
def make_lable():
    with open(r"F:\Datasets_Original\COCO_2017\COCO_annotations2017\stuff_train2017.json", "r", encoding="UTF-8") as f:
        f_open = json.load(f)
        #print(f_open.keys())  # dict_keys(['info', 'images', 'licenses', 'categories', 'annotations'])
        #print(f_open['annotations'][0].keys())  # ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
        # {'segmentation': {'counts': 'omh51Y=0ng31PXL01O10iW10ThN1PPY2', 'size': [426, 640]}, 'area': 11.0, 'iscrowd': 0, 'image_id': 139, 'bbox': [444.0, 226.0, 20.0, 11.0], 'category_id': 105, 'id': 20000002}

    with open(r"F:\Datasets_Original\COCO_2017\COCO_annotations2017\stuff_train2017_annotations.txt", "w") as w_f:
        for line in f_open['annotations']:
            w_f.write("{0}\n".format(line))

    with open(r"F:\Datasets_Original\COCO_2017\COCO_annotations2017\stuff_train2017_categories.txt", "w") as w_f:
        for line in f_open['categories']:
            w_f.write("{0}\n".format(line))
    with open(lable_save, "w") as w_f:
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
    line = linecache.getline(lable_save, 13).split()
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
    #make_lable()
    Show()