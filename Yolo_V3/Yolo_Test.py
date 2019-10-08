import numpy as np
import torch
import os
from PIL import Image,ImageDraw,ImageFont
import Yolo_V3.Yolo_Tools as tools
import matplotlib.pyplot as plt

def Select_Data(out_result,size,archor):
    archor=torch.Tensor(archor)
    out_result = out_result.permute(0, 2, 3, 1)
    out_result = torch.reshape(out_result, shape=(out_result.size(0), out_result.size(1), out_result.size(2), 3, -1))
    mask = out_result[..., 0] > 0.9
    idxes = mask.nonzero()
    if idxes.shape[0] == 0:
        return torch.Tensor([])
    out_have = out_result[mask]
    # print(out_have.shape,out_have)

    x_index=idxes[:,2]
    y_index=idxes[:,1]
    archor_idx=idxes[:,3]

    conf = out_have[:, 0]
    cx=(x_index.float()+out_have[:,1])*size
    cy=(y_index.float()+out_have[:,2])*size
    # print(archor_idx.shape,archor_idx)
    w=archor[archor_idx,0]*torch.exp(out_have[:,3])
    h=archor[archor_idx,1]*torch.exp(out_have[:,4])
    classify_num = torch.argmax(out_have[:,5:],dim=1)
    # print(classify_num)
    # print([conf.float(),cx.float(),cy.float(),w.float(),h.float(),classify_num.float()])
    box_center=torch.stack([conf.float(),cx.float(),cy.float(),w.float(),h.float(),classify_num.float(),archor_idx.float()],dim=1)
    boxes = box_center.numpy()
    save_boxes = tools.NMS(boxes, 0.3)
    return save_boxes

if __name__ == '__main__':
    dict_reverse = {0: '狗', 1: '人', 2: '羊驼', 3: '汽车', 4: '自行车', 5: '海豚', 6: '松鼠', 7: '马', 8: '猫'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.load(r"F:\jkl\Yolo_V3\Net_Save\Yolo_net_wen.pth")
    net = net.to(device)
    font=ImageFont.truetype((r"F:\jkl\Yolo\simkai.ttf"),25)

    img_path = r"F:\Datasets_Dispose\Yolo_Datasets\Wen\1.jpg"
    # for img_name in os.listdir(img_path):
    img_open = Image.open(img_path)
    img_draw = ImageDraw.Draw(img_open)
    img_data = tools.Trans_img(img_open)
    img_data = torch.unsqueeze(img_data, dim=0)
    img_data = img_data.to(device)

    out_13, out_26, out_52 = net(img_data)
    out_13, out_26, out_52 = out_13.cpu().data, out_26.cpu().data, out_52.cpu().data
    # print(out_13.shape,out_13)
    archor = tools.Archor()
    box_13 = Select_Data(out_13, 32, archor[13])
    box_26 = Select_Data(out_26, 16, archor[26])
    box_52 = Select_Data(out_52, 8, archor[52])
    boxes = torch.cat([box_13, box_26, box_52], dim=0)
    boxes = boxes.numpy()
    save_boxes = tools.NMS(boxes, 0.3, True)
    print(save_boxes)
    plt.ion()
    plt.clf()
    for box in save_boxes.numpy():
        img_draw.point((float(box[1]), float(box[2])), fill="red")
        img_draw.rectangle(
            (int(box[1] - box[3] / 2), int(box[2] - box[4] / 2), int(box[1] + box[3] / 2), int(box[2] + box[4] / 2)),outline="red", width=2)
        img_draw.text((int(box[1] - box[3] / 2), int(box[2] - box[4] / 2)), text=dict_reverse[int(box[5])]+"{:.3f}".format(box[0]),fill=(255, 0, 0), font=font)#+"{:.3f}".format(box[0])
    img_open.save(r"F:\Reply\YoloV3\1\{}".format(os.path.split(img_path)[1]))
    plt.imshow(img_open)
    plt.pause(0)
    plt.ioff()