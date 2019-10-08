import numpy
import torch
import torch.utils.data as data
import torch.nn as nn
from Yolo_V3.Yolo_Dataset import Y_Datasets
from Yolo_V3.Yolo_Net import DarkNet
import os
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import matplotlib.pyplot as plt
import Yolo_V3.Yolo_Tools as tools
import cv2

def Loss_design(out, feature_map, x):
    out = out.permute(0, 2, 3, 1)
    out = torch.reshape(out, shape=(out.size(0), out.size(1), out.size(2), 3, -1))

    mask_have = feature_map[..., 0] > 0
    mask_none = feature_map[..., 0] == 0

    loss_Logit = nn.MSELoss()
    loss_conf_h = loss_Logit(out[mask_have][..., 0], feature_map[mask_have][..., 0])
    loss_conf_n = loss_Logit(out[mask_none][..., 0], feature_map[mask_none][..., 0])
    loss_conf = loss_conf_h + (1 * loss_conf_n)

    loss_MSE = nn.MSELoss()
    loss_data_h = loss_MSE(out[mask_have][..., 1:5], feature_map[mask_have][..., 1:5])

    loss_class=nn.CrossEntropyLoss()
    #Softmax = nn.Softmax()
    loss_class_h = loss_class(out[mask_have][..., 5:], feature_map[mask_have][..., 5].long())
    # loss_class_n= loss_MSE(out[mask_none][..., 5:], feature_map[mask_none][..., 5:])
    Loss = loss_conf + loss_data_h + loss_class_h

    return Loss

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
    box_center=torch.stack([conf.float(),cx.float(),cy.float(),w.float(),h.float(),classify_num.float()],dim=1)
    return box_center

def Supervision(img_data,out_13,out_26,out_52,turn):
    out_13, out_26, out_52 =out_13.cpu().data, out_26.cpu().data, out_52.cpu().data
    archor = tools.Archor()
    box_13 = Select_Data(out_13, 32, archor[13])
    box_26 = Select_Data(out_26, 16, archor[26])
    box_52 = Select_Data(out_52, 8, archor[52])
    boxes = torch.cat([box_13, box_26, box_52], dim=0)
    boxes = boxes.numpy()
    boxes = tools.NMS(boxes, 0.3)
    #print(save_boxes)

    dict_reverse = {0: '狗', 1: '人', 2: '羊驼', 3: '汽车', 4: '自行车', 5: '海豚', 6: '松鼠', 7: '马', 8: '马'}
    img_data=np.array((img_data+0.5)*255,dtype=np.uint8)
    img_data=np.transpose(img_data[0],(1,2,0))

    img_back=Image.fromarray(img_data,"RGB")
    img_draw=ImageDraw.Draw(img_back)
    Font=ImageFont.truetype((r"F:\jkl\Yolo\simkai.ttf"),20)

    #if turn>300:
    plt.ion()
    plt.clf()
    for box in boxes:
        img_draw.point((float(box[1]), float(box[2])), fill="red")
        img_draw.rectangle((int(box[1] - box[3] / 2), int(box[2] - box[4] / 2), int(box[1] + box[3] / 2),
                            int(box[2] + box[4] / 2)), outline="red", width=2)
        img_draw.text((int(box[1] - box[3] / 2), int(box[2] - box[4] / 2)),
                      text=dict_reverse[int(box[5])] + "{:.3f}".format(box[0]),
                      fill=(255, 0,0), font=Font)
    plt.imshow(img_back)
    img_back.save(r"F:\Reply\YoloV3\Vedio_Ready\{}.jpg".format(turn))
    plt.pause(0.1)
    plt.ioff()

    # img_read=cv2.imread(img_back)
    # cv2.imshow("",img_read)
    # cv2.imwrite(r"F:\答辩\YoloV3\Vedio_Ready\{}.jpg".format(turn),img_read)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    img_path = r"F:\Datasets_Dispose\Yolo_Datasets\Wen"
    labels_path = r"F:\Datasets_Dispose\Yolo_Datasets\label_02_21.txt"
    net_path = r"F:\jkl\Yolo_V3\Net_Save\Yolo_net_wen.pth"

    datasets = Y_Datasets(labels_path, img_path, all_num=9)
    train_data = data.DataLoader(dataset=datasets, batch_size=1, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DarkNet(all_num=9)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters())

    # from torch.utils.tensorboard import SummaryWriter
    # summarywriter=SummaryWriter()

    if os.path.exists(net_path):
        net = torch.load(net_path)

    turn = 0
    while True:
        turn += 1
        for i, (img_data, feature_13, feature_26, feature_52) in enumerate(train_data):
            img_data_cuda = img_data.to(device)
            feature_13, feature_26, feature_52 = \
                feature_13.to(device), feature_26.to(device), feature_52.to(device)
            out_13, out_26, out_52 = net(img_data_cuda)

            Supervision(img_data,out_13,out_26,out_52,turn)
            # print(out_13.shape,out_13)
            # out_13, out_26, out_52=out_13.cpu().data, out_26.cpu().data, out_52.cpu().data
            archor=tools.Archor()
            loss_13 = Loss_design(out_13, feature_13, 0.9)
            loss_26 = Loss_design(out_26, feature_26, 0.9)
            loss_52 = Loss_design(out_52, feature_52, 0.9)

            Loss = loss_13 + loss_26 + loss_52

            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            print("{0}轮  --  Loss:{1}".format(turn, Loss.cpu().float()))
            # summarywriter.add_scalar("loss",Loss,global_step=turn)
            # summarywriter.add_histogram("weight",net.Dar_52[0].layer[0].weight.data,global_step=turn)

        if turn % 20 == 0:
            torch.save(net, net_path)
            print("Save Successfully!")

        # summarywriter.close()