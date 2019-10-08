# 读取图片
# 找到图片最长边
# 将最长边作为size，制作一张方形黑色背景
# 将图片贴在幕布中间
# 将贴好的图resize成416*416
# 将处理后的图片放到指定位置

from PIL import Image
import os


def pic_dispose(pic_path, pic_name, save_path):
    img = Image.open(os.path.join(pic_path, pic_name))
    w,h = img.size

    size = max(w,h)

    x = int((size-w)/2)
    y = int((size-h)/2)

    curtain = Image.new('RGB',(size,size),(255,255,255))
    curtain.paste(img,(x,y))
    # curtain.show()
    curtain = curtain.resize((416,416))
    curtain.save(os.path.join(save_path,pic_name))

if __name__ == '__main__':
    pic_path=r'F:\Yolo_Datasets'
    save_path = r'F:\Yolo_Datasets\pic_deal'
    for pic_name in os.listdir(r'F:\Yolo_Datasets'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pic_dispose(pic_path, pic_name, save_path)
