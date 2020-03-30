import ffmpeg
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms


def get_img(img_path):
    img = cv2.imread(img_path)
    return img


def get_frame_from_video(in_file, frame_num):
    """
    指定帧数读取任意帧
    """
    out, err = (
        ffmpeg.input(in_file)
            .filter('select', 'gte(n,{})'.format(frame_num))
            .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True)
    )
    if out is None:
        print("cannot find " + str(frame_num) + " in " + in_file)
    image_array = np.asarray(bytearray(out), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


# numpy img —> tensor   https://blog.csdn.net/qq_36955294/article/details/82888443
def toTensor(img):
    img = torch.from_numpy(img.transpose((0, 3, 1, 2)))
    return img.float().div(255).unsqueeze(0)


# tensor img —> numpy
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


def img2match_torch(img: np):
    img = cv2.resize(img, (105, 105))
    transform1 = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ]
    )
    img = transform1(img)
    if len(img.shape) == 3:  # 转化为batch
        img_a = img.unsqueeze(0)
    return img


def get_max_bbox(all_bbox):
    """
    寻找概率最大的框
    :param all_bbox: 是array，格式是(N,5),N个满足条件的框
                    每个框与5个值，前4个是位置信息，最后一个是概率值 0-1
    :return: 最大的框的array，前4个是位置信息，最后一个是概率值 0-1
    """
    all_bbox=np.array(all_bbox)
    probablity_list=all_bbox[:,4]
    max_index=np.array(probablity_list)
    return list(all_bbox[max_index])