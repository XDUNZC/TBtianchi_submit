import ffmpeg
import numpy as np
import cv2


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


def get_img(img_path):
    img=cv2.imread(img_path)
    if img is None:
        print("cannot find "+str(img_path)+" image")
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