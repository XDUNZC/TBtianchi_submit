from .mmdet.apis import init_detector, inference_detector
import numpy as np
import torch.nn as nn


def get_model(config_file='configs/my.py',
              checkpoint_file='work_dirs/retinanet_x101_64x4d_fpn_1x/latest.pth',
              device='cuda:0'):
    model = init_detector(config_file, checkpoint_file, device=device)
    return model


def get_result_box(img_path, model, score_thr=0.7):
    '''

    :param img_path: 图像的路径
    :param model: 加载好的预训练模型
    :param score_thr: 后处理，只输出概率值大于thr的框
    :return: bboxes_over_thr 是array，输出格式是(N,5),N个满足条件的框
             每个框与5个值，前4个是位置信息，最后一个是概率值 0-1
    '''
    result = inference_detector(model, img_path)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    inds = np.where(bboxes[:, -1] > score_thr)[0]
    bboxes_over_thr = bboxes[inds]
    return bboxes_over_thr


# todo 目前这个只支持onestage模型
def get_feature_extractor(config_file='configs/tbtc_fater_rcnn_voc.py',
                          checkpoint_file='checkpoints/faster_rcnn_x101_64x4d_fpn_1x20200324-ba5926a5.pth',
                          device='cuda:0'):
    '''

    :param config_file: 训好的网络参数设置
    :param checkpoint_file: 训好的网络权重
    :param device : 指定卡
    :return: feature_extractor
    '''
    model = get_model(config_file=config_file, checkpoint_file=checkpoint_file, device=device)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    del (model)
    return feature_extractor


# todo 封装时注意一个问题：不要把模型model读进来初始化，检测完丢掉，再读进来，再检测完再丢掉。注意节省时间
if __name__ == '__main__':
    # demo 检测图片 注意替换文件路径
    # config_file = 'configs/my.py'  # 替换为指定的configs/xx.py
    # checkpoint_file = 'work_dirs/retinanet_x101_64x4d_fpn_1x/latest.pth'  # 替换为预训练权重
    # device = 'cuda:0'  # GPU 卡号
    # model = get_model(config_file=config_file, checkpoint_file=checkpoint_file, device=device)
    # img_path = 'demo/demo.jpg'
    # bboxes = get_result_box(img_path=img_path, model=model, score_thr=0.7) # array，输出格式是(N,5),N个满足条件的框 每个框与5个值，前4个是位置信息，最后一个是概率值 0-1
    # print(bboxes)

    # demo 生成feature_extractor
    img = 'demo/tbtc_test.py'
    feature_extractor = get_feature_extractor(config_file='configs/tbtc_retinanet_voc.py',
                                              checkpoint_file='checkpoints/retinanet_x101_64x4d_fpn_1x20200322-53c08bb4.pth',
                                              device='cuda:0')
    feature_maps = feature_extractor(img)
    print(feature_maps)
