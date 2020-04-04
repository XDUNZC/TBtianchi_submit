from read_dataset import Reader
from save import Saver
from model.Resnet50.run import Worker as MatchWorker
from model.mmdetection_coco import run as DetectionWorker
import utils
import os
import mmcv
from classify_images import Classifier
import random

def main():
    # 初始化文件路径获得类
    reader = Reader(test_dataset_path='tcdata/',
                    img_path='tcdata/test_dataset_3w/image/',
                    video_path='tcdata/test_dataset_3w/video/')
    print("success init reader")
    # 初始化结果保存类
    saver = Saver()
    print("success init saver")
    # 执行匹配工作
    """初始化匹配模型"""
    # TODO 替换参数
    match_worker = MatchWorker(model_path='./model/Resnet50/models/model-inter-500001.pt')
    print("success load match model")
    """初始化获得框模型"""
    idx = 0
    config_file = ['./model/mmdetection_coco/configs/tbtc_fater_rcnn_voc.py',
                   'tbtc_retinanet_voc.py', 'tbtc_feature_exteactor_faster_rcnn.py',
                   'tbtc_feature_exteactor_faster_rcnn.py'][idx]
    checkpoint_file = ['./model/mmdetection_coco/checkpoints/faster_rcnn_x101_64x4d_fpn_1x20200324-ba5926a5.pth',
                       'retinanet_x101_64x4d_fpn_1x20200322-53c08bb4.pth'][idx]

    # TODO 替换参数
    coco_model = DetectionWorker.get_model(config_file=config_file,
                                           checkpoint_file=checkpoint_file)
    print("success load detection model")
    
    # 建立分类器,对图像库进行分类
    classifier = Classifier(coco_model, reader)
    # reader = classifier.reader
    print("success build classifier")

    # # 显示分类结果,正式提交的时候请注释
    # print('显示分类结果:')
    # classifier.show_classify_result()
    # for comm in reader.commodity_index_list:
    #     print(comm,': ', reader.commodity_index2img_path_list[comm])
    #
    #
    # """逐个视频运行""" 可以考虑切片,先只检测前1000个视频,节省时间
    print('开始逐视频进行检测: ')
    for video_path in mmcv.track_iter_progress(reader.video_path_list[:500]):
        video_index = video_path.split('/')[-1].split('.')[0]
        # print("匹配video，num=", video_index)
        max_match_value = -999999999
        max_match_video_frame_index = None
        max_match_commodity_index = None
        max_match_commodity_img_index = None
        max_match_commodity_img_path = None
        # 寻找最大匹配项
        video = mmcv.VideoReader(video_path)
        labels_in_this_video = {i: 0 for i in range(23)}
        frame_boxes_label_result = {}
        frame_index_list = list(range(0, 400, 80)) # 均匀采集5张图片
        for frame_index in frame_index_list:  # 逐个视频帧检测
            video_frame = video[frame_index]  # 获取视频帧
            result_over_thr, labels_over_thr, _ = DetectionWorker.get_result_and_feats(coco_model, video_frame) # 检测视频帧
            # 保存目标检测结果
            frame_boxes_label_result[frame_index]=(result_over_thr, labels_over_thr)
            # 记录检测出类别出现的频次
            for label in labels_over_thr:
                labels_in_this_video[label]+=1
        labels_in_this_video_list = sorted(labels_in_this_video.items(), key=lambda x: x[1], reverse=True)[:2] # [(1,12),(3,14)]
        # 删除调可能频次为0的类标
        for label,nums in labels_in_this_video_list:
            if nums==0:
                labels_in_this_video_list.remove((label,nums))
        labels_in_this_video_list = [i[0] for i in labels_in_this_video_list] # [1,3]
        # 如果该视频没有类标,则没有结果,跳过检测
        if len(labels_in_this_video_list)==0:
            # 从视频中检测不出类别 跳过这个视频
            continue
        else:
            # 进行匹配
            # 从5帧中选出2帧作为代表帧
            present_frames = []
            random.shuffle(frame_index_list)
            for frame_index in frame_index_list:
                result_over_thr, labels_over_thr = frame_boxes_label_result[frame_index]
                if [x for x in labels_in_this_video_list if x in labels_over_thr] != []:
                    present_frames.append(frame_index)
                if len(present_frames) == 2:  # 控制选择几帧
                    break
            # 生成待选的商品库
            candida_commoditys = set()
            for label in labels_in_this_video_list:
                # 将集合合并 生成候选匹配商品集
                candida_commoditys = candida_commoditys.union(classifier.class2commoditys[label])
            if len(candida_commoditys)==0:
                continue
            for frame_index in present_frames:
                video_frame = video[frame_index]
                for commodity_index in candida_commoditys:  # 逐个商品扫描
                    commodity_img_path_list = reader.commodity_index2img_path_list[commodity_index]
                    for img_path in commodity_img_path_list:  # 逐个图片
                        ci_index = img_path.split('/')[-1].split('.')[0]
                        commodity_img = utils.get_img(img_path)
                        video_frame_tensor = utils.img2match_torch(video_frame)
                        commodity_img_tensor = utils.img2match_torch(commodity_img)
                        match_value = match_worker.get_match_value(video_frame_tensor, commodity_img_tensor)  # 进行匹配
                        if match_value > max_match_value:  # 如果出现新的最大值，替换
                            max_match_value = match_value
                            max_match_video_frame_index = frame_index
                            max_match_commodity_index = commodity_index
                            max_match_commodity_img_index = str(ci_index)
                            max_match_commodity_img_path = img_path
            # 如果没有匹配到商品,跳过该视频
            if max_match_video_frame_index == None:
                continue
            # 保存匹配结果
            saver.save_match(video_index,
                             max_match_commodity_index,
                             max_match_video_frame_index,
                             max_match_commodity_img_index)
            # print("finish " + video_index + " match")
            # 进行画框
            # print(max_match_video_frame_index,"->",max_match_commodity_index)
            # print(max_match_video_frame)
            video_bbox, video_labels = frame_boxes_label_result[max_match_video_frame_index]
            ci_bbox, ci_labels = classifier.img_boxes_label_result[max_match_commodity_img_path]
            # print(video_bbox)
            # print(ci_bbox)

            # todo 优化生成最终提交框的方式
            # video_bbox,ci_bbox = utils.get_max_probability_box(video_bbox, video_labels, ci_bbox, ci_labels)
            video_bbox = utils.get_max_bbox(video_bbox)
            ci_bbox = utils.get_max_bbox(ci_bbox)

            # 保存画框结果
            saver.save_video_box(video_index,
                                 max_match_video_frame_index,
                                 video_bbox[:4])
            saver.save_item_box(video_index,
                                max_match_commodity_index,
                                max_match_commodity_img_index,
                                ci_bbox[:4])
            # print("finish " + video_index)
    """写出保存结果"""
    saver.write()
    print("successful finish all test")


if __name__ == "__main__":
    # success run
    # print("successful open run.py")

    main()

    # end run
    print("successful end run.py")
