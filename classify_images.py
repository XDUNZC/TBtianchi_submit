from read_dataset import Reader
from save import Saver
from model.Resnet50.run import Worker as MatchWorker
from model.mmdetection_coco import run as DetectionWorker
import utils
import os
import mmcv
import random

class Classifier():
    def __init__(self,classify_model,reader):
        self.classify_model = classify_model
        self.reader = reader
        self.class2commoditys = {i:set() for i in range(23)}
        self.img_boxes_label_result = {}
        self._classify_image()
    def _classify_image(self):
        print('开始检测所有商品图,并进行分类剪枝:')
        for commodity in mmcv.track_iter_progress(self.reader.commodity_index_list):
            labels_in_this_commodity = {i:0 for i in range(23)}
            imgs_in_this_commodity = list(self.reader.commodity_index2img_path_list[commodity])
            for img in imgs_in_this_commodity: 
                result_over_thr, labels_over_thr, _ = DetectionWorker.get_result_and_feats(self.classify_model, img)
                self.img_boxes_label_result[img] = (result_over_thr, labels_over_thr)
                for label in labels_over_thr:
                    labels_in_this_commodity[label]+=1
            labels_in_this_commodity_list = sorted(labels_in_this_commodity.items(), key=lambda x: x[1], reverse=True)[:2] # 取出现类标最多的两个
            for i,item  in enumerate(labels_in_this_commodity_list):
                label, appear_num = item
                if i!=0 and appear_num==0:
                    break
                self.class2commoditys[label].add(commodity) # 将商品加入到所属类标下
            # 选出具有代表性的图 剪枝商品图
            present_imgs = []
            random.shuffle(imgs_in_this_commodity)
            for img in imgs_in_this_commodity:
                result_over_thr, labels_over_thr = self.img_boxes_label_result[img]
                if [x for x in labels_in_this_commodity_list if x in labels_over_thr] != []:
                    present_imgs.append(img)
                if len(present_imgs) == 2 : # 控制选择几幅图
                    break
            self.reader.commodity_index2img_path_list[commodity] = present_imgs

    def show_classify_result(self):
        for label,commoditys in self.class2commoditys.items():
            print('lable: ',label,' commoditys: ',commoditys)


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
    # match_worker = MatchWorker(model_path='./model/Resnet50/models/model-inter-500001.pt')
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
    """逐个视频运行"""

    classifier = Classifier(coco_model,reader)
    print("success build classifier")

    # 显示分类结果,正式提交的时候请注释
    classifier.show_classify_result()




if __name__ == "__main__":
    # success run
    print("successful open run.py")

    main()

    # end run
    print("successful end test.py")
