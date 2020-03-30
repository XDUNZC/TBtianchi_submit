from read_dataset import Reader
from save import Saver
from model.Resnet50.run import Worker as MatchWorker
from model.mmdetection_coco import run as DetectionWorker
import utils


def main():
    # 初始化文件路径获得类
    reader = Reader()
    print("success init reader")
    # 初始化结果保存类
    saver = Saver()
    print("success init saver")
    # 执行匹配工作
    """初始化匹配模型"""
    # TODO 替换参数
    match_worker = MatchWorker(model_path='/root/4tb/shaohon/TBtianchi_submit/model/Resnet50/models/model-inter-500001.pt')
    print("success load match model")
    """初始化获得框模型"""
    # TODO 替换参数
    coco_model = DetectionWorker.get_model(config_file="./model/mmdetection_coco/configs/my.py",
            checkpoint_file="./model/mmdetection_coco/checkpoints/retinanet_x101_64x4d_fpn_1x20200322-53c08bb4.pth")
    print("success load detection model")
    """逐个视频运行"""
    for video_index, video_path in enumerate(reader.video_path_list, start=0):
        video_index = reader.video_name_list[video_index]
        print("匹配video，num=", video_path)
        max_match_value = 0
        max_match_video_frame_index = None
        max_match_commodity_index = None
        max_match_commodity_img_index = None
        max_match_commodity_img = None
        max_match_video_frame = None
        # 寻找最大匹配项
        for frame_index in range(0, 400, 20):  # 逐个视频帧匹配
            video_frame = utils.get_frame_from_video(video_path, frame_index)  # 获取视频帧
            for commodity_index in reader.commodity_index_list:  # 逐个商品扫描
                commodity_img_path_list = reader.commodity_index2img_path_list[commodity_index]
                for ci_index, img_path in enumerate(commodity_img_path_list, 0):  # 逐个图片
                    commodity_img = utils.get_img(img_path)
                    video_frame_tensor=utils.img2match_torch(video_frame)
                    commodity_img_tensor=utils.img2match_torch(commodity_img)
                    match_value = match_worker.get_match_value(video_frame_tensor, commodity_img_tensor)  # 进行匹配
                    if match_value > max_match_value:  # 如果出现新的最大值，替换
                        max_match_value = match_value
                        max_match_video_frame_index = frame_index
                        max_match_commodity_index = commodity_index
                        max_match_commodity_img_index = str(ci_index)
                        max_match_video_frame = video_frame
                        max_match_commodity_img = commodity_img
        # 保存匹配结果
        saver.save_match(video_index,
                         max_match_commodity_index,
                         max_match_video_frame_index,
                         max_match_commodity_img_index)
        # 进行画框
        video_bbox = DetectionWorker.get_result_box(max_match_video_frame, coco_model)
        ci_bbox = DetectionWorker.get_result_box(max_match_commodity_img, coco_model)
        video_bbox=utils.get_max_bbox(video_bbox)
        ci_bbox=utils.get_max_bbox(ci_bbox)
        # 保存画框结果
        saver.save_video_box(video_index,
                             max_match_video_frame_index,
                             video_bbox[:5])
        saver.save_item_box(video_index,
                            max_match_commodity_img_index,
                            ci_bbox[:,5])
    """写出保存结果"""
    saver.write()
    print("successful finish all test")

if __name__ == "__main__":
    # success run
    print("successful open run.py")

    main()

    # end run
    print("successful end test.py")
