import os


class Reader:
    def __init__(self,
                 test_dataset_path='/tcdata/',
                 img_path='/tcdata/test_dataset_3w/image/',
                 video_path='/tcdata/test_dataset_3w/video/'):
        # 初始化路径
        self.test_dataset_path = test_dataset_path
        self.img_path = img_path
        self.video_path = video_path
        # 判断路径是否存在
        if not os.path.exists(self.test_dataset_path):
            raise Exception("No test path:", self.test_dataset_path)
        if not os.path.exists(self.img_path):
            raise Exception("No test path:", self.img_path)
        if not os.path.exists(self.test_dataset_path):
            raise Exception("No test path:", self.test_dataset_path)

        # 读取直播片段
        self.video_name_list = os.listdir(self.video_path)  # 直播片段视频名
        self.video_path_list = [os.path.join(self.video_path, file) for file in self.video_name_list]  # 直播片段地址
        print("finish read video")
        # 读取商品类数
        _, self.commodity_index_list, _ = os.walk(self.img_path)  # 商品标签名
        self.img_dir_path_list = []  # 商品标签文件地址
        self.commodity_index2img_path_list = {}  # 根据商品标签索引商品图片地址
        for commodity_index in self.commodity_index_list:
            img_dir_path = os.path.join(self.img_path, commodity_index)
            self.img_dir_path_list.append(img_dir_path)
            img_file_in_dir_list = os.listdir(img_dir_path)
            img_file_in_dir_path_list = [os.path.join(img_dir_path, file) for file in img_file_in_dir_list]
            self.commodity_index2img_path_list[commodity_index] = img_file_in_dir_path_list
        print("finish read img")

    def get_img_id_num(self):
        """
        获得商品id数量
        """
        return len(self.commodity_index_list)

    def get_video_num(self):
        """
        获得视频id数量
        """
        return len(self.video_name_list)
