# 结果保存
import json
import os


class Saver:
    # 结果保存器
    def __init__(self):
        # 结果字典
        self.save = {}

    # 保存图片的结果
    def save_match(self, video_id, item_id, frame_index, img_name, cover_warning=True):
        # 判断有没有这个video_id，如果有就覆盖
        if cover_warning:
            if video_id in self.save:
                print("WARNING:Results coverage:" + video_id + ".frame_index:"
                      + self.save[video_id]["frame_index"] + "->" + frame_index)
                print("WARNING:Results coverage:" + video_id + ".item_id:"
                      + self.save[video_id]["item_id"] + "->" + item_id)
                print("WARNING:Results coverage:" + video_id + ".result.img_name:"
                      + self.save[video_id]["img_name"] + "->" + img_name)
        self.save[video_id] = {}
        self.save[video_id]["item_id"] = item_id
        self.save[video_id]["frame_index"] = frame_index
        self.save[video_id]["result"] = [{
            "img_name": img_name,
            "item_box": None,
            "frame_box": None
        }]

    def save_video_box(self, video_id,
                       frame_index,
                       # item_id,
                       # img_name,
                       frame_box,
                       check=True,
                       cover_warning=True):
        # 检查石佛普存在
        if not (video_id in self.save):
            raise Exception("ERROR:save " + video_id + " box error, cannot find this video id")
        # 判断内容是否对应
        if check:
            assert self.save[video_id]["frame_index"] == frame_index
            # assert self.save[video_id]["item_id"] == item_id
            # assert self.save[video_id]["result"][0]["img_name"]==img_name
        # 覆盖检查&警告
        if cover_warning:
            if not (self.save[video_id]["result"][0]["frame_box"] is None):
                print("WARNING:Results coverage:" + video_id + ".result.frame_index:"
                      + self.save[video_id]["frame_index"]
                      + self.save[video_id]["result"][0]["frame_box"] + "->" + frame_box)
        # 写入
        self.save[video_id]["result"][0]["frame_box"] = frame_box

    def save_item_box(self, video_id,
                      # frame_index,
                      item_id,
                      img_name,
                      item_box,
                      check=True,
                      cover_warning=True):
        # 检查石佛普存在
        if not (video_id in self.save):
            raise Exception("ERROR:save " + video_id + " box error, cannot find this video id")
        # 判断内容是否对应
        if check:
            # assert self.save[video_id]["frame_index"] == frame_index
            assert self.save[video_id]["item_id"] == item_id
            assert self.save[video_id]["result"][0]["img_name"] == img_name
        # 覆盖检查&警告
        if cover_warning:
            if not (self.save[video_id]["result"][0]["item_box"] is None):
                print("WARNING:Results coverage:" + video_id + ".result.img_index:"
                      + self.save[video_id]["item_index"]
                      + self.save[video_id]["result"][0]["item_box"] + "->" + item_box)
        # 写入
        self.save[video_id]["result"][0]["item_box"] = item_box

    def write(self, check=True):
        if os.path.exists('result.json') and check:
            print("WARNING: Cover result")
        with open('result.json', 'w') as f:
            f.write(json.dumps(self.save))
            print("Finish write result.json")
