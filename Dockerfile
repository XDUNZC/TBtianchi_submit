# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/mmdetection:pytorch1.4-cuda10.1-py3

# 安装tensorflow1.14
# RUN pip install tensorflow-gpu==1.14
# 安装常用库

RUN apt update && apt-get install -y software-properties-common && apt update && add-apt-repository -y ppa:jonathonf/ffmpeg-4 && apt install -y ffmpeg

RUN pip install python-gflags python-ffmpeg ffmpeg ffmpeg-python opencv-python -i https://pypi.douban.com/simple/
RUN pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

# 安装mmdetection需求
RUN pip install -r /model/mmdetection_coco/requirements/build.txt -i https://pypi.douban.com/simple/
# RUN cd model/mmdetection_coco/ && python setup.py develop && cd ../..
# RUN pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
