from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class MyDataset(CocoDataset):

    CLASSES =('长马甲', '古风', '短马甲', '背心上衣', '背带裤', '连体衣', '吊带上衣', '中裤', '短袖衬衫', '无袖上衣',
                 '长袖衬衫', '中等半身裙', '长半身裙', '长外套', '短裙', '无袖连衣裙', '短裤', '短外套',
                 '长袖连衣裙', '长袖上衣', '长裤', '短袖连衣裙', '短袖上衣')
    # CLASSES = (
    #     'chang ma jia',
    #      'gu feng',
    #      'duan ma jia',
    #      'bei xin shang yi',
    #      'bei dai ku',
    #      'lian ti yi',
    #      'diao dai shangyi',
    #      'zhong ku',
    #      'duan xiu chen shan',
    #      'wu xiu shang yi',
    #      'chang xiu chen shan',
    #      'zhong deng ban shen qun',
    #      'chang ban shen qun',
    #      'chang wai tao',
    #      'duan qun',
    #      'wu xiu lian yi qun',
    #      'duan ku',
    #      'duan wai tao',
    #      'chang xiu lian yi qun',
    #      'chang xiu shang yi',
    #      'chang ku',
    #      'duan xiu lian yi qun',
    #      'duan xiu shang yi'
    # )