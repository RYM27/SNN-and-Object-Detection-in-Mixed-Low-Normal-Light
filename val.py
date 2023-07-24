from custom_yolo import SwitchYOLO

if __name__ == "__main__":
    modelNormalLight = SwitchYOLO("./yolov8l-normal-coco/best.pt")
    results = modelNormalLight.val(
        #data='/home/rama/data rama/thesis/object detection/yolov8/data/custom-exdark.yaml',
        data='/home/rama/data rama/thesis/object detection/yolov8/data/custom-coco.yaml',
        #data='/home/rama/data rama/thesis/object detection/yolov8/data/custom-gabungan.yaml',
        #switch_model='vgg16',
        #switch_model='efficientformerv2_l',
        switch_model='efficientformerv2_s2',
        imgsz=640,
        batch=1,
        device=1,
        split='test',
        name='yolov8-switchEFV2S2-test-coco',
    )