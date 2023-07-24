from custom_yolo import SwitchYOLO

if __name__ == "__main__":
    modelNormalLight = SwitchYOLO("./yolov8l-normal-coco/best.pt")
    results = modelNormalLight.predict(
        source="/home/rama/data rama/thesis/object detection/yolov8/dataset-gabungan/test/images",
        #source="/home/rama/data rama/thesis/switch YOLO/low-light-test.mp4",
        switch_model='vgg16',
        #switch_model='efficientformerv2_l',
        #switch_model='efficientformerv2_s2',
        save=True,
        conf=0.5,
        device=1,
        name='predict-yolov8-switchVGG16-test-gabungan',
    )