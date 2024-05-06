from ultralytics import YOLO, settings
import os
import multiprocessing

# setting
settings.update({
    'runs_dir': '.\\runs', 
    'datasets_dir': '.\\datasets'
})
os.environ["KMPDUPLICATELIBOK"]="TRUE"
continue_learning = True
print(settings)

#Train the model
if __name__ == '__main__':
    multiprocessing.freeze_support()
    if not continue_learning:
        #모델 새 학습
        model = YOLO('yolov8m-oiv7.pt') 
        results = model.train(data='./datasets/data.yaml', epochs=100 ,imgsz=640,
                                conf=0.1,
                                lr0 = 0.00269,
                                lrf = 0.00288,
                                momentum = 0.73375,
                                weight_decay = 0.00015,
                                warmup_epochs = 1.22935,
                                warmup_momentum = 0.1525,
                                box = 18.27875,
                                cls = 1.32899,
                                dfl = 0.56016,
                                hsv_h = 0.01148, 
                                hsv_s = 0.53554, 
                                hsv_v = 0.13636, 
                                translate = 0.12431,
                                scale = 0.07643,
                              )
    else:
        #이어서 모델 학습
        model = YOLO('best.pt')
        results = model.train(resume=True)
