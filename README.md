# Yolov7-TensorRT

## Build

```bash
cd build
cmake ..
make -j$(nproc)
```

## Convert pytorch weight to onnx

```bash
cd <yolov7-pytorch>
python export.py --weights ./yolov7.pt --grid --simplify --img-size 640 640
```

## Convert onnx to Tensorrt

```bash
./TrtExec/TrtExec-bin \
    --onnx /home/emoi/Downloads/Miscellaneous/Yolov7-TensorRT/yolov7-80-class.onnx \
    --engine /home/emoi/Downloads/Miscellaneous/Yolov7-TensorRT/yolov7-80-class-int8.engine \
    --dataset "/home/emoi/Downloads/Miscellaneous/Yolov7-TensorRT/val2017/*.jpg" \
    --inputName "images" \
    --minShape 1x3x640x640 \
    --optShape 1x3x640x640 \
    --maxShape 1x3x640x640 \
    --workspace 4096 --int8
```

```bash
./TrtExec/TrtExec-bin \
    --onnx /home/emoi/Downloads/Miscellaneous/Yolov7-TensorRT/yolov7-80-class.onnx \
    --engine /home/emoi/Downloads/Miscellaneous/Yolov7-TensorRT/yolov7-80-class-fp16.engine \
    --inputName "images" \
    --minShape 1x3x640x640 \
    --optShape 1x3x640x640 \
    --maxShape 1x3x640x640 \
    --workspace 4096 --fp16
```

```bash
/usr/local/TensorRT/bin/trtexec \
    --onnx=/home/emoi/Downloads/Miscellaneous/Yolov7-TensorRT/yolov7-80-class.onnx \
    --calib=/home/emoi/Downloads/Miscellaneous/Yolov7-TensorRT/build/calibration.cache \
    --saveEngine=/home/emoi/Downloads/Miscellaneous/Yolov7-TensorRT/yolov7-80-class-int8.engine \
    --int8
```

# Result

![image](build/saved.jpg)
