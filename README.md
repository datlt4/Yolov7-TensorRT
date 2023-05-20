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

# Result

![image](build/saved.jpg)
