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
cd build/TrtExec
./TrtExec-bin \
    --onnx /home/kikai/Documents/Yolov7-TensorRT/yolov7-80-class.onnx \
    --engine /home/kikai/Documents/Yolov7-TensorRT/yolov7-80-class-dla.engine \
    --inputName "images" \
    --minShape 1x3x640x640 \
    --optShape 1x3x640x640 \
    --maxShape 1x3x640x640 \
    --workspace 1024 \
    --useDLACore 0


./TrtExec-bin \
    --onnx /home/kikai/Documents/Yolov7-TensorRT/yolov5s.onnx \
    --engine /home/kikai/Documents/Yolov7-TensorRT/yolov5s-dla.engine \
    --inputName "images" \
    --minShape 1x3x640x640 \
    --optShape 1x3x640x640 \
    --maxShape 1x3x640x640 \
    --workspace 1024 \
    --useDLACore 0
```

# Result

![image](build/saved.jpg)
