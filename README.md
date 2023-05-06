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
    --onnx /root/Software/Yolov7-TensorRT/yolov7-80-class.onnx \
    --engine /root/Software/Yolov7-TensorRT/yolov7-80-class-fp16.engine \
    --inputName "images" \
    --minShape 1x3x640x640 \
    --optShape 1x3x640x640 \
    --maxShape 1x3x640x640 \
    --workspace 2048
```

# Result

![image](build/saved.jpg)
