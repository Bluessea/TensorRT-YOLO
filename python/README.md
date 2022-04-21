# TensorRT-YOLO-Python
Deploy Yolo series algorithms on Jetson platform , including yolov5, yolox, etc

## PytorchToOnnx

**转换ONNX**

```bash
python export.py  --weights yolov5s.pt --opset 12 --simplify 
```

ONNX输出调整

yolov5 6.0 输出的list为4

```python
# 以输入640*640为例
（1,3,25,80,80）
（1,3,25,40,40）
（1,3,25,20,20）
（1,25200,25） # 三个维度拼接后，后处理部分在模型中计算
```

调整修改如下

```python
（1,3,80,80,25）
（1,3,40,40,25）
（1,3,20,20,25）
```

```python
   for i in range(self.nl):
       x[i] = self.m[i](x[i])  # conv
       bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,25)
       x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
       x[i] = x[i].sigmoid()  # 将sigmoid在GPU处理
   # 提前返回推理结果
   return x
```



## OnnxToEngine

Waiting...
