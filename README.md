# TR_infer_yolopose
本仓库是cpp下推理yolo的示例
## 关于onnx
onnx是一个标准的模型形式，我们可以用Netron来可视化onnx模型，重点关注输入输出
## 关于TensorRt模型
由onnx转化而成，有利于nvidia硬件下推理加速。由于要对硬件进行精度的补偿，所以转化过程要在推理设备上进行。
而非在训练模型的设备上进行。执行指令如下：
`trtexec --onnx=path_to_your_model.onnx --saveEngine=path_to_save_engine_file.engine`
## 关于代码
看注释很清楚