// #include <iostream>
// #include <fstream>
// #include <cassert>
// #include <string>
// #include <vector>
// #include <map>
// #include <functional>
// #include <cuda_runtime.h>
// #include <NvInfer.h>
// #include <NvOnnxParser.h>
// #include <memory>
// #include <opencv2/opencv.hpp>
// /**
//  * @brief 自定义日志器类，用于记录 TensorRT 的日志信息
//  */
// class Logger : public nvinfer1::ILogger {
// public:
//     /**
//      * @brief 实现日志记录函数
//      * @param severity 日志级别
//      * @param msg 日志信息
//      */
//     void log(Severity severity, const char* msg) noexcept override {
//         // 只记录警告及以上级别的日志
//         if (severity <= Severity::kWARNING) {
//             std::cout << msg << std::endl;
//         }
//     }
// };
//
// // 全局日志器实例
// static Logger gLogger;
//
// /**
//  * @brief 计算数据的哈希值，用于缓存模型
//  * @param data 数据指针
//  * @param size 数据大小
//  * @return size_t 哈希值
//  */
// size_t computeHash(const void* data, std::size_t size) {
//     if (data == nullptr || size == 0) {
//         throw std::invalid_argument("数据指针为空或大小为零");
//     }
//     return std::hash<std::string>()(std::string(static_cast<const char*>(data), size));
// }
//
// /**
//  * @brief 获取张量数据类型对应的字节数
//  * @param dtype 数据类型
//  * @return int 字节数
//  */
// int getTensorBytesPerComponent(nvinfer1::DataType dtype) {
//     switch (dtype) {
//         case nvinfer1::DataType::kINT64:
//             return 8;
//         case nvinfer1::DataType::kFLOAT:
//         case nvinfer1::DataType::kINT32:
//             return 4;
//         case nvinfer1::DataType::kHALF:
//         case nvinfer1::DataType::kBF16:
//             return 2;
//         case nvinfer1::DataType::kINT8:
//         case nvinfer1::DataType::kFP8:
//         case nvinfer1::DataType::kUINT8:
//         case nvinfer1::DataType::kBOOL:
//             return 1;
//         default:
//             throw std::invalid_argument("未知的数据类型");
//     }
// }
//
// /**
//  * @brief 输入输出张量信息结构体
//  */
// struct IOTensorInfo {
//     /**
//      * @brief 构造函数，初始化张量信息
//      * @param engine TensorRT 引擎指针
//      * @param index 张量索引
//      */
//     IOTensorInfo(nvinfer1::ICudaEngine* engine, int index) {
//         if (engine == nullptr) {
//             throw std::invalid_argument("引擎指针为空");
//         }
//         if (index < 0 || index >= engine->getNbIOTensors()) {
//             throw std::out_of_range("张量索引超出范围");
//         }
//
//         tensorIndex = index;
//         tensorName = engine->getIOTensorName(tensorIndex);
//         tensorDims = engine->getTensorShape(tensorName.c_str());
//         ioMode = engine->getTensorIOMode(tensorName.c_str());
//         format = engine->getTensorFormat(tensorName.c_str());
//         dataType = engine->getTensorDataType(tensorName.c_str());
//         bytesPerComponent = getTensorBytesPerComponent(dataType);
//
//         elementCount = 1;
//         for (int j = 0; j < tensorDims.nbDims; ++j) {
//             elementCount *= tensorDims.d[j];
//         }
//     }
//
//     /**
//      * @brief 重载输出运算符，便于打印张量信息
//      * @param os 输出流
//      * @param obj IOTensorInfo 对象
//      * @return std::ostream& 输出流
//      */
//     friend std::ostream& operator<<(std::ostream& os, const IOTensorInfo& obj) {
//         os << (obj.ioMode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output") << " " << obj.tensorIndex
//            << ": Name: " << obj.tensorName
//            << ", Format: " << static_cast<int32_t>(obj.format)
//            << ", DataType: " << static_cast<int32_t>(obj.dataType)
//            << ", BytesPerComponent: " << obj.bytesPerComponent
//            << ", ElementCount: " << obj.elementCount
//            << ", Dimensions: (";
//         for (int j = 0; j < obj.tensorDims.nbDims; ++j) {
//             os << obj.tensorDims.d[j];
//             if (j < obj.tensorDims.nbDims - 1) {
//                 os << ", ";
//             }
//         }
//         os << ")";
//         return os;
//     }
//
//     int tensorIndex;                 ///< 张量索引
//     std::string tensorName;          ///< 张量名称
//     nvinfer1::TensorIOMode ioMode;   ///< 输入或输出模式
//     nvinfer1::Dims tensorDims;       ///< 张量维度
//     nvinfer1::TensorFormat format;   ///< 张量格式
//     nvinfer1::DataType dataType;     ///< 数据类型
//     int32_t bytesPerComponent;       ///< 每个组件的字节数
//     int32_t elementCount;            ///< 元素总数
// };
//
// /**
//  * @brief 将 TensorRT 的模型保存到文件中
//  * @param hostMemory TensorRT 主机内存指针
//  * @param filename 文件名
//  * @return bool 是否保存成功
//  */
// bool saveIHostMemoryToFile(nvinfer1::IHostMemory* hostMemory, const std::string& filename) {
//     if (hostMemory == nullptr || filename.empty()) {
//         std::cerr << "无效的主机内存指针或文件名为空" << std::endl;
//         return false;
//     }
//     std::ofstream file(filename, std::ios::binary);
//     if (!file) {
//         std::cerr << "无法打开文件进行写入: " << filename << std::endl;
//         return false;
//     }
//     size_t size = hostMemory->size();
//     if (size == 0) {
//         std::cerr << "主机内存大小为零" << std::endl;
//         return false;
//     }
//     file.write(reinterpret_cast<const char*>(&size), sizeof(size_t)); // 写入大小
//     file.write(static_cast<const char*>(hostMemory->data()), size);    // 写入数据
//     file.close();
//     return true;
// }
//
// /**
//  * @brief 检查文件是否存在
//  * @param filename 文件名
//  * @return bool 是否存在
//  */
// bool fileExists(const std::string& filename) {
//     if (filename.empty()) {
//         return false;
//     }
//     std::ifstream file(filename);
//     return file.good();
// }
//
// /**
//  * @brief 读取二进制文件内容
//  * @param filename 文件名
//  * @return std::vector<char> 文件内容
//  */
// std::vector<char> readBinaryFile(const std::string& filename) {
//     if (filename.empty()) {
//         std::cerr << "文件名为空" << std::endl;
//         return {};
//     }
//     std::ifstream file(filename, std::ios::binary);
//     if (!file) {
//         std::cerr << "无法打开文件: " << filename << std::endl;
//         return {};
//     }
//     file.seekg(0, std::ios::end);
//     std::streamsize size = file.tellg();
//     if (size <= 0) {
//         std::cerr << "文件大小无效: " << filename << std::endl;
//         return {};
//     }
//     file.seekg(0, std::ios::beg);
//     std::vector<char> buffer(size);
//     if (!file.read(buffer.data(), size)) {
//         std::cerr << "读取文件失败: " << filename << std::endl;
//         return {};
//     }
//     file.close();
//     return buffer;
// }
//
// /**
//  * @brief TensorRT 推理类
//  */
// class TRTInferenceEngine {
// public:
//     // /**
//     //  * @brief 构造函数，初始化推理引擎
//     //  * @param serializedOnnxModel 序列化的 ONNX 模型数据指针
//     //  * @param modelSize 模型数据大小
//     //  */
//     // TRTInferenceEngine(const void* serializedOnnxModel, size_t modelSize) {
//     //     if (serializedOnnxModel == nullptr || modelSize == 0) {
//     //         throw std::invalid_argument("模型数据指针为空或大小为零");
//     //     }
//     //
//     //     runtime_ = nvinfer1::createInferRuntime(gLogger);
//     //     if (runtime_ == nullptr) {
//     //         throw std::runtime_error("创建 TensorRT Runtime 失败");
//     //     }
//     //
//     //     size_t hashValue = computeHash(serializedOnnxModel, modelSize);
//     //     cachePath_ = ".trt_cachemodel_" + std::to_string(hashValue) + ".engine";
//     //
//     //     if (!fileExists(cachePath_)) {
//     //         buildEngineFromONNX(serializedOnnxModel, modelSize);
//     //     } else {
//     //         loadEngineFromCache();
//     //     }
//     //
//     //     context_ = engine_->createExecutionContext();
//     //     if (context_ == nullptr) {
//     //         throw std::runtime_error("创建执行上下文失败");
//     //     }
//     //
//     //     int numBindings = engine_->getNbIOTensors();
//     //     for (int i = 0; i < numBindings; ++i) {
//     //         ioTensorInfo_.emplace_back(engine_, i);
//     //     }
//     //
//     //     for (const auto& info : ioTensorInfo_) {
//     //         std::cout << info << std::endl;
//     //     }
//     // }
//     TRTInferenceEngine(const std::string& filepath) {
//         if (loadEngineModel(filepath) == 0) {
//             std::cerr<<"failed to load engine"<<std::endl;
//         }
//
//     }
//     bool loadEngineModel(const std::string& filepath)
// 	{
// 		std::ifstream file(filepath, std::ios::binary);
// 		if (!file.good())
// 		{
// 			return false;
// 		}
//
// 		std::vector<char> data;
// 		try
// 		{
// 			file.seekg(0, file.end);
// 			const auto size = file.tellg();
// 			file.seekg(0, file.beg);
//
// 			data.resize(size);
// 			file.read(data.data(), size);
// 		}
// 		catch (const std::exception& e)
// 		{
// 			file.close();
// 			return false;
// 		}
// 		file.close();
//
// 		auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
// 		engine_ = runtime->deserializeCudaEngine(data.data(), data.size());
// 		if (!engine_)
// 		{
// 			return false;
// 		}
//
// 		context_ = (engine_->createExecutionContext());
// 		if (!context_)
// 		{
// 			return false;
// 		}
//
// 		int nbBindings = engine_->getNbIOTensors();
// 		assert(nbBindings == 2); // 输入和输出，一共是2个
//         ioTensorInfo_.push_back(IOTensorInfo(engine_,0));
//         ioTensorInfo_.push_back(IOTensorInfo(engine_,1));
// 		return true;
// 	}
//
//     /**
//      * @brief 析构函数，释放资源
//      */
//     ~TRTInferenceEngine() {
//         if (context_) delete context_;
//         if (engine_) delete engine_;
//         if (runtime_) delete runtime_;
//     }
//
//     /**
//      * @brief 获取张量总数
//      * @return int 张量数量
//      */
//     int getTensorCount() const {
//         return static_cast<int>(ioTensorInfo_.size());
//     }
//
//     /**
//      * @brief 获取指定索引的张量大小（字节数）
//      * @param index 张量索引
//      * @return int 张量大小（字节）
//      */
//     int getTensorSize(int index) const {
//         validateTensorIndex(index);
//         return ioTensorInfo_[index].bytesPerComponent * ioTensorInfo_[index].elementCount;
//     }
//
//     /**
//      * @brief 获取指定索引的张量元素数量
//      * @param index 张量索引
//      * @return int 元素数量
//      */
//     int getTensorElementCount(int index) const {
//         validateTensorIndex(index);
//         return ioTensorInfo_[index].elementCount;
//     }
//
//     /**
//      * @brief 判断指定索引的张量是否为输入张量
//      * @param index 张量索引
//      * @return bool 是否为输入张量
//      */
//     bool isInputTensor(int index) const {
//         validateTensorIndex(index);
//         return ioTensorInfo_[index].ioMode == nvinfer1::TensorIOMode::kINPUT;
//     }
//
//     /**
//      * @brief 执行推理
//      * @param buffers 输入和输出的设备内存指针数组
//      * @param bufferCount 缓冲区数量
//      * @param stream CUDA 流
//      * @return int 返回状态码
//      */
//     int infer(void** buffers, int bufferCount, cudaStream_t stream) {
//         if (buffers == nullptr || bufferCount != getTensorCount()) {
//             std::cerr << "缓冲区指针为空或数量不匹配" << std::endl;
//             return -1;
//         }
//         if (stream == nullptr) {
//             std::cerr << "CUDA 流为空" << std::endl;
//             return -1;
//         }
//
//         for (int i = 0; i < bufferCount; ++i) {
//             context_->setTensorAddress(ioTensorInfo_[i].tensorName.c_str(), buffers[i]);
//         }
//
//         bool status = context_->enqueueV3(stream);
//         if (!status) {
//             std::cerr << "推理执行失败" << std::endl;
//             return -1;
//         }
//         return 0;
//     }
//     void showIOInfo() {
//         for (int i = 0; i < getTensorCount(); ++i) {
//             std::cout<<ioTensorInfo_[i]<<std::endl;
//         }
//     }
// private:
//     /**
//      * @brief 从 ONNX 模型构建 TensorRT 引擎
//      * @param serializedOnnxModel 序列化的 ONNX 模型数据指针
//      * @param modelSize 模型数据大小
//      */
//     void buildEngineFromONNX(const void* serializedOnnxModel, size_t modelSize) {
//         std::cout << "构建 TensorRT 引擎..." << std::endl;
//         nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
//         if (builder == nullptr) {
//             throw std::runtime_error("创建 Builder 失败");
//         }
//
//         nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
//         if (network == nullptr) {
//             delete builder;
//             throw std::runtime_error("创建 Network 失败");
//         }
//
//         nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
//         if (parser == nullptr) {
//             delete network;
//             delete builder;
//             throw std::runtime_error("创建 Parser 失败");
//         }
//
//         if (!parser->parse(serializedOnnxModel, modelSize)) {
//             delete parser;
//             delete network;
//             delete builder;
//             throw std::runtime_error("解析 ONNX 模型失败");
//         }
//
//         nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
//         if (config == nullptr) {
//             delete parser;
//             delete network;
//             delete builder;
//             throw std::runtime_error("创建 BuilderConfig 失败");
//         }
//
//         // 如需使用 FP16 精度，可取消注释以下行
//         // config->setFlag(nvinfer1::BuilderFlag::kFP16);
//
//         nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
//         if (serializedModel == nullptr) {
//             delete config;
//             delete parser;
//             delete network;
//             delete builder;
//             throw std::runtime_error("构建序列化网络失败");
//         }
//
//         engine_ = runtime_->deserializeCudaEngine(serializedModel->data(), serializedModel->size());
//         if (engine_ == nullptr) {
//             delete serializedModel;
//             delete config;
//             delete parser;
//             delete network;
//             delete builder;
//             throw std::runtime_error("反序列化引擎失败");
//         }
//
//         // 保存序列化的模型到缓存文件
//         std::ofstream ofile(cachePath_, std::ios::binary);
//         if (!ofile) {
//             std::cerr << "无法打开文件进行写入: " << cachePath_ << std::endl;
//         } else {
//             ofile.write(static_cast<const char*>(serializedModel->data()), serializedModel->size());
//             std::cout << "模型已保存到: " << cachePath_ << std::endl;
//         }
//
//         // 释放资源
//         delete serializedModel;
//         delete config;
//         delete parser;
//         delete network;
//         delete builder;
//     }
//
//     /**
//      * @brief 从缓存加载 TensorRT 引擎
//      */
//     void loadEngineFromCache() {
//         std::cout << "从缓存加载模型: " << cachePath_ << std::endl;
//         std::vector<char> engineData = readBinaryFile(cachePath_);
//         if (engineData.empty()) {
//             throw std::runtime_error("读取缓存模型失败");
//         }
//         engine_ = runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
//         if (engine_ == nullptr) {
//             throw std::runtime_error("反序列化引擎失败");
//         }
//     }
//
//     /**
//      * @brief 验证张量索引的合法性
//      * @param index 张量索引
//      */
//     void validateTensorIndex(int index) const {
//         if (index < 0 || index >= static_cast<int>(ioTensorInfo_.size())) {
//             throw std::out_of_range("张量索引超出范围");
//         }
//     }
//
//     nvinfer1::IRuntime* runtime_ = nullptr;                       ///< TensorRT 运行时
//     nvinfer1::ICudaEngine* engine_ = nullptr;                     ///< TensorRT 引擎
//     nvinfer1::IExecutionContext* context_ = nullptr;              ///< 执行上下文
//     std::vector<IOTensorInfo> ioTensorInfo_;                      ///< 输入输出张量信息
//     std::string cachePath_;                                       ///< 缓存路径
// };
//
// /**
//  * @brief 计算均方误差（MSE）
//  * @param actual 实际值向量
//  * @param predicted 预测值向量
//  * @return float 均方误差
//  */
// float calculateMSE(const std::vector<float>& actual, const std::vector<float>& predicted) {
//     if (actual.size() != predicted.size() || actual.empty()) {
//         throw std::invalid_argument("实际值和预测值的大小不一致或为空");
//     }
//     float mse = 0.0f;
//     size_t n = actual.size();
//     std::cout << "计算 MSE，数据量为: " << n << std::endl;
//     for (size_t i = 0; i < n; ++i) {
//         float error = actual[i] - predicted[i];
//         if (i < 4) {
//             std::cout << "差值: " << error << " 实际值: " << actual[i] << " 预测值: " << predicted[i] << std::endl;
//         }
//         mse += error * error;
//     }
//     mse /= n;
//     return mse;
// }
//
// /**
//  * @brief 程序入口
//  * @param argc 参数数量
//  * @param argv 参数列表
//  * @return int 返回状态码
//  */
// int main() {
//
//
//         std::string inputImgPath="/home/wyh/Projects/OpencvTest/image_775.jpg";
//         cv::Mat img=cv::imread(inputImgPath,cv::IMREAD_COLOR);
//         // 定义目标大小
//         int width=640;
//         int height=640;
//         const float mean[3] = {0.485f, 0.456f, 0.406f};  // 归一化均值
//         const float stddev[3] = {0.229f, 0.224f, 0.225f}; // 归一化标准差
//
//         cv::resize(img, img, cv::Size(width, height));
//         cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//
//         // 归一化和标准化
//         float* input_data = new float[3 * width * height];
//         for (int c = 0; c < 3; ++c) {
//             for (int h = 0; h < height; ++h) {
//                 for (int w = 0; w < width; ++w) {
//                     int index = (c * height + h) * width + w;
//                     float pixel = img.at<cv::Vec3b>(h, w)[c] / 255.0f;
//                     input_data[index] = (pixel - mean[c]) / stddev[c];
//                 }
//             }
//         }
//
//         TRTInferenceEngine inferEngine("/home/wyh/Projects/OpencvTest/yolov8.engine");
//         inferEngine.showIOInfo();
//         // 创建 CUDA 流
//         cudaStream_t stream;
//         cudaError_t cudaStatus = cudaStreamCreate(&stream);
//         if (cudaStatus != cudaSuccess) {
//             std::cerr << "创建 CUDA 流失败: " << cudaGetErrorString(cudaStatus) << std::endl;
//             return -1;
//         }
//
//         // int tensorCount = inferEngine.getTensorCount();
//         // if (tensorCount != inputOutputCount) {
//         //     std::cerr << "输入输出数量不匹配: " << tensorCount << " vs " << inputOutputCount << std::endl;
//         //     cudaStreamDestroy(stream);
//         //     return -1;
//         // }
//
//         std::vector<void*> deviceBuffers(inferEngine.getTensorCount(), nullptr);
//         std::map<int, void*> hostMemoryMap;
//
//         // 为每个张量分配显存，并处理输入输出数据
//         for (int i = 0; i < inferEngine.getTensorCount(); ++i) {
//             int tensorSize = inferEngine.getTensorSize(i);
//             std::cout << "张量索引: " << i << " 大小: " << tensorSize << " 字节" << std::endl;
//
//             void* deviceBuffer = nullptr;
//             cudaStatus = cudaMalloc(&deviceBuffer, tensorSize);
//             if (cudaStatus != cudaSuccess) {
//                 std::cerr << "分配设备内存失败: " << cudaGetErrorString(cudaStatus) << std::endl;
//                 cudaStreamDestroy(stream);
//                 return -1;
//             }
//             deviceBuffers[i] = deviceBuffer;
//
//             if (inferEngine.isInputTensor(i)) {
//                 // std::vector<char> inputData = readBinaryFile(argv[2 + i]);
//                 // if (inputData.size() != static_cast<size_t>(tensorSize)) {
//                 //     std::cerr << "输入数据大小与张量大小不匹配: " << tensorSize << " vs " << inputData.size() << std::endl;
//                 //     cudaStreamDestroy(stream);
//                 //     return -1;
//                 // }
//                 cudaStatus = cudaMemcpy(deviceBuffer, input_data, 1*3*width*height*sizeof(float), cudaMemcpyHostToDevice);
//                 if (cudaStatus != cudaSuccess) {
//                     std::cerr << "拷贝输入数据到设备失败: " << cudaGetErrorString(cudaStatus) << std::endl;
//                     cudaStreamDestroy(stream);
//                     return -1;
//                 }
//             }
//             else {
//                 void* hostBuffer = malloc(tensorSize);
//                 if (hostBuffer == nullptr) {
//                     std::cerr << "分配主机内存失败" << std::endl;
//                     cudaStreamDestroy(stream);
//                     return -1;
//                 }
//                 hostMemoryMap[i] = hostBuffer;
//             }
//         }
//
//         // 执行推理
//         if (inferEngine.infer(deviceBuffers.data(), 2, stream) != 0) {
//             cudaStreamDestroy(stream);
//             return -1;
//         }
//
//         // 同步 CUDA 流，确保推理完成
//         cudaStatus = cudaStreamSynchronize(stream);
//         if (cudaStatus != cudaSuccess) {
//             std::cerr << "CUDA 流同步失败: " << cudaGetErrorString(cudaStatus) << std::endl;
//             cudaStreamDestroy(stream);
//             return -1;
//         }
//
//         // 处理输出数据，计算 MSE
//         for (const auto& pair : hostMemoryMap) {
//             int index = pair.first;
//             void* hostBuffer = pair.second;
//             int tensorSize = inferEngine.getTensorSize(index);
//             int elementCount = inferEngine.getTensorElementCount(index);
//             float* output_data = new float[elementCount];
//             cudaStatus = cudaMemcpy(output_data, deviceBuffers[index], tensorSize, cudaMemcpyDeviceToHost);
//             if (cudaStatus != cudaSuccess) {
//                 std::cerr << "拷贝输出数据到主机失败: " << cudaGetErrorString(cudaStatus) << std::endl;
//                 free(hostBuffer);
//                 cudaStreamDestroy(stream);
//                 return -1;
//             }
//
//             for (int i=0;i<10;i++) {
//                 for (int j=0;j<18;j++) {
//                     std::cout<<output_data[i*18+j]<<" ";
//                 }
//                 std::cout<<std::endl;
//             }
//
//
//
//             free(hostBuffer);
//         }
//
//         // 释放显存
//         for (auto buffer : deviceBuffers) {
//             if (buffer != nullptr) {
//                 cudaFree(buffer);
//             }
//         }
//
//         // 销毁 CUDA 流
//         cudaStreamDestroy(stream);
//
//
//
//     return 0;
// }
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iomanip>
class Logger : public nvinfer1::ILogger {
public:
    /**
     * @brief 实现日志记录函数
     * @param severity 日志级别
     * @param msg 日志信息
     */
    void log(Severity severity, const char* msg) noexcept override {
        // 只记录警告及以上级别的日志
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

// 全局日志器实例
static Logger gLogger;

class Box {
    public:
    Box(std::vector<float> data) {
        x = data[0];
        y = data[1];
        w = data[2];
        h = data[3];
        maxConf=-1;
        cls=-1;
        for (int i=4;i<8;i++) {
            if (data[i]>maxConf) {
                maxConf=data[i];
                cls=i;
            }
            conf.push_back(data[i]);
        }

        for (int i=8;i<18;i++) {
            if (i%2==0)
                pointx.push_back(data[i]);
            else
                pointy.push_back(data[i]);
        }
        boxRect=cv::Rect(int(x-w/2),int(y-h/2),int(w),int(h));
    }

    void print() {
        std::cout<<"x: "<<x<<std::setw(5);
        std::cout<<"y: "<<y<<std::setw(5);
        std::cout<<"w: "<<w<<std::setw(5);
        std::cout<<"h: "<<h<<std::setw(5)<<std::endl;

        std::cout<<"conf: ";
        for (int i=0;i<conf.size();i++) {
            std::cout<<conf[i]<<" ";
        }
        std::cout<<std::endl;
        for (int i=0;i<pointx.size();i++) {
            std::cout<<pointx[i]<<" ";
            std::cout<<pointy[i]<<" ";
            std::cout<<"; ";
        }
        std::cout<<std::endl;
    }


    float x,y,w,h;
    float maxConf;
    int cls;
    cv::Rect boxRect;
    std::vector<float> conf;
    std::vector<float> pointx;
    std::vector<float> pointy;
};
int main() {
    // 加载 TensorRT 引擎文件
    std::ifstream file("/home/wyh/Projects/OpencvTest/yolov8.engine", std::ios::binary);
    if (!file) {
        std::cerr << "无法打开引擎文件: model.engine" << std::endl;
        return -1;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // 创建 TensorRT 运行时和引擎
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), size);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // 获取输入张量的名称和类型
    const char* input_name = engine->getIOTensorName(0);
    nvinfer1::DataType input_type = engine->getTensorDataType(input_name);

    // 读取并预处理图片
    cv::Mat img = cv::imread("/home/wyh/Projects/OpencvTest/image_775.jpg");
    if (img.empty()) {
        std::cerr << "无法加载图片: /home/wyh/Projects/OpencvTest/image_775.jpg" << std::endl;
        return -1;
    }

    //输入大小是640*640,装甲板的输入大小和它不一样，要改
    //但是注意，这里面为了简化，预处理很草率，也没有后处理将预测内容还原成原尺寸，原来的代码里写的挺好的，按照那个改
    cv::resize(img, img, cv::Size(640, 640));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 预处理：归一化和标准化
    const float mean[3] = {0.485f, 0.456f, 0.406f};  // 归一化均值
    const float stddev[3] = {0.229f, 0.224f, 0.225f}; // 归一化标准差
    float* input_data = new float[3 * 640 * 640];
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 640; ++h) {
            for (int w = 0; w < 640; ++w) {
                int index = (c * 640 + h) * 640 + w;
                float pixel = img.at<cv::Vec3b>(h, w)[c] / 255.0f;
                input_data[index] = (pixel - mean[c]) / stddev[c];
            }
        }
    }

    // 分配 GPU 内存并复制数据
    void* input_mem;
    cudaMalloc(&input_mem, sizeof(float) * 3 * 640 * 640);
    cudaMemcpy(input_mem, input_data, sizeof(float) * 3 * 640 * 640, cudaMemcpyHostToDevice);
    delete[] input_data;

    // 设置输入张量并执行推理
    context->setTensorAddress(input_name, input_mem);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaStreamSynchronize(stream);

    // 获取输出张量信息
    const char* output_name = engine->getIOTensorName(1); // 假设只有一个输出张量
    nvinfer1::Dims output_dims = engine->getTensorShape(output_name);
    size_t output_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size *= output_dims.d[i];
    }
    nvinfer1::DataType output_type = engine->getTensorDataType(output_name);

    // 分配 GPU 内存用于输出
    void* output_mem;
    cudaMalloc(&output_mem, sizeof(float) * output_size);
    context->setTensorAddress(output_name, output_mem);

    // 执行推理
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    // 将输出数据从 GPU 复制到主机内存
    //这里注意一下模型的数据类型，有的是float有的是uint8
    float* output_data = new float[output_size];
    cudaMemcpy(output_data, output_mem, sizeof(float) * output_size, cudaMemcpyDeviceToHost);
    std::vector<Box> boxes;
    // for (int i=0;i<10;i++) {
    //
    //     for (int j=0;j<18;j++) {
    //         std::cout<<output_data[i+j*8400]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    //此处开始将数据整理至向量中
    //输出内容是一个被转化成一维数组的二维矩阵，形状为18行8400列，以此将第i行数据连接到第i+1行末尾形成一维数组。也就是说，原先的每一列是一个box
    //对于一个box的数据而言，结构为：x,y,w,h,conf0,conf1,conf2,conf3,x0,y0...x4,y4。具体内容见上面Box类。
    //这部分与原来的装甲板的数据处理不同
    for (int i=0;i<8400;i++) {
        std::vector<float> box_data;
        for (int j=0;j<18;j++) {
            box_data.push_back(output_data[i+j*8400]);
        }
        boxes.push_back(Box(box_data));
    }
    std::vector<cv::Rect> boxesRects;
    std::vector<float> scores;
    std::vector<int> indices; // 存放被选中的边界框的索引

    //将数据拆成非极大值抑制所需求的形式
    for (int i=0;i<boxes.size();i++) {
        boxesRects.push_back(boxes[i].boxRect);
        scores.push_back(boxes[i].maxConf);
    }

    // 设置NMS的参数
    static const float score_threshold = 0.65; // 分数阈值
    static const float nms_threshold = 0.4; // 非极大值抑制阈值

    // 执行NMS算法（原来的代码里有非极大值抑制的代码，注意找找）
    cv::dnn::NMSBoxes(boxesRects, scores, score_threshold, nms_threshold, indices);

    // 输出结果
    for (int i = 0; i < indices.size(); i++) {
        std::cout << "Selected box index: " << indices[i] << std::endl;
        boxes[indices[i]].print();
        std::cout << std::endl;
        // 绘制边界框
        cv::rectangle(img, boxes[indices[i]].boxRect,  cv::Scalar(0,255,0), 2);
        for (int j=0;j<5;j++) {
            cv::circle(img, cv::Point(int(boxes[indices[i]].pointx[j]), int(boxes[indices[i]].pointy[j])), 2, cv::Scalar(0,255,0), -1);
        }
    }
    cv::imshow("boxes", img);
    cv::waitKey(0);
    // 释放资源
    delete[] output_data;
    cudaFree(input_mem);
    cudaFree(output_mem);
    cudaStreamDestroy(stream);

    return 0;
}