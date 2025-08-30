📌 Get Started（入门）

Install SGLang：如何安装，包含 pip、源码安装或容器化部署方式。

Basic Usage：最基础的使用方法，跑通第一个 demo。

📌 Sending Requests（请求调用方式）

这一节主要讲 如何调用模型，以及和不同生态的兼容性：

OpenAI-Compatible APIs：直接用和 OpenAI 一样的 API 格式调用。

Offline Engine API：离线调用接口，不依赖云端。

SGLang Native APIs：SGLang 自己的原生 API，更灵活。

Sampling Parameters：采样参数（temperature、top_k、top_p 等）的设置。

DeepSeek / GPT OSS / Llama4 Usage：如何用这些具体模型。

📌 Advanced Features（高级功能）

这里是 进阶功能和优化手段：

Server Arguments：运行 server 时可配置的参数。

Hyperparameter Tuning：超参数调优方法。

Speculative Decoding：推测解码（加速推理的一种方法）。

Structured Outputs：约束模型输出结构（JSON、schema 等）。

Reasoning Models & Parser：专门针对推理模型的结构化输出。

Tool and Function Calling：让模型调用工具/函数。

Quantization：量化方法，降低显存占用。

LoRA Serving：部署 LoRA 微调模型。

PD Disaggregation / Observability / Attention Backend：一些更底层的优化与可观测性支持。

Query Vision Language Model：支持多模态模型（文字 + 图片输入）。

SGLang Router：多模型路由（根据请求自动选择模型）。

📌 Supported Models（支持的模型类型）

Large Language Models（大语言模型）

Multimodal Language Models（多模态语言模型）

Embedding Models（向量嵌入模型）

Reward Models（奖励模型）

Rerank Models（重排序模型）

How to Support New Models：如何接入新模型。

Transformers fallback：在不支持的情况下回落到 HuggingFace Transformers。

Use Models From ModelScope：接入 ModelScope 上的模型。

📌 Hardware Platforms（硬件平台支持）

SGLang 强调跨硬件：

AMD GPUs

Blackwell GPUs（英伟达最新架构）

CPU Servers

TPU

NVIDIA Jetson Orin（边缘设备）

Ascend NPUs（华为昇腾）

📌 Developer Guide（开发者指南）

Contribution Guide：如何贡献代码。

Development Guide Using Docker：用 Docker 搭建开发环境。

Benchmark and Profiling：基准测试与性能分析。

Bench Serving Guide：如何进行 serving 压测。

📌 References（参考与附录）

Troubleshooting and FAQ：常见问题排查。

Environment Variables：环境变量说明。

Production Metrics：生产环境监控指标。

Multi-Node Deployment：多节点部署指南。

Custom Chat Template：自定义对话模板。

Frontend Language：前端调用示例。

Learn more：更多资源链接。