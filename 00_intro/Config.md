## Config 配置信息


```yaml
server_args:
  model_path: "/home/buding/models/qwen2.5-7B-instruct"          # 模型权重路径（本地或HF样式）；必需
  tokenizer_path: "/home/buding/models/qwen2.5-7B-instruct"      # 分词器路径；默认与模型一致时仍可显式指定
  tokenizer_mode: "auto"                                         # 分词器模式：auto/slow/fast；auto 自动选择
  skip_tokenizer_init: false                                      # 启动时是否跳过分词器初始化以加快冷启动
  load_format: "auto"                                             # 权重加载格式：auto/pt/safetensors/gguf/…；auto 自动检测
  model_loader_extra_config: "{}"                                 # 以 JSON 字符串传入的额外加载配置
  trust_remote_code: false                                        # 允许从远端模型仓库执行自定义代码（有安全风险）
  context_length: null                                            # 上下文长度上限（覆盖模型默认）；null 使用模型默认
  is_embedding: false                                             # 以嵌入模型模式运行而非生成模式
  enable_multimodal: null                                         # 启用多模态推理（图像/音频等）；null 表示按模型能力自动
  revision: null                                                  # 指定模型版本/分支/修订号（如HF revision）
  model_impl: "auto"                                              # 后端实现：auto/vllm/pt/…；auto 自动选择

  host: "0.0.0.0"                                                 # 监听地址
  port: 33333                                                     # 监听端口
  skip_server_warmup: false                                       # 是否跳过服务启动后的预热
  warmups: null                                                   # 预热配置（提示/样本等）；null 表示无
  nccl_port: null                                                 # 指定 NCCL 通讯端口（多机多卡用）
  dtype: "auto"                                                   # 计算精度：auto/float16/bfloat16/float32/int8/…；auto 依据硬件/量化
  quantization: null                                              # 量化方式（如 awq/gptq/gguf…）；null 不量化
  quantization_param_path: null                                   # 量化额外参数文件路径
  kv_cache_dtype: "auto"                                          # KV 缓存精度：auto/float16/bfloat16/…；影响显存占用
  mem_fraction_static: 0.868                                      # 预留给模型的静态显存比例（0~1）
  max_running_requests: null                                      # 同时运行的最大请求数；null 由调度器自适应
  max_queued_requests: 9223372036854775807                        # 请求队列上限（极大值≈不限制）
  max_total_tokens: null                                          # 单请求总 token 上限（输入+输出）；null 使用默认
  chunked_prefill_size: 2048                                      # 分块预填充大小（prefill 分段长度）
  max_prefill_tokens: 16384                                       # 单请求最大 prefill token 数（长提示保护）
  schedule_policy: "fcfs"                                         # 调度策略：fcfs/… 先来先服务
  schedule_conservativeness: 1.0                                  # 调度保守系数（>1 更保守，减少拥塞）
  page_size: 1                                                    # 解码“页”大小（实现相关，批处理粒度）
  hybrid_kvcache_ratio: null                                      # 混合 KV 缓存比例（如显存+主存混合）；null 关闭
  swa_full_tokens_ratio: 0.8                                      # 逐步激活（SWA）完全令牌比例阈值
  disable_hybrid_swa_memory: false                                # 禁用混合 SWA 内存路径
  device: "cuda"                                                  # 设备类型：cuda/cpu/mps/…；优先 GPU
  tp_size: 1                                                      # 张量并行大小（Tensor Parallelism）
  pp_size: 1                                                      # 流水并行阶段数（Pipeline Parallelism）
  max_micro_batch_size: null                                      # 训练/推理微批上限；null 自适应
  stream_interval: 1                                              # 流式输出间隔（每多少 token 推一次）
  stream_output: false                                            # 是否启用服务端流式输出
  random_seed: 981563837                                          # 随机种子（采样/调度等）
  constrained_json_whitespace_pattern: null                       # 受约束 JSON 生成时允许的空白正则
  watchdog_timeout: 300                                           # 守护超时时间（秒），防止卡死
  dist_timeout: null                                              # 分布式初始化/通信超时（秒）
  download_dir: null                                              # 权重下载目录（需要远端拉取时）
  base_gpu_id: 0                                                  # 第一个使用的 GPU ID（多卡排布起点）
  gpu_id_step: 1                                                  # GPU ID 步长（多机/错位编号时有用）
  sleep_on_idle: false                                            # 空闲时是否进入休眠以降功耗
  log_level: "info"                                               # 全局日志级别：debug/info/warn/error
  log_level_http: null                                            # HTTP 层日志级别（覆盖全局）
  log_requests: false                                             # 是否记录每个请求的详细信息
  log_requests_level: 2                                           # 请求日志详细度（数值越大越详细）
  crash_dump_folder: null                                         # 进程崩溃时的转储目录
  show_time_cost: false                                           # 在日志中显示各阶段耗时
  enable_metrics: false                                           # 导出监控指标（如 Prometheus）
  enable_metrics_for_all_schedulers: false                        # 为所有调度器实例统一导出指标
  bucket_time_to_first_token: null                                # 指标直方图桶：首 token 延迟
  bucket_inter_token_latency: null                                # 指标直方图桶：token 间延迟
  bucket_e2e_request_latency: null                                # 指标直方图桶：端到端时延
  collect_tokens_histogram: false                                 # 收集生成 token 的长度分布
  decode_log_interval: 40                                         # 解码阶段日志间隔（步数）
  enable_request_time_stats_logging: false                        # 记录请求时间统计（均值/分位数）
  kv_events_config: null                                          # KV 事件跟踪配置
  gc_warning_threshold_secs: 0.0                                  # GC 警告阈值（秒）；>0 打印慢 GC 提醒
  api_key: null                                                   # 接口鉴权密钥（如需）
  served_model_name: "/home/buding/models/qwen2.5-7B-instruct"    # 对外暴露的模型名（路由名/展示名）
  weight_version: "default"                                       # 权重版本标签（多版本共存时）
  chat_template: null                                             # 聊天模板（Jinja/自定义）；null 用模型内置
  completion_template: null                                       # 补全模板（非聊天）
  file_storage_path: "sglang_storage"                             # 服务端文件存储目录（上传/缓存等）
  enable_cache_report: false                                      # 输出 KV/响应缓存命中统计
  reasoning_parser: null                                          # 推理轨迹解析器（如 CoT 解析）
  tool_call_parser: null                                          # 工具调用解析器（结构化函数调用）
  tool_server: null                                               # 外部工具服务端地址

  dp_size: 1                                                      # 数据并行大小（Data Parallel）
  load_balance_method: "round_robin"                              # 多副本负载均衡方法
  dist_init_addr: null                                            # 分布式初始化地址（如 tcp://host:port）
  nnodes: 1                                                       # 集群节点数
  node_rank: 0                                                    # 当前节点序号（0-based）
  json_model_override_args: "{}"                                  # 用 JSON 字符串覆盖模型内部配置
  preferred_sampling_params: null                                 # 默认采样参数（top_p/temperature 等）覆盖
  enable_lora: null                                               # 启用 LoRA 微调适配（null 自动/按路径）
  max_lora_rank: null                                             # LoRA 秩上限
  lora_target_modules: null                                       # LoRA 注入目标模块列表
  lora_paths: null                                                # 要加载的 LoRA 权重路径列表
  max_loaded_loras: null                                          # 允许同时加载的 LoRA 数上限
  max_loras_per_batch: 8                                          # 单批可并行的 LoRA 变体数
  lora_backend: "triton"                                          # LoRA 后端：triton/torch/…
  attention_backend: "torch_native"                               # 统一注意力后端（回退用）
  decode_attention_backend: "torch_native"                        # 解码阶段注意力内核
  prefill_attention_backend: "torch_native"                       # 预填充阶段注意力内核
  sampling_backend: "pytorch"                                     # 采样实现后端
  grammar_backend: "xgrammar"                                     # 语法约束后端（Outlines/xgrammar 等）
  mm_attention_backend: null                                      # 多模态注意力后端；null 自动/关闭

  speculative_algorithm: null                                     # 选择性采样/推测解码算法（eagle/medusa/…）
  speculative_draft_model_path: null                              # 推测解码的草稿模型路径
  speculative_num_steps: null                                     # 推测解码每轮步数
  speculative_eagle_topk: null                                    # EAGLE 草稿选择 top-k
  speculative_num_draft_tokens: null                              # 每轮生成草稿 token 数
  speculative_accept_threshold_single: 1.0                        # 单 token 接受阈值
  speculative_accept_threshold_acc: 1.0                           # 累积接受阈值
  speculative_token_map: null                                     # 草稿到主模型 token 映射策略

  ep_size: 1                                                      # Expert 并行大小（MoE）
  moe_a2a_backend: "none"                                         # MoE all-to-all 通信后端
  moe_runner_backend: "auto"                                      # MoE 运行时后端选择
  flashinfer_mxfp4_moe_precision: "default"                       # FlashInfer MoE mxfp4 精度模式
  enable_flashinfer_allreduce_fusion: false                       # 融合 all-reduce 以优化通信
  deepep_mode: "auto"                                             # DeepEP 模式（自动/禁用/强制）
  ep_num_redundant_experts: 0                                     # 冗余专家数（容错）
  ep_dispatch_algorithm: "static"                                 # 专家调度算法：static/… 
  init_expert_location: "trivial"                                 # 专家初始分布策略
  enable_eplb: false                                              # 启用专家负载均衡（EPLB）
  eplb_algorithm: "auto"                                          # EPLB 算法选择
  eplb_rebalance_num_iterations: 1000                             # 负载均衡最大迭代次数
  eplb_rebalance_layers_per_chunk: null                           # 每次重平衡涉及的层数
  expert_distribution_recorder_mode: null                         # 记录专家分布模式
  expert_distribution_recorder_buffer_size: 1000                  # 专家分布缓冲区大小
  enable_expert_distribution_metrics: false                       # 导出专家分布监控
  deepep_config: null                                             # DeepEP 自定义配置
  moe_dense_tp_size: null                                         # MoE 中稠密部分的张量并行大小

  enable_hierarchical_cache: false                                # 启用分层缓存（显存/主存/存储）
  hicache_ratio: 2.0                                              # 分层缓存容量系数（相对模型需要）
  hicache_size: 0                                                 # 分层缓存固定大小（字节，0 表示自动）
  hicache_write_policy: "write_through_selective"                 # 写入策略（直写/选择性直写等）
  hicache_io_backend: "kernel"                                    # I/O 后端（内核/用户态）
  hicache_mem_layout: "layer_first"                               # 内存布局：按层优先/按序列优先
  hicache_storage_backend: null                                   # 存储后端（如 nvme/ufs/…）
  hicache_storage_prefetch_policy: "best_effort"                  # 存储预取策略
  hicache_storage_backend_extra_config: null                      # 存储后端额外配置

  enable_double_sparsity: false                                   # 启用双稀疏（稀疏注意力+权重）
  ds_channel_config_path: null                                    # 稀疏通道配置文件
  ds_heavy_channel_num: 32                                        # 重通道数量
  ds_heavy_token_num: 256                                         # 标记为重通道的 token 数阈值
  ds_heavy_channel_type: "qk"                                     # 重通道类型（qk/v/…）
  ds_sparse_decode_threshold: 4096                                # 稀疏解码启动阈值（序列长度）
  cpu_offload_gb: 0                                               # 向 CPU offload 的上限（GB）
  offload_group_size: -1                                          # offload 分组大小（-1 自动）
  offload_num_in_group: 1                                         # 每组 offload 的并发数
  offload_prefetch_step: 1                                        # 预取步数（降低等待）
  offload_mode: "cpu"                                             # offload 目标：cpu/nvme/…
  disable_radix_cache: false                                      # 禁用 radix 前缀缓存

  cuda_graph_max_bs: 8                                            # CUDA Graph 记录的最大 batch size
  cuda_graph_bs: null                                             # 固定 Graph 用的 batch size（null 自适应）
  disable_cuda_graph: true                                        # 禁用 CUDA Graph（为 true 时不使用）
  disable_cuda_graph_padding: false                               # 禁用为 Graph 对 batch 做 padding
  enable_profile_cuda_graph: false                                # 对 CUDA Graph 进行性能剖析
  enable_cudagraph_gc: false                                      # 对 CUDA Graph 产生的资源做 GC
  enable_nccl_nvls: false                                         # 启用 NCCL NVLS（新型通信优化）
  enable_symm_mem: false                                          # 启用对称内存（多进程共享）
  disable_flashinfer_cutlass_moe_fp4_allgather: false             # 禁用 MoE FP4 allgather 优化（兼容开关）
  enable_tokenizer_batch_encode: false                            # 分词器批量编码优化
  disable_outlines_disk_cache: false                              # 禁用 Outlines 语法缓存的落盘
  disable_custom_all_reduce: false                                # 禁用自定义 all-reduce（回退到默认）
  enable_mscclpp: false                                           # 启用 MSCCL++ 通信库
  disable_overlap_schedule: false                                 # 禁用计算/通信重叠调度
  enable_mixed_chunk: false                                       # 允许不同 chunk 策略混合
  enable_dp_attention: false                                      # 数据并行下开启跨卡注意力
  enable_dp_lm_head: false                                        # 数据并行下共享 LM Head
  enable_two_batch_overlap: false                                 # 启用双批重叠（prefill/decode 并行）
  tbo_token_distribution_threshold: 0.48                          # 双批重叠时 token 分布阈值
  enable_torch_compile: false                                     # 启用 torch.compile 加速
  torch_compile_max_bs: 32                                        # torch.compile 可支持的最大 batch
  torchao_config: ""                                              # TorchAO（低精/稀疏）配置字符串
  enable_nan_detection: false                                     # 检测并报告 NaN
  enable_p2p_check: false                                         # 启用 P2P 连通性自检
  triton_attention_reduce_in_fp32: false                          # Triton 注意力归约使用 FP32（提高稳定性）
  triton_attention_num_kv_splits: 8                               # Triton 注意力 KV 分片数
  num_continuous_decode_steps: 1                                  # 连续解码步数（增大可减少同步开销）
  delete_ckpt_after_loading: false                                # 加载后删除临时/下载的 ckpt 文件
  enable_memory_saver: false                                      # 启用内存节省策略（可能影响吞吐）
  allow_auto_truncate: false                                      # 超过上下文时允许自动截断输入
  enable_custom_logit_processor: false                            # 启用自定义 logit 处理（温度/屏蔽等）
  flashinfer_mla_disable_ragged: false                            # 禁用 FlashInfer MLA 的 ragged 支持
  disable_shared_experts_fusion: false                            # 禁用共享专家融合优化
  disable_chunked_prefix_cache: false                             # 禁用分块前缀缓存
  disable_fast_image_processor: false                             # 禁用快速图像处理管线（多模态）
  enable_return_hidden_states: false                              # 是否返回隐藏状态（供分析/可视化）
  scheduler_recv_interval: 1                                      # 调度器消息接收间隔（毫秒/步，实现相关）
  debug_tensor_dump_output_folder: null                           # 张量转储输出目录
  debug_tensor_dump_input_file: null                              # 从转储输入文件重放调试
  debug_tensor_dump_inject: false                                 # 将转储张量注入运行图（复现实验）
  debug_tensor_dump_prefill_only: false                           # 仅在 prefill 阶段转储
  disaggregation_mode: "null"                                     # 计算/权重解耦（null/… 实验特性）
  disaggregation_transfer_backend: "mooncake"                     # 解耦传输后端
  disaggregation_bootstrap_port: 8998                             # 解耦模式引导端口
  disaggregation_decode_tp: null                                  # 解码阶段的 TP 设置（覆盖全局）
  disaggregation_decode_dp: null                                  # 解码阶段的 DP 设置（覆盖全局）
  disaggregation_prefill_pp: 1                                    # 预填充阶段的 PP 设置（覆盖全局）
  disaggregation_ib_device: null                                  # 解耦使用的 IB 网卡设备名

  num_reserved_decode_tokens: 512                                 # 为解码阶段预留的 token 空间
  pdlb_url: null                                                  # 外部负载均衡/调度服务地址
  custom_weight_loader: []                                        # 自定义权重加载器列表
  weight_loader_disable_mmap: false                               # 禁用 mmap 加载（兼容某些文件系统）
  enable_pdmux: false                                             # 启用 PD 多路复用（实验/吞吐优化）
  sm_group_num: 3                                                 # SM 分组数量（核资源分组）
  enable_ep_moe: false                                            # 启用 EP-MoE（专家并行）
  enable_deepep_moe: false                                        # 启用 DeepEP MoE
  enable_flashinfer_cutlass_moe: false                            # 启用 FlashInfer Cutlass MoE 内核
  enable_flashinfer_trtllm_moe: false                             # 启用 FlashInfer TensorRT-LLM MoE 内核
  enable_triton_kernel_moe: false                                 # 启用 Triton MoE 内核
  enable_flashinfer_mxfp4_moe: false                              # 启用 FlashInfer mxfp4 MoE 精度
```