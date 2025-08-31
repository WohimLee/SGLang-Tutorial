# 禁用自定义 CUDA 算子（RMSNorm 等用 PyTorch 实现）
# export SGLANG_DISABLE_CUSTOM_OPS=1
# 单卡
# export CUDA_VISIBLE_DEVICES=0,1

python3 -m sglang.launch_server \
  --model-path /home/buding/models/qwen2.5-7B-instruct \
  --host 0.0.0.0 --port 33333 \
  --disable-cuda-graph \
  --attention-backend torch_native \
  --prefill-attention-backend torch_native \
  --decode-attention-backend torch_native \
  --sampling-backend pytorch