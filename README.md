# InferKernelLab

InferKernelLab 是一个面向 LLM 推理优化的可审计实验项目，重点展示 paged KV cache、
decode attention Triton kernel、真实多层 Llama adapter、正确性验证和性能测量。它与
MiniTrainBench 互补：后者侧重训练/预训练基础设施，本项目侧重推理时的 KV 内存访问与
decode kernel。

本项目是推理优化实验室，不是完整 serving 框架：runtime/scheduler 不隐式下载外部模型，
synthetic decoder 保留固定权重闭环；可选 Transformers adapter 进一步把 paged KV cache
接入真实的多层 Llama-like decoder。

## 当前能力

- 物理 KV block 分配、逻辑 block table、slot mapping、读写与资源释放。
- 支持 MHA、GQA、MQA 的 PyTorch reference decode attention。
- PyTorch SDPA dense baseline，以及 paged gather + SDPA baseline。
- 可选 Triton paged decode attention 和 KV append kernel。
- 模型无关的单层 synthetic decoder：embedding、Q/K/V 投影、paged KV append、decode attention、输出投影和 greedy token 生成。
- 可选 `Transformers/Llama` adapter：tiny random `LlamaConfig`、本地 checkpoint、
  多层 RMSNorm/RoPE/GQA、paged KV cache 和真实 decoder layer 手动执行。
- 使用 Transformers eager full-sequence path 做 logits/token correctness oracle，并输出
  prefill、decode P50/P95、TTFT/TPOT、KV memory 和 backend/kernel 配置。
- runtime 可选 `prefill_fn`/`decode_fn` hook，保留原有无模型 `step()` 接口。
- benchmark 输出各路径的误差、P50/P95、环境信息和形状级相对性能。
- attention microbenchmark 和端到端 decode benchmark 分开记录，避免混淆比较对象。
- A100 参数 sweep、Nsight Compute profiling 脚本及面试记录材料。

## 快速开始

所有 CUDA/Triton 依赖都在 Docker 容器中，不需要修改宿主 Python 环境。默认从本地 `nano-vllm:optimized` 构建，其中包含 PyTorch、Triton、CUDA 工具链和 Nsight Compute：

```bash
./scripts/docker_build.sh
./scripts/docker_test.sh
```

运行一组 decode benchmark：

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD:/workspace" -w /workspace inferkernellab:cuda \
  python3 -m inferkernellab.benchmark \
  --device cuda --backend auto --dtype float16 \
  --batch-size 4 --context-len 512 \
  --num-heads 32 --num-kv-heads 8 --head-dim 64 \
  --block-size 16 --num-blocks 159 --warmup 20 --iterations 100
```

运行完整 54 组 sweep：

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD:/workspace" -w /workspace inferkernellab:cuda \
  python3 scripts/run_benchmark_sweep.py \
  --output docs/benchmark_results/local_sweep.jsonl
```

运行单层 synthetic decoder 的端到端 sweep：

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD:/workspace" -w /workspace inferkernellab:cuda \
  python3 scripts/run_decode_sweep.py \
  --output docs/benchmark_results/local_synthetic_decode.jsonl \
  --dtypes float16,bfloat16 \
  --batch-sizes 1,4,16 --context-lens 128,512 \
  --num-kv-heads-list 32,8,1 --max-new-tokens 16 \
  --warmup 1 --repeats 3
```

`run_decode_sweep.py` 默认比较 `paged_reference`、`paged_sdpa` 和 `triton_paged`。其中 `paged_reference` 是 correctness oracle；`paged_sdpa` 的时间包含 paged gather；Triton 路径直接读取 paged cache。可用 `--include-triton-append` 额外测量 Triton KV append，但该结果单独标记，不混入 attention-only 对照。

运行真实多层 tiny Llama 的 paged decode smoke：

```bash
IMAGE=inferkernellab:cuda scripts/run_llama_paged_smoke.sh
```

也可以在已安装 `inferkernellab[hf]` 的环境中直接运行无网络 random model：

```bash
python3 -m inferkernellab.hf_benchmark \
  --model-source random --device cuda --dtype bfloat16 \
  --backend paged_sdpa --batch-size 2 --prompt-length 128 \
  --max-new-tokens 8 --warmup 10 --iterations 50 \
  --output docs/benchmark_results/llama_paged.json
```

`--model-source path --model-path PATH` 只读取本地 Transformers checkpoint，使用
`local_files_only=True`，不会隐式联网。设计说明见
[`docs/hf_llama_adapter_design.md`](docs/hf_llama_adapter_design.md)。

项目镜像需要本地存在 `nano-vllm:optimized`。如需其他基础镜像，可通过 `BASE_IMAGE` 覆盖；若基础镜像没有 `ncu`，profiling 脚本不会自动在宿主机安装工具。

## 架构

```text
src/inferkernellab/
  cache.py       物理 block allocator 与 paged KV 存储
  attention.py   PyTorch reference、dense SDPA、paged SDPA
  triton_ops.py  Triton KV append 与 paged decode attention
  decode.py      单层 synthetic decoder 与端到端 decode runner
  hf_benchmark.py 可选 Transformers/Llama 多层 paged decode adapter
  scheduler.py   prefill/decode token-budget 调度示例
  runtime.py     请求、KV block 生命周期和模型执行 hook
  benchmark.py   正确性、延迟和环境信息采集
```

KV cache 张量布局为：

```text
[num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

逻辑 token 位置 `p` 映射到物理 slot 的过程为：

```text
logical_block = p // block_size
physical_block = block_table[logical_block]
slot = physical_block * block_size + p % block_size
```

## 基准测试口径

- `dense` 与 `paged_reference` 是 PyTorch 数学参考路径，不应当作生产级性能基线。
- `dense_sdpa` 调用 PyTorch SDPA 处理连续 KV；`paged_sdpa` 将 paged KV gather 成连续张量后再调用 SDPA，计时包含 gather。
- `triton_paged` 直接通过 block table 读取 paged KV。比较时应优先看相同配置下相对 `paged_sdpa` 的结果。
- benchmark 构造交错分配的物理 block，并在逻辑 block 之间留出物理空洞，避免只测到连续分配特例；`num_blocks` 需包含这些保留块。
- 等长 batch 的 dense SDPA 预先堆叠连续 KV，计时期间使用单次 batched SDPA；paged SDPA 在计时期间做 batched gather，再执行 batched SDPA。
- `estimated_kv_read_gbps` 是按 K/V 理论读取字节数计算的估算值，不是硬件计数器。DRAM 吞吐只能来自 Nsight Compute 等 profiler。
- CUDA Event 测量的是 GPU 时间线上的设备执行区间，不等同于含 Python 调度、排队和网络开销的线上请求延迟。
- 输出 `kernel_config` 记录 Triton autotuner 选择的 `block_n`、`num_warps` 和 `num_stages`。
- 端到端 benchmark 同时记录 wall-clock 和 CUDA Event 设备时间；`TTFT` 定义为全部 prefill step 加第一个 decode step，`TPOT` 显式记录平均 decode step 时间，同时报告 decode step P50/P95。
- `tokens_per_sec` 是整个 batch 生成 token 数除以 decode 阶段耗时，不等同线上服务吞吐。

当前端到端结果保存在 `docs/benchmark_results/a100_synthetic_decode.jsonl`。A100 上 108 条记录全部通过 dtype-aware logits `allclose`；但 greedy token 是离散 argmax，接近相等的 logits 可能在不同数值路径下产生不同 token，因此报告同时记录 `token_match_rate`，不能只看 token 一致率判断浮点实现是否正确。

当前代表性 synthetic decoder sweep 的结论是：在本项目的单层模型无关闭环中，Triton direct-paged attention 的 decode step P50 中位数约为 paged SDPA 的 `0.841x`（即当前约慢 18.9%），没有胜过成熟的 PyTorch SDPA backend；这不否定它在 attention-only benchmark 中避免显式 gather 的价值。性能数字只适用于报告中的 A100、dtype、batch 和 context 配置。

当前容器中的 `ncu` 可以启动，但驱动报告 `ERR_NVGPUCTRPERM`，说明 GPU performance counter 权限由宿主驱动策略限制。脚本不会修改宿主机权限；启用该权限后可运行：

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PWD:/workspace" -w /workspace inferkernellab:cuda \
  bash scripts/profile_attention.sh
```

## 面试材料

- `docs/change_log.md`：重要改动、设计取舍、测试与实测结果记录。
- `docs/interview_guide.md`：项目讲解路径、核心知识点与常见追问。
- `docs/design.md`：内存布局、attention kernel 和 benchmark 设计。
- `docs/benchmark_report.md`：实际测量配置、结果、限制和复现命令。
- `docs/benchmark_results/a100_synthetic_decode.jsonl`：单层 synthetic decoder 的原始端到端结果。
- `docs/benchmark_results/a100_synthetic_decode_triton_append.jsonl`：一个选定形状的 Triton append 组合结果。

简历只引用报告中可复现的形状级结果，并明确比较对象、GPU、精度和测量口径。不要把 reference 对比结果概括成普遍的推理加速比。
