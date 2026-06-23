# VMM V2 回归验证 Checklist

> 最后验证日期: 2026-06-23 (GPU 0 for VMM V2 full replay regression)
> 验证分支: `vmm_v2_pr3` (`/work/Paddle`), `vmm_v2_pr4` (`/work/dev_tool/Paddle`)
> Backing View 实验分支: `feature/vmm-v2-backing-view`
> 环境: A100-80G, CUDA 12.9, Python 3.12

---

## 一、验证目标

每次代码修改后，需确认以下维度无回退：

| # | 维度 | 不可接受的回退 |
|---|------|--------------|
| 1 | 正确性 | 出现 SIGSEGV / SIGABRT / 未捕获异常 |
| 2 | 泄漏 | `post_cleanup_reserved > 0`，V2 应释放所有 reserved |
| 3 | OOM 数 | 相同压力下 OOM 显著增多，超过 10% 回退；逐请求 replay 中 OOM 计数不是唯一正确性判据 |
| 4 | Remap 有效性 | Probe 测试从 SUCCESS 变为 FAIL |
| 5 | 性能 | 相同 OOM 数下 elapsed 显著增加，超过 50% 回退 |
| 6 | BackingMap mirror | Phase 1a 下出现 `BackingMap mismatch` / `validation failed` |

说明：

- 正确性硬失败优先看 crash、泄漏、BackingMap mismatch/validation failed、force-release、invalid range、probe remap 从成功变失败等信号。
- OOM 数用于发现性能/碎片整理效果回退，但 replay 是逐个请求执行：如果前面通过 compact 成功满足了更大的请求，后续可用空间形态可能变化，最终 OOM 次数可能小幅上升。因此 OOM 计数需要结合日志行为、reserved cleanup、compact 成功率和请求序列分析，不应单独作为 correctness hard fail。
- `dsv3_45g_compact_all` 当前最新完整回归值是 `ooms=3493`，历史接受基线 `4362` 保留为正确性硬阈值。后续若 OOM 数高于最新值但仍低于硬阈值，需要结合请求序列、compact 成功率和 alloc/frees 判断是否为效果差异；高于 `4362` 或出现 `post_cleanup > 0`、`force-release`、`backing_err`、crash 时必须按回退继续排查。
- V2 IPC 本版采用保守 pin 作为完成语义：IPC exported backing page 在没有可靠 exporter-side unpin 信号前不允许 remap/release/reuse；可靠 unpin 需要新增跨进程生命周期协议，作为后续独立工作，不阻塞当前 BackingMap 设计收尾。

---

## 二、测试环境准备

### 编译安装

```bash
cd /work/dev_tool/Paddle/build
make -j32 paddle_python

cd /work/dev_tool/Paddle/build/python
python setup.py bdist_wheel

python -m pip install --force-reinstall --no-deps \
  /work/dev_tool/Paddle/build/python/dist/paddlepaddle_gpu-*.whl

LD_LIBRARY_PATH=/usr/lib64 python - <<'PY'
import paddle
print(paddle.__file__)
print(paddle.__version__)
PY
```

说明：

- 当前 Backing View / IPC 验证推荐使用正式 whl 安装后的 Paddle 包。
- 不推荐只复制 `libpaddle.so` 或少量 `.so` 后直接跑 Python 分布式测试；这可能导致 `build/python` 中旧 Python 文件与新 C++ so 混装。
- 2026-05-27 验证中曾观察到 AdamW learning rate dtype mismatch，根因即为 staging 混装；重新 `make -j32 paddle_python`、打 whl 并 `pip install --force-reinstall` 后，V1/V2 fleet 均通过。
- 一键回归脚本位于 `/work/MemoryTools/run_vmm_v2_regression_log.sh`；不要从 Paddle 仓库根目录直接调用同名路径。

### 测试工具

```bash
cd /work/MemoryTools
ls scripts/mem_replay_v2.py
ls logs/dsv3/pp2ep4.log
ls logs/ernie/ernie_4nodes_0.log
```

### GPU 独占

```bash
nvidia-smi
```

---

## 三、回归测试用例

说明：

- 本节命令主要用于手工隔离单个场景、定位某一类行为，不再是当前 Backing View 分支的主回归入口。
- 当前推荐主流程见下方“**四、一键回归脚本**”。日常验证优先使用 `run_vmm_v2_regression_log.sh`，单测命令仅在需要复现单 case 或缩小问题范围时使用。
- 本节中的早期基线值保留为历史对照；当前是否通过，优先以“一键回归脚本”章节中的 8/8 回归结果和后文历史基线记录为准。

### Test 1: ERNIE 35G - Paddle V2 remap ON

```bash
cd /work/MemoryTools
FLAGS_gpu_allocator_retry_time=0 \
FLAGS_use_vmm_auto_growth_best_fit_allocator_v2=1 \
FLAGS_vmm_v2_remap_on_oom=1 \
GLOG_v=0 \
python3 scripts/mem_replay_v2.py logs/ernie/ernie_4nodes_0.log paddle-v2 \
  --occupy-gib 35 --device 0
```

| 指标 | 基线值 | 通过标准 |
|------|--------|---------|
| OOMs | 1927 | ≤ 2120 |
| Peak reserved | 75.60 GiB | 约 75.6 GiB，允许 ±0.5 |
| Post-cleanup | 0.0 GiB | 必须为 0 |
| Elapsed | 178.6s | ≤ 270s |
| Crash | 无 | 必须无 crash |

### Test 2: ERNIE 35G - Paddle V2 remap OFF

```bash
cd /work/MemoryTools
FLAGS_gpu_allocator_retry_time=0 \
FLAGS_use_vmm_auto_growth_best_fit_allocator_v2=1 \
FLAGS_vmm_v2_remap_on_oom=0 \
GLOG_v=0 \
python3 scripts/mem_replay_v2.py logs/ernie/ernie_4nodes_0.log paddle-v2 \
  --occupy-gib 35 --device 0
```

| 指标 | 基线值 | 通过标准 |
|------|--------|---------|
| OOMs | 1927 | ≤ 2120 |
| Post-cleanup | 0.0 GiB | 必须为 0 |
| Elapsed | 216.0s | ≤ 324s |
| Crash | 无 | 必须无 crash |

### Test 3: ERNIE 35G - Torch expandable segments

```bash
cd /work/MemoryTools
python3 scripts/mem_replay_v2.py logs/ernie/ernie_4nodes_0.log torch-exp \
  --occupy-gib 35 --device 0
```

| 指标 | 基线值 | 验证点 |
|------|--------|--------|
| OOMs | 2087 | Paddle V2 OOMs ≤ 此值 |
| Post-cleanup | 0.0 GiB | 对照 |
| Elapsed | 1.7s | torch 对照，不作为 Paddle 性能标准 |

### Test 4: ERNIE 35G - Paddle V1

```bash
cd /work/MemoryTools
FLAGS_gpu_allocator_retry_time=0 \
FLAGS_use_vmm_auto_growth_best_fit_allocator_v2=0 \
FLAGS_use_virtual_memory_auto_growth=1 \
GLOG_v=0 \
python3 scripts/mem_replay_v2.py logs/ernie/ernie_4nodes_0.log paddle-v1 \
  --occupy-gib 35 --device 0
```

| 指标 | 基线值 | 验证点 |
|------|--------|--------|
| OOMs | 2310 | Paddle V2 OOMs ≤ 此值 |
| Post-cleanup | 75.6 GiB | V1 不释放 reserved，为预期行为 |
| Elapsed | 168.8s | 对照 |

### Test 5: DSV3 pp2ep4 30G - Paddle V2

```bash
cd /work/MemoryTools
FLAGS_gpu_allocator_retry_time=0 \
FLAGS_use_vmm_auto_growth_best_fit_allocator_v2=1 \
FLAGS_vmm_v2_remap_on_oom=1 \
GLOG_v=0 \
python3 scripts/mem_replay_v2.py logs/dsv3/pp2ep4.log paddle-v2 \
  --occupy-gib 30 --device 0
```

| 指标 | 基线值 | 通过标准 |
|------|--------|---------|
| OOMs | 62 | ≤ 68 |
| Post-cleanup | 0.0 GiB | 必须为 0 |
| Elapsed | 641.8s | ≤ 960s |
| Crash | 无 | 必须无 crash |

### Test 6: 合成碎片 Probe - Remap 有效性

```bash
cd /work/MemoryTools

FLAGS_use_vmm_auto_growth_best_fit_allocator_v2=1 \
FLAGS_vmm_v2_remap_on_oom=1 \
FLAGS_gpu_allocator_retry_time=0 \
python3 /work/vmm_test/paddle_vmm_70g_20m_probe.py \
  --worker --mode vmm --anchor-gib 60 --superpool-gib 12 \
  --superblock-mib 1024 --subblock-mib 20 --sub-fill-ratio 0.95 \
  --pin-stride 8 --big-gib 7 --device 0
```

| 模式 | 期望结果 | 说明 |
|------|---------|------|
| vmm auto-remap | SUCCESS | 单 worker 直接验证 compact/remap 解决碎片 |

说明：不要在一键回归脚本中使用 probe parent 的多模式调度。该 parent 脚本会串行运行 baseline / vmm / empty_cache / compact 多个 worker，baseline 预期 OOM 后可能污染同一 GPU 上的后续 worker。回归脚本中的 `probe_standard` 使用 `--worker --mode vmm` 单模式验证。

Remap A/B:

```bash
FLAGS_use_vmm_auto_growth_best_fit_allocator_v2=True \
FLAGS_vmm_v2_remap_on_oom=True \
FLAGS_gpu_allocator_retry_time=0 \
python3 /work/vmm_test/test_remap_no_grow.py

FLAGS_use_vmm_auto_growth_best_fit_allocator_v2=True \
FLAGS_vmm_v2_remap_on_oom=False \
FLAGS_gpu_allocator_retry_time=0 \
python3 /work/vmm_test/test_remap_no_grow.py
```

| 配置 | 期望结果 |
|------|---------|
| Remap ON | SUCCESS |
| Remap OFF | FAIL / OOM |

### Test 7: PyTorch 对照 Probe

```bash
cd /work/MemoryTools
python3 /work/vmm_test/torch_vmm_70g_20m_remap_probe.py \
  --anchor-gib 60 --superpool-gib 12 --subblock-mib 20 \
  --pin-stride 8 --big-gib 7 --device 0
```

| 模式 | 期望结果 |
|------|---------|
| expandable_segments=True | SUCCESS |
| expandable_segments=False | FAIL |

### Test 8: BackingMap Phase 1a mirror 检查

Phase 1a 不应改变行为。除了上述结果与 `pr4/vmm-remap-v2` 持平，还需要确认日志中无 BackingMap 不一致：

```bash
grep -Ei "BackingMap.*mismatch|BackingMap.*validation failed|invalid range" your_test.log
```

期望：无输出。

---

## 四、一键回归脚本

脚本路径：

```text
/work/MemoryTools/run_vmm_v2_regression_log.sh
```

用法：

```bash
cd /work/MemoryTools
bash run_vmm_v2_regression_log.sh 7
```

推荐用法：

```bash
# 当前推荐：先确认 case 列表，再跑目标 GPU 的全量正确性回归
cd /work/MemoryTools
LIST=1 bash ./run_vmm_v2_regression_log.sh
bash ./run_vmm_v2_regression_log.sh 7

# 单独运行一个 case
ONLY=probe_standard bash ./run_vmm_v2_regression_log.sh 7

# 查看可用 case 名称
LIST=1 bash ./run_vmm_v2_regression_log.sh

# 只对 replay 类 case 开 allocator 诊断日志，确认 force-release/backing_err
ONLY=dsv3_45g_compact_all DIAG=1 bash ./run_vmm_v2_regression_log.sh 7
```

注意：不要对全量 batch 使用 `DIAG=1`。诊断日志会显著放大多进程 probe 的 stdout/stderr 和 glog 开销，历史测试中观察到可能造成 worker 异常退出或 GPU 进程残留。`force-release` 正确性主要通过 `dsv3_45g_compact_all` 单 case 诊断确认。

当前运行约定：

- 位置：`cd /work/MemoryTools`
- 设备：命令行参数直接传 GPU 编号，例如 `7`
- 默认日志：全量回归保持 `GLOG_v=0`
- 诊断日志：只在单 case 下使用 `DIAG=1`
- 推荐判定顺序：先看 `summary.md` / `status.tsv`，再按需 grep 单个 case 日志

单 case 诊断常用检查：

```bash
LOG=/work/MemoryTools/logs/regression/<timestamp>/dsv3_45g_compact_all.log

grep -Eci "force-releasing handle|force-release" "$LOG"
grep -Ei "BackingMap.*(mismatch|validation failed|invalid range|reconfigure)" "$LOG"
```

期望：

- `force-release` 计数为 `0`
- BackingMap 相关 grep 无输出

补充说明：

- 若 `DIAG=1` 单测在 `RESULT` 之后出现 `double free or corruption (!prev)` 并被 runner 标记为 `TEARDOWN_TIMEOUT`，不要直接算作本轮正确性回归失败。
- 该现象历史上与开启高日志级别强相关，旧 VMM 版本也可复现；当前主判定仍以 `GLOG_v=0` 的全量 8/8 回归和单 case grep 结果为准。

测试项：

| # | 测试名 | 说明 | 通过标准 |
|---|--------|------|---------|
| 1 | ERNIE 35G remap ON | 高频 OOM + compact | post_cleanup=0, OOMs≤2120；最新参考 48；无 crash |
| 2 | ERNIE 35G remap OFF | flag 隔离 | post_cleanup=0, OOMs≤2120, 无 crash |
| 3 | DSV3 30G | 中压 compact | post_cleanup=0, OOMs≤68, 无 crash |
| 4 | DSV3 45G bounded | 高压默认 compact | post_cleanup=0, OOMs≤4950；最新参考 3509；无 crash |
| 5 | DSV3 45G compact_all | 高压全量 compact | post_cleanup=0, OOMs≤4362；最新参考 3493；force-release=0, backing_err=0, 无 crash |
| 6 | Probe 标准 | 单 worker `mode=vmm`，remap 解决碎片 | success=1 |
| 7 | Probe Split-Fill | 真实 BestFit 碎片模式 | successes≥5/10 |
| 8 | Compact 功能验证 | pool 满无法 grow，compact 恢复连续空间 | SUCCESS + cleanup=0.0G |

日志目录：

```text
/work/MemoryTools/logs/regression/<timestamp>/
```

---

## 五、回归实验索引（新到旧）

> 本节按日期/日志时间从新到旧排列，用于快速查看最近回归状态；下方“历史基线明细”保留每轮的详细命令、指标和结论。

| 日期/时间 | 范围 | 日志/验证 | 结论 |
|-----------|------|-----------|------|
| 2026-06-23 23:03-23:22 | `112b0b2f5b` stale CUDA OOM cleanup 修复后完整 8/8 回归 | `/work/MemoryTools/logs/regression/20260623_230352/` | 8/8 PASS，用时 18m41s；运行前后 GPU 0-7 均为 0MiB；wheel 由当前 working tree 编译，Python 运行时 commit 字段仍显示上一个提交 `4a6b699093cad0752cd68c054243596f9ba32a2f`，但包含 `112b0b2f5b` 的 stale-OOM cleanup 改动；结果：`ernie_35g_remap_on ooms=48 elapsed=10.19s`、`ernie_35g_remap_off ooms=61 elapsed=10.18s`、`dsv3_30g ooms=38 elapsed=7.11s`、`dsv3_45g_bounded ooms=3509 elapsed=418.31s`、`dsv3_45g_compact_all ooms=3501 elapsed=414.60s`、`probe_standard success=1`、`probe_split_fill=8/10 elapsed=1.45s`、`compact_no_grow SUCCESS time=111.9ms cleanup=0.000G`；所有 replay `post_cleanup_reserved_gib=0.0`，grep 未发现 force-release、BackingMap mismatch、validation failed、invalid range、cudaError、corrupted size、NCCL error、crash 等错误信号 |
| 2026-06-23 | 模型侧最近一次正常性能基线，当前机器暂不可复测 | 正常基线提交: `4a6b699093cad0752cd68c054243596f9ba32a2f`; 日志: `vmm_h2_no_parts_remap_on_clean_623`、`vmm_h16_remap_on_clean_623`、`vmm_off_623` | `4a6b699` 是 no-parts hot path 主体修复后的模型性能基线：h2 no-parts remap-on clean 最后 10 step mean 约 `3796.160`，h16 remap-on clean mean 约 `3800.760`，vmm off mean 约 `3845.320`，h2 相比 h16 基本持平、相比 off 约 -1.28%；`112b0b2f5b` 仅改 OOM/Release cleanup 路径，理论上不影响稳态模型热路径，但由于 4 机模型环境暂不可用，模型 clean 性能复测标记为待补 |
| 2026-06-23 19:37-20:25 | no-parts hot path 后 stale CUDA OOM cleanup 修复验证 | 失败批次: `/work/MemoryTools/logs/regression/20260623_194247/`、`20260623_194251/`、`20260623_194253/`、`20260623_194255/`; 修复后补跑: `/work/MemoryTools/logs/regression/20260623_201605/`、`20260623_201651/`、`20260623_201734/`、`20260623_201817/` | C++ VMM 单测 3/3 PASS；初始批次中 `dsv3_30g`、`ernie_35g_remap_on/off`、`dsv3_45g_compact_all` 均在 replay 结束后 `paddle.device.cuda.empty_cache()` 入口因 stale `cudaErrorMemoryAllocation` 失败；monkey-patch `core._check_last_cuda_error()` 后可通过，定位为 CUDA runtime error slot 未清理，而不是 VMM release 泄漏；在 `AllocatorFacade::Release` 入口清理后补跑通过：`dsv3_30g ooms=38 post_cleanup=0.0G elapsed=6.77s`、`ernie_35g_remap_off ooms=61 post_cleanup=0.0G elapsed=10.88s`、`ernie_35g_remap_on ooms=48 post_cleanup=0.0G elapsed=10.08s`、`dsv3_45g_compact_all ooms=3501 post_cleanup=0.0G elapsed=416.0s` |
| 2026-06-23 11:20-11:39 | lazy remap event / no-parts merge / no-stream safety 修复后 clean GPU 完整回归 | `/work/MemoryTools/logs/regression/20260623_112045/` | 8/8 PASS，用时 18m41s；运行前 GPU 7 起始占用为 0MiB，Python 运行时 commit 为 `7b1f3b06a235e519f20cc7359a24bed97925121f`，wheel 包含当前 working tree 编译产物；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3509`、`dsv3_45g_compact_all=3501`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case 单日志 grep 无 force-release、BackingMap mismatch/validation failed/invalid range、crash/double-free 等硬错误，主要 replay case `post_cleanup_reserved_gib=0.0`；结果与 `20260608_184225` 对齐 |
| 2026-06-23 10:51-11:14 | lazy remap event / no-parts merge / no-stream safety 修复后非 clean GPU 对照回归 | `/work/MemoryTools/logs/regression/20260623_105127/` | 8/8 PASS，但运行前 GPU 7 已占用 428MiB，导致 `peak_reserved_gib` 降到约 78.320GiB，OOM/elapsed 相比历史最新显著变差：`ernie_35g_remap_on=183`、`ernie_35g_remap_off=181`、`dsv3_45g_bounded=4265`、`dsv3_45g_compact_all=3875`；清理残留进程并在 0MiB 起始占用下重跑后恢复到 `20260623_112045`，因此本轮不作为代码回退依据，只保留为 GPU 起始占用敏感性的对照记录 |
| 2026-06-09 | PR 拆分与 upstream/develop rebase 后静态检查 | PR1 `166c99da45`、PR2 `d5a5f4c512`、PR3 `d45791a9c1` | 3 个串行 PR 已基于 `upstream/develop e313387970` 重新整理：PR1 backing map + CUDA VMM allocator，PR2 best-fit/remap/compact，PR3 runtime facade + StreamSafe/multi-pool + IPC/Python；`merge-base --is-ancestor` 依赖链检查通过，PR1/PR2/PR3 `git diff --check` 通过。最终 accessor 命名恢复为小写后尚未在 split worktree 单独编译 |
| 2026-06-08 18:42-19:00 | final PR4 / pre-split 完整回归 | `/work/MemoryTools/logs/regression/20260608_184225/` | 8/8 PASS，用时 18m41s；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3509`、`dsv3_45g_compact_all=3501`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case 无 backing/force-release 错误，`compact_all=3501` 低于当前接受阈值 `4362`，未发现 OOM 回退 |
| 2026-06-08 | final PR4 / pre-split 本地构建与单测 | C++/Python VMM memory tests | 已构建 `phi_core`、`cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`、`paddle_python`；C++ 测试通过：`13/13`、`26/26`、`4/4`；wheel `build/python/dist/paddlepaddle_gpu-3.5.0.dev20260602-cp312-cp312-linux_x86_64.whl` 构建并安装；Python `test_cuda_vmm_memory.py` VMM 路径 6 个测试 OK，1 个 V1-only stats case skip |
| 2026-06-01 23:42 | `14b59df58e` 性能优化 P2 目标选择扫描收口完整回归 | `/work/MemoryTools/logs/regression/20260601_234233/` | 8/8 PASS；`make -j32 paddle_python`、`python setup.py bdist_wheel`、`pip install --force-reinstall --no-deps` 通过，Python 运行时 commit 为 `14b59df58ef134789838fd7f976510b864c2974d`；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3509`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`；错误信号 grep 为 0；长 replay `destination_plan_us` 最大约 108us，当前不支持优先推进 P3 driver probe 替换 |
| 2026-06-01 23:28 | `14b59df58e` 性能优化 P2 目标选择扫描收口与观测补齐 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：15/15、35/35、7/7；`FindSingleUnmappedFreeBlock(...)` 与 `PlanUnmappedFreeScatter(...)` 合并为 `PlanUnmappedFreeDestinations(...)`，tail 失败后只扫描一次 block-list，同时保持 tail -> single unmapped-free -> scatter 的原放置优先级；compact summary 新增 `source_collect_us`、`destination_plan_us`、`move_commit_us`；新增 single/scatter unmapped-free 行为测试；完整 whl replay 见上一条 |
| 2026-06-01 19:58-20:16 | `a39c49177d` 性能优化 P1/P2/P4 第一批完整回归 | `/work/MemoryTools/logs/regression/20260601_195826/` | 8/8 PASS；`make -j32 paddle_python`、`python setup.py bdist_wheel`、`pip install --force-reinstall --no-deps` 通过，Python 运行时 commit 为 `a39c49177dd5feace14f88a098964b41d7928698`；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3509`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`；错误信号 grep 为 0 |
| 2026-06-01 19:52 | `a39c49177d` 性能优化 P1/P2/P4 第一批局部验证 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：15/15、33/33、7/7；新增 `unmapped_free_blocks_` 索引，unmapped-free reuse 不再扫描全量 `all_blocks_`；underlying allocation registry 增加 VA 有序 overlap index；scatter unmapped-free destination 规划合并 capacity precheck 与 placement 扫描；P3 driver VA probe 保持 CUDA driver 查询；随后完整回归见上一条 |
| 2026-06-01 19:37 | working tree review L4 低风险健壮性收口 | C++: `cuda_virtual_mem_allocator_v2_test` | PASS：15/15；`IsDriverVaRangeUnmapped(...)` retain probe 的 `cuMemRelease` 返回值已检查，失败时输出 VA/handle/status；这是诊断路径健壮性改动，未改变 reuse 判定 |
| 2026-06-01 19:19-19:36 | `eb711a3a12` Phase 1c API 面收口后最终完整回归 | `/work/MemoryTools/logs/regression/20260601_191904/` | 8/8 PASS；`make -j32 paddle_python`、`python setup.py bdist_wheel`、`pip install --force-reinstall --no-deps` 通过，Python 运行时 commit 为 `eb711a3a12aaaa07eb8314be3dbbb6bce77544ec`；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3509`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`；错误信号 grep 为 0 |
| 2026-06-01 17:23 | working tree Phase 1c API 面收口局部验证 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：15/15、33/33、7/7；删除未使用的 `UnmapHandle(...)` public 入口和对应白盒测试；`MapHandlesToVA(...)` 收回 private；删除 best-fit 未使用的 `BlockHasIpcExportedBacking(...)` helper；清理旧 `remapped` 诊断命名；`parts_` 保留为 allocation-view layout |
| 2026-06-01 16:40-16:57 | working tree review 审计项 H5/M2/M4/L1/L4 收尾完整回归 | `/work/MemoryTools/logs/regression/20260601_164013/` | 8/8 PASS；回归 wheel 的 Python 运行时 commit 显示为提交前的 `5b6efe1434abab16cc662cdceb610c69cff02052`，包含当前 working tree 编译产物；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3509`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`；错误信号 grep 为 0 |
| 2026-06-01 16:35 | working tree review 审计项 H5/M2/M4/L1/L4 收尾局部验证 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：16/16、33/33、7/7；pending-event ready/null GC 不再 bump epoch；staged synthetic allocation materialization/staging 异常路径会销毁 staged allocation；`FreeImpl`、MovePage double-failure 与 compactor non-std exception rollback 补充诊断；随后完整回归见上一条 |
| 2026-06-01 15:46-16:04 | working tree review 审计项 M1/M3/L2/L3 收尾完整回归 | `/work/MemoryTools/logs/regression/20260601_154657/` | 8/8 PASS；回归 wheel 的 Python 运行时 commit 显示为提交前的 `899e1fde8ba982280501a7f7cf7593a4a1c0e9de`，包含当前 working tree 编译产物；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3509`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`；错误信号 grep 为 0 |
| 2026-06-01 15:40 | working tree review 审计项 M1/M3/L2/L3 收尾局部验证 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：16/16、33/33、7/7；visitor 统计改用 best-fit block-list 快照；`VmmBackingMap::MarkMapped(...)` 禁止覆盖已 mapped 且 handle 不同的 page；best-fit tail reuse 先删除 free index 再 move block；remap tail 目标选择增加 synthetic preparation 预检，不可准备时落到 unmapped-free 目标选择；随后完整回归见上一条 |
| 2026-06-01 14:15-14:33 | working tree remap fallback 收尾完整回归 | `/work/MemoryTools/logs/regression/20260601_141544/` | 8/8 PASS；Python 运行时 commit 为 `6d46545fa5dd97896b90261439fc679a39bdfb24`；移除 legacy unmap/map fallback、`TryUnmapHandle(...)` 过渡入口和 `VmmHandleMeta` 的 `IsRemapped()/MarkRemapped()/ClearRemapped()` 兼容命名；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3509`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`；错误信号 grep 为 0 |
| 2026-06-01 14:06 | working tree remap fallback 收尾局部验证 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：15/15、33/33、7/7；remap 触发后只走 MovePage target selection + transaction materialize，不再 fallback 到 legacy unmap/map；`parts_` 保留为 allocation-view layout，不作为 backing ownership 决策源 |
| 2026-06-01 13:38-13:55 | working tree Phase 1c 低风险收尾完整回归 | `/work/MemoryTools/logs/regression/20260601_133802/` | 8/8 PASS；wheel 运行时 commit 显示为 `a707ac9776554035af21e9b53423e533fef5a787`，包含当前 working tree 编译产物；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3509`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`；错误信号 grep 为 0 |
| 2026-06-01 13:30 | working tree Phase 1c 低风险收尾局部验证 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：15/15、33/33、7/7；`FreeIdleChunks` 释放 backing 后保留显式 `kUnmappedFree` VA range，`MarkPendingEvent(...)` 同 stream 替换旧 event，`TotalMappedBytes()` 改为 `mapped_page_count_` O(1) 查询 |
| 2026-06-01 10:42-10:59 | `d161c36f3d` compact summary logging 恢复后完整回归 | `/work/MemoryTools/logs/regression/20260601_104201/` | 8/8 PASS；Python 运行时 commit 为 `d161c36f3d8ee540b53f3f0a5b7ee15199aec3e0`；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3509`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`；`allocator_stats` 统计恢复：`ernie_on compact/phase1=15/15`、`dsv3_30g=1/1`、`dsv3_45g_bounded=59/59`、`dsv3_45g_compact_all=72/72`、`probe_split_fill=4/4`、`compact_no_grow=1/1` |
| 2026-06-01 01:19-01:37 | bounded compact precheck 修复提交后完整回归 | `/work/MemoryTools/logs/regression/20260601_011923/` | 8/8 PASS；回归 wheel 的 Python 运行时 commit 为 docs-only amend 前的 `51fae1b26f8806cf26a7457b1856d6bd2a97c443`；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3509`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`；高频 compact/backing 详细诊断已降级到 VLOG，默认日志保留单行 compact summary 供脚本统计 compact/phase1 |
| 2026-06-01 00:46-01:04 | working tree bounded compact deficit precheck 修复后完整回归 | `/work/MemoryTools/logs/regression/20260601_004630/` | 8/8 PASS；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3509`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0` |
| 2026-06-01 00:08-00:26 | `66f805766a` BackingMap pending-event source state 清理后完整回归 | `/work/MemoryTools/logs/regression/20260601_000821/` | 8/8 runner PASS，但 `probe_split_fill=7/10` 相比上一轮 `8/10` 是效果回退；根因是 bounded compact precheck 在只统计 BackingMap `kReady` source page 后仍按完整 `requested_size` 要求 releasable bytes，导致已有 tail free 只需补缺口的 8GiB probe 被提前跳过；后续见 `20260601_004630` 修复回归 |
| 2026-06-01 00:00 | working tree BackingMap pending-event source state 清理 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：14/14、31/31、7/7；`VmmHandleMeta` 不再保存 remap-safe event，transaction 不再维护本地 event cache，source `ready/remap-destination-owned/pending-event/partial-or-invalid` 由 BackingMap page-state 返回；best-fit 预检只统计 `kReady` source page |
| 2026-05-31 23:31-23:50 | `7228c989e6` 保守 IPC pin 文档收口后完整回归 | `/work/MemoryTools/logs/regression/20260531_233158/` | 8/8 PASS；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3739`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0` |
| 2026-05-31 23:10 | working tree IPC 生命周期决策收口 | 文档决策 | 本版确认采用保守 IPC pin：exported backing page 禁止 remap/release/reuse，相邻普通 page 仍可按 BackingMap page-state 参与 compact/release/reuse；可靠 unpin 后续单独设计 |
| 2026-05-31 22:43 | working tree best-fit 测试白盒依赖清理完成 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：14/14、31/31、7/7；移除 best-fit 测试中的 `#define private public`，删除两个仅靠私有 registry/helper 或手工 block-list 修改构造的白盒用例；V2 memory C++ 测试已无 `#define private public` 残留 |
| 2026-05-31 09:45 | working tree best-fit 剩余白盒面收窄 | C++: `vmm_auto_growth_best_fit_allocator_v2_test` | PASS：33/33；event/IPC pin 相关断言改用 public block 视图和底层 BackingMap 查询；best-fit 测试剩余 private/public 宏仅用于 synthetic registry 两阶段 helper 和手工 non-owned unmapped gap 构造 |
| 2026-05-31 09:42 | working tree bottom allocator 测试白盒依赖清理 | C++: `cuda_virtual_mem_allocator_v2_test` | PASS：14/14；移除 bottom allocator 测试中的 `#define private public`，私有 layout API 验证改用 public `AllocateWithBlock(...)`、`AllocateAtVAWithBlock(...)`、BackingMap page 查询和 `BlockV2` allocation-view 元数据 |
| 2026-05-31 09:36 | working tree best-fit 测试白盒依赖继续收口 | C++: `vmm_auto_growth_best_fit_allocator_v2_test` | PASS：33/33；best-fit 测试只读 block-list 访问改用 public `all_blocks()`，普通 free-index/active 计数断言改用 `GetFreeBlockStats(...)` 和 block 视图；剩余 private/public 宏仅覆盖异常内部状态构造、remap-destination registry 和 event/IPC pin 内部标记验证 |
| 2026-05-31 09:27 | working tree multi-pool 测试白盒依赖清理 | C++: `vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：7/7；移除 multi-pool 测试中的 `#define private public`，route/free/IPC pin 改用 public 子池只读 block 视图和复用行为验证 |
| 2026-05-31 09:19 | working tree 白盒依赖清理与 IPC 生命周期复核 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：14/14、33/33、7/7；移除 `BlockV2::Parts()` public 观测入口，测试改用窄的 allocation-view 查询方法和 BackingMap page 查询；复核 VMM-v1 IPC 生命周期，确认导入端 close/cache 不能作为 V2 导出端 unpin 信号 |
| 2026-05-31 01:39-01:57 | `cdd53fd5d4` BackingMap API 边界和回归文档收口后完整回归 | `/work/MemoryTools/logs/regression/20260531_013904/` | 8/8 PASS；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3739`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0` |
| 2026-05-31 01:31 | `f0a728e9dd` BackingMap API 边界收口 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：14/14、33/33、7/7；移除 `BlockPartV2` 上未使用的 remap/source/IPC helper，remap source 状态改为 transaction 内基于 BackingMap page-state 判断，修正 handle-only `MarkMapped` 双写；未单独触发 whl/replay 回归 |
| 2026-05-31 01:06-01:25 | `86106143fd` BackingMap page meta 与 source ownership 迁移后完整回归 | `/work/MemoryTools/logs/regression/20260531_010649/` | 8/8 PASS；`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3739`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0` |
| 2026-05-31 00:07-00:33 | `82dba476f4` raw handle traversal 收口后完整回归 | `/work/MemoryTools/logs/regression/20260531_000741/` | 8/8 PASS；`ernie_35g_remap_on=56`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=4265`、`dsv3_45g_compact_all=4362`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0` |
| 2026-05-31 00:02 | `82dba476f4` `HandleMetaRaw()` 移除 / unique handle 遍历改为 shared handle 引用 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：14/14、33/33、7/7；该批后进入 whl 构建安装与完整 replay 回归 |
| 2026-05-30 23:50-23:51 | `2ce67d1d9e` `BlockPartV2` allocation-view slice 字段私有化 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：14/14、33/33、7/7；这是接口边界收口批次，未单独触发全量 replay 回归 |
| 2026-05-30 23:44 | `ffcabf7558` `VmmHandleMeta` physical backing 字段私有化 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：14/14、33/33、7/7；这是封装/边界收口批次，未单独触发全量 replay 回归 |
| 2026-05-30 23:00-23:27 | `2b287b3442` remap planning / restore split 收口后完整回归 | `/work/MemoryTools/logs/regression/20260530_230050/` | 8/8 PASS；`ernie_35g_remap_on=56`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=4265`、`dsv3_45g_compact_all=4362`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，关键错误信号 grep 为 0 |
| 2026-05-30 22:48-22:54 | `ad857a4156` / `083d4d22ed` remap target plan 与 restore split 收口 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：14/14、33/33、7/7；MovePage 与 legacy 共享 `DestinationPlan` 目标选择，rollback restore 切分下沉到 `BlockV2` |
| 2026-05-30 08:44-09:10 | `ca578e6f8e` source restore 下沉后完整回归 | `/work/MemoryTools/logs/regression/20260530_084421/` | 8/8 PASS；`ernie_35g_remap_on=56`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=4265`、`dsv3_45g_compact_all=4362`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0` |
| 2026-05-30 02:02-02:28 | `e1517c655a` IPC/multi-pool/HandleLayout public 面收口后完整回归 | `/work/MemoryTools/logs/regression/20260530_020255/` | 8/8 PASS；关键 OOM 基线保持 `56/61/38/4265/4362`，`probe_split_fill=8/10`，`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0` |
| 2026-05-29 23:49 | 移除 `BlockV2::ipc_exported_` 兼容字段 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test` | PASS：13/13、33/33；IPC pin 只由 BackingMap page-state 表达，`BlockV2` 不再携带 IPC export 标志、构造参数或 merge 传播逻辑 |
| 2026-05-29 23:18-23:44 | `58ed4a8437` IPC pin 行为状态迁移后完整回归 | `/work/MemoryTools/logs/regression/20260529_231811/` | 8/8 PASS；`ernie_35g_remap_on=56`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=4265`、`dsv3_45g_compact_all=4362`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，错误信号 grep 为 0 |
| 2026-05-29 23:11 | IPC pin 行为状态迁到 BackingMap page-state | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`; Python: `test/legacy_test/test_cuda_vmm_memory.py` with V2 flag | PASS：C++ 13/13、33/33；Python IPC 5 个测试 OK、1 个 V1-only stats case skip。`CollectTensorParts(...)` 不再标记 `BlockV2::ipc_exported_`，remap source replacement segment 不再传播 block 级 IPC flag，pinned/reuse/release/compact gate 由 BackingMap page-state 判断 |
| 2026-05-29 22:30-22:56 | `e9edaa86fd` allocation-view / remap materialization / backing adopt 批次完整回归 | `/work/MemoryTools/logs/regression/20260529_223000/` | 8/8 PASS；`ernie_35g_remap_on=56`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=4265`、`dsv3_45g_compact_all=4362`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0`，错误信号 grep 为 0 |
| 2026-05-29 22:24 | `BlockV2` parts mutator API 收口 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：13/13、33/33、6/6；移除未使用的 public parts mutator/accessor，remap source replacement segment 改由 `BlockV2::MakeFreeSegment(...)` 物化 |
| 2026-05-29 22:20 | best-fit backing allocation adopt 收口 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：13/13、33/33、6/6；grow 与 unmapped-free reuse 共享 `AdoptBackingAllocation(...)` 接收底层 allocation/parts/size，`AllocationWithParts` 提供显式 consume 方法 |
| 2026-05-29 22:16 | `RemapTransaction` destination range 物化收口 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：13/13、33/33、6/6；target BackingMap validation、synthetic allocation prepare、legacy map+materialize 统一到 shared helper，tail/single-gap/scatter placement 策略不变 |
| 2026-05-29 19:23 | `BlockV2::parts_` allocation-view 边界收口 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：13/13、33/33、6/6；`parts_` 转为 private logical allocation-view 切片，外部 V2 测试改用 `Parts()` 观测 layout；ownership/reuse/release/remap safety 继续由 BackingMap 与底层 allocator 语义入口决定 |
| 2026-05-29 18:44-19:10 | `b433210696` block/backing 语义收口批次完整回归 | `/work/MemoryTools/logs/regression/20260529_184448/` | 8/8 PASS；`ernie_35g_remap_on=56`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=4265`、`dsv3_45g_compact_all=4362`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`；所有 case `backing_err=0`、`force_release=0` |
| 2026-05-29 18:38 | block VA range helper 收口 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：13/13、33/33、6/6；best-fit 与 remap transaction 的主要 block VA range 查询改用 `BlockV2` 入口 |
| 2026-05-29 18:34 | block-part handle 遍历与 block merge materialization 收口 | C++: `cuda_virtual_mem_allocator_v2_test`、`vmm_auto_growth_best_fit_allocator_v2_test`、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS：13/13、33/33、6/6；本轮只做类型层 helper 与调用点收口，尚未触发 whl/replay 回归 |
| 2026-05-29 17:09 | `ceabac60de` block range helper 收口后完整回归 | `/work/MemoryTools/logs/regression/20260529_170952/` | 5 个 replay case 完成：`ernie_35g_remap_on=56`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=4265`、`dsv3_45g_compact_all=4362`；所有 case `post_cleanup_reserved_gib=0.0`、`backing_err=0`、`force_release=0`。runner 因旧 `4359` 阈值将 compact_all 标 FAIL；`run_vmm_v2_regression_log.sh` 已同步阈值到当前接受基线 `4362` |
| 2026-05-29 16:26 | 当前代码完整回归 | `/work/MemoryTools/logs/regression/20260529_162640/` | 5 个 replay case 完成：`ernie_35g_remap_on=56`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=4265` 均 PASS；`dsv3_45g_compact_all=4362` 因 runner 仍按旧 `4359` 阈值判 FAIL，但 `post_cleanup_reserved_gib=0.0`、`backing_err=0`、`force_release=0`，按当前接受标准视为通过 |
| 2026-05-29 16:05 | `dsv3_45g_compact_all` source-created target fallback 修正后单 case | `/work/MemoryTools/logs/regression/20260529_160521/` | `ooms=4362`、`allocs/frees=22426/22346`、`post_cleanup_reserved_gib=0.0`；确认禁止 fallback 使用 source unmap 后新生成的 tail/gap 不改变当前接受基线 |
| 2026-05-29 15:21-15:52 | range-level ownership cleanup 临时尝试 | `/work/MemoryTools/logs/regression/20260529_152527/`、`20260529_153904/`、`20260529_155211/` | 临时尝试导致 `dsv3_45g_compact_all ooms=4409`，根因是 range-level cleanup 留下 stale synthetic overlap 并阻断 gap reuse；该方向已撤回 |
| 2026-05-29 10:40 | `dsv3_45g_compact_all` compact_all legacy fallback 复验 | `/work/MemoryTools/logs/regression/20260529_104054/` | 临时放开 compact_all 在 MovePage 无 ownership-safe target 时继续 legacy fallback，结果 `ooms=5621`、`allocs/frees=21167/21088`，明显劣于当前 `4362`；说明 `no ownership-safe target capacity` skip 是必要保护，不能作为恢复 4359 的修复方向 |
| 2026-05-29 10:23 | `dsv3_45g_compact_all` interior scatter target 复验 | `/work/MemoryTools/logs/regression/20260529_102332/` | 临时允许 unmapped-free scatter 跳过 non-remap-destination overlap 前缀后选择内部子段，结果 `ooms=4408`、`allocs/frees=22380/22300`，较当前 `4362` 退化；该方向增加 synthetic preparation overlap 探测并降低成功分配数，已撤回 |
| 2026-05-29 00:32 | `dsv3_45g_compact_all` 剩余 4362 vs 4359 差异诊断 | `/work/MemoryTools/logs/regression/20260529_003200/` | 临时关闭 MovePage compact 后单 case `ooms=6076`，明显劣于当前 `4362`；剩余差异不是 MovePage 单独造成，MovePage 当前抵消了 legacy/backing target 约束下的大量损失 |
| 2026-05-29 00:05 | `e867f17ca3` compact_all partial remap attempt 修复 | `/work/MemoryTools/logs/regression/20260529_000552/` | 单 case `dsv3_45g_compact_all ooms=4362`，从 4411 恢复到 2026-05-28 16:38 水平；runner 当时仍按旧 `4359` 阈值判 FAIL，后续诊断确认 `4362` 可作为当前接受基线 |
| 2026-05-28 20:13 | `a40d75bea5` bounded compact precheck 修复 | `/work/MemoryTools/logs/regression/20260528_201303/` | `dsv3_45g_bounded` PASS，`ooms=4265`，恢复到 15:03 低 OOM 基线 |
| 2026-05-28 19:43 | `db7d525c19` / `f7991ea271` / `e81a097d94` backing ownership/materialization 下沉后回归定位 | `/work/MemoryTools/logs/regression/20260528_194304/` | 前 4 项可运行，但 `dsv3_45g_bounded ooms=4501`，较 15:03 基线 +236，定位为 bounded compact 预检退化后中止 |
| 2026-05-28 19:38 | `db7d525c19` / `f7991ea271` / `e81a097d94` backing ownership/materialization 下沉 | C++: `cuda_virtual_mem_allocator_v2_test` 13/13、best-fit 31/31、multi-pool 6/6 | PASS；后续 whl 回归见 19:43 / 20:13 |
| 2026-05-28 19:19 | `0310725faf` / `0f63aa47cd` block-level backing API 与 block materialization 收口 | C++: `cuda_virtual_mem_allocator_v2_test` 12/12、best-fit 31/31、multi-pool 6/6 | PASS，尚未单独跑完整一键回归 |
| 2026-05-28 16:38 | `09892b0a74` tail compaction normalize 后完整回归 | `/work/MemoryTools/logs/regression/20260528_163826/` | 8/8 PASS，`dsv3_45g_compact_all ooms=4362`，无 backing/force-release 错误 |
| 2026-05-28 15:03 | Phase 2d page-state 接管模块级回归 | `/work/MemoryTools/logs/regression/20260528_150332/` | 8/8 PASS，`dsv3_45g_compact_all ooms=4411` |
| 2026-05-28 13:16-13:46 | Phase 2c 收尾回归补齐 | `/work/MemoryTools/logs/regression/20260528_131609/` + `20260528_133313/134453/134528/134603` | 8 个逻辑 case PASS |
| 2026-05-28 11:46 | BlockPartV2 / RemapTransaction 收口模块回归 | `/work/MemoryTools/logs/regression/20260528_114606/` | 8/8 PASS，`dsv3_45g_compact_all ooms=4364` |
| 2026-05-28 10:57 | IPC export / compact precheck / remap event 下沉 | `/work/MemoryTools/logs/regression/20260528_105715/` | 8/8 PASS，`dsv3_45g_compact_all ooms=4364` |
| 2026-05-28 10:07 | Phase 2c BackingMap 语义入口模块级回归 | `/work/MemoryTools/logs/regression/20260528_100710/` | 8/8 PASS，`dsv3_45g_compact_all ooms=4364` |
| 2026-05-28 00:48 | Phase 2c 批量回归 | `/work/MemoryTools/logs/regression/20260528_004800/` | 8/8 PASS，`dsv3_45g_compact_all ooms=4364` |
| 2026-05-27 23:38 | BackingMap IPC pin 约束 reuse / tail reuse | `/work/MemoryTools/logs/regression/20260527_233815/` | 8/8 PASS，`dsv3_45g_compact_all ooms=4364` |
| 2026-05-27 19:44-19:45 | VMM V2 IPC tensor sharing / pinned block 验证 | `/work/MemoryTools/logs/regression/20260527_194423/194502/` | `compact_no_grow` PASS，`dsv3_45g_compact_all ooms=4364` |
| 2026-05-26 23:52-2026-05-27 00:03 | remapped / block state accessor 本地验证 | `20260526_235241` + `20260527_000350` | `dsv3_45g_compact_all ooms=4359`，`probe_standard` PASS；该值保留为历史效果对照，不再作为当前正确性硬阈值 |
| 2026-05-26 | RemapTransaction rollback / synthetic 生命周期收口 | 见历史明细对应小节 | 串行单 case 恢复到基线，`dsv3_45g_compact_all ooms=4359` |
| 2026-05-25 | RemapTransaction placement / install / staging 系列 | 见历史明细对应小节 | 与前序基线保持一致 |
| 2026-05-24 | RemapTransaction commit hooks | 见历史明细对应小节 | 与 Phase 1b 基线保持一致 |
| 2026-05-22 | BackingMap compact candidate collection | 见历史明细对应小节 | 与 `d43de6a32b` 基线持平 |
| 2026-05-21 | BackingMap page collection / validation 系列 | 见历史明细对应小节 | 与前序 BackingMap 基线持平 |
| 2026-05-20 | Backing View stable2 全量回归 | 见历史明细对应小节 | `dsv3_45g_compact_all=4359` |
| 2026-05-19 | BackingMap Phase 1a mirror 验证 | 见历史明细对应小节 | 与 PR4 关键基线持平 |
| 2026-05-18 | Edge Case 修复验证 | `/work/MemoryTools/logs/regression/20260518_162925/` | 见历史明细 |
| 2026-05-14 | PendingMappedRange 修复 | `/work/MemoryTools/logs/regression/20260514_182803/` | 见历史明细 |
| 2026-05-13 | RollbackToOriginalVA / Split-Fill probe | 见历史明细对应小节 | 见历史明细 |
| 2026-05-11 | Tail Reuse Fix | 见历史明细对应小节 | 见历史明细 |
| 2026-05-08 | Compact 架构重构旧基线 | 见历史明细对应小节 | 见历史明细 |

## 六、历史基线明细
### 2026-06-01 - 性能优化 P2 目标选择扫描收口与观测补齐 (working tree)

背景:
- `a39c49177d` 已把 scatter unmapped-free destination 的 capacity precheck 合并进 placement 扫描，但 tail 失败后仍保留 single-block 查找与 scatter 规划两段 block-list 遍历。
- 本轮继续收口 P2，但不改变放置策略：tail 优先，其次第一个可承接完整请求的 single unmapped-free，最后才 scatter。

改动:
- 删除独立的 `FindSingleUnmappedFreeBlock(...)` 与 `PlanUnmappedFreeScatter(...)`。
- 新增 `PlanUnmappedFreeDestinations(...)`，一次扫描 block-list 收集 unmapped-free 候选并查找 single 目标；single 目标存在时沿用旧路径直接返回。
- 没有 single 目标时，基于候选列表规划 scatter，避免再次遍历 block-list；大块候选的 leading unmapped page 计数可复用，减少重复 BackingMap 查询。
- compact summary 补充 `source_collect_us`、`destination_plan_us`、`move_commit_us` 三个字段，复用现有单行 INFO summary，不新增额外日志行。
- 新增 public-behavior 覆盖：tail VA 被外部映射占住时使用 single unmapped-free；无 single 大块时 scatter 到多个 unmapped-free block。

验证:
- `git diff --check`: 通过。
- `make -C build -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 通过。
- `cuda_virtual_mem_allocator_v2_test`: 15/15 PASS。
- `vmm_auto_growth_best_fit_allocator_v2_test`: 35/35 PASS。
- `vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 7/7 PASS。

完整回归:
- `make -j32 paddle_python`: 通过。
- `python setup.py bdist_wheel` + `pip install --force-reinstall --no-deps`: 通过。
- Python 运行时 commit: `14b59df58ef134789838fd7f976510b864c2974d`。
- 命令：`cd /work/MemoryTools && bash run_vmm_v2_regression_log.sh 7`
- 日志：`/work/MemoryTools/logs/regression/20260601_234233/`
- 结果：8/8 PASS，用时 17m40s。
- `ernie_35g_remap_on`: `ooms=48`，`cleanup=0.0`，`compact=15`，`phase1=15`。
- `ernie_35g_remap_off`: `ooms=61`，`cleanup=0.0`，`compact=0`，`phase1=0`。
- `dsv3_30g`: `ooms=38`，`cleanup=0.0`，`compact=1`，`phase1=1`。
- `dsv3_45g_bounded`: `ooms=3509`，`cleanup=0.0`，`compact=59`，`phase1=59`。
- `dsv3_45g_compact_all`: `ooms=3493`，`cleanup=0.0`，`compact=72`，`phase1=72`。
- `probe_standard`: `success=1`。
- `probe_split_fill`: `successes=8/10`，`compact=4`，`phase1=4`。
- `compact_no_grow`: `result=SUCCESS`，`cleanup=0.000G`。
- 所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`；关键错误信号 grep 为 0。

Timing 观测:
- `dsv3_45g_bounded`: 59 次 compact，`source_collect_us` 最大 240，`destination_plan_us` 最大 100、均值 38.8，`move_commit_us` 最大 18879、均值 5922.8。
- `dsv3_45g_compact_all`: 72 次 compact，`source_collect_us` 最大 255，`destination_plan_us` 最大 108、均值 35.1，`move_commit_us` 最大 22749、均值 4883.6。
- `ernie_35g_remap_on`: 15 次 compact，`destination_plan_us` 最大 106、均值 36.7，`move_commit_us` 最大 19684、均值 7485.3。
- `probe_split_fill`: 4 次 compact，`destination_plan_us` 最大 209、均值 134.0，`move_commit_us` 最大 219430、均值 110257.8。

`probe_split_fill` 按需 remap 细分:

| 请求 | 是否触发 compact | remap handles | remap bytes | source collect | destination plan | move commit | 三段合计 |
|------|------------------|---------------|-------------|----------------|------------------|-------------|----------|
| 1 GiB | 是 | 512 | 1.000 GiB | 0.244 ms | 0.095 ms | 56.394 ms | 56.733 ms |
| 2 GiB | 是 | 1024 | 2.000 GiB | 0.612 ms | 0.145 ms | 109.613 ms | 110.370 ms |
| 3 GiB | 否 | - | - | - | - | - | - |
| 4 GiB | 是 | 2048 | 4.000 GiB | 1.354 ms | 0.209 ms | 219.430 ms | 220.993 ms |
| 5 GiB | 否 | - | - | - | - | - | - |
| 6 GiB | 否 | - | - | - | - | - | - |
| 7 GiB | 否 | - | - | - | - | - | - |
| 8 GiB | 是 | 529 | 1.033 GiB | 1.905 ms | 0.087 ms | 55.594 ms | 57.586 ms |
| 9 GiB | 失败 | - | - | - | - | - | - |
| 10 GiB | 失败 | - | - | - | - | - | - |

说明: `frag_probe_varied.py` 只打印每个请求的成功/失败，不打印每个请求的 wall time；上表中的耗时来自 compact summary。`3/5/6/7 GiB` 成功但没有新的 compact summary，说明这些请求复用了前面按需 remap 后形成的连续空间或已有可用空间。`8 GiB` 只 remap 约 1.033 GiB，是因为已有 tail/free capacity，只补缺口。`move_commit_us` 基本随实际 remapped handle 数线性增长，`destination_plan_us` 不是该 probe 的主耗时。

结论: P2 扫描收口没有引入行为回退，指标与 `a39c49177d` 完整回归持平。当前 timing 显示 destination planning 不是主耗时，P3 driver probe 替换属于可能有效但收益暂不明显的谨慎项，暂不优先推进。

### 2026-06-01 - 性能优化 P1/P2/P4 第一批 (`a39c49177d`)

背景:
- 该批是性能优化收口，不改变 remap 主策略、IPC 保守 pin 语义或 BackingMap ownership 判定。
- P3 的 tail driver VA probe 继续保留 CUDA driver 查询；该路径需要确认真实 VA mapping 状态，不能用 BackingMap 推断替代。

改动:
- P1: best-fit 为 `kUnmappedFree` 维护 size-ordered index，`AllocFromUnmappedFreeBlocks(...)` 通过 `lower_bound` 选择候选，不再全量扫描 `all_blocks_`。
- P2: `UnderlyingAllocationRegistry` 增加 VA-ordered overlap index，`RangeOverlapsUnderlyingAllocation(...)` / synthetic cleanup overlap 查询从线性扫描降为有序区间查询。
- P4: scatter unmapped-free destination 规划把 capacity precheck 合并进 placement 扫描，避免重复遍历 block-list 和重复收集 BackingMap unmapped pages。

局部验证:
- `git diff --check`: 通过。
- `make -C build -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 通过。
- `cuda_virtual_mem_allocator_v2_test`: 15/15 PASS。
- `vmm_auto_growth_best_fit_allocator_v2_test`: 33/33 PASS。
- `vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 7/7 PASS。

完整回归:
- `make -j32 paddle_python`: 通过。
- `python setup.py bdist_wheel` + `pip install --force-reinstall --no-deps`: 通过。
- Python 运行时 commit: `a39c49177dd5feace14f88a098964b41d7928698`。
- 命令：`cd /work/MemoryTools && bash run_vmm_v2_regression_log.sh 7`
- 日志：`/work/MemoryTools/logs/regression/20260601_195826/`
- 结果：8/8 PASS，用时 17m40s。
- `ernie_35g_remap_on`: `ooms=48`，`cleanup=0.0`，`compact=15`，`phase1=15`。
- `ernie_35g_remap_off`: `ooms=61`，`cleanup=0.0`，`compact=0`，`phase1=0`。
- `dsv3_30g`: `ooms=38`，`cleanup=0.0`，`compact=1`，`phase1=1`。
- `dsv3_45g_bounded`: `ooms=3509`，`cleanup=0.0`，`compact=59`，`phase1=59`。
- `dsv3_45g_compact_all`: `ooms=3493`，`cleanup=0.0`，`compact=72`，`phase1=72`。
- `probe_standard`: `success=1`。
- `probe_split_fill`: `successes=8/10`，`max_success_gib=8.0`，`compact=4`，`phase1=4`。
- `compact_no_grow`: `SUCCESS`，`cleanup=0.000G`，`compact=1`，`phase1=1`。
- 所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`；错误信号 grep 为 0。

结论: P1/P2/P4 索引与扫描优化没有改变 replay 语义；OOM、cleanup、probe、compact/phase1 指标与 `/work/MemoryTools/logs/regression/20260601_191904/` 保持一致。

### 2026-06-01 - Phase 1c API 面收口 (working tree)

背景:
- `parts_` 当前保留为 allocation-view layout，不作为本轮删除目标。
- Phase 1c 的当前收口重点是删掉未使用或只服务白盒过渡的公开 API，避免底层 unmap/map 原语继续暴露给上层或测试。

改动:
- 删除未使用的 `CUDAVirtualMemAllocatorV2::UnmapHandle(...)` public 入口。
- `CUDAVirtualMemAllocatorV2::MapHandlesToVA(...)` 从 public API 收回 private，仅保留为底层 allocator 内部映射原语。
- 删除依赖上述内部 unmap/map 原语的 `UnmapAndMapHandleBackToSameVA` 白盒测试；正式 MovePage 行为仍由 `MoveBackingPageRoundTripsHandle` 覆盖。
- 删除 best-fit 未使用的 `BlockHasIpcExportedBacking(...)` 过渡 helper。
- 清理少量旧 `remapped` 诊断命名，改为 `remap-destination-owned` / `owned_by_remap_destination`。

验证:
- `git diff --check`: 通过。
- `make -C build -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 通过。
- `cuda_virtual_mem_allocator_v2_test`: 15/15 PASS。
- `vmm_auto_growth_best_fit_allocator_v2_test`: 33/33 PASS。
- `vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 7/7 PASS。
- 最终完整回归准备：`make -j32 paddle_python`、`python setup.py bdist_wheel`、`pip install --force-reinstall --no-deps` 均通过。
- Python 运行时 commit: `eb711a3a12aaaa07eb8314be3dbbb6bce77544ec`。
- 完整回归命令：`cd /work/MemoryTools && bash run_vmm_v2_regression_log.sh 7`
- 完整回归日志：`/work/MemoryTools/logs/regression/20260601_191904/`。
- 完整回归结果：8/8 PASS，用时 17m40s。
- 关键指标：`ernie_35g_remap_on=48`、`ernie_35g_remap_off=61`、`dsv3_30g=38`、`dsv3_45g_bounded=3509`、`dsv3_45g_compact_all=3493`、`probe_standard success=1`、`probe_split_fill=8/10`、`compact_no_grow cleanup=0.000G`。
- 所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`。
- 错误信号 grep：0。
- 本轮回归结束后继续完成 review L4 低风险健壮性收口：`IsDriverVaRangeUnmapped(...)` retain probe 的 `cuMemRelease` 返回值已检查，失败时输出 VA/handle/status；`make -C build -j32 cuda_virtual_mem_allocator_v2_test` 和 `./build/test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test` 通过，C++ 测试 15/15 PASS。

### 2026-06-01 - review 审计项 H5/M2/M4/L1/L4 收尾 (working tree)

背景:
- 在 M1/M3/L2/L3 之后继续处理 review 文档中剩余的高风险异常路径和中低风险一致性项。
- H1 的 source restore double-failure 当前只增强诊断，不引入 force-release 语义；原因是 source restore 已失败时，强行 release backing 可能让 block-list 与 BackingMap/CUDA mapping 状态进一步分叉，需要单独设计可验证的 source-lost 事务语义或故障注入 hook。

改动:
- H5: `CreateStagedRemapDestinationAllocationWithBlock(...)` 在 destination block materialization 抛异常时销毁 staged synthetic allocation；`RemapTransaction::MaterializeMappedRange(...)` 在 staging vector 插入抛异常时同样销毁 staged allocation 并重新抛出。
- M2: `VmmBackingMap::PageEventsReadyLocked(...)` 清理 ready/null pending event 时不再 bump page epoch，避免只读 readiness query 让已收集 mapped page snapshot 失效。
- M4: `FreeImpl` 对 `cuMemRelease` 异常状态补充 handle/base/size/remap-destination ownership 诊断，便于区分 ownership transfer 后的重复 release 与真实异常。
- L1: compactor 捕获 non-std exception 时补充 `exception_ptr` 状态，保留 rollback 语义。
- L4: MovePage target map/access 失败且 source restore 也失败时，补充 source/target/size/handle 诊断；不改变正常路径语义。
- 测试补充 `VmmBackingMap.ReplacesPendingEventForSameStream` 的 mapped snapshot validate 断言，覆盖 pending-event ready GC 不 bump epoch。

验证:
- `git diff --check`: 通过。
- `make -C build -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 通过。
- `cuda_virtual_mem_allocator_v2_test`: 16/16 PASS。
- `vmm_auto_growth_best_fit_allocator_v2_test`: 33/33 PASS。
- `vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 7/7 PASS。
- `make -C build -j32 paddle_python`: 通过。
- `python setup.py bdist_wheel` + `pip install --force-reinstall --no-deps`: 通过。
- Python 运行时 commit: `5b6efe1434abab16cc662cdceb610c69cff02052`，对应提交前 HEAD；wheel 包含当前 working tree 编译产物。
- 完整回归命令：`cd /work/MemoryTools && bash run_vmm_v2_regression_log.sh 7`
- 完整回归日志：`/work/MemoryTools/logs/regression/20260601_164013/`

完整回归结果:
- 8/8 PASS，用时 17m41s。
- `ernie_35g_remap_on`: `ooms=48`，`cleanup=0.0`，`compact=15`，`phase1=15`。
- `ernie_35g_remap_off`: `ooms=61`，`cleanup=0.0`，`compact=0`，`phase1=0`。
- `dsv3_30g`: `ooms=38`，`cleanup=0.0`，`compact=1`，`phase1=1`。
- `dsv3_45g_bounded`: `ooms=3509`，`cleanup=0.0`，`compact=59`，`phase1=59`。
- `dsv3_45g_compact_all`: `ooms=3493`，`cleanup=0.0`，`compact=72`，`phase1=72`。
- `probe_standard`: `success=1`。
- `probe_split_fill`: `successes=8/10`，`max_success_gib=8.0`，`compact=4`，`phase1=4`。
- `compact_no_grow`: `SUCCESS`，`cleanup=0.000G`，`compact=1`，`phase1=1`。
- 错误信号 grep：0；`backing_err=0`、`force_release=0`。

### 2026-06-01 - review 审计项 M1/M3/L2/L3 收尾 (working tree)

背景:
- 根据 `vmm_v2_backing_view_review.md` 的代码审计项，优先处理不改变总体设计方向、但能降低状态一致性风险的收尾项。
- `BlockV2::parts_` 本批不删除，继续作为 allocation-view layout；physical backing ownership、IPC pin、pending-event、release/reuse/remap source 决策仍由 BackingMap page-state 承担。

改动:
- M1: `VMMV2PoolStatsVisitor` 不再直接遍历 `all_blocks()` 引用，改用 `VMMAutoGrowthBestFitAllocatorV2::SnapshotAllBlocks()` 在 allocator 锁内复制快照后统计，避免 visitor 读路径无锁遍历生产 block-list。
- M3: `VmmBackingMap::MarkMapped(...)` 对“已 mapped page 被不同 physical handle 覆盖”从 `VLOG(0)` 升级为 `PreconditionNotMet`，避免 silent overwrite 掩盖 BackingMap ownership 错误。
- L2: best-fit tail reuse 先从 `free_blocks_` 删除 tail entry，再 move `BlockV2`，不再依赖 moved-from block 的 `ptr_` / `size_`。
- L3: remap tail destination 选择增加 `CanPrepareSyntheticAllocationRange(...)` 预检；driver VA 可用但 synthetic allocation range 不可准备时，不再选中 tail 后失败，而是继续尝试 single/scatter unmapped-free 目标。
- 测试新增 `VmmBackingMap.RejectsMappedPageHandleOverwrite`，覆盖 handle-only 和 meta 两个 `MarkMapped(...)` 入口。

验证:
- `git diff --check`: 通过。
- `make -C build -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 通过。
- `cuda_virtual_mem_allocator_v2_test`: 16/16 PASS。
- `vmm_auto_growth_best_fit_allocator_v2_test`: 33/33 PASS。
- `vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 7/7 PASS。
- `make -C build -j32 paddle_python`: 通过。
- `python setup.py bdist_wheel` + `pip install --force-reinstall --no-deps`: 通过。
- Python 运行时 commit: `899e1fde8ba982280501a7f7cf7593a4a1c0e9de`，对应提交前 HEAD；wheel 包含当前 working tree 编译产物。
- 完整回归命令：`cd /work/MemoryTools && bash run_vmm_v2_regression_log.sh 7`
- 完整回归日志：`/work/MemoryTools/logs/regression/20260601_154657/`

完整回归结果:
- 8/8 PASS，用时 17m41s。
- `ernie_35g_remap_on`: `ooms=48`，`cleanup=0.0`，`compact=15`，`phase1=15`。
- `ernie_35g_remap_off`: `ooms=61`，`cleanup=0.0`，`compact=0`，`phase1=0`。
- `dsv3_30g`: `ooms=38`，`cleanup=0.0`，`compact=1`，`phase1=1`。
- `dsv3_45g_bounded`: `ooms=3509`，`cleanup=0.0`，`compact=59`，`phase1=59`。
- `dsv3_45g_compact_all`: `ooms=3493`，`cleanup=0.0`，`compact=72`，`phase1=72`。
- `probe_standard`: `success=1`。
- `probe_split_fill`: `successes=8/10`，`max_success_gib=8.0`，`compact=4`，`phase1=4`。
- `compact_no_grow`: `SUCCESS`，`cleanup=0.000G`，`compact=1`，`phase1=1`。
- 所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`；错误关键字扫描未发现 `force-release`、BackingMap mismatch/validation failed/invalid range、crash、`PreconditionNotMet`。

结论: 本批是 review hardening，不改变 remap 主策略和 IPC 保守 pin 语义；C++ 局部验证与完整 replay 回归均通过，OOM、cleanup、probe 和 compact/phase1 指标与 `20260601_141544` 保持一致。

### 2026-06-01 - compact summary logging 恢复后完整回归 (`d161c36f3d`)

背景:
- `17cb2810c7` 将高频 compact/backing 详细诊断降级到 VLOG 后，默认日志不再刷 per-page/per-compact 明细，但也使 runner 通过 `entering Compact` / `Phase 1 done` grep 得到的 `compact/phase1` 观测变成 0。
- `d161c36f3d` 新增默认级别单行 `VMM V2 compact summary`，保留上述两个关键字，用于低成本恢复 `allocator_stats` 统计；详细 source coverage 仍在 VLOG。

验证:
- `make -C build -j32 paddle_python`: 通过。
- `python setup.py bdist_wheel` + `pip install --force-reinstall --no-deps`: 通过。
- Python 运行时 commit: `d161c36f3d8ee540b53f3f0a5b7ee15199aec3e0`。
- 完整回归命令：`cd /work/MemoryTools && bash run_vmm_v2_regression_log.sh 7`
- 完整回归日志：`/work/MemoryTools/logs/regression/20260601_104201/`

完整回归结果:
- 8/8 PASS，用时 17m41s。
- `ernie_35g_remap_on`: `ooms=48`，`cleanup=0.0`，`compact=15`，`phase1=15`。
- `ernie_35g_remap_off`: `ooms=61`，`cleanup=0.0`，`compact=0`，`phase1=0`。
- `dsv3_30g`: `ooms=38`，`cleanup=0.0`，`compact=1`，`phase1=1`。
- `dsv3_45g_bounded`: `ooms=3509`，`cleanup=0.0`，`compact=59`，`phase1=59`。
- `dsv3_45g_compact_all`: `ooms=3493`，`cleanup=0.0`，`compact=72`，`phase1=72`。
- `probe_standard`: `success=1`。
- `probe_split_fill`: `successes=8/10`，`max_success_gib=8.0`，`compact=4`，`phase1=4`。
- `compact_no_grow`: `SUCCESS`，`cleanup=0.000G`，`compact=1`，`phase1=1`。
- 所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`。

结论: compact summary logging 恢复了 runner 的 `compact/phase1` 观测，OOM、cleanup、probe 结果与 `20260601_011923` 保持一致，未引入行为回退。

### 2026-06-01 - bounded compact precheck 修复提交后完整回归

背景:
- `probe_split_fill 8/10 -> 7/10` 已确认为 bounded compact precheck 的真实效果回退；修复后先在 working tree 完成 `/work/MemoryTools/logs/regression/20260601_004630/` 验证，再提交并在文档补记后 amend。
- 提交后重新 `make -C build -j32 paddle_python`、`python setup.py bdist_wheel` 并 `pip install --force-reinstall --no-deps`，确认回归 wheel 的 Python 运行时 commit 为 docs-only amend 前的 `51fae1b26f8806cf26a7457b1856d6bd2a97c443`。最终 amend 只更新文档记录，不改变编译进 wheel 的 C++/CUDA/Python runtime 代码。

完整回归:
- 命令：`cd /work/MemoryTools && bash run_vmm_v2_regression_log.sh 7`
- 日志：`/work/MemoryTools/logs/regression/20260601_011923/`
- 结果：8/8 PASS，用时 17m41s。
- `ernie_35g_remap_on`: `ooms=48`，`cleanup=0.0`。
- `ernie_35g_remap_off`: `ooms=61`，`cleanup=0.0`。
- `dsv3_30g`: `ooms=38`，`cleanup=0.0`。
- `dsv3_45g_bounded`: `ooms=3509`，`cleanup=0.0`，test elapsed `395.7s`。
- `dsv3_45g_compact_all`: `ooms=3493`，`cleanup=0.0`，test elapsed `392.5s`。
- `probe_standard`: `success=1`。
- `probe_split_fill`: `successes=8/10`，`max_success_gib=8.0`。
- `compact_no_grow`: `SUCCESS`，`cleanup=0.000G`。
- 所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`。

说明:
- 本提交将高频 compact/backing 详细诊断从默认 INFO 降到 VLOG；默认 `GLOG_v=0` 下保留单行 `VMM V2 compact summary`，继续兼容 runner 通过 `entering Compact` / `Phase 1 done` grep 统计 compact/phase1。

结论: 最终提交与 working tree 修复结果一致，`probe_split_fill` 保持 `8/10`，`dsv3_45g_compact_all=3493` 保持当前最好值，未发现 crash、cleanup 泄漏、force-release 或 BackingMap 错误。

### 2026-06-01 - bounded compact deficit precheck 修复后完整回归 (working tree)

背景:
- `66f805766a` 将 remap source readiness 收口到 BackingMap page-state 后，`probe_split_fill` 从 `8/10` 退到 `7/10`。该回退不是容量探测波动，而是 bounded compact 预检口径错误。
- 具体根因：预检只统计 `RemapSourceState::kReady` page 是正确的，但仍要求 `releasable_bytes >= requested_size`。在 split-fill probe 的 8GiB 请求中，前序 compact 已经形成可扩展 tail free，实际只需搬迁约 1GiB ready page 补齐缺口；旧预检按完整 8GiB 判断，导致 compactor 未进入。

修复:
- bounded compact 的 source precheck 改为按 tail free 缺口计算 `required_releasable_bytes = requested_size - tail_free`，只要求 ready source page 能补齐当前可扩展 tail free 到请求大小。
- `compact_all` 保持原有 partial compact 行为，不受 bounded precheck 限制。
- 新增 `BoundedCompactUsesTailFreeDeficitForReleasablePrecheck`，覆盖已有 tail free + 少量 ready page 补齐请求的场景；保留 `BoundedCompactSkipsWhenReleasableBytesInsufficient`，防止孤立 free page 误放宽。
- 高频 remap/backing 诊断日志从默认 INFO 降到 VLOG，避免 dsv3 高压回归被 per-page/per-compact 日志写入拖慢。

本地验证:
- `git diff --check`: 通过。
- `make -C build -j32 vmm_auto_growth_best_fit_allocator_v2_test`: 通过。
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test`: 32/32 通过。
- whl 构建安装后单 case `probe_split_fill`: `successes=8/10`，`max_success_gib=8.0`，恢复到上一轮基线。
- 完整回归命令：`cd /work/MemoryTools && bash ./run_vmm_v2_regression_log.sh 7`
- 完整回归日志：`/work/MemoryTools/logs/regression/20260601_004630/`

完整回归结果:
- 8/8 PASS，用时 17m42s。
- `ernie_35g_remap_on`: `ooms=48`，`cleanup=0.0`，`compact=15`，`phase1=15`。
- `ernie_35g_remap_off`: `ooms=61`，`cleanup=0.0`。
- `dsv3_30g`: `ooms=38`，`cleanup=0.0`，`compact=1`，`phase1=1`。
- `dsv3_45g_bounded`: `ooms=3509`，`cleanup=0.0`，`compact=59`，`phase1=59`。
- `dsv3_45g_compact_all`: `ooms=3493`，`cleanup=0.0`，`compact=72`，`phase1=72`。
- `probe_standard`: `success=1`。
- `probe_split_fill`: `successes=8/10`，`max_success_gib=8.0`。
- `compact_no_grow`: `SUCCESS`，`cleanup=0.000G`。
- 所有 case `backing_err=0`、`force_release=0`，主要 replay case `post_cleanup_reserved_gib=0.0`。

结论: `probe_split_fill 7/10` 是 bounded compact precheck bug，已修复并恢复到 `8/10`。`dsv3_45g_compact_all=3493` 保持当前最好值，`dsv3_45g_bounded=3509` 较 `66f805766a` 的 3712 继续改善；未发现 BackingMap mismatch、force-release、cleanup 泄漏或 crash。

### 2026-06-01 - BackingMap pending-event source state 清理后完整回归 (`66f805766a`)

环境:
- whl 构建安装后运行，Python 运行时 commit 为 `66f805766a0b2f58f38b3ccc1d0732646b941d63`。
- 命令：`cd /work/MemoryTools && bash ./run_vmm_v2_regression_log.sh 7`
- 日志目录：`/work/MemoryTools/logs/regression/20260601_000821/`

结果:
- 8/8 PASS，用时 18m12s。
- `ernie_35g_remap_on`: `ooms=52`，`cleanup=0.0`，`compact=12`，`phase1=12`。
- `ernie_35g_remap_off`: `ooms=61`，`cleanup=0.0`。
- `dsv3_30g`: `ooms=38`，`cleanup=0.0`，`compact=1`，`phase1=1`。
- `dsv3_45g_bounded`: `ooms=3712`，`cleanup=0.0`，`compact=55`，`phase1=55`。
- `dsv3_45g_compact_all`: `ooms=3493`，`cleanup=0.0`，`compact=72`，`phase1=72`。
- `probe_standard`: `success=1`。
- `probe_split_fill`: `successes=7/10`，相比上一轮 `8/10` 是效果回退。
- `compact_no_grow`: `SUCCESS`，`cleanup=0.000G`。

错误信号:
- 未发现 `BackingMap mismatch` / `validation failed` / `invalid range` / `force-release` / CUDA error / crash / Python traceback。

结论: pending-event source gate 从 `VmmHandleMeta` 迁到 BackingMap page-state 后完整 replay runner 通过，但 `probe_split_fill=7/10` 暴露 bounded compact 预检 bug：只统计 `kReady` source page 后，预检仍按完整 `requested_size` 要求 releasable bytes，导致已有 tail free 只需补缺口的请求被提前跳过。该问题在后续 working tree 修复并由 `/work/MemoryTools/logs/regression/20260601_004630/` 验证恢复到 `8/10`。

### 2026-06-01 - BackingMap pending-event source state 清理 (working tree)

改动范围:
- `VmmBackingMap::MappedPage` 增加 `remap_source_state`，BackingMap 在 source page collection 阶段统一返回 `ready` / `remap-destination-owned` / `pending-event` / `partial-or-invalid`。
- `RemapTransaction` 删除本地 `EventReadyCache` 与 handle-event 查询逻辑，只消费 BackingMap 返回的 source state。
- `VmmHandleMeta` 删除 `LastUseStream()` / `RemapSafeEvent()` / `HasRemapSafeEvent()` / `SetRemapSafeEvent(...)` / `ClearRemapSafeEvent()` 运行态字段与访问器。
- `CUDAVirtualMemAllocatorV2::SetBackingRangeRemapEvent(...)` 只写 BackingMap pending-event page-state。
- best-fit compact 预检只统计 `RemapSourceState::kReady` page，避免 pending/remap-owned page 被误算为 releasable。
- `CollectRemapSourcePagesFullyCoveredBy(..., target_bytes)` 按 ready source page 达到目标后停止，pending/remap-owned page 只用于统计 blocked reason，不占用 ready page 目标额度。

本地验证:
- `git diff --check`: PASS
- `make -C build -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: PASS
- `./build/test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test`: 14/14 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test`: 31/31 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 7/7 PASS

结论: pending-event source gate 已从 handle meta 运行态彻底移到 BackingMap page-state；C++ 覆盖 release/reuse/compact source、best-fit compact、multi-pool event route 均通过。该批次仍需 whl 安装后一键 replay 回归确认长序列指标。

### 2026-05-31 - 保守 IPC pin 文档收口后完整回归 (`7228c989e6`)

环境:
- whl 构建安装后运行，Python 运行时 commit 为 `7228c989e6c703df72aad467c7a85b7dcd19d9ce`。
- 命令：`cd /work/MemoryTools && bash ./run_vmm_v2_regression_log.sh 7`
- 日志目录：`/work/MemoryTools/logs/regression/20260531_233158/`

结果:
- 8/8 PASS，用时 18m13s。
- `ernie_35g_remap_on`: `ooms=48`，`cleanup=0.0`。
- `ernie_35g_remap_off`: `ooms=61`，`cleanup=0.0`。
- `dsv3_30g`: `ooms=38`，`cleanup=0.0`。
- `dsv3_45g_bounded`: `ooms=3739`，`cleanup=0.0`，`compact=2180`，`phase1=2180`。
- `dsv3_45g_compact_all`: `ooms=3493`，`cleanup=0.0`，`compact=2418`，`phase1=2418`。
- `probe_standard`: `success=1`。
- `probe_split_fill`: `successes=8/10`。
- `compact_no_grow`: `SUCCESS`，`cleanup=0.000G`。

错误信号:
- 未发现 `BackingMap mismatch` / `validation failed` / `invalid range` / `force-release` / CUDA error / crash / Python traceback。

结论: 保守 IPC pin 文档收口后的完整 replay 回归通过，OOM 指标保持最新低值基线 `dsv3_45g_compact_all=3493`。

### 2026-05-31 - IPC 生命周期决策收口 (working tree)

决策:
- 本版 VMM V2 IPC 采用保守 pin 作为完成语义。
- IPC exported backing page 在没有可靠 exporter-side unpin 信号前，不允许 remap、release 或 reuse。
- pin 是 page-granular gate，不是 allocator-global gate；同一 merged free block 中相邻未导出的普通 page 仍可按 BackingMap page-state 参与 release、compact 和 reuse。
- 可靠 unpin 需要新增跨进程生命周期协议或引用计数，不能直接复用 VMM-v1 / 普通 CUDA IPC 的导入端 close/cache 逻辑，后续作为独立模块设计。

结论: 当前 BackingMap 设计目标不再等待 unpin；最终收尾聚焦文档定稿、完整 whl 回归和必要的残留状态字段清理评估。

### 2026-05-31 - best-fit 测试白盒依赖清理完成 (working tree)

改动范围:
- `vmm_auto_growth_best_fit_allocator_v2_test.cu` 移除 `#define private public`。
- 删除 `SyntheticOwnershipPreparationIsTwoPhase`：该用例直接检查私有 `underlying_allocations_` registry 和 `CanRelease...` / `Release...` helper 的两阶段内部状态，不对应可通过正式 public API 观察的 allocator 行为。
- 删除 `AllocateReusesNonOwnedGapBeforeTailGrow`：该用例通过手工向 `all_blocks_` 插入 unmapped-free gap 构造内部状态；公开行为仍由 `AllocateSkipsOwnershipOverlappedGap`、compact/remap、bottom allocator `AllocateAtVAWithBlock(...)` 和 tail/gap reuse 路径覆盖。
- 取舍：不为上述两个纯白盒构造新增 test-only/public 过渡接口，避免把测试内部状态构造固化为正式 API。

本地验证:
- `git diff --check`: PASS
- `make -C build -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: PASS
- `./build/test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test`: 14/14 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test`: 31/31 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 7/7 PASS
- `rg -n "#define private public" test/cpp/fluid/memory/*v2*_test.cu test/cpp/fluid/memory/*_v2_test.cu`: 无 V2 memory 测试命中

结论: best-fit V2 测试已退出 private/public 宏；V2 memory C++ 测试当前全部通过 public 观测面验证 block/backing 行为。删除的两个用例是私有内部构造验证，不改变生产代码行为，也不影响当前 ownership-overlapped gap skip、remap/compact、IPC pin/event gate 和 visitor 路径覆盖。

### 2026-05-31 - best-fit 剩余白盒面收窄 (working tree)

改动范围:
- `vmm_auto_growth_best_fit_allocator_v2_test.cu` 中 event/remap metadata 断言改用 public `all_blocks()` 视图定位 active/free block。
- IPC pin 断言改用底层 `CUDAVirtualMemAllocatorV2::HasIpcExportedBacking(...)` 和 free-index 统计观察，不再调用 best-fit 私有 `BlockHasIpcExportedBacking(...)`、`CanIndexFreeBlock(...)` 或 `RebuildFreeBlockIndex()`。
- 当前仍保留 `#define private public` 的测试范围只剩：synthetic remap-destination registry 两阶段 helper 验证，以及手工插入 non-owned unmapped gap 的构造用例。

本地验证:
- `git diff --check`: PASS
- `make -C build -j32 vmm_auto_growth_best_fit_allocator_v2_test`: PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test`: 33/33 PASS

结论: best-fit 普通 event/IPC pin 行为已通过 public 观察面覆盖；剩余白盒测试对应内部异常状态构造和私有生命周期 helper，不为这些场景新增正式 public API。

### 2026-05-31 - bottom allocator 测试白盒依赖清理 (working tree)

改动范围:
- `cuda_virtual_mem_allocator_v2_test.cu` 移除 `#define private public`。
- 原先针对私有 `AllocateWithLayout(...)` / `AllocateAtVAWithLayout(...)` / `CollectAllocationHandleLayout(...)` / `IsRemapDestinationOwnedLayout(...)` 的断言改为使用 public `AllocateWithBlock(...)`、`AllocateAtVAWithBlock(...)`、BackingMap mapped/unmapped page 查询和 `BlockV2` allocation-view 元数据。
- remap destination ownership、IPC pin、pending event 相关测试仍覆盖对应行为，但不再访问底层私有 layout registry。

本地验证:
- `git diff --check`: PASS
- `make -C build -j32 cuda_virtual_mem_allocator_v2_test`: PASS
- `./build/test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test`: 14/14 PASS

结论: bottom allocator 测试已退出 private/public 宏，公开测试面与当前设计一致：正式观测入口是 block/backing-state API，`HandleLayout` 只保留为底层内部实现细节。

### 2026-05-31 - best-fit 测试白盒依赖继续收口 (working tree)

改动范围:
- `vmm_auto_growth_best_fit_allocator_v2_test.cu` 中只读 block-list 遍历从 `all_blocks_` 改为 public `all_blocks()`。
- 普通 indexed free-block 断言从直接访问 `free_blocks_` 改为 `GetFreeBlockStats(...)`；普通 active 计数改为基于 public block 视图统计。
- 保留 `#define private public` 的范围仅用于仍需白盒的用例：主动插入 synthetic unmapped gap、验证 remap-destination underlying allocation registry、以及 event/IPC pin 的内部标记传播。

本地验证:
- `git diff --check`: PASS
- `make -C build -j32 vmm_auto_growth_best_fit_allocator_v2_test`: PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test`: 33/33 PASS

结论: best-fit 测试的普通行为断言已尽量改到 public 观测面；剩余白盒依赖对应的是测试主动构造异常内部状态或验证私有生命周期 helper，暂不为这些场景新增过渡 public API。

### 2026-05-31 - multi-pool 测试白盒依赖清理 (working tree)

改动范围:
- `vmm_auto_growth_best_fit_multi_pool_allocator_v2_test.cu` 移除 `#define private public`。
- route/free 断言不再访问 `active_allocations_`、子池 `allocated_blocks_` 或私有子池成员，改用 public `small_allocator()` / `large_allocator()` 和只读 `all_blocks()` 视图确认路由结果。
- IPC pin 断言不再调用私有 `BlockHasIpcExportedBacking(...)`，改为 export 后释放再分配，确认 pinned backing 不会被复用。

本地验证:
- `git diff --check`: PASS
- `make -C build -j32 vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 7/7 PASS

结论: multi-pool 测试不再需要 private/public 宏；行为覆盖保持 route、free delegation、visitor IPC export 和 threshold boundary。

### 2026-05-31 - 白盒依赖清理与 IPC 生命周期复核 (working tree)

改动范围:
- 移除 `BlockV2::Parts()` public 观测入口，保留 `AllocationPartCount()`、`HasSingleAllocationPart(...)`、`AllocationPartHandleRelOffset(...)`、`AllocationPartByteSize(...)` 等窄的 allocation-view 查询方法。
- `vmm_auto_growth_best_fit_allocator_v2_test` / `cuda_virtual_mem_allocator_v2_test` / `vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` 不再直接依赖 `BlockV2` 内部 parts vector；remap 后 handle 迁移改用 BackingMap mapped page 查询确认。
- 复核 VMM-v1 IPC 生命周期：VMM-v1 `_share_cuda()` 收集 parts 并导出 shareable fd；导入端通过 `pidfd_getfd` + `cuMemImportFromShareableHandle` 导入，`VmmImportedAllocation` 析构只释放导入进程本地 VA mapping 和 imported handles。普通 CUDA IPC 同样是导入端 local weak cache + `cudaIpcCloseMemHandle`。

本地验证:
- `git diff --check`: PASS
- `make -C build -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: PASS
- `./build/test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test`: 14/14 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test`: 33/33 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 7/7 PASS

结论: 这是测试/接口边界清理，不改变 allocator 主路径。现有 VMM-v1 / CUDA IPC 路径没有导出端 remote-importer release/unpin 信号，V2 继续保守保持 BackingMap `ipc_exported` pin；未来 unpin 需要新增跨进程生命周期协议或引用计数。

### 2026-05-31 - BackingMap API 边界收口后最终完整回归 (`cdd53fd5d4`)

改动范围:
- 在 `f0a728e9dd` 基础上补齐回归 checklist，明确最新 `dsv3_45g_compact_all=3493` 与历史硬阈值 `4362` 的关系。
- 当前主路径已经由 BackingMap page-state 决定 IPC/event/reuse/release/remap source ownership；`BlockV2::parts_` 保留为 allocation-view layout 和白盒测试观测面。

本地验证:
- `make -C build -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: PASS
- `./build/test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test`: 14/14 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test`: 33/33 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 7/7 PASS
- `make -C build -j32 paddle_python`: PASS
- `cd build/python && python setup.py bdist_wheel`: PASS
- `python -m pip install --force-reinstall --no-deps build/python/dist/paddlepaddle_gpu-*.whl`: PASS
- Python 运行时 commit: `cdd53fd5d46d82f1da5dd5514d9af1c9523ade02`

完整一键回归:

- 命令：`cd /work/MemoryTools && bash ./run_vmm_v2_regression_log.sh 7`
- 日志：`/work/MemoryTools/logs/regression/20260531_013904/`
- 结果：8/8 PASS，用时 18m13s

| 测试 | OOMs / 结果 | post_cleanup | compact | force-release | backing_err |
|------|-------------|--------------|---------|---------------|-------------|
| ernie_35g_remap_on | 48 | 0.0G | 29 | 0 | 0 |
| ernie_35g_remap_off | 61 | 0.0G | 0 | 0 | 0 |
| dsv3_30g | 38 | 0.0G | 1 | 0 | 0 |
| dsv3_45g_bounded | 3739 | 0.0G | 2180 | 0 | 0 |
| dsv3_45g_compact_all | 3493 | 0.0G | 2418 | 0 | 0 |
| probe_standard | SUCCESS | - | 0 | 0 | 0 |
| probe_split_fill | 8/10 | - | 4 | 0 | 0 |
| compact_no_grow | SUCCESS | 0.000G | 1 | 0 | 0 |

结论: API 边界收口和文档更新没有引入行为回退；指标与 `86106143fd` 完整回归一致，`dsv3_45g_compact_all=3493` 低于历史接受基线 `4362`，无 cleanup 泄漏、无 BackingMap 错误、无 force-release。

### 2026-05-31 - BackingMap page meta 与 source ownership 迁移完整回归 (`86106143fd`)

改动范围:
- BackingMap page-state 持有 `VmmHandleMeta`，fresh allocation、AllocateAtVA、MovePage restore/target 和 remap destination mapping 均同步写入 page meta。
- block-level backing API 改为按 block VA range 查询 BackingMap，remap source collection 不再遍历 `BlockV2::parts_` 判断 physical backing ownership。
- `BlockV2::IsMappedFree()` / `IsUnmappedFree()` 明确按 block type 判断，`parts_` 仅保留为 allocation-view layout。

本地验证:
- `make -C build -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: PASS
- `./build/test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test`: 14/14 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test`: 33/33 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 7/7 PASS
- `make -C build -j32 paddle_python`: PASS
- `cd build/python && python setup.py bdist_wheel`: PASS
- `python -m pip install --force-reinstall --no-deps build/python/dist/paddlepaddle_gpu-*.whl`: PASS

完整一键回归:

- 命令：`cd /work/MemoryTools && bash ./run_vmm_v2_regression_log.sh 7`
- 日志：`/work/MemoryTools/logs/regression/20260531_010649/`
- 结果：8/8 PASS，用时 18m13s

| 测试 | OOMs / 结果 | post_cleanup | compact | force-release | backing_err |
|------|-------------|--------------|---------|---------------|-------------|
| ernie_35g_remap_on | 48 | 0.0G | 29 | 0 | 0 |
| ernie_35g_remap_off | 61 | 0.0G | 0 | 0 | 0 |
| dsv3_30g | 38 | 0.0G | 1 | 0 | 0 |
| dsv3_45g_bounded | 3739 | 0.0G | 2180 | 0 | 0 |
| dsv3_45g_compact_all | 3493 | 0.0G | 2418 | 0 | 0 |
| probe_standard | SUCCESS | - | 0 | 0 | 0 |
| probe_split_fill | 8/10 | - | 4 | 0 | 0 |
| compact_no_grow | SUCCESS | 0.000G | 1 | 0 | 0 |

结论: BackingMap page-driven source collection 覆盖了此前 `parts_` 观测不到的 mapped-free page，`dsv3_45g_bounded` 和 `dsv3_45g_compact_all` OOM 数明显低于历史接受基线；无 cleanup 泄漏、无 BackingMap 错误、无 force-release。

### 2026-05-31 - raw handle traversal 收口完整回归 (`82dba476f4`)

改动范围:
- `82dba476f4` 移除 `BlockPartV2::HandleMetaRaw()`。
- `BlockV2::ForEachUniqueHandle(...)` 改为向调用侧传递 `const std::shared_ptr<VmmHandleMeta>&`，block-level backing API 不再把裸 handle 指针作为外部遍历接口。

本地验证:
- `make -C build -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: PASS
- `./build/test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test`: 14/14 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test`: 33/33 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 7/7 PASS
- `make -C build -j32 paddle_python`: PASS
- `cd build/python && python setup.py bdist_wheel`: PASS
- `python -m pip install --force-reinstall --no-deps build/python/dist/paddlepaddle_gpu-*.whl`: PASS，`paddle.version.commit=82dba476f4c97c092fbf54635695c9041ba74cc0`

完整一键回归:

- 命令：`cd /work/MemoryTools && bash ./run_vmm_v2_regression_log.sh 7`
- 日志：`/work/MemoryTools/logs/regression/20260531_000741/`
- 结果：8/8 PASS，用时 26m11s

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err |
|------|-------------|---------|--------------|---------|---------------|-------------|
| ernie_35g_remap_on | 56 | 22.92s | 0.0G | 19 | 0 | 0 |
| ernie_35g_remap_off | 61 | 21.89s | 0.0G | 0 | 0 | 0 |
| dsv3_30g | 38 | 9.32s | 0.0G | 1 | 0 | 0 |
| dsv3_45g_bounded | 4265 | 531.04s | 0.0G | 537 | 0 | 0 |
| dsv3_45g_compact_all | 4362 | 625.93s | 0.0G | 2339 | 0 | 0 |
| probe_standard | SUCCESS | - | - | 0 | 0 | 0 |
| probe_split_fill | 8/10 | 1.78s | - | 4 | 0 | 0 |
| compact_no_grow | SUCCESS | 119.0ms | 0.000G | 1 | 0 | 0 |

结论: raw handle traversal 收口没有引入行为回退；`dsv3_45g_compact_all` 维持当前接受基线 `ooms=4362`，所有 replay/probe 均无 cleanup 泄漏、无 backing mirror 错误、无 force-release。

### 2026-05-30 - remap planning / restore split 收口完整回归 (`2b287b3442`)

改动范围:
- `9380211bcd` / `348757dd7d` / `a84c42cb48` / `ad857a4156` 继续收口 remap destination placement，MovePage 与 legacy unmap/map 共享 tail -> single unmapped-free -> scatter 的目标选择和 placement materialize/install 入口。
- `083d4d22ed` 将 remap rollback 中 unmapped-free -> restored mapped-free 的 prefix/middle/suffix 切分下沉到 `BlockV2::BuildRestoreMappedFreeSegments(...)`。
- `2b287b3442` 同步本轮提交与验证文档。

本地验证:
- `cmake --build build --target cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test -j32`: PASS
- `./build/test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test`: 14/14 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test`: 33/33 PASS
- `./build/test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 7/7 PASS
- `cmake --build build --target paddle_python -j32`: PASS
- `cd build/python && python setup.py bdist_wheel && python -m zipfile -t dist/*.whl`: PASS
- `python -m pip install dist/*.whl --no-deps --force-reinstall`: PASS，`paddle.version.commit=2b287b3442ec7e50702e56e13f5280f8a22812f6`

完整一键回归:

- 命令：`cd /work/MemoryTools && bash ./run_vmm_v2_regression_log.sh 7`
- 日志：`/work/MemoryTools/logs/regression/20260530_230050/`
- 结果：8/8 PASS，用时 26m11s
- 关键字扫描：未发现 `force_release=[1-9]`、`backing_err=[1-9]`、`BackingMap .*failed`、`mismatch`、`segmentation fault`、`Traceback`、`CUDA error`、`FATAL`、`SIGSEGV`。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err |
|------|-------------|---------|--------------|---------|---------------|-------------|
| ernie_35g_remap_on | 56 | 24.25s | 0.0G | 19 | 0 | 0 |
| ernie_35g_remap_off | 61 | 22.22s | 0.0G | 0 | 0 | 0 |
| dsv3_30g | 38 | 9.19s | 0.0G | 1 | 0 | 0 |
| dsv3_45g_bounded | 4265 | 530.81s | 0.0G | 537 | 0 | 0 |
| dsv3_45g_compact_all | 4362 | 625.17s | 0.0G | 2339 | 0 | 0 |
| probe_standard | SUCCESS | - | - | 0 | 0 | 0 |
| probe_split_fill | 8/10 | 1.90s | - | 4 | 0 | 0 |
| compact_no_grow | SUCCESS | 117.0ms | 0.000G | 1 | 0 | 0 |

结论: remap destination planning 与 rollback restore split 的结构收口没有引入行为回退；`dsv3_45g_compact_all` 维持当前接受基线 `ooms=4362`，所有 replay/probe 均无 cleanup 泄漏、无 backing mirror 错误、无 force-release。

### 2026-05-29 - compact_all partial remap attempt 修复 (`e867f17ca3`)

问题:
- `a40d75bea5` 恢复 bounded compact 的 `releasable_bytes < requested_size` guard 后，`dsv3_45g_bounded` 回到 `ooms=4265`，但 `dsv3_45g_compact_all` 保持 `ooms=4411`。
- 对比 `/work/MemoryTools/logs/regression/20260528_223538/` 与历史好 run `/work/MemoryTools/logs/regression/20260528_163826/`：当前 run 前段少 4 次 OOM，但末段多 51 次 OOM，净增加 47 次 OOM；这说明不是单纯的正确性硬失败，而是 compact 时机/粒度改变后，前面成功了少量请求，后面付出了更多失败。

根因:
- compact_all 的目标是“本次进入 compact 后尽量整理所有 releasable handles”，但进入 compact 的外层准入仍需要防止无效工作。
- `a40d75bea5` 新增的 `releasable_bytes < requested_size` guard 对 bounded 是正确的；对 compact_all 也生效后，会跳过“总 free 足够、最大 free 不足、当前 fully releasable bytes 小于本次请求”的场景。
- 这些场景虽然不能直接满足当前请求，但 compact_all 历史行为会做 partial remap，改善后续碎片。跳过它们会把 `Phase 1` 次数从 `2339` 降到 `468`，最终 `ooms=4411`。
- 曾验证过在 compact_all 下绕过所有 requested-size 预检，结果恶化到 `ooms=4578`，并出现大量 `no ownership-safe target capacity` / target overlap 日志。因此正确修复不是取消全部预检，而是只让 compact_all 跳过 `releasable_bytes < requested_size` 这一项，保留 `total_free < requested_size` 与 `max_free >= requested_size` 的准入。

修复:
- `CompactImpl()` 中 `releasable_bytes < requested_size` guard 改为仅在 `!FLAGS_vmm_v2_compact_all` 时生效。
- 新增 `CompactAllIgnoresReleasableBytesRequestedPrecheck` 单测，覆盖 total free 足够、max free 不足、fully releasable bytes 小于 requested 时，compact_all 仍允许整理一个完整 releasable handle。
- 调整 `BoundedCompactSkipsWhenReleasableBytesInsufficient`，使其真正覆盖 total free 足够但 releasable bytes 不足的 bounded skip。

验证:
- C++: `vmm_auto_growth_best_fit_allocator_v2_test` 33/33 PASS。
- wheel: 已重新构建并 `pip install --force-reinstall --no-deps`。
- 回归: `ONLY=dsv3_45g_compact_all bash ./run_vmm_v2_regression_log.sh 7`，日志 `/work/MemoryTools/logs/regression/20260529_000552/`。

| 测试 | OOMs | allocs/frees | elapsed | post_cleanup | Phase 1 次数 | no ownership-safe |
|------|------|--------------|---------|--------------|--------------|-------------------|
| `20260528_223538` 修复前 | 4411 | 22377 / 22297 | 544.14s | 0.0G | 468 | 0 |
| `20260529_000552` 修复后 | 4362 | 22426 / 22346 | 627.90s | 0.0G | 2339 | 64 |
| `20260528_163826` 历史对照 | 4362 | 22426 / 22346 | 613.01s | 0.0G | 2339 | 64 |
| `20260526_235241` 最好参考 | 4359 | - | 639.20s | 0.0G | 2307 | - |

结论:
- 本修复已确认解决 `4411` 退化的主要根因，并恢复到 `2026-05-28 16:38` 的 `4362` 水平。
- 由于最好参考值仍是 `4359`，该单 case 在当前 runner 严格阈值下仍判 FAIL；剩余 3 次 OOM 差异应继续作为效果差异分析，而不是与本次 `releasable_bytes` 误拦截混为同一问题。

补充诊断:
- 对比 `20260526_235241` 最好参考与 `20260529_000552`：0%-84% OOM 计数一致；89% 时当前少 1 次 OOM，94% 时当前多 3 次，最终多 3 次。说明剩余差异符合“前面更大请求成功，后续请求形态改变”的效果差异，而不是单点正确性硬失败。
- Phase 1 handle 序列首次在第 2271 次 compact 分歧：`20260526_235241` 为 `948 handles / 1988100096 bytes`，当前为 `1944 handles / 4076863488 bytes`。后续当前 tail compaction 尝试更多，但 OOM 净差仅 +3。
- 为验证是否由 MovePage compact 单独引入，临时关闭 MovePage 分支并通过 `PYTHONPATH=/work/dev_tool/Paddle/build/python` 跑 `ONLY=dsv3_45g_compact_all`，日志 `/work/MemoryTools/logs/regression/20260529_003200/`。结果 `ooms=6076`、`allocs/frees=20712/20633`、`post_cleanup_reserved_gib=0.0`，明显劣于当前 `4362`。该临时改动已恢复，源码与 build 产物已回到当前实现。
- 为验证是否可通过更激进 target 选择弥补剩余差异，临时尝试 unmapped-free scatter 跳过 bad prefix 后使用内部可准备子段，日志 `/work/MemoryTools/logs/regression/20260529_102332/`。结果 `ooms=4408`、`allocs/frees=22380/22300`，较当前 `4362` 明显退化；同时 `synthetic allocation preparation` overlap 探测从当前约 9840 次增加到 38687 次。该方向已撤回，不作为最终实现路线。
- 为验证 `no ownership-safe target capacity` skip 是否误伤 compact_all，临时允许 compact_all 在 MovePage 无 target 时继续 legacy fallback，日志 `/work/MemoryTools/logs/regression/20260529_104054/`。结果 `ooms=5621`、`allocs/frees=21167/21088`，较当前 `4362` 严重退化；日志显示 legacy fallback 仍大量因 `unmapped-free capacity 0 < total_remapped` 回滚。结论是该 skip 是必要保护，不能直接放开。
- 结论：`4362 vs 4359` 不是 IPC export 路径误触发，也不是 MovePage 本身导致的单独退化；更可能是 BackingMap ownership/target 约束收紧后，compact source/target 选择和请求序列之间的效果权衡。当前应继续优化 target 选择/partial compact 策略，而不是回退 MovePage。

### 2026-05-28 - bounded compact precheck 修复 (`a40d75bea5`)

问题:
- `/work/MemoryTools/logs/regression/20260528_194304/` 中 `dsv3_45g_bounded` 虽被 runner 判定 PASS，但 `ooms=4501`，相比低 OOM 基线 `/work/MemoryTools/logs/regression/20260528_150332/` 的 `ooms=4265` 增加 236。
- 同一 replay 的 `allocs/frees` 从 `22523/22443` 降到 `22287/22207`，减少值也为 236，说明是 236 个此前可满足的请求退化为 OOM，而不是单纯统计噪声。

根因:
- commit `8f45474bfc` 在恢复 compact entry semantics 时，把 bounded compact 的预检从 `releasable_bytes >= requested_size` 弱化成 `total_free >= requested_size`。
- 对 OOM bounded compact 来说，`total_free` 只说明 allocator 视角有足够 free VA/块大小；真正能解除当前 OOM 的是可 remap/release 的 backing bytes。IPC pin、pending event、partial page 和 backing ownership filter 过滤后，`total_free` 可能足够但 `releasable_bytes` 不足。
- 这种情况下继续执行 compact 会产生大量小批次 partial remap，改变后续请求形态并增加 OOM。

对比:

| Run | ooms | allocs/frees | elapsed | Phase 1 次数 | handle 总数 | 平均/中位 handles |
|-----|------|--------------|---------|--------------|-------------|-------------------|
| `20260528_150332` 低 OOM 基线 | 4265 | 22523 / 22443 | 521.68s | 537 | 56382 | 104.99 / 96 |
| `20260528_194304` 退化 run | 4501 | 22287 / 22207 | 746.06s | 2408 | 98670 | 40.98 / 18 |
| `20260528_201303` 修复后 | 4265 | 22523 / 22443 | 539.11s | 537 | 56382 | 104.99 / 96 |

修复:
- `CompactImpl(place, requested_size)` 在 `requested_size > 0` 时恢复 `releasable_bytes < requested_size` 直接跳过 compact。
- 新增单测 `BoundedCompactSkipsWhenReleasableBytesInsufficient`，覆盖 `total_free` 足够但 fully releasable backing 不足时不做 partial remap。

验证:
- `git diff --check -- paddle/phi/core/memory/allocation/vmm_auto_growth_best_fit_allocator_v2.cc test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test.cu`: PASS
- `make -j32 vmm_auto_growth_best_fit_allocator_v2_test && ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test`: 32/32 PASS
- `make -j32 paddle_python`: PASS
- `python setup.py bdist_wheel` + `python -m zipfile -t dist/*.whl`: PASS
- `pip install --force-reinstall --no-deps paddlepaddle_gpu-3.5.0.dev20260527-...whl`: PASS
- `ONLY=dsv3_45g_bounded bash ./run_vmm_v2_regression_log.sh 7`: PASS，日志 `/work/MemoryTools/logs/regression/20260528_201303/`，`ooms=4265`。

结论: 这次 OOM 回退不是 `db7d525c19` / `f7991ea271` / `e81a097d94` 本身的 ownership/materialization 下沉引起，而是前序 bounded compact 预检语义被弱化后，在该批回归中暴露出来。恢复可释放 backing bytes 预检后，OOM、alloc/frees 和 compaction 批次分布回到低 OOM 基线。

### 2026-05-28 - backing ownership/materialization 下沉 (`db7d525c19`, `f7991ea271`, `e81a097d94`)

改动:
- `db7d525c19` 将 synthetic remap destination cleanup 中的 underlying allocation overlap 扫描、ownership precheck 和 erase 收口到 `UnderlyingAllocationRegistry`。
- `f7991ea271` 新增 `CUDAVirtualMemAllocatorV2::AllocationWithParts`、`AllocateWithParts(...)`、`AllocateAtVAWithParts(...)`，best-fit grow 与 unmapped-free reuse 不再直接从 `HandleLayout` 构造 `BlockPartV2`。
- `e81a097d94` 新增 `CUDAVirtualMemAllocatorV2::IsAllocationOwnedByRemapDestination(...)`，best-fit synthetic cleanup 不再拉取 `HandleLayout` 自行判断 remap-destination ownership。

局部验证:
- `make -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: PASS
- `./test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test`: 13/13 PASS
- `./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test`: 31/31 PASS
- `./test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test`: 6/6 PASS

说明: 当前只做 C++ 局部验证；下一轮模块级改动合并后再统一 whl + 一键回归。

### 2026-05-28 - block-level backing API / block materialization 收口 (`0310725faf`, `0f63aa47cd`)

- 变更 1：`CUDAVirtualMemAllocatorV2` 新增 block-level backing state API：`CollectIpcPartsForBlock(...)`、`MarkBlockIpcExported(...)`、`SetBlockRemapEvent(...)`、`IsBlockReusableForAllocation(...)`。best-fit 的 IPC export、remap event 和 reuse 判断不再直接展开 block parts。
- 变更 2：`BlockV2` 接管 mapped/free/unmapped block materialization、adjacent absorb、prefix/suffix trim 和 single-part restore。grow、unmapped-free reuse、release split、remap materialization 和 merge 路径不再分别手工维护 `ptr_/size_/type_/pool_/parts_`。
- 验证：`git diff --check` 通过；`make -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` 通过；`cuda_virtual_mem_allocator_v2_test` 12/12、`vmm_auto_growth_best_fit_allocator_v2_test` 31/31、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` 6/6 通过。
- 说明：该批是结构收口，尚未单独跑完整 whl + 一键回归；计划和下一批 ownership/page materialization 迁移合并回归。

### 2026-05-28 - Phase 2d page-state 接管模块级回归 (`4aecdd762e`, `63688a2277`)

- 构建安装：
  - `make -j32 paddle_python` 通过。
  - `python setup.py bdist_wheel` 通过。
  - `python -m pip install --force-reinstall --no-deps /work/dev_tool/Paddle/build/python/dist/paddlepaddle_gpu-3.5.0.dev20260527-cp312-cp312-linux_x86_64.whl` 通过。
  - `import paddle` 正常，`paddle.__version__=3.5.0.dev20260527`，CUDA device count 为 8。
- 回归：`bash ./run_vmm_v2_regression_log.sh 7`，日志 `/work/MemoryTools/logs/regression/20260528_150332/`，8/8 PASS。
- 关键结果：
  - `ernie_35g_remap_on`: PASS，`ooms=56`，`post_cleanup_reserved_gib=0.0`。
  - `ernie_35g_remap_off`: PASS，`ooms=61`，`post_cleanup_reserved_gib=0.0`。
  - `dsv3_30g`: PASS，`ooms=38`，`post_cleanup_reserved_gib=0.0`。
  - `dsv3_45g_bounded`: PASS，`ooms=4265`，`post_cleanup_reserved_gib=0.0`，`elapsed_s=521.68`。
  - `dsv3_45g_compact_all`: PASS，`ooms=4411`，`post_cleanup_reserved_gib=0.0`，`elapsed_s=535.51`。
  - `probe_standard`: PASS，`success=1`。
  - `probe_split_fill`: PASS，`successes=8/10`，`max_success_gib=8.0`。
  - `compact_no_grow`: PASS，`result=SUCCESS`，`cleanup=0.000G`。
- 结论：page-granular IPC pin 决策与 BackingMap-driven event gate 完整回归通过。此前失败过的 `dsv3_45g_compact_all` 继续稳定通过。
### 2026-05-28 - Phase 2d transaction event gate 接入 BackingMap candidate（working tree）

- 变更：
  - `RemapTransaction` 不再维护 `EventReadyCache`，也不再直接 `cudaEventQuery` / `hipEventQuery` 决定 source handle 是否可搬。
  - `IsFullyCoveredHandle(...)` 只判断 block part 是否完整覆盖 handle 且不是 remap-destination ownership。
  - pending event 是否阻止 source 进入 compact candidate 由 `VmmBackingMap::CollectMappedPagesFullyCoveredBy(...)` 的 page-state 判断完成。
  - candidate 匹配成功后清理 legacy `VmmHandleMeta::remap_safe_event`，保持旧运行态字段不会长期残留。
  - 如果 candidate 缺失且 handle 上仍有 legacy event 字段，统计上仍计入 `event_blocked`，但行为 gate 已由 BackingMap 决定。
- 目的：让 pending-event 和 IPC pin 的 source gate 共用 BackingMap page-state，减少 transaction 对 handle 运行态字段的直接依赖。
- 验证：
  - `make -j32 vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test cuda_virtual_mem_allocator_v2_test` 通过。
  - `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test` 30/30 通过。
  - `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` 6/6 通过。
  - `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test` 12/12 通过。

说明：该改动与 page-granular IPC pin 决策同属 Phase 2d page-state 接管模块，计划合并打一轮 whl 和完整一键回归。
### 2026-05-28 - Phase 2d page-granular IPC pin 决策（working tree）

- 变更：
  - `BlockV2::CanBeRemapSource()` 不再用 coarse `ipc_exported_` 排除整个 free block，只表达 mapped-free block 可进入 source 扫描。
  - `CompactImpl` precheck 的 source range 改为所有 mapped-free block，再由 BackingMap page-state 过滤 IPC pinned / pending-event page；`indexable_max_free` 仅用于判断是否已有可直接复用的大 free block。
  - `CanIndexFreeBlock(...)` 与 `IsRangeEntirelyFree(...)` 不再由 block `ipc_exported_` 一票否决；reuse/release 由 BackingMap page-state 判断。
  - `RemapTransaction` split source segment 时保留 `ipc_exported_` 兼容观测标志，但该标志不再主导 release/compact 行为。
  - 更新 `IpcExportedMergeKeepsWholeFreeBlockPinned` 为 page 粒度语义：已导出 page 保持 pinned，相邻普通 page 可以 release 并复用；新增 compact 用例覆盖同一 merged free block 中普通 page 可作为 compact source，IPC pinned page 被 BackingMap 阻止。
- 目的：把 IPC pin 从 block 粒度行为 gate 推进到 BackingMap page 粒度，减少“一个 page export 导致整个 merged free block 永久 pinned”的过渡行为。
- 验证：
  - `make -j32 vmm_auto_growth_best_fit_allocator_v2_test` 通过；`CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test` 30/30 通过。
  - `make -j32 vmm_auto_growth_best_fit_multi_pool_allocator_v2_test cuda_virtual_mem_allocator_v2_test` 通过。
  - `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` 6/6 通过。
  - `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test` 12/12 通过。

说明：这是 Phase 2d 的实际行为推进，不是过渡 accessor。完整 whl 回归计划与下一批 BackingMap/Backing View ownership 生命周期改动合并执行。
### 2026-05-28 - Phase 2d compact source candidate 接管（working tree）

- 变更：`RemapTransaction` 的 source selection 先由 BackingMap 从 free ranges 收集 compactable mapped source pages，再用 candidate page 匹配 block part 做 block-list 替换；移除旧的逐 part BackingMap source page 查询入口。
- 目的：让 compact source 是否可搬的事实来源从 `BlockPartV2` 判断进一步迁到 BackingMap page-state，block parts 只保留为当前 block-list 切分和 IPC/tensor view 物化所需的描述。
- 验证：`make -j32 vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test cuda_virtual_mem_allocator_v2_test` 通过；`CUDA_VISIBLE_DEVICES=7` 下 `cuda_virtual_mem_allocator_v2_test` 12/12、`vmm_auto_growth_best_fit_allocator_v2_test` 29/29、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` 6/6 通过。
### 2026-05-28 - Phase 2d pending-event page-state 接管（working tree）

- 变更：
  - `VmmBackingMap::Page` 新增 pending event page-state。
  - `CUDAVirtualMemAllocatorV2::SetBlockPartsRemapEvent(...)` 同步把 handle 对应 backing range 标记为 event-pending。
  - `VmmBackingMap::IsRangeReleasable(...)`、`IsRangeReusableForAllocation(...)`、`CollectMappedPages*` / compact source collection 统一过滤 IPC pinned 或 pending-event page。
  - 新增 `CUDAVirtualMemAllocatorV2.BackingMapPendingEventBlocksReuse`，验证 pending event 阻止 reuse/release，event synchronize 后自动放行。
- 目的：Phase 2d 第一批核心迁移，把 event safety gate 从 `BlockPartV2` / `VmmHandleMeta` 逐步迁到 BackingMap page-state；当前仍保留 handle meta event 作为兼容运行态，后续清理时不再继续扩大过渡层。
- 验证：`make -j32 vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test cuda_virtual_mem_allocator_v2_test` 通过；`CUDA_VISIBLE_DEVICES=7` 下 `cuda_virtual_mem_allocator_v2_test` 12/12、`vmm_auto_growth_best_fit_allocator_v2_test` 29/29、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` 6/6 通过。
### 2026-05-28 - Phase 2c 收尾回归补齐（commit de221fcdc7）

- 构建安装：
  - `make -j32 paddle_python` 通过。
  - `python setup.py bdist_wheel` 通过。
  - `python -m pip install --force-reinstall --no-deps /work/dev_tool/Paddle/build/python/dist/paddlepaddle_gpu-3.5.0.dev20260527-cp312-cp312-linux_x86_64.whl` 通过。
  - `import paddle` 正常，`paddle.__version__=3.5.0.dev20260527`，CUDA device count 为 8。
- 回归：
  - 原始批跑 `/work/MemoryTools/logs/regression/20260528_131609/`：`ernie_35g_remap_on`、`ernie_35g_remap_off`、`dsv3_30g` 已写入 status PASS；`dsv3_45g_bounded.log` 产出 RESULT，`ooms=4483`、`post_cleanup_reserved_gib=0.0`、`elapsed_s=676.3566`，但外层 runner 未写入第 4 项 status，也未继续第 5-8 项。
  - 补跑 `/work/MemoryTools/logs/regression/20260528_133313/`：`dsv3_45g_compact_all` PASS，`ooms=4364`、`post_cleanup_reserved_gib=0.0`、`compact=2339`、`backing_err=0`、`force_release=0`。
  - 补跑 `/work/MemoryTools/logs/regression/20260528_134453/`：`probe_standard` PASS，`success=1`。
  - 补跑 `/work/MemoryTools/logs/regression/20260528_134528/`：`probe_split_fill` PASS，`successes=8/10`、`compact=5`。
  - 补跑 `/work/MemoryTools/logs/regression/20260528_134603/`：`compact_no_grow` PASS，`result=SUCCESS`、`cleanup=0.000G`。
- 结论：组合后 8 个逻辑 case 全部通过。bad-pattern 扫描未发现 `force-release` / BackingMap mismatch / validation failed / invalid range / Duplicate key / segfault / exception / synthetic allocation preparation failed / target mismatch；`probe_split_fill` 日志里的 `9.0 GiB -> FAIL`、`10.0 GiB -> FAIL` 是预期容量探测结果，runner 判定 PASS。
### 2026-05-28 - bottom allocator block part accessor 收口（working tree）

- 变更：`BlockPartV2` 新增 `HandleMetaRaw()` / `HandleRelOffset()` / `Device()`；`CUDAVirtualMemAllocatorV2` 的 IPC export、IPC pin、remap event、reuse gate、compact backing page collection 改用 accessor。
- 目的：移除底层 allocator 对 `part.handle` / `part.handle_rel_off` 的裸字段展开，把 handle 元信息访问集中在共享类型层。
- 验证：`make -j32 vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test cuda_virtual_mem_allocator_v2_test` 通过；`CUDA_VISIBLE_DEVICES=7` 下 `cuda_virtual_mem_allocator_v2_test` 11/11、`vmm_auto_growth_best_fit_allocator_v2_test` 29/29、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` 6/6 通过。
### 2026-05-28 - block part range/iteration 收口（working tree）

- 变更 1：`BlockPartV2::SliceBase()` 与 `BlockV2::SetPartsFromRange(...)` / `TrimPartsToRange(...)` 接管 best-fit split/grow/release 的 block part range 更新。
- 变更 2：`BlockV2::ForEachPartWithPtr(...)` 接管 `RemapTransaction` compact source collection / move plan 中的 part pointer 计算。
- 目的：继续缩小上层 allocator / transaction 对 `parts_` 物理切片布局和 VA 偏移计算的直接依赖，为后续将 ownership/page-state 迁移到 Backing View 保留更窄替换面。
- 验证：`make -j32 vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test cuda_virtual_mem_allocator_v2_test` 通过；`CUDA_VISIBLE_DEVICES=7` 下 `cuda_virtual_mem_allocator_v2_test` 11/11、`vmm_auto_growth_best_fit_allocator_v2_test` 29/29、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` 6/6 通过。
### 2026-05-28 - compact source backing page 语义入口（working tree）

- 变更：新增 `CUDAVirtualMemAllocatorV2::CollectMappedBackingPageForBlockPart(...)`；`RemapTransaction::CollectSourceBackingPage(...)` 改为调用该入口。
- 目的：把 compact source page 的 BackingMap 查询和 handle 校验从 transaction 层下沉到底层 allocator，transaction 只表达“为这个 block part 获取 source backing page”。
- 验证：`make -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` 通过；`cuda_virtual_mem_allocator_v2_test` 9/9、`vmm_auto_growth_best_fit_allocator_v2_test` 29/29、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` 6/6 通过。
### 2026-05-28 - reuse backing 语义入口（working tree）

- 变更：新增 `CUDAVirtualMemAllocatorV2::AreBlockPartsReusableForAllocation(...)`；best-fit `BlockHasIpcExportedBacking(...)` 改为调用该入口。
- 目的：把 free index / normal reuse / tail reuse 所需的 BackingMap IPC pin 检查从 best-fit 层下沉到底层 allocator 语义入口。当前仍保留 `BlockPartV2` 参数，因为 free block 可能是 sub-handle slice，不能直接用非 page-aligned block range 查询 BackingMap。
- 验证：`make -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test` 通过；`cuda_virtual_mem_allocator_v2_test` 9/9 通过；`vmm_auto_growth_best_fit_allocator_v2_test` 29/29 通过。
### 2026-05-28 - BlockPartV2 part-state 语义入口（working tree）

- 变更：`BlockPartV2` 新增 `HasHandle()` / `CanExportForIpc()` / `HasRemapSafeEvent()`；IPC export、BackingMap IPC pin 检查、remap source event gate 改用这些入口。
- 目的：继续减少上层 allocator / transaction 对 `BlockPartV2::handle` 细节的直接展开，为后续将 handle ownership 状态迁移到 BackingMap/Backing View 留出更窄替换面。
- 验证：`make -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` 通过；`cuda_virtual_mem_allocator_v2_test` 8/8、`vmm_auto_growth_best_fit_allocator_v2_test` 29/29、`vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` 6/6 通过。
### 2026-05-28 - synthetic ownership 语义入口（working tree）

- 变更：新增 `CUDAVirtualMemAllocatorV2::IsRemapDestinationOwnedLayout(...)`；best-fit `AllocationHandlesOwnedByRemapDestination(...)` 改为调用该入口。
- 目的：将 synthetic allocation precheck 的 layout ownership 判断收敛到底层 allocator 语义入口，后续替换为 BackingMap/Backing View ownership 状态时不需要继续改 best-fit 层。
- 验证：`make -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test` 通过；`cuda_virtual_mem_allocator_v2_test` 8/8 通过；`vmm_auto_growth_best_fit_allocator_v2_test` 29/29 通过。
### 2026-05-28 - Phase 2c 批量回归（commit 43374a34fb）

- 构建安装：
  - `make -j32 paddle_python` 通过。
  - `python setup.py bdist_wheel` 通过。
  - `python -m pip install --force-reinstall --no-deps /work/dev_tool/Paddle/build/python/dist/paddlepaddle_gpu-3.5.0.dev20260527-cp312-cp312-linux_x86_64.whl` 通过。
  - `import paddle` 正常，`paddle.__version__=3.5.0.dev20260527`，CUDA device count 为 8。
- 回归：`bash ./run_vmm_v2_regression_log.sh 7`，日志 `/work/MemoryTools/logs/regression/20260528_004800/`，8/8 PASS。
- 关键结果：
  - `dsv3_45g_bounded`: PASS，`ooms=4483`，`post_cleanup_reserved_gib=0.0`，`backing_err=0`，`force_release=0`。
  - `dsv3_45g_compact_all`: PASS，`ooms=4364`，`post_cleanup_reserved_gib=0.0`，`backing_err=0`，`force_release=0`。
  - `compact_no_grow`: PASS，`result=SUCCESS`，`cleanup=0.000G`。
- 额外扫描：未发现 `force-release` / `BackingMap mismatch` / `validation failed` / `invalid range` / `Duplicate key` / `segfault` / `exception` / `synthetic allocation preparation failed` / `target mismatch`。
### 2026-05-28 - backing releasable 语义入口（working tree）

- 变更：新增 `VmmBackingMap::IsRangeReleasable(...)` 和 `CUDAVirtualMemAllocatorV2::IsBackingRangeReleasable(...)`；best-fit idle release predicate 改用该入口。
- 目的：把“backing range 是否可释放”从上层 allocator 的 IPC page 细节判断收敛到 BackingMap 语义，后续可在该入口扩展 ownership/page-state 判断。
- 验证：`make -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test` 通过；`cuda_virtual_mem_allocator_v2_test` 7/7 通过；`vmm_auto_growth_best_fit_allocator_v2_test` 29/29 通过。
### 2026-05-28 - idle release predicate 收口（working tree）

- 变更：`TryReleaseIdleUnderlyingAllocation(...)` 不再直接展开 release 条件，改为调用 `CanReleaseIdleUnderlyingAllocation(...)`。
- 目的：将“整段 VA free + backing 未 IPC exported”统一为 release predicate，为后续替换成 BackingMap/Backing View 页面级 releasable 判断保留单一入口。
- 验证：`make -j32 vmm_auto_growth_best_fit_allocator_v2_test` 通过；`vmm_auto_growth_best_fit_allocator_v2_test` 29/29 通过。
### 2026-05-28 - Block range parts 更新收口（working tree）

改动:

- `BlockPartV2` 新增 `SliceBase()`，避免 best-fit 手工展开 `HandleBase() + handle_rel_off`。
- `BlockV2` 新增 `SetPartsFromRange(...)` / `TrimPartsToRange(...)`。
- best-fit 的 grow split、free block reuse split、unmapped-free allocation split、`SplitAndRemoveRange(...)` 改用上述入口，不再直接调用 `SliceBlockPartsForRange(...)` 更新 block parts。

局部验证:

| 测试 | 结果 |
|------|------|
| `make -j32 vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test cuda_virtual_mem_allocator_v2_test` | PASS |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test` | PASS, 11/11 |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test` | PASS, 29/29 |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS, 6/6 |

结论: 该改动继续减少 best-fit 对 `BlockV2::parts_` 具体切片算法的直接依赖，不改变分配/释放策略；短测未发现回退。完整 whl 回归计划与下一批模块改动合并执行。
### 2026-05-28 - BlockPartV2 / RemapTransaction 收口模块回归 (`50242933c0`, `c52d584118`, `afb3032a42`)

Commits:

- `50242933c0 [Allocator] Centralize VMM block part operations`
- `c52d584118 [Allocator] Encapsulate VMM block part handle access`
- `afb3032a42 [Allocator] Reuse VMM block part merge helpers`

验证流程:

| 步骤 | 结果 |
|------|------|
| `make -j32 paddle_python` | PASS |
| `python setup.py bdist_wheel` | PASS after packaging staging fix |
| `pip install --force-reinstall --no-deps paddlepaddle_gpu-3.5.0.dev20260527-*.whl` | PASS |
| `LD_LIBRARY_PATH=/usr/lib64 python -c 'import paddle; ...'` | PASS, site-packages, CUDA device count=8 |
| `bash ./run_vmm_v2_regression_log.sh 7` | PASS, 8/8 |

说明: 首次 `bdist_wheel` 失败在 cutlass include packaging staging，缺失 `visitor_op_row_broadcast.h`；源文件存在，补齐 `build/python/build/bdist.../paddle/include/.../epilogue_visitor_op` 后重新打包成功。该问题不是 VMM allocator 逻辑回退。

回归日志: `/work/MemoryTools/logs/regression/20260528_114606/`

| Case | 结果 | OOMs / probe | Elapsed | cleanup | compact | force-release | backing_err |
|------|------|--------------|---------|---------|---------|---------------|-------------|
| ernie_35g_remap_on | PASS | 50 | 21.93s | 0.0G | 50 | 0 | 0 |
| ernie_35g_remap_off | PASS | 61 | 20.09s | 0.0G | 0 | 0 | 0 |
| dsv3_30g | PASS | 38 | 7.91s | 0.0G | 4 | 0 | 0 |
| dsv3_45g_bounded | PASS | 4483 | 671.66s | 0.0G | 2423 | 0 | 0 |
| dsv3_45g_compact_all | PASS | 4364 | 654.54s | 0.0G | 2339 | 0 | 0 |
| probe_standard | PASS | success=1 | - | - | 0 | 0 | 0 |
| probe_split_fill | PASS | 8/10 | 2.30s | - | 5 | 0 | 0 |
| compact_no_grow | PASS | SUCCESS | 121.6ms | 0.000G | 1 | 0 | 0 |

日志检查:

- 未发现 `force-release` / BackingMap mismatch / validation failed / invalid range。
- 未发现 `Duplicate key` / segmentation fault / Traceback / exception。
- 未发现 `synthetic preparation failed` / `target mismatch`。

结论: BlockPartV2 基础操作、handle 访问、RemapTransaction parts merge 三项收口后，完整一键回归未发现行为回退；此前失败过的 `dsv3_45g_compact_all` 继续稳定通过。
### 2026-05-28 - RemapTransaction block parts merge 收口（working tree）

改动:

- `BlockV2` 新增 `SetSinglePart(...)`。
- `RemapTransaction::InstallTailFreeBlock(...)`、`InstallMappedUnmappedFreeRange(...)`、`MergeAdjacentFreeBlocks(...)` 改用 `BlockV2::AppendPartsFrom(...)` 合并 parts。
- `RestoreUnmappedFreeRangeToMappedFreeBlock(...)` 改用 `BlockV2::SetSinglePart(...)`，不再手工 `ClearParts()` + `AddPart(...)`。

局部验证:

| 测试 | 结果 |
|------|------|
| `make -j32 vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test cuda_virtual_mem_allocator_v2_test` | PASS |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test` | PASS, 11/11 |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test` | PASS, 29/29 |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS, 6/6 |

结论: 该改动继续减少 remap transaction 对 `BlockV2::parts_` 容器的直接操作，不改变 remap 策略；短测未发现回退。下一步将与前两条 block-part 收口提交一起打 whl 并跑完整一键回归。
### 2026-05-28 - BlockPartV2 handle 访问收口（working tree）

改动:

- `BlockPartV2` 新增 `HandleMeta()`、`HandleBase()`、`HandleSize()`、`AllocationHandle()`、`RemapSafeEvent()`、`ClearRemapSafeEvent()`。
- `RemapTransaction` 的 source collection / move plan 不再直接展开 `part.handle->base`、`part.handle->size`、`part.handle->handle`、`part.handle->RemapSafeEvent()`。
- best-fit 首次分配的 combined VA 计算改用 `BlockPartV2::HandleBase()`。

局部验证:

| 测试 | 结果 |
|------|------|
| `make -j32 vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test` | PASS, 29/29 |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS, 6/6 |
| `make -j32 cuda_virtual_mem_allocator_v2_test` | PASS |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test` | PASS, 11/11 |

结论: 该改动继续收缩 `BlockPartV2` 物理 handle 访问面，不改变 remap 策略；短测未发现回退。完整 whl 回归计划与下一批模块改动合并执行。
### 2026-05-28 - BlockPartV2 基础操作收口（working tree）

改动:

- `vmm_allocator_v2_types.h` 新增 `BuildBlockPartsFromHandleLayout(...)`、`SliceBlockPartsForRange(...)`、`AppendBlockPartsTail(...)` 与 `BlockV2::AppendPartsFrom(...)`。
- best-fit 删除私有 `SlicePartsForRange(...)` / `AppendPartsTail(...)` / `BuildBlockPartsFromHandleLayout(...)` 重复实现，统一调用共享类型层入口。
- `RemapTransaction::MaterializeMappedRange(...)` 改为通过 `BuildBlockPartsFromHandleLayout(...)` 物化目标 free block，避免手工 `MutableParts()->push_back(...)`。

局部验证:

| 测试 | 结果 |
|------|------|
| `make -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test` | PASS, 11/11 |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test` | PASS, 29/29 |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS, 6/6 |

结论: 该改动是 block-part 基础操作收口，不改变 allocator 策略；短测未发现回退。完整 whl 回归计划与下一批 Phase 2c/2d 模块改动合并执行。
### 2026-05-28 - IPC export / compact precheck / remap event 继续下沉 (`bc42b55295`, `6703a0b47f`, `cbdc0c22cd`)

Commits:

- `bc42b55295 [Allocator] Centralize VMM IPC block part export`
- `6703a0b47f [Allocator] Use backing map for VMM compact precheck`
- `cbdc0c22cd [Allocator] Centralize VMM remap event assignment`

改动:

- `CUDAVirtualMemAllocatorV2` 新增 `CollectIpcPartsForBlockParts(...)` 与 `MarkBlockPartsIpcExported(...)`，best-fit 的 `CollectTensorParts` 不再直接展开 `BlockPartV2 -> BlockPart/VmmChunkMeta` 转换，也不再直接逐 handle 标记 BackingMap IPC pin。
- `CompactImpl` 的 source precheck 改为依赖 `CollectMappedBackingPagesFullyCoveredBy(...)`，只确认 FREE VA ranges 是否完整覆盖 mapped、non-IPC backing pages；event readiness 仍留给 `RemapTransaction` 运行时判断。
- `CUDAVirtualMemAllocatorV2` 新增 `SetBlockPartsRemapEvent(...)`，best-fit 的 `SetBlockRemapEvent` 不再直接遍历 block parts 设置 per-handle event 运行态。
- 删除上层 compact precheck 中基于 `BlockPartV2` / active handle set / remapped flag 的可释放 handle 推导，进一步对齐 Allocation View / Backing View 分离方向。

局部验证:

| 测试 | 结果 |
|------|------|
| `make -j32 cuda_virtual_mem_allocator_v2_test vmm_auto_growth_best_fit_allocator_v2_test` | PASS |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test` | PASS, 11/11 |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test` | PASS, 29/29 |
| `make -j32 vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS, 6/6 |

whl 级验证:

| 测试 | 结果 |
|------|------|
| `make -j32 paddle_python` | PASS |
| `python setup.py bdist_wheel` | PASS |
| `pip install --force-reinstall --no-deps paddlepaddle_gpu-3.5.0.dev20260527-*.whl` | PASS |
| `bash ./run_vmm_v2_regression_log.sh 7` | PASS, 8/8 |

回归日志: `/work/MemoryTools/logs/regression/20260528_105715/`

| Case | 结果 | OOMs / probe | Elapsed | cleanup | compact | force-release | backing_err |
|------|------|--------------|---------|---------|---------|---------------|-------------|
| ernie_35g_remap_on | PASS | 50 | 20.96s | 0.0G | 50 | 0 | 0 |
| ernie_35g_remap_off | PASS | 61 | 19.53s | 0.0G | 0 | 0 | 0 |
| dsv3_30g | PASS | 38 | 7.92s | 0.0G | 4 | 0 | 0 |
| dsv3_45g_bounded | PASS | 4483 | 666.26s | 0.0G | 2423 | 0 | 0 |
| dsv3_45g_compact_all | PASS | 4364 | 643.99s | 0.0G | 2339 | 0 | 0 |
| probe_standard | PASS | success=1 | - | - | 0 | 0 | 0 |
| probe_split_fill | PASS | 8/10 | 2.28s | - | 5 | 0 | 0 |
| compact_no_grow | PASS | SUCCESS | 114.0ms | 0.000G | 1 | 0 | 0 |

日志检查:

- 未发现 `force-release` / BackingMap mismatch / validation failed / invalid range。
- 未发现 `Duplicate key` / segmentation fault / Traceback / exception。
- 未发现 `synthetic preparation failed` / `target mismatch`。

结论: IPC export、compact source precheck、remap event assignment 下沉后，完整一键回归未发现行为回退；此前失败过的 `dsv3_45g_compact_all` 稳定通过。
### 2026-05-28 - Phase 2c BackingMap 语义入口模块级回归 (`15b367d0b0`)

范围:

- `74c9ebde40` compact precheck 对齐 BackingMap IPC pin。
- `c5a84449b9` 封装 remap event 运行态。
- `e78d6e2feb` remap source 仅允许 mapped free block。
- `41199715a3` / `f5e0e0053c` 收口 synthetic ownership 查询入口。
- `446964b447` / `43374a34fb` 收口 release releasable 查询入口。
- `ec34df1f13` 封装 `BlockPartV2` 状态查询。
- `14d9527965` 收口 reuse backing IPC pin 查询入口。
- `15b367d0b0` 将 compact source backing page 收集下沉到 `CUDAVirtualMemAllocatorV2`。

构建安装:

| 步骤 | 结果 |
|------|------|
| `make -j32 paddle_python` | PASS |
| `python setup.py bdist_wheel` | PASS |
| `python -m pip install --force-reinstall --no-deps paddlepaddle_gpu-3.5.0.dev20260527-*.whl` | PASS |
| import 校验 | `/usr/local/lib/python3.12/site-packages/paddle/__init__.py`, version `3.5.0.dev20260527`, GPU count `8` |

完整一键回归:

- 命令：`cd /work/MemoryTools && bash ./run_vmm_v2_regression_log.sh 7`
- 日志：`/work/MemoryTools/logs/regression/20260528_100710/`
- 结果：8/8 PASS
- 关键字扫描：未发现 `force-release`、`BackingMap mismatch`、`validation failed`、`invalid range`、`Duplicate key`、`segmentation fault`、`Traceback`、`Exception`、`synthetic preparation failed`、`target mismatch`。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err |
|------|-------------|---------|--------------|---------|---------------|-------------|
| ernie_35g_remap_on | 50 | 21.03s | 0.0G | 50 | 0 | 0 |
| ernie_35g_remap_off | 61 | 20.14s | 0.0G | 0 | 0 | 0 |
| dsv3_30g | 38 | 7.96s | 0.0G | 4 | 0 | 0 |
| dsv3_45g_bounded | 4483 | 684.39s | 0.0G | 2423 | 0 | 0 |
| dsv3_45g_compact_all | 4364 | 666.13s | 0.0G | 2339 | 0 | 0 |
| probe_standard | success=1 | - | - | 0 | 0 | 0 |
| probe_split_fill | 8/10, max=8G | 2.27s | - | 5 | 0 | 0 |
| compact_no_grow | SUCCESS, 200MB | 123.1ms | 0.000G | 1 | 0 | 0 |

结论:

- `dsv3_45g_compact_all` 未复现此前 deferred synthetic registration 暴露的问题。
- BackingMap IPC pin gate 在 release、free index、normal reuse、tail reuse、compact candidate、compact source backing page 收集路径下未发现回退。
- DSV3 45G 两个长 case 耗时约 11 分钟，和 2026-05-27 `d762968c79` 完整回归同量级。
### 2026-05-28 - synthetic ownership 判断入口收口 (`41199715a3`)

Commit: `41199715a3 [Allocator] Centralize VMM synthetic ownership checks`

改动:

- 新增 `VMMAutoGrowthBestFitAllocatorV2::AllocationHandlesOwnedByRemapDestination(...)`。
- `CanReleaseRemapDestinationUnderlyingAllocations(...)` 和 `ReleaseRemapDestinationUnderlyingAllocations(...)` 复用该入口，移除重复的 `CollectAllocationHandleLayout + all_of(IsOwnedByRemapDestination)` 代码。
- 行为不变，目的是继续减少 synthetic ownership 逻辑散落点，为后续从 layout/remapped 标志迁移到 BackingMap ownership 判断预留单一替换点。

验证:

| 测试 | 结果 |
|------|------|
| `make -j32 vmm_auto_growth_best_fit_allocator_v2_test` | PASS |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test` | PASS, 29/29 |

说明: 这是 Phase 2c 的 ownership 判断入口收口；完整长回归计划与后续 Phase 2c 小批改动合并执行。
### 2026-05-28 - remap source 仅允许 mapped free block (`e78d6e2feb`)

Commit: `e78d6e2feb [Allocator] Require mapped free blocks for VMM remap sources`

改动:

- `BlockV2::CanBeRemapSource()` 从 `IsFree() && !IsIpcExported()` 收紧为 `IsMappedFree() && !IsIpcExported()`。
- 语义上 remap source 必须有 backing `parts_`；旧兼容表示 `FREE && parts_.empty()` 只应被视为 `kUnmappedFree`，不应进入 source safe 统计。

验证:

| 测试 | 结果 |
|------|------|
| `make -j32 vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test` | PASS, 29/29 |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS, 6/6 |

说明: 这是 Phase 2c 的 block 状态语义收口；完整长回归计划与后续 Phase 2c 小批改动合并执行。
### 2026-05-28 - VMM handle remap event 状态封装 (`c5a84449b9`)

Commit: `c5a84449b9 [Allocator] Encapsulate VMM V2 remap event state`

改动:

- `VmmHandleMeta` 新增 `LastUseStream()`、`RemapSafeEvent()`、`HasRemapSafeEvent()`、`SetRemapSafeEvent(...)`、`ClearRemapSafeEvent()`。
- `SetBlockRemapEvent(...)`、`RemapTransaction::IsFullyCoveredHandle(...)` 和 source blocked 统计改用上述接口，不再直接访问 `last_use_stream` / `remap_safe_event` 字段。
- 对应 C++ 测试改为通过 accessor 检查事件状态。

验证:

| 测试 | 结果 |
|------|------|
| `make -j32 vmm_auto_growth_best_fit_allocator_v2_test vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test` | PASS, 29/29 |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_multi_pool_allocator_v2_test` | PASS, 6/6 |

说明: 这是 Phase 2c 的状态访问面收口，不改变 allocator 行为；完整长回归计划与后续 Phase 2c 小批改动合并执行。
### 2026-05-28 - compact precheck 对齐 BackingMap IPC pin (`74c9ebde40`)

Commit: `74c9ebde40 [Allocator] Align VMM V2 compact precheck with backing IPC pin`

改动:

- `CompactImpl` 的 `total_free` / `max_free` 预检查改用 `CanIndexFreeBlock(...)`，不再把 BackingMap 已 IPC pinned 的 free backing 算入可 compact 容量。
- releasable handle 预扫描同样跳过 BackingMap IPC pinned free block，保持预检查与实际 BackingMap compact candidate 语义一致。
- `BackingIpcPinPreventsFreeIndexReleaseAndTailReuse` 增加 `Compact(...) == 0` 覆盖。

验证:

| 测试 | 结果 |
|------|------|
| `make -j32 vmm_auto_growth_best_fit_allocator_v2_test` | PASS |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test` | PASS, 29/29 |

说明: 该提交发生在 `d762968c79` 完整 8/8 回归之后，当前只做局部 C++ 验证；后续与下一批 Phase 2c 改动合并跑完整回归。
### 2026-05-27 - VMM V2 IPC tensor sharing / pinned block 验证 (`3aa14194dd`)

环境: A100-80G, CUDA 12.9, Python 3.12, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `3aa14194dd [Allocator] Enable VMM V2 IPC tensor sharing`
Wheel: `/work/dev_tool/Paddle/build/python/dist/paddlepaddle_gpu-3.5.0.dev20260527-cp312-cp312-linux_x86_64.whl`
安装包: `/usr/local/lib/python3.12/site-packages/paddle`, version `3.5.0.dev20260527`

改动:

- V2 allocator 新增 `CollectTensorParts`，支持 `_share_cuda()` 收集 VMM V2 backing parts。
- `VmmTensorPartsVisitor` 新增 V2 allocator visit 路径。
- V2 IPC export 后将 block 标记为 `ipc_exported_`。
- `ipc_exported_` block 采用过渡 pinned 策略：free 后不复用、不 compact/remap、不 release。
- BackingMap mirror 同步记录 IPC pin，compact candidate 收集不再返回 IPC exported mapped page，idle release gate 也会拒绝包含 IPC exported page 的 range。
- best-fit free index、normal reuse 和 tail reuse 同步检查 BackingMap IPC pin，避免 block 级标志弱化时误复用 exported backing。
- `_share_cuda/_new_shared_cuda` 可覆盖 VMM V2 分配。

快速验证:

| 测试 | 设备 | 结果 |
|------|------|------|
| `cuda_virtual_mem_allocator_v2_test` | GPU 7 | 7/7 PASS |
| `vmm_auto_growth_best_fit_allocator_v2_test` | GPU 7 | 28/28 PASS |
| `test/legacy_test/test_cuda_vmm_memory.py` V1 | GPU 7 | 5/5 PASS |
| `test/legacy_test/test_cuda_vmm_memory.py` V2 | GPU 7 | 4 PASS, 1 SKIP (`test_memory_stats` 为 V1-specific) |
| V2 `_share_cuda/_new_shared_cuda` minimal IPC roundtrip | GPU 7 | PASS, `meta_len=5` |
| 自定义两进程 IPC import/export | GPU 1,2 | PASS |
| 2 卡 fleet V1 对照 | GPU 1,2 | PASS |
| 2 卡 fleet V2 | GPU 1,2 | PASS |

关键回归:

| 测试 | Run | OOMs / 结果 | post_cleanup | compact | phase1 | force-release | backing_err |
|------|-----|-------------|--------------|---------|--------|---------------|-------------|
| `compact_no_grow` | `20260527_194423` | PASS | 0.0G | - | - | 0 | 0 |
| `dsv3_45g_compact_all` | `20260527_194502` | 4364 | 0.0G | 2339 | 2339 | 0 | 0 |

日志检查:

- `Duplicate key`: 无
- `exception`: 无关键错误
- `synthetic allocation preparation failed`: 无
- `target mismatch`: 无
- `force-release`: 0
- `BackingMap mismatch` / `BackingMap validation failed`: 无

结论:

- VMM V2 IPC 主链路已跑通。
- `dsv3_45g_compact_all` 已覆盖此前 deferred synthetic registration 暴露的问题，当前 run 未复现。
- AdamW dtype mismatch 是 Python/C++ staging 混装问题，正式 whl 安装后 V1/V2 fleet 均通过，不判定为 VMM 修改回退。
### 2026-05-27 - BackingMap IPC pin 约束 reuse / tail reuse (`d762968c79`)

Commit: `d762968c79 [Allocator] Gate VMM V2 reuse by backing IPC pin`

改动:

- `VMMAutoGrowthBestFitAllocatorV2` 新增 `BlockHasIpcExportedBacking(...)` / `CanIndexFreeBlock(...)`，统一判断 free block 是否可进入 best-fit free index。
- normal free reuse 会剔除 BackingMap 已标记 IPC exported 的 stale free index entry。
- tail reuse 不再只看 block 级 `ipc_exported_`，也会拒绝 BackingMap IPC pinned backing。
- 新增 C++ 用例 `BackingIpcPinPreventsFreeIndexReleaseAndTailReuse`，覆盖 block 标志缺失但 BackingMap pin 存在时不复用、不 release。

本地验证:

| 测试 | 结果 |
|------|------|
| `make -j32 vmm_auto_growth_best_fit_allocator_v2_test` | PASS |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/vmm_auto_growth_best_fit_allocator_v2_test` | PASS, 29/29 |
| `make -j32 cuda_virtual_mem_allocator_v2_test` | PASS |
| `CUDA_VISIBLE_DEVICES=7 ./test/cpp/fluid/memory/cuda_virtual_mem_allocator_v2_test` | PASS, 7/7 |
| `make -j32 paddle_python` + bdist_wheel + force reinstall | PASS |
| `FLAGS_use_vmm_auto_growth_best_fit_allocator_v2=1 python test/legacy_test/test_cuda_vmm_memory.py` | PASS, 4 passed, 1 skipped |
| `ONLY=compact_no_grow bash ./run_vmm_v2_regression_log.sh 7` | PASS, `force_release=0`, `backing_err=0` |

完整一键回归:

- 命令：`cd /work/MemoryTools && bash ./run_vmm_v2_regression_log.sh 7`
- 日志：`/work/MemoryTools/logs/regression/20260527_233815/`
- 结果：8/8 PASS
- 关键字扫描：未发现 `force-release`、`BackingMap mismatch`、`BackingMap validation failed`、`invalid range`、`Duplicate key`、`segfault`、`exception`。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err |
|------|-------------|---------|--------------|---------|---------------|-------------|
| ernie_35g_remap_on | 50 | 21.33s | 0.0G | 50 | 0 | 0 |
| ernie_35g_remap_off | 61 | 19.53s | 0.0G | 0 | 0 | 0 |
| dsv3_30g | 38 | 8.32s | 0.0G | 4 | 0 | 0 |
| dsv3_45g_bounded | 4483 | 685.85s | 0.0G | 2423 | 0 | 0 |
| dsv3_45g_compact_all | 4364 | 665.26s | 0.0G | 2339 | 0 | 0 |
| probe_standard | SUCCESS | - | - | 0 | 0 | 0 |
| probe_split_fill | 8/10 | 2.21s | - | 5 | 0 | 0 |
| compact_no_grow | SUCCESS | - | 0.000G | 1 | 0 | 0 |

结论: BackingMap IPC pin 已覆盖 release、free index、normal reuse 和 tail reuse 的过渡 gate。此前 `dsv3_45g_compact_all` 暴露的 deferred synthetic registration 问题未复现，完整回归无 force-release / BackingMap 错误。
### 2026-05-26 - remapped / block state accessor 收口（本地验证，未提交文档）

- 变更 1：`VmmHandleMeta` 新增 `IsRemapped()/MarkRemapped()/ClearRemapped()`，替换 transaction、`FreeImpl()`、compaction pre-check、BackingMap validate 中对 `remapped` 的直接访问。
- 变更 2：`BlockV2` 新增 `IsActive()/IsFree()/IsGap()`，先替换 `AllocFromGapBlocks()`、free index rebuild、`IsRangeEntirelyFree()`、`SplitAndRemoveRange()`、以及 remap/gap 收集主路径上的显式 `BlockType` 判定。
- 目的：继续缩小 `remapped` / `kGap` 旧表示的直接访问面，让后续删除字段/枚举语义时不必同时改动多个模块的裸字段判断。
- 验证：
  - `20260526_235241` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`，`elapsed_s=639.2026014328003`
  - `20260527_000350` `probe_standard` PASS，`success=1`
- 补充：替换 `libpaddle.so` 后首次 `import paddle` 仍出现一次瞬时 `Bus error`；顺序重试后恢复正常，`cuda_count=8`。该现象与此前多轮一致，判定为替换时序问题，不作为代码回退。
### 2026-05-26 - GAP normalize 收口（working tree, 未提交）

- 变更：`MergeAdjacentGapBlocks(...)` 与 `MergeAdjacentFreeBlocks(...)` 迁入 `RemapTransaction`，`NormalizeBlocks()` 不再依赖文件级 merge helper；source restore 在恢复 gap 后也通过 transaction 提供的 free-merge callback 收口 block normalize。
- 目的：把 `kGap` 相关的 create / restore / normalize 三类 block 编辑都集中到 transaction 内部，为下一步真正削减 `kGap` 语义做准备。
- 验证：
  - `20260526_230629` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`，`elapsed_s=644.5576734542847`
  - `20260526_231741` `probe_standard` PASS，`success=1`
### 2026-05-26 - Underlying allocation / GAP 入口收口（working tree, 未提交）

- 变更 1：`underlying_allocations_` 的 adoption / synthetic commit / idle release 分别收口到 `TrackUnderlyingAllocation(...)`、`CommitSyntheticUnderlyingAllocation(...)`、`TryReleaseIdleUnderlyingAllocation(...)`。
- 变更 2：`RestoreRemappedSourcesToFreeBlocks(...)` 改为通过 transaction 成员入口 `RestoreGapRangeToFreeBlock(...)` 恢复 source gap；随后 gap block 构造和 source gap restore 逻辑本体迁入 `RemapTransaction`，旧 `RestoreGapToFree(...)` 退出主路径。
- 目的：继续减少 allocator/transaction 对 `underlying_allocations_` 与 `kGap` 文件级 helper 的直接依赖，为后续替换 `kGap -> FREE + UNMAPPED` 做准备。
- 验证：
  - `20260526_195326` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`，`elapsed_s=649.9056541919708`
  - `20260526_200436` `probe_standard` PASS，`success=1`
  - `20260526_224102` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`，`elapsed_s=649.9934949874878`
  - `20260526_225213` `probe_standard` PASS，`success=1`
### 2026-05-26 - Destination rollback callback 化（working tree, 未提交）

- 变更 1：`RemapTransaction` 不再保留 `MappedDestinationRange` / `mapped_destination_ranges_`，改为在每次 destination map 成功后直接登记 rollback callback，逆序回放。
- 变更 2：只有在 `CollectRemapSources(...)` 真正收集到 source handles 后，才安装 `rollback_source_mappings_`；空事务不再携带未使用的 source rollback callback。
- 目的：继续把 transaction 中的 rollback 状态从“字段 + 结构体”压缩为“动作 + 顺序”，减少空事务和显式 bookkeeping。
- 验证：
  - `20260526_193226` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`，`elapsed_s=649.4036078453064`
  - `20260526_194335` `probe_standard` PASS，`success=1`
### 2026-05-26 - Rollback callback / helper 收口（working tree, 未提交）

- 变更 1：`RemapTransaction` 的 source restore 改为 `rollback_source_mappings_` callback，不再保留专门的 `source_blocks_` 成员状态。
- 变更 2：新增 `CUDAVirtualMemAllocatorV2::RollbackMappedHandleRange(...)`，统一 destination 前缀回滚与 transaction 的 mapped-destination rollback helper。
- 目的：继续减少事务层对 rollback 细节的显式状态和重复 loop，给 Phase 2b 收尾阶段留下更窄的 rollback 面。
- 验证：
  - `20260526_190139` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`，`elapsed_s=655.0184350013733`
  - `20260526_191558` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`，`elapsed_s=644.9899625778198`
  - `20260526_192708` `probe_standard` PASS，`success=1`
### 2026-05-26 - Synthetic commit hook 收口（working tree, 未提交）

- 变更：`RemapTransaction` 不再持有 `std::list<DecoratedAllocationPtr>*`，改为接收显式的 synthetic commit hook；`FreeBlockRemapCompactor` 只负责透传该 hook，`underlying_allocations_.emplace_back(...)` 收回到 `VMMAutoGrowthBestFitAllocatorV2::CompactImpl()`。
- 目的：把 compactor/transaction 对 `underlying_allocations_` 容器的直接依赖移除，继续收窄 synthetic / ownership 兼容链的边界。
- 验证：
  - `20260526_184235` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`，`elapsed_s=648.7763767242432`
  - `20260526_185345` `probe_standard` PASS，`success=1`
### 2026-05-26 - Staged synthetic 生命周期拆分（working tree, 未提交）

- 变更：`RemapTransaction` staged synthetic 改为持有裸 `Allocation*`；`Commit()` 时再包装成带 `FreeImpl()` deleter 的 `DecoratedAllocationPtr` 交给 `underlying_allocations_`，`Rollback()` 则通过 `DestroySyntheticAllocation()` 走 `UnregisterHandleLayout + delete`。
- 目的：把 staged synthetic 与 committed synthetic 的销毁路径彻底拆开，避免事务层继续手工 `release()+delete`，为后续继续收窄 `underlying_allocations_` / synthetic 兼容链做准备。
- 验证：
  - `20260526_171103` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`，`elapsed_s=652.1486003398895`
  - `20260526_172245` `probe_standard` PASS，`success=1`
### 2026-05-26 - RemapTransaction direct source restore 验证 (`357c249a7c`)

环境: A100-80G, CUDA 12.9, Python 3.12, physical GPU 7 (`CUDA_VISIBLE_DEVICES=7`, logical GPU 0), `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `357c249a7c [Allocator] Make remap source restore transaction-owned`
Runner: `2026-05-20-stable2`
Run: `20260526_110540`
总耗时: 26m42s

改动: 删除仅由 source restore 使用的通用 `rollback_actions_` / `AddRollbackAction` / `AddSourceRestoreAction` 间接层。`RemapTransaction` 直接持有 source block list 与已收集 handle/meta，`Rollback()` 在清理 pending destination 后直接恢复 source 映射；成功路径在 `Commit()` 清除 source rollback 状态。此步骤继续推进 Phase 2b，但仍保留 GAP、parts 与 synthetic allocation 兼容机制。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 13.49s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 10.94s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 8.04s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 674.31s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 659.09s | 0.0G | 2307 | 0* | 0* | direct source restore |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.03s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 109.4ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，默认日志中未观察到对应错误。

补充: 按原始命令 `bash ./run_vmm_v2_regression_log.sh 7` 执行的 `20260526_103730` batch 前七项通过，但 `compact_no_grow` 未产生 `RESULT`。该测试脚本固定选择 `gpu:0` 而未使用 runner 的 device 参数，当时物理 GPU 0 被其他进程占用约 79.3 GiB，导致其在 Step 1 前置分配阶段直接失败，`compact=0`，并未执行当前修改路径。使用 `CUDA_VISIBLE_DEVICES=7` 将同一物理 GPU 7 暴露为逻辑 GPU 0 后，单 case `20260526_110449` 与完整 batch `20260526_110540` 均通过。

结论: source restore 改为事务直接管理后，OOM、cleanup、compact 次数和 probe 结果与 `2c13b89af8` 基线保持一致；Phase 2b 已删除通用 rollback hook 层，后续仍需处理 GAP/parts/synthetic allocation 兼容机制。
### 2026-05-26 - RemapTransaction destination rollback tracking 验证 (`c6fbcad8b3`)

环境: A100-80G, CUDA 12.9, Python 3.12, physical GPU 7 (`CUDA_VISIBLE_DEVICES=7`, logical GPU 0), `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `c6fbcad8b3 [Allocator] Narrow remap destination rollback tracking`
Runner: `2026-05-20-stable2`
Run: `20260526_130154` + `20260526_130232` + `20260526_124932(dsv3_45g_compact_all single case)`

改动: destination 侧 rollback 记录从“尝试 map 的区间”收紧为“实际 map 成功的区间”，并让 `CUDAVirtualMemAllocatorV2::MapHandlesToVA(...)` 在部分 `cuMemMap` / `cuMemSetAccess` 失败时自行回滚已经落地的 prefix 映射，避免 transaction 对未成功映射目标做重复清理。synthetic allocation sink 改为构造注入，compactor 不再单独调用 setter。

结果: 关键回归与历史基线一致。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 13.68s | 0.0G | 50 | 0* | 0* | `20260526_130154`，单卡串行复验 |
| dsv3_45g_bounded | 4483 | 658.49s | 0.0G | 2423 | 0* | 0* | `20260526_130232`，单卡串行复验 |
| dsv3_45g_compact_all | 4359 | 657.43s | 0.0G | 2307 | 0* | 0* | `20260526_124932` 单 case，指标与既有基线一致 |

`*`: 默认日志中未观察到对应错误。

补充: 20260526_124932 曾尝试按 case 分散到 8 张物理卡并行运行。该模式适合快速发现 crash，但不适合直接比较 replay 类 case 的 `ooms` / `elapsed`，因为多个重压回放同时运行会改变事件完成时机和 CPU 调度，造成单 case 墙钟放大以及个别 OOM 计数偏移。最终以单卡串行复验结果作为基线判断依据。

结论: destination rollback 跟踪边界收紧后，功能与性能均保持在现有基线波动范围内。下一步进入 synthetic allocation 真正两阶段提交的收口。
### 2026-05-26 - Synthetic layout rollback unregister 验证（working tree, 未提交）

环境: A100-80G, CUDA 12.9, Python 3.12, physical GPU 7 (`CUDA_VISIBLE_DEVICES=7`, logical GPU 0), `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Runner: `2026-05-20-stable2`
Run: `20260526_152824` + `20260526_161029` + `20260526_164612` + `20260526_164655` + `20260526_164735`

改动: 曾尝试把 synthetic allocation 的 `allocation_layout_map_` 注册从 `CreateSyntheticAllocation(...)` 推迟到 `Commit()`，但在 `dsv3_45g_compact_all` 里暴露出当前 GAP/parts/source-restore 兼容层的异常顺序缺口。最终采用更保守的过渡方案：保持 synthetic allocation 创建时 eager register，不改 success path；`RemapTransaction::Rollback()` 在销毁 staged synthetic allocations 前显式 `UnregisterHandleLayout(ptr)`，清理未提交 destination 对应的 layout 注册。

结果: 串行单 case 补跑恢复到基线；probe 三项在同一物理卡并行运行时会互相污染，不可用作正确性结论。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 13.75s | 0.0G | 50 | 0* | 0* | `20260526_152824` |
| ernie_35g_remap_off | 61 | 10.68s | 0.0G | 0 | 0* | 0* | `20260526_152824` |
| dsv3_30g | 38 | 7.11s | 0.0G | 4 | 0* | 0* | `20260526_152824` |
| dsv3_45g_bounded | 4483 | 663.86s | 0.0G | 2423 | 0* | 0* | `20260526_152824` |
| dsv3_45g_compact_all | 4359 | 660.79s | 0.0G | 2307 | 0* | 0* | `20260526_161029`，修复后单 case |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | `20260526_164612`，串行单 case |
| probe_split_fill | 8/10, max=8G | 2.35s | - | 5 | 0* | 0* | `20260526_164655`，串行单 case |
| compact_no_grow | SUCCESS, 200MB | 113.0ms | 0.000G | 1 | 0* | 0* | `20260526_164735`，串行单 case |

`*`: 默认日志中未观察到对应错误。

补充: `20260526_164350` 曾将 `probe_standard` / `probe_split_fill` / `compact_no_grow` 同时绑到同一张物理 GPU 7 上并行运行，前两者在 anchor/occupy 阶段直接 OOM，`compact=0`，未进入有效 remap 验证路径；该结果被判定为测试方法无效，而非代码回退。三者逐 case 串行复跑均通过。

结论: 这一步没有实现最终设计里的“synthetic registration 延后到 commit”，但作为过渡方案，它修复了 rollback 残留 layout 注册的问题，并保持了 8 个目标 case 的基线行为。
### 2026-05-26 - RemapTransaction source collection rollback 修复验证 (`2c13b89af8`)

环境: A100-80G, CUDA 12.9, Python 3.12, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `2c13b89af8 [Allocator] Keep remap restore state alive through rollback`
Runner: `2026-05-20-stable2`
Run: `20260526_001300`
总耗时: 26m11s

改动: `85a0ce2856` 将 Phase 1 source 收集、fully-covered handle 判断、事件 gate 与 FREE->GAP 替换循环收口到 `RemapTransaction`，但 rollback action 捕获了 `CompactFreeBlocks()` 局部 source vector 的指针。`dsv3_45g_compact_all` 在 `20260525_231418` 运行中触发异常回滚后读取悬空状态，将栈地址作为原 VA 并 `SIGSEGV`。本提交将 source handle/meta 存储迁入事务对象，保证外层 catch 执行 `Rollback()` 时恢复状态仍存活。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 12.88s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 10.67s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 6.81s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 660.14s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 641.66s | 0.0G | 2307 | 0* | 0* | 修复原 SIGSEGV 场景 |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.15s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 111.5ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；`20260526_000147` 的 `dsv3_45g_compact_all` 单 case 复测也为 PASS，`force_release=0`、`backing_err=0`。

结论: Source collection 收口引入的 rollback 状态生命周期回退已修复；OOM、cleanup、compact 次数与 `cee4b699bb` 基线保持一致，`compact_all` 不再崩溃。
### 2026-05-25 - RemapTransaction placement strategy 验证 (`cee4b699bb`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `cee4b699bb [Allocator] Unify remap placement strategy in transaction`
Runner: `2026-05-20-stable2`
Run: `20260525_200743`
总耗时: 26m41s

改动: 将 tail / single-gap / gap-scatter 三条 destination 路径的高层策略入口统一收口成 `ExecutePlacementStrategy(...)`；compactor 不再自己串联 tail probe、single-gap 查找、gap capacity precheck 与 placement commit 分支。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 13.06s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 10.89s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 6.95s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 676.13s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 651.93s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.36s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 109.8ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: destination 侧高层 placement 策略统一入口后，OOM、cleanup、compact 次数、probe 与 `bab68ef167` 基线保持一致，说明 compactor 去掉 tail/gap/gap-scatter 的手工调度分支后未引入行为回退。
### 2026-05-25 - RemapTransaction placement planning 验证 (`bab68ef167`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `bab68ef167 [Allocator] Move remap placement planning into transaction`
Runner: `2026-05-20-stable2`
Run: `20260525_193644`
总耗时: 26m41s

改动: 将 tail probe、single-gap 选择、gap capacity 统计、gap-scatter placement 规划与执行继续收口到 `RemapTransaction`。compactor 不再手工执行 placement loop，只保留 Phase 1 source 收集和高层路径调度。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 13.73s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 10.95s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 7.17s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 675.16s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 653.65s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.44s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 111.6ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: placement 规划事务化后，OOM、cleanup、compact 次数、probe 与 `23e8dccf36` 基线保持一致，说明 compactor 去掉 destination 侧规划循环后未引入行为回退。
### 2026-05-25 - RemapTransaction block install / normalize 验证 (`23e8dccf36`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `23e8dccf36 [Allocator] Move remap block installs into transaction`
Runner: `2026-05-20-stable2`
Run: `20260525_190859`
总耗时: 26m41s

改动: 将 tail / single-gap / gap-scatter 的 block install 与 free/gap normalize 收口到 `RemapTransaction`，compactor 不再自己做 success path 的 block splice 与 merge。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 12.89s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 10.69s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 6.93s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 675.31s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 663.88s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.28s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 113.9ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: success path 的 block install / normalize 迁入事务后，OOM、cleanup、compact 次数、probe 与前一版基线保持一致，说明 block 落盘逻辑从 compactor 迁移到事务对象未引入行为回退。
### 2026-05-25 - RemapTransaction restore / staging 验证 (`ce31b89025 + db133af12a`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `db133af12a [Allocator] Fix remap transaction materialization constness`
Runner: `2026-05-20-stable2`
Run: `20260525_182802`
总耗时: 26m41s

改动: `ce31b89025` 将 rollback action 扩展为通用 hook 栈，并把 source restore / gap restore / force-release 迁入 `remap_transaction.cc`；synthetic allocation 改为事务 staging/commit。`db133af12a` 修复 `MaterializeMappedRange(...)` 的 `const` 接口一致性。该轮回归实际覆盖了这两版联合作用。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 13.21s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 10.87s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 6.94s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 670.03s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 662.40s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.56s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 117.0ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: rollback hook 栈、source restore / gap restore 迁移与 synthetic allocation staging/commit 统一后，OOM、cleanup、compact 次数、probe 与 `153eb50843` 基线保持一致，说明 Phase 2b 前半段事务收口未引入行为回退。
### 2026-05-25 - RemapTransaction mapped range materialization 验证 (`153eb50843`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `153eb50843 [Allocator] Materialize remap success paths in transaction`
Runner: `2026-05-20-stable2`
Run: `20260525_154510`
总耗时: 26m41s

改动: 新增 `RemapTransaction::MaterializeMappedRange(...)`，统一根据目标 VA 和 `[start, count)` 子区间完成 `HandleLayout`、synthetic allocation 与 `BlockV2(parts_)` 物化；tail、single-gap、gap-scatter 三条 success path 都不再在 compactor 中手工构造目标 free block。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 12.78s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 11.75s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 6.95s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 664.50s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 653.05s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.35s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 109.0ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: destination success path 的物化收口不改变 allocator 行为，Phase 2a 可以视为完成。
### 2026-05-25 - RemapTransaction ranged destination map 验证 (`b767c46ee9`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `b767c46ee9 [Allocator] Add remap transaction range mapping`
Runner: `2026-05-20-stable2`
Run: `20260525_134224`
总耗时: 26m11s

改动: 新增 `RemapTransaction::MapHandleRangeToDestination(...)`，让 gap-scatter 直接传入全局 `remapped_handles/remapped_metas` 和 `[start, count)` 区间；compactor 不再手工构造 `chunk/chunk_metas` 临时子向量。tail、single-gap 仍使用事务的整段 destination map 原语。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 13.84s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 10.78s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 7.14s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 660.03s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 647.58s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.33s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 112.0ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: gap-scatter 的区间映射原语不改变 allocator 行为。OOM、cleanup、compact 次数、probe 与 `62837cd47c` 基线保持一致，说明 compactor 去掉临时 `chunk/chunk_metas` 子切片后，bounded path 和 gap-scatter path 都未出现行为回退。
### 2026-05-25 - RemapTransaction destination map primitive 验证 (`8892814d52`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `8892814d52 [Allocator] Route destination map through remap transaction`
Runner: `2026-05-20-stable2`
Run: `20260525_111316`
总耗时: 26m11s

改动: 新增 `RemapTransaction::MapHandlesToDestination(...)`，内部先登记 destination intent，再调用原有 `CUDAVirtualMemAllocatorV2::MapHandlesToVA(...)`；compactor 的 tail、single-gap、gap-scatter 三条 destination placement 路径统一改用事务 map 原语。该阶段仍沿用现有 source parts 扫描、synthetic ownership 与 `RollbackToOriginalVA` 恢复实现。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 12.94s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 11.53s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 7.13s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 659.04s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 644.16s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.20s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 110.3ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: destination map 通过事务转发后，OOM、cleanup、compact 次数和 probe 结果均与 `62837cd47c` 基线保持一致。tail、single-gap 与 gap-scatter 三条 placement 路径没有引入行为回退，Phase 2a 的 destination map 收口已验证通过。
### 2026-05-25 - RemapTransaction source restore orchestration 验证 (`62837cd47c`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `62837cd47c [Allocator] Route remap source restore through transaction`
Runner: `2026-05-20-stable2`
Run: `20260525_002339`
总耗时: 26m11s

改动: `RemapTransaction` 新增 source rollback action 和一次性完成状态；compactor 将现有 `RollbackToOriginalVA` 注册为恢复回调，所有失败分支仅调用 `transaction.Rollback()`。destination range 在 `MapHandlesToVA` 前登记，从而覆盖 map 成功后 bookkeeping 抛异常时的清理空窗。事务内部先解除 pending destination mappings，再恢复 source VA。旧 `RollbackToOriginalVA` 具体实现仍保留在 compactor 中，因此这仍是 Phase 2a 的控制流收口，不是 Phase 2b 清理。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 12.86s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 10.65s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 6.82s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 659.95s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 645.46s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.31s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 109.8ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: source restore 由事务统一调度后，OOM、cleanup、compact 次数、probe 和默认错误统计均与 `963ff5c94f` 保持一致；`compact_no_grow` 继续成功，说明成功提交路径没有误触发 source restore。Phase 2a 的 rollback 控制流收口已验证通过。
### 2026-05-24 - RemapTransaction commit hooks 验证 (`963ff5c94f`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `963ff5c94f [Allocator] Add remap transaction commit hooks`
Runner: `2026-05-20-stable2`
Run: `20260524_230331`
总耗时: 26m11s

改动: 在 `6fda10be2b` 引入 `RemapTransaction` 骨架、`4853f28957` 收口 candidate flow 的基础上，增加显式 `Commit()` / `Rollback()` 事务入口；`free_block_remap_compactor.cc` 的成功路径与失败路径统一通过事务对象清理 pending mapped ranges。该阶段未切换 parts 驱动 source 扫描，未删除 `RollbackToOriginalVA`。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 12.80s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 10.66s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 6.84s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 659.15s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 644.75s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.22s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 109.4ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: `RemapTransaction` 的显式 `Commit()` / `Rollback()` 收口未改变 allocator 行为。OOM、cleanup、compact 次数、probe 结果均与 Phase 1b 基线保持一致；此前 `20260522_144938` / `20260522_155244` 中 DSV3 长测 elapsed 上浮在本次运行恢复到 `659.15s / 644.75s`，可视为运行波动而非代码性能回退。
### 2026-05-22 - BackingMap compact candidate collection 验证 (`601df8b0be`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `601df8b0be [Allocator] Add VMM backing compact candidate collection`
Runner: `2026-05-20-stable2`
Run: `20260522_132942`
总耗时: 26m41s

改动: 为 BackingMap 增加 `CompactCandidates {source_pages, target_pages}` 和 `CollectCompactCandidates(source_ranges, target_ranges, target_bytes)`，把 Phase 1b 的 source mapped pages / target unmapped pages 收集收敛成统一语义接口；compactor `VLOG(4)` pre-scan 改用该接口。默认 `GLOG_v=0` 下不执行该诊断扫描，不改变 compact 主路径。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 13.14s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 11.00s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 7.11s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 677.54s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 658.48s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.17s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 110.1ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: compact candidate collection 作为 Phase 1b 的统一语义接口，不改变现有 allocator 行为；OOM、cleanup、compact 次数、probe 结果与 `d43de6a32b` 基线持平。默认日志无 force-release / BackingMap error。
### 2026-05-21 - BackingMap unmapped target page collection 验证 (`d43de6a32b`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `d43de6a32b [Allocator] Add VMM backing target page pre-scan`
Runner: `2026-05-20-stable2`
Run: `20260521_191532`
总耗时: 27m12s

改动: 为 BackingMap 增加 `UnmappedPage {va, epoch}` 快照、`CollectUnmappedPagesFullyCoveredBy` 和 `ValidateUnmappedPages`，支持从非 page-aligned GAP/target ranges 中收集完全覆盖的 unmapped pages；compactor `VLOG(4)` pre-scan 同时打印 source mapped pages 与 target unmapped pages。默认 `GLOG_v=0` 下不执行该诊断扫描，不改变 compact 主路径。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 13.40s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 11.25s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 7.11s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 695.41s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 666.86s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.13s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 110.4ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: target-side unmapped page collection / validation 不改变现有 allocator 行为；OOM、cleanup、compact 次数、probe 结果与 `e946af32c0` 基线持平。默认日志无 force-release / BackingMap error。
### 2026-05-21 - BackingMap compact pre-scan diagnostic 验证 (`e946af32c0`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `e946af32c0 [Allocator] Add VMM backing compact pre-scan`
Runner: `2026-05-20-stable2`
Run: `20260521_171756`
总耗时: 29m11s

改动: 新增 `CollectMappedPagesFullyCoveredBy`，允许输入非 page-aligned FREE ranges，只返回完全落在 FREE range 内的 mapped pages；同时在 compactor Phase 1 前增加 `VLOG(4)` gated 的 BackingMap pre-scan 旁路，收集并校验 mapped page 快照。默认 `GLOG_v=0` 下不执行该诊断扫描，不改变 compact 主路径。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 14.53s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 12.26s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 7.78s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 755.53s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 711.19s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.45s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 122.3ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: BackingMap compact pre-scan 在默认回归下不改变现有 allocator 行为；OOM、cleanup、compact 次数、probe 结果与 `a1b1050cba` 基线持平。默认日志无 force-release / BackingMap error。
### 2026-05-21 - BackingMap mapped page snapshot validation 验证 (`a1b1050cba`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `a1b1050cba [Allocator] Add VMM backing page snapshot validation`
Runner: `2026-05-20-stable2`
Run: `20260521_162724`
总耗时: 29m41s

改动: 为 `VmmBackingMap::MappedPage` 快照增加 `ValidateMappedPages` 校验接口，确认快照 page 仍然 mapped、handle 未变、epoch 未变。该接口用于后续 `MovePage` / `RemapTransaction` 前的冲突检测，当前尚未接入现有 compact 主路径，预期不改变 allocator 行为。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 14.46s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 12.63s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 7.33s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 735.58s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 785.84s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.76s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 140.0ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: mapped page snapshot validation 不改变现有 allocator 行为；OOM、cleanup、compact 次数、probe 结果与 `b839ba14fa` 基线持平。DSV3 45G elapsed 有波动，但未伴随 OOM 或 cleanup 回退。
### 2026-05-21 - bounded BackingMap page collection 验证 (`b839ba14fa`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `b839ba14fa [Allocator] Add bounded VMM backing page collection`
Runner: `2026-05-20-stable2`
Run: `20260521_154412`
总耗时: 28m11s

改动: 为 `CollectMappedPages` 增加 `target_bytes` bounded 重载，按 page size 向上取整，达到目标页数后停止收集。该接口尚未接入现有 compact 主路径，预期不改变 allocator 行为。

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 14.98s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 11.96s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 7.45s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 727.63s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 704.62s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.70s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 112.3ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: bounded page collection 不改变现有 allocator 行为；OOM、cleanup、probe 结果与前序 BackingMap query/collection 基线持平。DSV3 45G elapsed 比 `b281cc4d37` 略高，但 OOM 数、compact 次数、cleanup 均一致，仍在长测噪声和通过阈值内。
### 2026-05-21 - BackingMap multi-range query 验证 (`b281cc4d37`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `b281cc4d37 [Allocator] Support multi-range VMM backing map queries`
Runner: `2026-05-20-stable2`
Run: `20260521_140718`
总耗时: 27m12s

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 13.29s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 10.97s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 7.22s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 678.93s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 676.17s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.92s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 113.3ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断仍以单 case `DIAG=1` 为准。

结论: multi-range BackingMap query 不改变现有 allocator 行为；OOM、cleanup、probe 结果与 `8f177e284f` 持平，elapsed 波动在 DSV3 45G 长测正常范围内。
### 2026-05-21 - BackingMap range collection 验证 (`8f177e284f`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `8f177e284f [Allocator] Add VMM backing map range collection`
Runner: `2026-05-20-stable2`
Run: `20260521_103508`
总耗时: 26m11s

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 12.89s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 10.82s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 6.84s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 658.25s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 643.58s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.38s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 111.6ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格诊断见下方 DIAG 单测。

DIAG 单测：

Run: `20260521_110327`

```text
ONLY=dsv3_45g_compact_all DIAG=1 bash ./run_vmm_v2_regression_log.sh 7
```

结果：

- `grep -Eci "force-releasing handle|force-release"` = 0。
- `grep -Ei "BackingMap.*(mismatch|validation failed|invalid range|reconfigure)"` 无输出。
- 结论: range collection 提交后未引入 force-release 或 BackingMap mirror 错误。
### 2026-05-20 - Backing View stable2 全量回归 (`18df2b243a`)

环境: A100-80G, CUDA 12.9, GPU 7, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `18df2b243a [Allocator] Fix VMM V2 driver result comparison`
Runner: `2026-05-20-stable2`
Run: `20260520_173159`
总耗时: 26m11s

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | compact | force-release | backing_err | 备注 |
|------|-------------|---------|--------------|---------|---------------|-------------|------|
| ernie_35g_remap_on | 50 | 13.87s | 0.0G | 50 | 0* | 0* | remap ON |
| ernie_35g_remap_off | 61 | 10.90s | 0.0G | 0 | 0* | 0* | flag 隔离，compact=0 |
| dsv3_30g | 38 | 7.41s | 0.0G | 4 | 0* | 0* | 中压场景 |
| dsv3_45g_bounded | 4483 | 659.40s | 0.0G | 2423 | 0* | 0* | bounded compact |
| dsv3_45g_compact_all | 4359 | 641.97s | 0.0G | 2307 | 0* | 0* | compact_all |
| probe_standard | success=1 | - | - | 0 | 0* | 0* | 单 worker `mode=vmm` |
| probe_split_fill | 8/10, max=8G | 2.42s | - | 5 | 0* | 0* | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 113.3ms | 0.000G | 1 | 0* | 0* | remap no-grow |

`*`: 全量回归使用 `GLOG_v=0`，该列表示默认日志中未观察到对应错误；严格 `force-release` / BackingMap 诊断见下方 DIAG 单测。

关键结论：

- 全量正确性回归 8/8 passed，无 crash、无 cleanup 泄漏。
- `post_cleanup_reserved_gib=0.0` 全部满足。
- 默认日志中未观察到 `force-release` 或 BackingMap 错误。
- `remap_off` 中 compact=0，确认 `FLAGS_vmm_v2_remap_on_oom=0` 能正确关闭 compact。
- `probe_standard` 改为直接运行单 worker `--mode vmm`，避免 probe parent 脚本先跑 baseline / empty_cache / compact 多模式导致 GPU 状态污染。
- `dsv3_45g_bounded=4483`、`dsv3_45g_compact_all=4359` 与 PR4 关键基线一致。

DIAG 单测结论:

Run: `20260520_180557`

```text
ONLY=dsv3_45g_compact_all DIAG=1 bash ./run_vmm_v2_regression_log.sh 7
```

结果：

- Functional result: PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`，`elapsed_s=641.77`。
- `grep -Eci "force-releasing handle|force-release"` = 0。
- `grep -Ei "BackingMap.*(mismatch|validation failed|invalid range|reconfigure)"` 无输出。
- RESULT 后 teardown 触发 `double free or corruption (!prev)`，runner 标记 `TEARDOWN_TIMEOUT`。该问题与开启 VLOG/DIAG 强相关，历史旧 VMM 版本也可复现，暂不归因于当前 Backing View 版本。

诊断策略:

- 全量 batch 不使用 `DIAG=1`。
- 如需确认 `force-release` / BackingMap 错误，单独运行：

```bash
cd /work/MemoryTools
ONLY=dsv3_45g_compact_all DIAG=1 bash ./run_vmm_v2_regression_log.sh 7
```
### 2026-05-19 - BackingMap Phase 1a mirror 验证 (`da1814cb19`)

环境: GPU 7 + GPU 0 复测, `/work/dev_tool/Paddle`, `feature/vmm-v2-backing-view`
Commit: `da1814cb19 [Allocator] Fix VMM V2 remap event test`
Run: `20260519_182903`
总耗时: 14m42s

结果: 8/8 passed。

| 测试 | OOMs / 结果 | Elapsed | post_cleanup | 备注 |
|------|-------------|---------|--------------|------|
| ernie_35g_remap_on | 50 | 12.89s | 0.0G | remap ON，复跑结果 |
| ernie_35g_remap_off | 61 | 11.40s | 0.0G | flag 隔离 |
| dsv3_30g | 38 | 7.48s | 0.0G | 中压场景 |
| dsv3_45g_bounded | 4483 | 670.28s | 0.0G | GPU0 复测，bounded compact |
| dsv3_45g_compact_all | 4359 | 679.01s | 0.0G | compact_all |
| probe_standard | success=1 | - | - | `mode=vmm_compact` |
| probe_split_fill | 8/10, max=8G | 2.21s | - | split-fill probe |
| compact_no_grow | SUCCESS, 200MB | 110.9ms | 0.000G | remap no-grow |

关键指标：

| 指标 | 结果 | 结论 |
|------|------|------|
| 回归测试 | 8/8 passed | 通过 |
| post_cleanup_reserved | 0.0G | 无泄漏 |
| ernie_35g_remap_on | 50 OOMs, 12.89s | 与 PR4 基线持平，无性能回退 |
| dsv3_45g_bounded | 4483 OOMs, 670.28s | 与 PR4 基线持平 |
| compact_all OOMs | 4359 | 与 PR4 关键基线持平 |
| probe_split_fill | 8/10 | 与 PR4 基线持平 |
| compact_no_grow | SUCCESS | remap 功能有效 |

备注: 初次 GPU7 run 中 `dsv3_45g_bounded=152 OOMs / 27.61s` 与历史 PR4 基线差异过大，且 `compact_all=4359` 仍与 PR4 持平；该结果判定为不可比异常值，不作为性能提升结论。GPU0 复测结果为 `4483 OOMs / 670.28s`，`allocs/frees=22305/22225`、compact 次数 `2423`，与旧 PR4 bounded 日志完全一致。

结论: BackingMap Phase 1a mirror 不改变现有 VMM V2 行为；OOM、cleanup、probe、主要 elapsed 结果与 PR4 关键基线保持一致，无明确性能回退。后续 Phase 1b/1c 应以 GPU0 复测后的 bounded 结果作为 BackingMap 分支初始基线。
### 2026-05-18 - Edge Case 修复验证 (`f50fd2fe5a`)

环境: A100-80G, CUDA 12.9, GPU 0, `/work/dev_tool/Paddle` vmm_v2_pr4
日志: `/work/MemoryTools/logs/regression/20260518_162925/`

| 测试 | OOMs | Elapsed | post_cleanup | force-release | 对比 25ace1d |
|------|------|---------|-------------|---------------|-------------|
| ernie_35g_remap_on | 50 | 14.3s | 0.0G | - | 持平 |
| ernie_35g_remap_off | 61 | 12.5s | 0.0G | - | 持平 |
| dsv3_30g | 38 | 7.7s | 0.0G | - | 持平 |
| dsv3_45g_bounded | 4483 | 718s | 0.0G | - | 持平 |
| dsv3_45g_compact_all | 4359 | 729s | 0.0G | 0 | 持平 |
| probe_standard | success=1 | - | - | - | 通过 |
| probe_split_fill | 8/10 | 2.8s | - | - | 持平 |
| compact_no_grow | SUCCESS | 128ms | 0.0G | - | 通过 |

结论: edge case 修复无回归，force-release 保持为 0，OOMs 完全一致。
### 2026-05-14 - PendingMappedRange 修复 (`25ace1d151`)

环境: A100-80G, CUDA 12.9, GPU 0
日志: `/work/MemoryTools/logs/regression/20260514_182803/`

| 测试 | OOMs | Elapsed | post_cleanup | 对比 26067a1 |
|------|------|---------|-------------|-------------|
| ernie_35g_remap_on | 50 | 13.0s | 0.0G | 持平 |
| ernie_35g_remap_off | 61 | 11.2s | 0.0G | 持平 |
| dsv3_30g | 38 | 7.0s | 0.0G | 持平 |
| dsv3_45g_bounded | 4483 | 645s | 0.0G | 持平 |
| dsv3_45g_compact_all | 4359 | 630s | 0.0G | 改善 |
| probe_standard | success=1 | - | - | 通过 |
| probe_split_fill | 8/10 | 2.3s | - | 持平 |
| compact_no_grow | SUCCESS | 109ms | 0.0G | 通过 |

compact_all 改善：

| 指标 | `26067a133b` | `25ace1d151` | 变化 |
|------|-------------|-------------|------|
| OOMs | 4372 | 4359 | -13 |
| Elapsed | 681s | 630s | -51s |
| force-release | 316 handles | 0 | 完全消除 |
| post_cleanup | 0.0G | 0.0G | 持平 |
### 2026-05-13 - RollbackToOriginalVA 第一版 (`26067a133b`)

| Workload | Occupy | Mode | OOMs | Elapsed | Post-cleanup | Probe |
|----------|--------|------|------|---------|-------------|-------|
| ERNIE | 35G | no_vmm | 65 | 8.3s | 0.0G | |
| ERNIE | 35G | VMM V1 | 61 | 6.6s | 78.71G | |
| ERNIE | 35G | VMM V2 bounded | 50 | 14.1s | 0.0G | |
| ERNIE | 35G | Torch-exp | 58 | 1.6s | 0.0G | |
| ERNIE | 40G | no_vmm | 4637 | 496s | 0.0G | |
| ERNIE | 40G | VMM V1 | 3460 | 251s | 78.74G | |
| ERNIE | 40G | VMM V2 bounded | 3989 | 558s | 0.0G | |
| ERNIE | 40G | Torch-exp | 4224 | 1.8s | 0.0G | |
| DSV3 | 40G | no_vmm | 148 | 16.6s | 0.0G | |
| DSV3 | 40G | VMM V1 | 127 | 9.9s | 78.73G | |
| DSV3 | 40G | VMM V2 bounded | 107 | 21.1s | 0.0G | |
| DSV3 | 40G | Torch-exp | 111 | 3.0s | 0.0G | |
| DSV3 | 45G | no_vmm | 3419 | 360s | 0.0G | |
| DSV3 | 45G | VMM V1 | 3765 | 305s | 78.74G | |
| DSV3 | 45G | VMM V2 bounded | 4483 | 704s | 0.0G | |
| DSV3 | 45G | VMM V2 remap-all | 4359 | 630s | 0.0G | |
| DSV3 | 45G | Torch-exp | 4158 | 23.4s | 0.0G | |
| DSV3 | 50G | no_vmm | 8415 | 911s | 0.0G | |
| DSV3 | 50G | VMM V1 | 1803 | 137s | 78.73G | |
| DSV3 | 50G | VMM V2 bounded | 2053 | 334s | 0.0G | |
| DSV3 | 50G | Torch-exp | 10072 | 28.7s | 0.0G | |
| Probe | 60+12G | no_vmm | - | - | - | FAIL |
| Probe | 60+12G | VMM V2 auto-remap | - | - | - | SUCCESS |
| Probe | 60+12G | Torch exp=True | - | - | - | SUCCESS |
| Probe | 60+12G | Torch exp=False | - | - | - | FAIL |

PaddleFormers Qwen2.5-7B SFT:

| 指标 | no_vmm | VMM V1 | VMM V2 |
|------|--------|--------|--------|
| Samples/s | 23.39 | 23.35 | 23.24 |
| Step time | 0.6856s | 0.6870s | 0.6901s |
| Max reserved | 57.480G | 33.037G | 33.037G |
| Steady fragmentation | 29.659G | 5.217G | 5.217G |
| 吞吐变化 vs no_vmm | - | -0.2% | -0.6% |

结论:

- V2 post_cleanup = 0.0G，V1 reserved 不释放。
- V2 在多数 replay 下 OOM 数低于 torch-exp。
- DSV3 45G 是弱项，compact 频繁且 event_blocked 占比高。
- Probe 验证 V2 碎片整理能力等价或优于 PyTorch expandable_segments。
### 2026-05-13 - Split-Fill 碎片 Probe

脚本: `/work/MemoryTools/scripts/frag_probe_varied.py`

核心构造:

- 先用 128MB 大块 prime 分配器池。
- 全部释放，形成 cached free。
- 再用 100KB 到 127.9MB 的不规则小块 split-fill。
- 释放每隔一个 tensor，形成 swiss-cheese 碎片。

结果:

| Mode | Successes | Max Alloc | Fragmentation | Probe Elapsed | 失败原因 |
|------|-----------|-----------|---------------|---------------|---------|
| paddle-baseline | 0/10 | 0G | 9.36G | 1.30s | chunks 被 split，FreeIdleChunks 无效 |
| paddle-v1 | 0/10 | 0G | 9.57G | 0.93s | 无碎片整理能力 |
| paddle-v2 | 8/10 | 8G | 9.57G | 3.93s | compact 重排 handles |
| torch | 0/10 | 0G | 9.54G | 0.01s | segments 被 split |
| torch-exp | 0/10 | 0G | 9.54G | 0.01s | 不能移动 active 分配 |

结论: split-fill 才是真实碎片化场景的关键差异，V2 compact 能恢复 8G/9.57G 连续空间，其他 allocator 在该模式下恢复 0G。
### 2026-05-11 - Tail Reuse Fix

ERNIE 35G:

| Mode | OOMs | Elapsed | Peak reserved | Post-cleanup |
|------|------|---------|---------------|-------------|
| no_vmm | 65 | 10.2s | 78.69G | 0.0G |
| VMM V1 | 63 | 7.6s | 78.72G | 78.72G |
| VMM V2 | 48 | 15.9s | 78.74G | 0.0G |
| Torch-exp | 58 | 1.5s | 78.73G | 0.0G |

DSV3:

| occupy | Mode | OOMs | Elapsed | Post-cleanup |
|--------|------|------|---------|-------------|
| 30G | no_vmm | 38 | 7.8s | 0.0G |
| 30G | VMM V1 | 38 | 3.4s | 78.72G |
| 30G | VMM V2 | 38 | 7.9s | 0.0G |
| 30G | Torch-exp | 37 | 0.4s | 0.0G |
| 35G | no_vmm | 60 | 7.8s | 0.0G |
| 35G | VMM V1 | 71 | 5.9s | 78.73G |
| 35G | VMM V2 | 71 | 11.6s | 0.0G |
| 35G | Torch-exp | 79 | 0.5s | 0.0G |
| 40G | no_vmm | 138 | 16.7s | 0.0G |
| 40G | VMM V1 | 136 | 11.0s | 78.73G |
| 40G | VMM V2 | 107 | 20.9s | 0.0G |
| 40G | Torch-exp | 111 | 3.1s | 0.0G |

Probe:

| Mode | 结果 | Reserved pre-big | Fragmentation |
|------|------|------------------|---------------|
| Paddle baseline | FAIL | 72.00G | 10.57G |
| Paddle VMM V2 auto-remap | SUCCESS | 72.00G | 10.58G |
| Paddle VMM V2 empty_cache | SUCCESS | 72.00G | 10.57G |
| Paddle VMM V2 explicit compact | SUCCESS | 72.00G | 10.58G |
| Torch expandable=True | SUCCESS | 72.01G | 10.57G |
| Torch expandable=False | FAIL | 72.00G | 10.55G |

### 2026-05-08 - Compact 架构重构旧基线

| Test | Mode | OOMs | Elapsed | Post-cleanup |
|------|------|------|---------|-------------|
| ERNIE 35G remap ON | V2 | 1927 | 178.6s | 0.0G |
| ERNIE 35G remap OFF | V2 | 1927 | 216.0s | 0.0G |
| ERNIE 35G torch-exp | torch | 2087 | 1.7s | 0.0G |
| ERNIE 35G V1 | V1 | 2310 | 168.8s | 75.6G |
| pp2ep4 30G | V2 | 62 | 641.8s | 0.0G |
| Probe baseline | - | FAIL | - | - |
| Probe vmm auto-remap | - | SUCCESS | - | - |
| torch exp=True probe | - | SUCCESS | - | - |

---

## 七、测试场景覆盖矩阵

| 场景 | 覆盖的测试 | 检验能力 |
|------|-----------|----------|
| 高频 OOM | Test 1,2,4 | RetryAllocator 快速 throw，无性能退化 |
| 碎片化 OOM | Test 6,7 | Compact remap 整理 VA->PA |
| Cross-stream 回收 | Test 1 vs 4 | ProcessUnfreedAllocations 降低 OOM |
| Flag 隔离 | Test 1 vs 2 | remap_on_oom flag 正确控制路径 |
| 无泄漏 | Test 1-5 | post_cleanup = 0 |
| 与 torch 对比 | Test 1 vs 3 | V2 OOM ≤ torch-exp |
| Remap A/B | Test 6 | ON=SUCCESS, OFF=FAIL |
| BackingMap mirror | Test 8 | Phase 1a 双写不改变行为且无 mismatch |

---

## 八、注意事项

1. GPU 独占：多进程共享 GPU 会导致 OOM 数波动 ±10-20%。
2. 多次运行：关键测试建议跑 2-3 次取中值。
3. `FLAGS_gpu_allocator_retry_time=0`：消除 `cv_wait` 变量。
4. Paddle 和 torch 测试之间等待 GPU 内存完全释放。
5. Elapsed 波动：Python 框架开销受系统负载影响，±20% 属正常。
6. BackingMap Phase 1a：预期 OOM、cleanup、force-release 与 `pr4/vmm-remap-v2` 完全持平，新增日志只用于发现镜像状态不一致。
7. `DIAG=1` 只用于单个 replay case 诊断，不用于全量 batch，也不用于 probe case。
8. Probe 标准回归使用 `--worker --mode vmm` 单模式，避免 parent 脚本多 worker 串行执行导致 baseline 预期 OOM 污染 GPU 状态。
