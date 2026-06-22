# VMM V2 架构升级：Allocation View / Backing View 分离

> 日期: 2026-05-18
> 状态: 设计中；Phase 2b 过渡实现已完成 VMM V2 IPC 主链路
> 前序: `vmm_allocator_v2_final_plan.md`, `vmm_defrag_v2_free_block_remap.md`
> 目标: 降低 compactor 复杂度和 bug 面，为 IPC/多设备扩展打基础

---

## 一、动机

### 1.1 当前架构的核心矛盾

`BlockV2::parts_` 同时承载两种信息：

```
逻辑信息: 这段 VA 是 FREE/ACTIVE，大小是多少，可否被 split/merge
物理信息: 这段 VA 背后是哪个 handle，handle 的 rel_off/len，是否被 remap，event 是否安全
```

这导致每次 **split/merge/remap/release** 都需要同步维护双方状态，产生了一系列耦合 bug：

| Bug | 本质 | 根因 |
|-----|------|------|
| Bug 5: re-remap SIGSEGV | parts 携带 stale handle ref | 逻辑视图未清理物理引用 |
| Bug 6: meta poisoning | CreateTailFreeBlock 用旧 metas | remap 后物理信息残留在逻辑 block |
| Synthetic ownership 链 | compact 需转移 handle lifecycle | 物理 ownership 绑定在逻辑 allocation 上 |
| force-release 316 handles | rollback 时原始 VA 被占用 | remap 操作散落在多处，无统一事务 |
| FreeIdleChunks 与 remapped 冲突 | IsRangeEntirelyFree 要检查 remapped 状态 | 释放判断需同时查逻辑+物理 |
| tail_offset stale | synthetic allocation 与底层 allocator 交错 | 物理映射范围由两个 owner 管理 |

**根本问题**: 6 个 bug 全部源自逻辑/物理耦合。双层分离后，这些 bug 在设计层面不存在。

### 1.2 设计目标

1. **Block 只管 VA 空间**：`{ptr, size, state}`，不持有 handle/event/remapped
2. **BackingMap 只管 VA→PA 映射**：page 粒度，统一 handle 生命周期
3. **Compact 查询方向反转**：从"物理页→查逻辑状态"替代"逻辑碎片→推导物理页是否完整"
4. **事务化 remap**：所有 cuMemMap/Unmap 通过 RemapTransaction，失败一行 Rollback
5. **渐进迁移**：不推翻 PR4，双写验证后再移除旧逻辑

### 1.3 2026-06 性能问题发现：h=2 下 parts eager split 成为热路径瓶颈

2026-06 的 4 机 MoE/DeepEP 训练压测发现：在不触发 remap 的 steady-state 下，VMM V2 使用较小 handle size（尤其 h=2MB）时吞吐明显低于 h=16MB 和 VMM off。该问题最初表现为：

- `FLAGS_vmm_v2_remap_on_oom=0` 时仍存在 h=2 稳态吞吐下降。
- `dispatch` / `moe-mlp` 的 wall time 显著增加，但 DeepEP microbenchmark 中 CUDA event 时间对 handle size 不敏感。
- `h=16` clean 与 VMM off 接近，说明 VMM V2 的基础路径并非一定慢；问题集中在 h=2 带来的 handle/parts 粒度放大。

关键实验数据（step 11-20，`tokens/s/card`）：

| 实验 | tokens/s/card mean | 相对 h16 clean | 相对 h2 clean | dispatch | combine | moe-mlp | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| h2 clean | 3561.88 | -6.41% | 0.00% | 312.40ms | 141.76ms | 971.83ms | h2 存在真实稳态性能损失 |
| h2 skip remap-safety | 3609.75 | -5.16% | +1.34% | 309.58ms | 141.47ms | 967.25ms | remap-safety hot path 不是主因 |
| h2 split-once | 3627.42 | -4.69% | +1.84% | 297.53ms | 139.57ms | 954.54ms | mapped-free split 路径有真实成本 |
| h2 fake split parts | 3702.82 | -2.71% | +3.96% | 241.48ms | 118.88ms | 827.90ms | 主因基本锁定到 allocation split parts |
| h2 fast no-parts | 3704.10 | -2.68% | +3.99% | 243.18ms | 120.61ms | 828.86ms | 与 fake split parts 几乎一致 |
| h16 clean | 3806.00 | 0.00% | +6.85% | 252.09ms | 120.07ms | 838.30ms | h16 基线接近 VMM off |
| h2 lazy parts clean | 3655.12 | -3.96% | +2.62% | 待补充 | 待补充 | 待补充 | lazy parts 已生效，但未达到 fake/fast no-parts 收益 |

由此得到的直接结论：

1. h=2 的主要性能损失不是 remap，也不是 IPC/tensor-info，也不是 remap-safety 标记。
2. `fast_no_parts` 的收益几乎完全可由“只 fake mapped-free allocation split parts”复现。
3. 当前瓶颈在 `AllocFromFreeBlocks()` 复用 mapped-free block 时调用的：
   - `BlockV2::SplitMappedFreeSuffixFromPrefix()`
   - `BlockV2::SplitPartsAt()`
   - `SliceBlockPartsForRange()`
4. h=2 时同样大小的 logical block 被切成更多 `BlockPartV2`，每次 allocation split 都 eager materialize prefix/suffix parts vector；该 CPU 热路径被训练中的高频 alloc/free 放大，最终表现为 dispatch/moe-mlp wall time 增加和吞吐下降。
5. 初版 `FLAGS_vmm_v2_lazy_block_parts=1` 只恢复约一半 gap：相对 h2 clean 提升 `+2.62%`，但仍比 h16 clean 低 `-3.96%`。这说明 lazy parts 路径已绕过主要 eager parts slice，但为了保持正确性仍保留的 block 级 `ipc_exported_` / remap-safety 元数据传播，或尚未迁移的 release/split range 残余 parts 路径，仍可能贡献剩余开销。此前 fake/fast 实验不能直接视为最终正确实现的性能，因为它们可能同时绕过了部分正确性元数据维护。

这组性能实证进一步支持本设计文档的核心方向：`BlockV2::parts_` 不应继续作为正常 allocator hot path 的必备状态。物理 backing 信息应由 Backing View / `backing_map_` 统一维护，Allocation View 只维护逻辑 VA 区间。

### 1.4 过渡期优化目标：lazy parts / backing view 查询

完整 Allocation View / Backing View 分离是最终目标，但当前代码仍处于渐进迁移阶段。为了先解决 h=2 性能问题，可以先引入过渡期的 `lazy parts` 策略：

- allocator 正常 alloc/free/split/merge 热路径不 eager 维护完整 `BlockV2::parts_`。
- `BlockV2` 只保留逻辑 VA 信息：`ptr_`、`size_`、`type_`、`pool_type_`、必要的 remap-safety / IPC gate 状态。
- `backing_map_` 作为 physical backing truth，记录 VA page 到 handle/meta/event/ipc 的映射。
- 只有 IPC、tensor-info、remap 事务等确实需要 backing 明细时，才按 VA range 从 `backing_map_` materialize parts 或 page list。

该过渡策略的关键约束是：不能简单把 `parts_` 清空后保留现有所有调用。只要仍有代码从 no-parts block 调用 `MakeMapped*SubBlock()` / `SplitPartsAt()` / `SliceBlockPartsForRange()`，就会出现不完整 parts slice，例如：

```text
Invalid VMM V2 block-part slice range: requested ... bytes at offset 0,
but only sliced ... bytes from ... parts.
```

因此 lazy parts 必须与消费点迁移一起完成。

#### IPC 路径可行性

底层 VMM V2 已经支持基于 VA range 查询 IPC parts，不要求 `BlockV2::parts_` 是完整状态：

```cpp
CUDAVirtualMemAllocatorV2::CollectBlockIpcParts(block, parts)
  -> CollectIpcParts(block.BeginVA(), block.Size(), parts)

CUDAVirtualMemAllocatorV2::MarkBlockIpcExported(block)
  -> MarkIpcExported(block.BeginVA(), block.Size())
```

但 best-fit 层当前仍会先构造临时 tensor block：

```cpp
BlockV2 tensor_block = block_it->MakeMappedActiveSubBlock(block_offset, size);
underlying_allocator_->CollectBlockIpcParts(tensor_block, &collected);
underlying_allocator_->MarkBlockIpcExported(tensor_block);
```

lazy parts 下应改为直接按目标 VA range 访问 backing map：

```cpp
underlying_allocator_->CollectIpcParts(target_va, size, &collected);
underlying_allocator_->MarkIpcExported(target_va, size);
```

因此 IPC 方向可行，但必须移除 best-fit 层对 `MakeMappedActiveSubBlock()` 的依赖。

实现时必须保留 `CollectIpcPartDescriptorsLocked` 的 fail-fast 语义。该函数在 page 级遇到以下状态时直接返回 `false`：

- `!page.mapped`
- `page.meta == nullptr`
- `page.meta->IsOwnedByRemapDestination()`

去掉 hot-path parts 后，这条 page 级校验会成为 IPC 正确性的主要防线：只有 backing map 确认目标 VA range 覆盖的每个 page 都是已映射、meta 有效且不属于 remap-destination 临时 ownership，才能导出 IPC metadata。这也是 IPC 可以脱离 `BlockV2::parts_` 的前提；校验本来就应该发生在 page/backing 维度，而不是 parts vector 维度。

#### Remap 路径可行性

remap 的核心 source 选择已经主要基于 free VA ranges 和 `backing_map_`：

```cpp
CollectFreeRanges(blocks)
underlying_allocator_->CollectRemapSourcePages(source_ranges, requested_size)
```

`SetBlockRemapEvent()` 也已经下沉为 VA range 操作：

```cpp
SetBlockRemapEvent(block)
  -> SetRemapEvent(block.BeginVA(), block.Size(), stream, event)
```

这说明 remap source 选择、page readiness、event gate 和 handle move 本身不需要 `BlockV2::parts_` 作为 truth。

仍需迁移的残余点是 remap transaction 中的 block list 重建：

```cpp
AppendMappedFreeSubRange()
  -> source.MakeMappedFreeSubBlock(...)

MergeAdjacentFreeBlocks()
  -> AbsorbAdjacentBlock(...)
```

这些调用会 eager slice/append `BlockV2::parts_`。lazy parts 下应构造 no-parts mapped-free block，或者由 `backing_map_` 在确需 parts 时按 VA range 临时 materialize。

同时，去 parts 时不能误删 remap 残留段的 block 级元数据传播。`MakeMappedFreeSubBlock()` 当前除了 slice parts，还会做：

- `ipc_exported_ = source.ipc_exported_`
- `CopyRemapSafetyFrom(source)`

这些与 parts 无关，仍然必须保留。lazy parts 只是移除 backing parts 的 eager materialization，不等于移除 block 级 IPC gate 或 remap-safety 状态。remap transaction 在生成残留 free segment、合并相邻 free block 时，仍需正确传播：

- `ipc_exported_`：合并时按 OR 传播，避免被 IPC export 过的 backing 被误复用、误搬迁或误释放。
- `owning_stream_` / `remap_safe_event_` / `remap_pending_states_`：残留段必须继承 source 的 remap-safety 约束，合并时也必须保守合并。

#### 过渡实现建议

第一阶段建议引入受控开关，例如：

```bash
FLAGS_vmm_v2_lazy_block_parts=1
```

在该开关下：

1. `AllocFromFreeBlocks()` 复用 mapped-free block 时，不调用 `SplitMappedFreeSuffixFromPrefix()`；直接按 VA/size 生成 active prefix 和 free suffix。
2. `TryMerge()` 只合并 VA range、IPC gate 和 remap-safety 状态，不 append parts vector。
3. grow / tail reuse 产生的 active/remainder block 也允许 no-parts view。
4. `CollectTensorParts()` 改为直接 `CollectIpcParts(target_va, size)`。
5. `MarkBlockIpcExported()` 改为直接 `MarkIpcExported(target_va, size)`。
6. `remap_transaction.cc::AppendMappedFreeSubRange()` 生成 no-parts mapped-free block。
7. release / split range helper 遇到 no-parts block 时不得调用 parts slice，应继续生成 no-parts block。
8. no-parts subblock / merge 仍必须复制或合并 `ipc_exported_` 和 remap-safety metadata；只跳过 parts vector。
9. IPC export 仍必须依赖 `CollectIpcPartDescriptorsLocked` 的 page 级 fail-fast 校验，不能因为去 parts 而放宽导出条件。

该开关只用于过渡验证；默认仍可保持旧 eager parts 路径，直到 IPC/remap/tensor-info smoke 全部通过。

#### 验证标准

lazy parts 方案需要同时通过性能和正确性验证：

| 验证 | 目标 |
|---|---|
| h2 lazy clean | 吞吐接近 `h2 fake split parts` / `h2 fast no-parts`，即约 `3700+ tokens/s/card` |
| h16 lazy clean | 不比 h16 clean 明显下降 |
| h2 lazy stats | `mapped_free_split_us` 显著下降，且 block/free index 数量稳定 |
| IPC smoke | `_share_cuda()` / `CollectIpcParts(target_va, size)` 成功，不依赖完整 `BlockV2::parts_`；未 mapped / meta 为空 / remap-destination ownership 的 page 必须 fail-fast |
| remap smoke | 触发一次 compact，source collect / move / rollback / restore 正常；残留 free segment 的 `ipc_exported_` 和 remap-safety metadata 必须保留 |
| tensor-info smoke | `vmm_tensor_info` 从 backing map 查询，不触发 parts eager slice 或显存泄漏 |

若上述验证通过，说明可以把 `BlockV2::parts_` 从 allocator hot path 中正式移除，并把 backing 信息收敛到 Backing View。

---

## 二、架构总览

```
┌──────────────────────────────────────────────────────┐
│                  User / Framework                     │
│              Alloc(size) / Free(ptr)                  │
└─────────────────────────┬────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│            Allocation View (BestFit)                  │
│                                                      │
│  数据: std::list<AllocationBlock>                     │
│        FreeIndex: size → iterator                    │
│                                                      │
│  职责: split / merge / best-fit 查找                  │
│        返回 VA 区间给用户                              │
│        不知道 handle, 不知道 physical backing          │
│                                                      │
│  接口:                                               │
│    Alloc(size) → ptr                                 │
│    Free(ptr)                                         │
│    GetFreeRanges() → [{va, size}, ...]               │
│    IsRangeFree(va, size) → bool                      │
└─────────────────────────┬────────────────────────────┘
                          │ "ensure backing" / "release backing"
                          ▼
┌──────────────────────────────────────────────────────┐
│            Backing View (Page Table)                  │
│                                                      │
│  数据: std::vector<BackingPage> pages_               │
│        (flat array, index = (va - base) / page_size) │
│                                                      │
│  职责: VA page → physical handle 映射                 │
│        handle 生命周期 (create / release)              │
│        event 安全性判断                               │
│        IPC export 标记                               │
│                                                      │
│  接口:                                               │
│    EnsureBacking(va, size)                           │
│    ReleaseBacking(va, size)                          │
│    CollectCompactablePages(free_ranges) → pages      │
│    IsPageSafe(va) → bool                            │
│    AllPagesReleasable(va, size) → bool              │
│    GetHandle(va) → handle                           │
└─────────────────────────┬────────────────────────────┘
                          │ cuMemMap / cuMemUnmap / cuMemCreate / cuMemRelease
                          ▼
┌──────────────────────────────────────────────────────┐
│              CUDA Driver (VMM API)                    │
└──────────────────────────────────────────────────────┘
```

---

## 三、数据结构设计

### 3.0 核心概念：三态模型

**关键设计决策**: VA 空间中的每一页同时拥有两个独立状态维度：

```
AllocationState: ACTIVE / FREE
BackingState:    MAPPED / UNMAPPED / EVENT_PENDING / IPC_PINNED
```

组合矩阵：

| AllocationState | BackingState | 含义 | 可操作 |
|----------------|-------------|------|--------|
| ACTIVE | MAPPED | 用户持有的活跃分配 | 正常使用 |
| FREE | EVENT_PENDING | 用户已 Free，但 GPU 仍可能访问该 page | 不可分配、不可 compact、不可 release，等待 event 完成 |
| FREE | MAPPED | 已释放且物理 backing 仍在，events 已完成 | **Alloc 可直接分配** |
| FREE | UNMAPPED | 已释放且物理 backing 已被搬走或释放 | **需 EnsureBacking 后才能分配** |
| FREE | IPC_PINNED | 已释放但被 IPC export 锁定 | Compact 不可搬，Release 不可回收 |
| ACTIVE | UNMAPPED | **非法状态** | 不可能出现 |
| ACTIVE | EVENT_PENDING | **非法状态** | event pending 只出现在 Free 后的 page 保护阶段 |

**核心规则**：
1. `Alloc(size)` 返回给用户前必须满足 **FREE + MAPPED + events_complete**；如果候选区域是 FREE + UNMAPPED，必须先 `EnsureBacking` 成功
2. `Compact` 搬迁的源页必须是 **FREE + MAPPED**（搬走后变为 FREE + UNMAPPED）
3. `Compact` 搬迁的目标 VA 必须是 **FREE + UNMAPPED**（放入后变为 FREE + MAPPED）
4. `FreeIdleChunks` 回收的是 **FREE + MAPPED + events_complete + !ipc_exported** 页（回收后 VA reservation 保留，变为 FREE + UNMAPPED）
5. `Grow` 在 tail 新增 VA 时：Allocation View 扩展 FREE block，Backing View 映射新 handle → 结果 FREE + MAPPED
6. **Gap reuse 优先级**：正常分配先复用已有 `FREE+MAPPED`，再原地填充 `FREE+UNMAPPED` gap，只有没有合适 VA 候选时才 tail grow；OOM compact 复用 gap 作为 remap target，不新建 backing

**GAP 消除**：当前的 `BlockType::kGap` 等价于新架构中的 **FREE + UNMAPPED**——不再需要单独的 block type。

### 3.0.1 Phase 2b 过渡状态：IPC pinned block

完整设计中，IPC 状态属于 Backing View 的 page/handle 维度，即上表中的 `BackingState::IPC_PINNED`。不过当前 VMM V2 仍处于渐进迁移阶段，`BlockV2::parts_` 还承担一部分物理 backing 描述和 ownership 兼容职责，Backing View 尚未完全接管所有 handle 生命周期。因此 Phase 2b 先采用 block 级 `ipc_exported_` 作为行为 gate；同时 BackingMap mirror 已记录 page 级 IPC pin，作为后续迁移落点。

当前触发条件：

1. Tensor 调用 `_share_cuda()` 时，通过 `VmmTensorPartsVisitor` 访问 VMM allocator。
2. V1 继续走 `VirtualMemoryAutoGrowthBestFitAllocator::CollectTensorParts`。
3. V2 新增 `VMMAutoGrowthBestFitAllocatorV2::CollectTensorParts`。
4. V2 收集成功后，把对应 active block 标记为 `ipc_exported_ = true`。
5. 同步调用 `MarkBackingIpcExported`，把该 block 覆盖的 backing handle pages 标记为 IPC exported。

当前 `ipc_exported_` 的语义：

| 行为 | 处理方式 | 原因 |
|------|----------|------|
| free 后复用 | 不进入普通 free index；reuse/tail reuse 同时检查 BackingMap IPC pin | 当前进程释放 Tensor 不代表远端 importer 已关闭 |
| compact/remap source | 跳过该 block | IPC handle 已暴露，不能移动 physical backing |
| BackingMap compact candidate | 不返回 IPC exported mapped page | 为后续 page-level gate 准备一致语义 |
| release | 不回收该 block 的 backing | 避免 `cuMemRelease` 破坏远端导入方 |
| BackingMap release gate | 若 range 含 IPC exported page，则拒绝 idle release | 防止后续 block gate 弱化时误释放 |
| BackingMap reuse gate | 若 block `parts_` 指向的 handle 在 BackingMap 中含 IPC exported page，则不进入 free index、不走 normal/tail reuse | 防止 block 级标志缺失时误复用 |
| free block merge | `ipc_exported_` 通过 OR 传播 | merge 后无法只保护子区间，按保守粒度处理 |

这是一种保守策略：只要某段 backing 被 IPC export 过，就认为它可能被其他进程持有，直到进程退出或后续引入可靠 unpin 机制之前，不主动 reuse/remap/release。该策略牺牲部分内存回收能力，但可以保证 Phase 2b 的 IPC correctness。

当前限制：

- pin 粒度是 block 级，不是最终设计中的 page/handle 级。
- 一个小 slice export 后，merge 得到的较大 free block 也会被整体 pin。
- 当前没有远端 importer 引用计数，也没有 unpin 回调。
- 后续应将 `ipc_exported_` 行为 gate 继续下沉到 Backing View，并由 `CollectCompactablePages` / `AllPagesReleasable` / `CanReuseForAlloc` 统一判断 `!ipc_exported`。当前 compact candidate、idle release 和 reuse/tail reuse 的 IPC gate 已接入 BackingMap mirror。

### 3.1 Allocation View

```cpp
// 纯逻辑 block，不持有任何物理信息
struct AllocationBlock {
  void* ptr;           // VA 起始地址
  size_t size;         // 字节数
  BlockState state;    // ACTIVE / FREE
  PoolType pool_type;  // kSmall / kLarge
};

// Allocation View 只负责 VA 空间管理
class AllocationView {
 public:
  // 查找候选 FREE block，但不修改 block list。
  // 调用方必须先确认 backing/event 条件，再 CommitAlloc。
  void* FindFreeCandidate(size_t size);
  // 提交分配：split → 标记 ACTIVE → 返回 VA
  void* CommitAlloc(void* ptr, size_t size);
  // 仅在本轮查找中跳过候选，不修改 block list。
  void SkipCandidate(void* ptr, size_t size);
  // 释放：标记 FREE → merge 相邻 FREE blocks
  void Free(void* ptr);
  // 获取所有 FREE 区间（供 Compact / Release 使用）
  std::vector<std::pair<VmmDevicePtr, size_t>> GetFreeRanges() const;
  // 查询某 VA range 是否全部 FREE（供 Backing View 判断 page 是否可搬）
  bool IsRangeFree(VmmDevicePtr va, size_t size) const;
  // 扩展 VA 空间尾部（grow 时追加新 FREE block）
  void ExtendTail(VmmDevicePtr va, size_t size);
  // 尾部 FREE block 大小
  size_t TailFreeSize() const;
  // 尾部 FREE block 起始地址
  void* TailFreeStart() const;

 private:
  std::list<AllocationBlock> blocks_;
  // size → free block iterator 的有序索引
  std::map<size_t, std::set<std::list<AllocationBlock>::iterator>> free_index_;
};
```

**重要**: AllocationView **不区分** FREE+MAPPED / FREE+UNMAPPED / FREE+EVENT_PENDING。这个区分由 BackingMap 负责。分配必须采用两阶段提交，避免 `EnsureBacking` 失败后污染 AllocationView：
```
1. AllocationView::FindFreeCandidate(size) → 返回候选 ptr，但不 split、不标记 ACTIVE
2. BackingMap::CanReuseForAlloc(ptr, size) → 确认无 pending events / IPC pin
3. BackingMap::EnsureBacking(ptr, size) → 在候选 ptr 原 VA 上确保该区域有物理 backing
   - 如果已经 MAPPED: no-op
   - 如果 UNMAPPED: 对这些 pages 原地 `cuMemCreate + cuMemMap`
   - 禁止在该接口中改用 tail grow；tail grow 只能由上层在没有候选 VA 时显式触发
   - 如果失败: 不修改 AllocationView，继续找下一个候选或返回 OOM
4. AllocationView::CommitAlloc(ptr, size) → split + 标记 ACTIVE
5. 返回 ptr 给用户
```

**对比当前**：

| 当前 BlockV2 | 新 AllocationBlock |
|-------------|-------------------|
| `ptr_`, `size_`, `type_` | `ptr`, `size`, `state` |
| `pool_type_` | `pool_type` |
| `parts_` (vector\<BlockPartV2\>) | **移除** |
| 隐含的 merge 逻辑需遍历 parts | 纯 VA merge，O(1) |
| 需要区分 kFree / kGap | 统一 FREE (backing 状态由 BackingMap 管) |

### 3.2 Backing View

```cpp
// 每个 page 可能被多个 stream 使用（multi-stream pending events）
struct PendingEvent {
  gpuStream_t stream;
  std::shared_ptr<CudaEventGuard> event;
};

struct BackingPage {
  VmmDevicePtr va;                          // 该 page 对应的 VA (page-aligned)
  size_t size;                              // page_size (通常 2MB)
  CUmemGenericAllocationHandle handle;      // physical backing handle (0 if unmapped)
  bool mapped;                              // 当前是否 cuMemMap 到 va
  bool ipc_exported;                        // 是否被 IPC export (不可搬迁)
  // 多 stream 事件: 一个 page 可能被多个 stream 使用,
  // 所有 pending events 都完成后才能搬迁
  std::vector<PendingEvent> pending_events;
  uint64_t epoch;                           // 映射 epoch (用于事务冲突检测)
};

class BackingMap {
 public:
  explicit BackingMap(VmmDevicePtr va_base, size_t page_size, int device);

  // --- 基本操作 ---

  // 原地 backing 填充：只处理调用方传入的 [va, va+size)。
  // 如果 page 已 mapped，no-op；如果 page unmapped，则在原 VA 上 cuMemCreate + cuMemMap。
  // 注意：该接口不得自行 tail grow 或返回其他 VA；失败时返回 false，调用方不得 CommitAlloc。
  bool EnsureBacking(VmmDevicePtr va, size_t size);

  // Alloc reuse gate: 所有 pages 必须无 pending events、非 IPC pinned；
  // mapped pages 可直接复用，unmapped pages 可由 EnsureBacking 填充。
  bool CanReuseForAlloc(VmmDevicePtr va, size_t size) const;

  // 判断 [va, va+size) 是否已有连续 mapped backing，可直接返回给用户。
  bool HasContiguousMappedBacking(VmmDevicePtr va, size_t size) const;

  // Release: unmap + cuMemRelease 指定 VA 范围的 pages
  // 前提: AllocationView 确认该范围全部 FREE 且所有 events 完成
  // 注意: VA reservation 保留 (不 cuMemAddressFree)，只释放物理 handle
  void ReleaseBacking(VmmDevicePtr va, size_t size);

  // --- Compact 支持 ---

  // 收集可搬迁的 pages: 在 free_ranges 中找 mapped && events_complete && !ipc 的 pages
  std::vector<size_t> CollectCompactablePages(
      const std::vector<std::pair<VmmDevicePtr, size_t>>& free_ranges,
      const AllocationView& alloc_view) const;

  // 判断某 page 是否可 compact
  bool IsPageCompactable(size_t page_idx, const AllocationView& alloc_view) const;

  // 收集 unmapped 的 VA 区域 (作为 compact 目标)
  std::vector<std::pair<VmmDevicePtr, size_t>> CollectUnmappedRanges(
      const std::vector<std::pair<VmmDevicePtr, size_t>>& free_ranges) const;

  // --- 查询 ---

  bool IsPageMapped(VmmDevicePtr va) const;
  bool AllPagesReleasable(VmmDevicePtr va, size_t size) const;
  BackingPage& GetPage(VmmDevicePtr va);
  BackingPage& GetPage(size_t page_idx);
  const BackingPage& GetPage(VmmDevicePtr va) const;
  size_t PageIndex(VmmDevicePtr va) const { return (va - va_base_) / page_size_; }
  size_t page_size() const { return page_size_; }

  // --- Event ---

  // 记录 event: 同一 page 可被多个 stream 使用
  // 每个 stream 只保留最新 event (替换旧的)
  void RecordEvent(VmmDevicePtr va, size_t size,
                   gpuStream_t stream,
                   std::shared_ptr<CudaEventGuard> event);
  // 判断该 page 所有 pending events 是否都已完成
  bool AllEventsComplete(VmmDevicePtr va) const;

  // --- 统计 ---

  size_t TotalMappedBytes() const;
  size_t TotalUnmappedPages() const;
  size_t TotalPages() const { return pages_.size(); }

 private:
  VmmDevicePtr va_base_;
  size_t page_size_;       // 2MB (CUDA granularity)
  int device_;
  std::vector<BackingPage> pages_;   // flat array, O(1) access
  size_t mapped_count_ = 0;         // 快速统计
};
```

**多 Stream Event 模型说明**:
- 一个 allocation 可能在 stream A 上 compute，又在 stream B 上 D2D copy
- Free 后该 allocation 覆盖的 pages 可能有来自两个 stream 的 pending events
- Compact 必须等待**所有** pending events 完成才能搬迁该 page
- 每个 stream 只保留最新 event（因为同 stream 内是顺序执行，旧 event 完成 ⊂ 新 event 完成）
- `AllEventsComplete` 对 `pending_events` 逐一 `cudaEventQuery`，全部完成则可搬

**容量估算** (A100-80G):
- 80 GiB / 2 MiB = 40960 pages
- `sizeof(BackingPage)` ≈ 80 bytes (含 vector 开销)
- 总内存: ~3.2 MB (negligible)

### 3.3 GAP 消除（三态视角）

当前 `BlockType::kGap` 表示"VA 已预留但 handle 被 remap 走了的空洞"。

三态模型下的等价表达：
- **kGap** = FREE + UNMAPPED（VA 仍属于 pool，但无物理 backing）
- **kFree** = FREE + MAPPED（VA 属于 pool 且有物理 backing 可分配）
- **kActive** = ACTIVE + MAPPED（正常使用中）

双层架构下：
- AllocationView 只有 FREE/ACTIVE，**不区分** MAPPED/UNMAPPED
- BackingMap 记录每个 page 是 mapped 还是 unmapped
- **Alloc 必须检查 backing**：在 AllocationView 返回候选 ptr 后，调用 `EnsureBacking` 原地确保该区域有 backing
  - 如果 ptr 落在已有 backing 的区域 (FREE+MAPPED) → no-op
  - 如果 ptr 落在无 backing 的区域 (FREE+UNMAPPED) → 在 ptr 原 VA 上新建 handle 并 map
  - 如果原地 `EnsureBacking` 因物理显存不足失败 → 不提交 allocation；后续 OOM compact 可把其他 FREE+MAPPED backing 搬到该 gap

```
当前:  [ACTIVE 4MB][GAP 2MB][FREE 6MB][ACTIVE 2MB]
       GAP = VA 占位但无 physical backing

三态:  AllocationView:  [ACTIVE 4MB][FREE 8MB][ACTIVE 2MB]
       BackingMap:       page[2] = UNMAPPED  (原 GAP 位置, FREE+UNMAPPED)
                        page[3..5] = MAPPED  (原 FREE 位置, FREE+MAPPED)

Compact 后:
       AllocationView:  [ACTIVE 4MB][FREE 8MB][ACTIVE 2MB]  (不变)
       BackingMap:       page[2] = MAPPED  (handle 从 page[5] 搬来, FREE+MAPPED)
                        page[5] = UNMAPPED (handle 被搬走, FREE+UNMAPPED)
```

**Compact 语义（三态表述）**：
- 源: FREE + MAPPED → 搬走后变为 FREE + UNMAPPED
- 目标: FREE + UNMAPPED → 放入后变为 FREE + MAPPED
- 目标选择策略: 优先选尾部连续的 UNMAPPED 区域，使高地址 pages 变为连续 MAPPED → 创造大块可分配空间

**Gap reuse 的两条路径**：
1. 正常分配路径：`FindFreeCandidate` 命中 `FREE+UNMAPPED` gap 后，`EnsureBacking(candidate_va, size)` 在原 VA 上创建并映射新 backing，然后 `CommitAlloc`。
2. Compact 路径：`CollectUnmappedRanges(free_ranges)` 将 `FREE+UNMAPPED` gap 作为 remap target，`MovePage(source, gap_va)` 把已有 backing 搬入 gap，不创建新 backing。

---

## 四、Compact 流程（双层版）

### 4.1 当前流程 vs 新流程

**当前 (PR4)**:
```
1. 遍历 FREE blocks → 遍历 parts → IsFullyCoveredHandle
2. 收集 fully-covered handles → cudaDeviceSynchronize
3. UnmapHandle (从源 VA)
4. 选择目标: tail / single-gap / gap-scatter
5. MapHandlesToVA (到目标 VA)
6. CreateSyntheticAllocation (转移 ownership)
7. 创建新 FREE block with new metas
8. 设旧 metas remapped=true
9. 失败: PendingMappedRange + RollbackToOriginalVA + RestoreGapToFree
```

**新 (三态 + Backing View)**:
```
1. AllocationView::GetFreeRanges()
2. BackingMap::CollectCompactablePages(free_ranges)
   → 源: FREE + MAPPED + events_complete + !ipc
3. BackingMap::CollectUnmappedRanges(free_ranges)
   → 目标: FREE + UNMAPPED 区域
4. RemapTransaction::MovePage(source_idx, target_va) for each page
   → Unmap source + Map target + 源标记 UNMAPPED + 目标标记 MAPPED
5. Commit → 清空 undo log
6. 失败: RemapTransaction::Rollback() (一行)
```

步骤从 9 步降到 6 步。**关键语义变化**: Compact 不涉及 AllocationView 的任何修改——
AllocationView 中的 FREE block 不变（它不关心 backing 状态），只有 BackingMap 中的 page 状态发生变化。

### 4.2 目标选择策略 (SelectTarget)

**当前**: 目标固定为 tail / gap-scatter（对应 kGap blocks）

**新**: 目标为 `FREE + UNMAPPED` 区域，选择策略:

```cpp
VmmDevicePtr SelectTarget(const BackingMap& backing_map,
                          const AllocationView& alloc_view,
                          size_t needed_pages,
                          CompactMode mode) {
  // 获取 FREE 范围中 UNMAPPED 的区域
  auto free_ranges = alloc_view.GetFreeRanges();
  auto unmapped_ranges = backing_map.CollectUnmappedRanges(free_ranges);

  // 策略: 优先选择最大的连续 UNMAPPED 区域（使 compact 后产生大块连续 MAPPED）
  // 这等价于当前的 "tail placement"，因为 tail 通常是最大连续 UNMAPPED
  VmmDevicePtr best_target = 0;
  size_t best_size = 0;
  for (auto& [va, size] : unmapped_ranges) {
    if (size >= needed_pages * backing_map.page_size() && size > best_size) {
      best_target = va;
      best_size = size;
    }
  }

  // OOM bounded compact 的目标必须连续，否则搬够 requested_size 后
  // 仍可能没有一个连续 FREE+MAPPED 区间可满足本次 allocation。
  if (mode == CompactMode::kBoundedForOom) {
    return best_target;
  }

  // compact_all / cleanup 可以 scatter，因为目标不是立即满足单次大 allocation。
  if (best_target == 0 && !unmapped_ranges.empty()) {
    best_target = unmapped_ranges[0].first;
  }
  return best_target;
}
```

**注意**: 与当前的 `ExtendTail` 不同，新架构中 Compact **不创建新的 AllocationView block**。
因为 Compact 只改变 backing 状态（UNMAPPED→MAPPED），不改变 allocation 状态（仍然是 FREE）。
AllocationView 中的 FREE block 覆盖范围不变——下次 Alloc 时只需确认 backing 即可。

### 4.3 详细流程

```cpp
size_t Compactor::Compact(AllocationView* alloc_view,
                          BackingMap* backing_map,
                          size_t requested_size) {
  // --- 预检 ---
  auto free_ranges = alloc_view->GetFreeRanges();
  size_t total_free = 0;
  size_t max_free = 0;
  for (auto& [va, size] : free_ranges) {
    total_free += size;
    max_free = std::max(max_free, size);
  }

  // 预检 1: 物理内存不足
  if (total_free < requested_size) return 0;
  // 预检 2: 已有连续大块 (注意: 需确认该 FREE 区域有 backing)
  if (max_free >= requested_size) {
    // 进一步检查: 是否有足够大的 FREE+MAPPED 连续区域
    // 如果有，无需 compact
    // 如果 max_free 区域是 UNMAPPED 的，仍然需要 compact 来填充它
    if (HasContiguousMappedFree(*alloc_view, *backing_map, requested_size)) {
      return 0;
    }
  }

  // --- 收集可搬迁 pages (源: FREE + MAPPED) ---
  auto compactable = backing_map->CollectCompactablePages(
      free_ranges, *alloc_view);

  // 预检 3: 无可搬迁 pages
  if (compactable.empty()) return 0;

  size_t page_size = backing_map->page_size();
  size_t total_compactable = compactable.size() * page_size;
  if (total_compactable < requested_size) return 0;

  // --- 选择目标 VA (目标: FREE + UNMAPPED) ---
  auto unmapped_ranges = backing_map->CollectUnmappedRanges(free_ranges);
  if (unmapped_ranges.empty()) return 0;  // 无可用目标

  // bounded OOM compact 必须先找到一个能容纳 requested_size 的连续目标区间。
  // scatter 只允许 compact_all/cleanup 使用，不能把分散 backing 当成本次 OOM 恢复成功。
  auto target_range = SelectContiguousUnmappedRange(unmapped_ranges, requested_size);
  if (!target_range.has_value()) return 0;

  // --- 执行 remap (事务化) ---
  RemapTransaction tx(backing_map);

  size_t moved = 0;
  VmmDevicePtr dst = target_range->first;
  VmmDevicePtr dst_end = dst + target_range->second;

  for (size_t page_idx : compactable) {
    if (dst + page_size > dst_end) break;

    if (!tx.MovePage(page_idx, dst)) {
      tx.Rollback();
      return 0;
    }
    dst += page_size;
    moved += page_size;

    // bounded 模式: 搬够 requested_size 就停
    if (moved >= requested_size) break;
  }
  if (moved < requested_size) {
    tx.Rollback();
    return 0;
  }

  tx.Commit();

  // AllocationView 不需要修改!
  // 因为 Compact 只改变 backing 状态，不改变 allocation 状态。
  // 之前连续 FREE+UNMAPPED 的目标区域现在变成连续 FREE+MAPPED。
  // bounded OOM compact 只有在形成 requested_size 连续 mapped free 后才算成功。
  PADDLE_ENFORCE(HasContiguousMappedFree(*alloc_view, *backing_map, requested_size));

  return moved;
}
```

**关键差异 vs 当前设计**:
1. **无 AllocationView 修改**: Compact 后不需要 `ExtendTail` 或任何 block list 操作
2. **无 ownership 转移**: BackingMap 始终持有所有 handle，`MovePage` 只是改变 page→VA 的映射关系
3. **无 GAP 创建**: 源页变为 UNMAPPED，不需要创建 kGap block
4. **Scatter 自然支持**: 目标可以是多个不连续的 UNMAPPED 区域

### 4.4 CollectCompactablePages 实现

```cpp
std::vector<size_t> BackingMap::CollectCompactablePages(
    const std::vector<std::pair<VmmDevicePtr, size_t>>& free_ranges,
    const AllocationView& alloc_view) const {
  std::vector<size_t> result;

  for (auto& [va, size] : free_ranges) {
    // 按 page 边界遍历
    VmmDevicePtr page_start = AlignUp(va, page_size_);
    VmmDevicePtr range_end = va + size;

    for (VmmDevicePtr pva = page_start; pva + page_size_ <= range_end;
         pva += page_size_) {
      size_t idx = PageIndex(pva);
      const auto& page = pages_[idx];

      if (!page.mapped) continue;              // UNMAPPED，不是源
      if (page.ipc_exported) continue;         // IPC pinned，不可搬
      if (!AllEventsComplete(pva)) continue;   // 有 pending events，不安全

      // 确认整个 page 在 Allocation View 中全是 FREE
      // (处理 page 跨 ACTIVE/FREE 边界的情况)
      if (!alloc_view.IsRangeFree(pva, page_size_)) continue;

      result.push_back(idx);
    }
  }
  return result;
}
```

**对比当前 `IsFullyCoveredHandle`**:

| 当前 | 新 |
|------|-----|
| 从 block.parts_ 推导 handle 是否完全被 FREE 覆盖 | 从 page VA 查 AllocationView 是否全 FREE |
| 需要 `handle_rel_off == 0 && len == handle_size` | 只需 `IsRangeFree(page_va, page_size)` |
| 要处理 remapped=true 的 skip | 不存在 remapped 概念 |
| 要处理 parts 跨 block 合并后的一致性 | 无关——page 是独立单元 |

---

## 五、RemapTransaction 设计

### 5.1 核心语义：MovePage

**关键设计决策**: RemapTransaction 使用 `MovePage` 而非分离的 `UnmapPage` + `MapPage`。

原因: 如果分开 Unmap 和 Map，在 Commit 时需要额外逻辑清理源页的 ownership。
`MovePage` 语义明确——一次性完成 "源 unmap + 目标 map + ownership 转移"。

```
MovePage(source_page_idx, target_va):
  1. unmap source VA
  2. map handle to target VA + SetAccess
  3. 记录 undo log: {source_va, target_va, handle}
  4. Commit 时: source page → UNMAPPED, target page → MAPPED with handle
  5. Rollback 时: unmap target + remap back to source
```

### 5.2 接口

```cpp
class RemapTransaction {
 public:
  explicit RemapTransaction(BackingMap* backing_map);

  // 搬迁一个 page: 从 source_page_idx 的 VA 搬到 target_va
  // 返回 false 表示 cuMemMap 失败（需要 Rollback）
  bool MovePage(size_t source_page_idx, VmmDevicePtr target_va);

  // 提交：更新 BackingMap 中源页和目标页的状态
  void Commit();

  // 回滚：逆序恢复所有操作 (unmap target + remap source)
  void Rollback();

  // 统计
  size_t MovedCount() const { return move_log_.size(); }

 private:
  struct MoveEntry {
    size_t source_page_idx;     // 源页 index
    VmmDevicePtr source_va;     // 源 VA
    VmmDevicePtr target_va;     // 目标 VA
    size_t target_page_idx;     // 目标页 index
    CUmemGenericAllocationHandle handle;  // 被搬迁的 handle
  };

  BackingMap* backing_map_;
  std::vector<MoveEntry> move_log_;
};
```

### 5.3 实现

```cpp
bool RemapTransaction::MovePage(size_t source_page_idx, VmmDevicePtr target_va) {
  size_t page_size = backing_map_->page_size();
  auto& source_page = backing_map_->GetPage(source_page_idx);
  PADDLE_ENFORCE(source_page.mapped, "Cannot move an unmapped page");
  PADDLE_ENFORCE(source_page.pending_events.empty(),
                 "Cannot move a page with pending events");
  PADDLE_ENFORCE(!source_page.ipc_exported, "Cannot move an IPC pinned page");

  size_t target_idx = backing_map_->PageIndex(target_va);
  auto& target_page = backing_map_->GetPage(target_idx);
  PADDLE_ENFORCE(!target_page.mapped, "Target page must be unmapped");
  PADDLE_ENFORCE(target_page.pending_events.empty(),
                 "Target page must not have pending events");
  PADDLE_ENFORCE(!target_page.ipc_exported, "Target page must not be IPC pinned");

  CUmemGenericAllocationHandle handle = source_page.handle;
  VmmDevicePtr source_va = source_page.va;

  // Reserve undo-log capacity before driver operations. After this point,
  // MoveEntry push_back should not allocate; otherwise rollback may lose
  // the only record needed to restore source mapping.
  move_log_.reserve(move_log_.size() + 1);

  // Step 1: Unmap from source VA
  phi::dynload::cuMemUnmap(source_va, page_size);

  // Step 2: Map to target VA
  CUresult status = phi::dynload::cuMemMap(target_va, page_size, handle, 0, 0);
  if (status != CUDA_SUCCESS) {
    // Map 失败: 尝试恢复源映射
    CUresult restore = phi::dynload::cuMemMap(source_va, page_size, handle, 0, 0);
    if (restore == CUDA_SUCCESS) {
      CUmemAccessDesc access = backing_map_->access_desc();
      phi::dynload::cuMemSetAccess(source_va, page_size, &access, 1);
    } else {
      VLOG(0) << "MovePage: cannot restore source mapping, handle orphaned";
    }
    return false;
  }

  // Step 3: SetAccess on target
  CUmemAccessDesc access = backing_map_->access_desc();
  status = phi::dynload::cuMemSetAccess(target_va, page_size, &access, 1);
  if (status != CUDA_SUCCESS) {
    // SetAccess 失败: unmap target, restore source
    phi::dynload::cuMemUnmap(target_va, page_size);
    CUresult restore = phi::dynload::cuMemMap(source_va, page_size, handle, 0, 0);
    if (restore == CUDA_SUCCESS) {
      phi::dynload::cuMemSetAccess(source_va, page_size, &access, 1);
    }
    return false;
  }

  // Step 4: 立即更新 BackingMap 状态 (Commit 不需要额外操作)
  source_page.mapped = false;
  source_page.handle = 0;
  source_page.pending_events.clear();  // 源页不再需要 events

  target_page.handle = handle;
  target_page.mapped = true;
  target_page.epoch++;

  // Step 5: 记录 undo log (供 Rollback 使用)
  move_log_.push_back({source_page_idx, source_va, target_va, target_idx, handle});

  return true;
}

void RemapTransaction::Commit() {
  // MovePage 已经实时更新了 BackingMap，Commit 只需清空 undo log
  // 这意味着: 如果后续不 Rollback，状态就是正确的
  move_log_.clear();
}

void RemapTransaction::Rollback() {
  size_t page_size = backing_map_->page_size();
  CUmemAccessDesc access = backing_map_->access_desc();

  // 逆序恢复：unmap target → remap to source
  for (auto it = move_log_.rbegin(); it != move_log_.rend(); ++it) {
    // Unmap from target
    phi::dynload::cuMemUnmap(it->target_va, page_size);
    auto& target_page = backing_map_->GetPage(it->target_page_idx);
    target_page.mapped = false;
    target_page.handle = 0;

    // Remap back to source
    CUresult status = phi::dynload::cuMemMap(
        it->source_va, page_size, it->handle, 0, 0);
    if (status == CUDA_SUCCESS) {
      phi::dynload::cuMemSetAccess(it->source_va, page_size, &access, 1);
      auto& source_page = backing_map_->GetPage(it->source_page_idx);
      source_page.handle = it->handle;
      source_page.mapped = true;
    } else {
      // Rollback 中 map 失败 = handle 彻底丢失
      VLOG(0) << "RemapTransaction::Rollback: cannot restore page at "
              << reinterpret_cast<void*>(it->source_va) << ", handle lost";
      platform::RecordedGpuMemRelease(
          it->handle, page_size, backing_map_->device());
    }
  }

  move_log_.clear();
}
```

### 5.4 MovePage vs 分离 Unmap+Map 的正确性论证

| 方面 | 分离 Unmap+Map | MovePage |
|------|---------------|----------|
| 源 ownership 清理 | Commit 时遍历 unmap_log 清理源页 handle | MovePage 内立即清理 |
| Rollback 时源状态 | unmap_log 已记录源 handle，可恢复 | move_log 记录源 handle，可恢复 |
| 中间状态一致性 | Unmap 后、Map 前: handle 既不在源也不在目标 | Map 失败时立即恢复源 |
| BackingMap 观察一致性 | 事务中 BackingMap 状态不一致（源已清，目标未设）| 每次 MovePage 后 BackingMap 始终一致 |

`MovePage` 的优势: **每次操作后 BackingMap 处于一致状态**——不存在 handle "悬空"在中间的时刻。

### 5.5 对比当前 rollback 机制

| 当前 | RemapTransaction |
|------|-----------------|
| `PendingMappedRange` + `UnmapPendingMappedRanges` | `move_log_` + `Rollback()` |
| `RollbackToOriginalVA` (~80 行) | `Rollback()` 逆序恢复 |
| `RestoreGapToFree` (找 GAP block → 改 block list) | 不需要——源页标 UNMAPPED 即可 |
| `UnmapPartialDestination` | 不需要——Rollback 逐页恢复 |
| `CreateSyntheticAllocation` (ownership 转移) | 不需要——BackingMap 统一持有 |
| force-release 兜底 (Rollback 中 map 失败) | Rollback 中 map 失败 → release + 日志 |
| ~200 行 rollback 相关代码 | **~40 行** |

---

## 六、Grow / Release 流程

### 6.1 Grow (分配时 free 不足)

```cpp
void* VMMBestFitAllocatorV2::Alloc(size_t size) {
  // 1. Allocation View: 尝试 BestFit 候选，但先不提交 ACTIVE 状态。
  // 分配优先级:
  //   a) FREE + MAPPED + events_complete: 直接复用
  //   b) FREE + UNMAPPED: EnsureBacking 在候选 VA 原地填充 gap
  //   c) 没有合适候选时才 tail grow
  void* ptr = nullptr;
  while ((ptr = alloc_view_.FindFreeCandidate(size)) != nullptr) {
    VmmDevicePtr va = reinterpret_cast<VmmDevicePtr>(ptr);
    if (!backing_map_.CanReuseForAlloc(va, size)) {
      alloc_view_.SkipCandidate(ptr, size);  // 仅本轮查找跳过，不修改 block list
      continue;
    }
    if (backing_map_.EnsureBacking(va, size)) {  // 原地填充；已 mapped 则 no-op
      return alloc_view_.CommitAlloc(ptr, size);
    }
    alloc_view_.SkipCandidate(ptr, size);
  }

  // 2. 只有没有可用 FREE VA 候选，或所有候选都无法原地 EnsureBacking 时，才 grow。
  size_t grow_size = AlignUp(size, page_size_);

  // Tail reuse: 如果 Allocation View 末尾是 FREE block，
  // 只需补差额的 backing
  size_t tail_free = alloc_view_.TailFreeSize();
  size_t need_new_va = (grow_size > tail_free) ? (grow_size - tail_free) : 0;

  if (need_new_va > 0) {
    VmmDevicePtr new_va = va_base_ + total_reserved_size_;
    // 扩展 VA reservation (cuMemAddressReserve 或使用预留的 VA 池)
    ReserveMoreVA(new_va, need_new_va);
    alloc_view_.ExtendTail(new_va, need_new_va);
    total_reserved_size_ += need_new_va;
  }

  // 确保目标区域有 backing (包括 tail reuse 部分可能是 UNMAPPED 的)。
  // EnsureBacking 仍然是原地填充，不会选择其他 VA。
  VmmDevicePtr alloc_va = reinterpret_cast<VmmDevicePtr>(
      alloc_view_.TailFreeStart());
  PADDLE_ENFORCE(backing_map_.EnsureBacking(alloc_va, grow_size));

  // 3. 直接提交 tail 候选，避免再次 best-fit 选中其他 pending/unmapped 区域。
  PADDLE_ENFORCE(backing_map_.CanReuseForAlloc(alloc_va, size));
  return alloc_view_.CommitAlloc(reinterpret_cast<void*>(alloc_va), size);
}
```

### 6.2 Release (empty_cache)

**关键设计决策**: `FreeIdleChunks` **只释放物理 backing，保留 VA reservation**。

原因:
1. VA reservation (`cuMemAddressReserve`) 是虚拟地址空间，不占用物理资源
2. 保留 VA 使得后续 Grow 可以复用相同 VA range，避免 VA 碎片化
3. 当前的 `ShrinkRange` (释放 VA + 从 block list 移除) 导致 VA 空间收缩后无法恢复
4. 保留的 FREE+UNMAPPED 区域可以被 Compact 用作目标（填充 handles）或被 Grow 重新 EnsureBacking

```cpp
void VMMBestFitAllocatorV2::FreeIdleChunks() {
  auto free_ranges = alloc_view_.GetFreeRanges();

  for (auto& [va, size] : free_ranges) {
    // 只释放 FREE + MAPPED + events_complete + !ipc_exported pages 的物理 backing
    // VA reservation 保留，block list 中的 FREE block 保留
    backing_map_.ReleaseBacking(va, size);
    // ReleaseBacking 内部:
    //   对 [va, va+size) 范围内每个 mapped page:
    //     若 pending events 未完成: skip
    //     若 ipc_exported: skip
    //     cuMemUnmap(page.va, page_size)
    //     cuMemRelease(page.handle)
    //     page.mapped = false
    //     page.handle = 0
    //   跳过已经 unmapped 的 pages (no-op)
  }

  // 注意: AllocationView 不需要任何修改!
  // FREE block 仍然存在（VA 仍属于 pool），只是 backing 被释放了
  // 下次 Alloc 落到这些区域时，EnsureBacking 会重新创建 handles
}
```

**对比当前**:

| 当前 | 新设计 |
|------|--------|
| `IsRangeEntirelyFree` + `SplitAndRemoveRange` | `ReleaseBacking` (只操作 BackingMap) |
| 释放 VA + 从 block list 移除 | **只释放 backing，保留 VA** |
| 需要检查 `remapped` 状态 | 不需要——UNMAPPED pages 自动跳过 |
| `underlying_allocations_` 追踪 | BackingMap 统一管理 |
| VA 空间收缩 (不可恢复) | VA 空间保持 (可复用) |

---

## 七、Event 管理（多 Stream 模型）

### 7.1 当前问题

Event 附着在 `BlockV2` 或 `BlockPartV2` 上（`remap_safe_event_`）。问题：
1. Block split/merge 时 event 语义模糊：一个 event 保护的是哪些具体的物理页？
2. 只支持单 event：如果一个 allocation 被多个 stream 使用（D2D copy + compute），只记录最后一个
3. event 归属于 block 而非物理页：Compact 搬迁后 event 可能指向错误的物理位置

### 7.2 新设计：多 Stream Events 属于 Page

```cpp
// StreamSafeCUDAAllocator::FreeImpl 中：
void StreamSafeCUDAAllocator::FreeImpl(Allocation* alloc,
                                        gpuStream_t stream) {
  // 记录 event 到覆盖的所有 pages
  auto event = RecordEvent(stream);
  backing_map_->RecordEvent(
      reinterpret_cast<VmmDevicePtr>(alloc->ptr()),
      alloc->size(),
      stream,
      event);

  // 通知 Allocation View 标记为 FREE
  alloc_view_->Free(alloc->ptr());
}
```

```cpp
void BackingMap::RecordEvent(VmmDevicePtr va, size_t size,
                             gpuStream_t stream,
                             std::shared_ptr<CudaEventGuard> event) {
  size_t start_idx = PageIndex(AlignDown(va, page_size_));
  size_t end_idx = PageIndex(AlignUp(va + size, page_size_));
  for (size_t i = start_idx; i < end_idx; ++i) {
    auto& page = pages_[i];
    // 同 stream 的旧 event 替换为新 event (同 stream 内顺序保证)
    bool found = false;
    for (auto& pe : page.pending_events) {
      if (pe.stream == stream) {
        pe.event = event;
        found = true;
        break;
      }
    }
    if (!found) {
      page.pending_events.push_back({stream, event});
    }
  }
}
```

Compact 时：
```cpp
bool BackingMap::AllEventsComplete(VmmDevicePtr va) const {
  auto& page = pages_[PageIndex(va)];
  if (page.pending_events.empty()) return true;  // 从未被 GPU 使用

  for (auto& pe : page.pending_events) {
    if (cudaEventQuery(pe.event->event()) != cudaSuccess) {
      return false;  // 至少一个 stream 的操作未完成
    }
  }
  return true;  // 所有 stream 的操作都已完成
}

// 定期清理已完成的 events (避免 pending_events 无限增长)
void BackingMap::GarbageCollectEvents(VmmDevicePtr va) {
  auto& page = pages_[PageIndex(va)];
  page.pending_events.erase(
      std::remove_if(page.pending_events.begin(), page.pending_events.end(),
                     [](const PendingEvent& pe) {
                       return cudaEventQuery(pe.event->event()) == cudaSuccess;
                     }),
      page.pending_events.end());
}
```

### 7.3 多 Stream 场景示例

```
Stream A: matmul(tensor_X)  → event_A recorded on pages[10..15]
Stream B: D2D_copy(tensor_X, tensor_Y) → event_B recorded on pages[10..15]
Free(tensor_X):
  pages[10..15].pending_events = [{A, event_A}, {B, event_B}]

Compact 检查 page[10]:
  AllEventsComplete(page[10].va)?
    cudaEventQuery(event_A) → SUCCESS
    cudaEventQuery(event_B) → NOT_READY
  → return false, 不搬迁

(等 Stream B 完成后再次 compact)
  AllEventsComplete(page[10].va)?
    cudaEventQuery(event_A) → SUCCESS
    cudaEventQuery(event_B) → SUCCESS
  → return true, 可搬迁
```

### 7.4 Event GC 策略

`pending_events` 可能随时间增长（多个 stream 交替使用同一页）。GC 策略：
1. **Compact 入口时 bulk GC**: 对所有候选 pages 做一次 `GarbageCollectEvents`，清理已完成的 events
2. **Alloc 时 lazy GC**: `CanReuseForAlloc` 检查候选 pages 时清理已完成 events；仍有 pending event 的候选本轮跳过
3. **阈值 GC**: `pending_events.size() > 4` 时强制清理（实践中很少超过 2-3 个 stream）

**优势 vs 当前**：
- Event 粒度明确（page level，非 block level）
- Split/merge block 不影响 event 状态
- 多 stream 场景正确处理（不会丢失任一 stream 的 pending event）
- Compact 搬迁后源页 events 自动清除（在 `MovePage` 中 `pending_events.clear()`）

---

## 八、IPC 支持预留

```cpp
// Export: 标记 pages 为 ipc_exported，不可被 compact 搬迁
void BackingMap::MarkIpcExported(VmmDevicePtr va, size_t size) {
  ForEachPage(va, size, [](BackingPage& page) {
    page.ipc_exported = true;
  });
}

// Import: 从另一个进程的 handle 创建映射
void BackingMap::ImportBacking(VmmDevicePtr va, CUmemGenericAllocationHandle handle) {
  size_t idx = PageIndex(va);
  pages_[idx] = {va, page_size_, handle, true, true, {}, 0};
}
```

当前架构下 IPC 需要侵入 `BlockV2::parts_` 添加 pin 标记，且 compact 要额外检查每个 part 的 IPC 状态。双层后 IPC 完全在 Backing View 内闭环。

---

## 九、实施计划

### 9.0 分支策略与固定基线

Backing View 是架构级重构，不继续叠加在 `pr4/vmm-remap-v2` 的保底修复分支上实现。建议新建独立实验分支：

```bash
git checkout pr4/vmm-remap-v2
git checkout -b feature/vmm-v2-backing-view
```

固定对比基线：

| 分支 | 角色 | 用途 |
|------|------|------|
| `pr4/vmm-remap-v2` | PR4 保底版本 | correctness / OOM 次数 / cleanup / force-release / 耗时的基线 |
| `feature/vmm-v2-backing-view` | Backing View 实验版本 | 分阶段验证新架构收益与回归风险 |

每个阶段至少对比以下指标：

| 指标 | 目的 |
|------|------|
| OOM 次数 | 验证 remap/compact 是否实际提升恢复能力 |
| compact 成功次数 / 返回 0 次数 | 区分碎片可整理比例与无效 compact |
| cleanup_reserved / post_cleanup_reserved | 验证 backing ownership 无泄漏 |
| force-release 次数 | 验证 rollback 不再丢 handle |
| remap 耗时 / 总 elapsed | 验证复杂度下降没有换来性能退化 |
| target 是否连续 mapped free | 验证 bounded OOM compact 是否真的能满足本次 allocation |

推荐 A/B 测试命令保持同一 replay 脚本、同一 occupy size、同一 FLAGS，只切换分支二进制，避免把框架初始化显存差异、tail reuse 修复和 Backing View 重构混在一起。

### Phase 1a: 引入 BackingMap（双写，只读验证）

**目标**: 不改变行为，验证 BackingMap 模型正确

| 改动 | 文件 |
|------|------|
| 新增 `BackingMap` 类 | `vmm_backing_map.h`, `vmm_backing_map.cc` |
| `MapHandlesToVA` 时同步写 BackingMap | `cuda_virtual_mem_allocator_v2.cc` |
| `FreeImpl` (unmap) 时同步写 BackingMap | `cuda_virtual_mem_allocator_v2.cc` |
| DCHECK: BackingMap 与 parts_ handle 一致 | `vmm_auto_growth_best_fit_allocator_v2.cc` |
| 查询接口: `IsRangeMapped` / `IsRangeUnmapped` | `vmm_backing_map.h`, `vmm_backing_map.cc` |
| 区间收集接口: `CollectMappedRanges` / `CollectUnmappedRanges` | `vmm_backing_map.h`, `vmm_backing_map.cc` |

**验证**: 回归 8/8 + DCHECK 无 fire

**进展**:

- 2026-05-20: `feature/vmm-v2-backing-view` 在 `18df2b243a` 上完成 stable2 全量回归，`20260520_173159` 8/8 passed；默认日志无 `force-release` / BackingMap 错误。
- 2026-05-20: `dsv3_45g_compact_all DIAG=1` 单测功能路径通过，`force-release=0`，BackingMap 错误=0；RESULT 后 teardown 出现 `double free or corruption (!prev)`，与 VLOG/DIAG 强相关，历史旧 VMM 也可复现，暂不作为当前版本 regression。
- 2026-05-20: 补充 BackingMap range 查询与 mapped/unmapped 区间 coalesce 能力，以及基础状态转换单测，为 Phase 1b 的 `CollectCompactablePages` / `CollectUnmappedRanges` 做准备；该改动不改变现有 allocator 行为。
- 2026-05-21: 扩展 `CollectMappedRanges` / `CollectUnmappedRanges` 支持多个 VA range 输入，并跨相邻输入 range 合并连续输出区间；该接口形态与未来 AllocationView 的 `GetFreeRanges()` 输出对齐，仍不改变现有 allocator 行为。

### Phase 1b: Compact 改用 BackingMap 驱动

**目标**: 建立 BackingMap 驱动的 compact source/target 候选收集语义，先以旁路诊断验证，不改变现有 remap 主路径

| 改动 | 文件 |
|------|------|
| source 页快照: `MappedPage {va, handle, epoch}` | `vmm_backing_map.h`, `vmm_backing_map.cc` |
| target 页快照: `UnmappedPage {va, epoch}` | `vmm_backing_map.h`, `vmm_backing_map.cc` |
| bounded source 收集: `CollectMappedPages(..., target_bytes)` | `vmm_backing_map.h`, `vmm_backing_map.cc` |
| 非对齐 FREE range source 收集: `CollectMappedPagesFullyCoveredBy` | `vmm_backing_map.h`, `vmm_backing_map.cc` |
| 非对齐 GAP/target range 收集: `CollectUnmappedPagesFullyCoveredBy` | `vmm_backing_map.h`, `vmm_backing_map.cc` |
| 快照校验: `ValidateMappedPages` / `ValidateUnmappedPages` | `vmm_backing_map.h`, `vmm_backing_map.cc` |
| compact 语义封装: `CollectCompactCandidates(source_ranges, target_ranges, target_bytes)` | `vmm_backing_map.h`, `vmm_backing_map.cc` |
| compactor `VLOG(4)` pre-scan 旁路统计 source/target candidates | `free_block_remap_compactor.cc` |

**验证**: 回归 8/8，compact OOMs 与旧逻辑一致

**进展**:

- 2026-05-21: 补充 `CollectMappedPages`，从多个 FREE VA range 中返回 mapped page 的 `{va, handle, epoch}` 只读快照；这是后续 `CollectCompactCandidates` / `MovePage` 事务冲突检测的输入基础，不改变现有 allocator 行为。
- 2026-05-21: 为 `CollectMappedPages` 增加 `target_bytes` bounded 收集重载，按 page 粒度向上取整并在达到目标后停止；这与当前 bounded compact 的按需整理策略对齐，仍未接入现有 compact 主路径。
- 2026-05-21: `b839ba14fa` 完成 stable2 全量回归，`20260521_154412` 8/8 passed；OOM 数保持 `50 / 61 / 38 / 4483 / 4359`，`post_cleanup=0`，默认日志无 `force-release` / BackingMap error。说明 bounded page collection 作为只读能力未改变现有 allocator 行为。
- 2026-05-21: `a1b1050cba` 补充 `ValidateMappedPages`，用于校验 `CollectMappedPages` 返回的 `{va, handle, epoch}` 快照仍然有效；后续 `MovePage` / `RemapTransaction` 可在提交前用它检测 page 是否被并发 unmap/remap/release。该接口仍为只读验证能力，未接入现有 compact 主路径。
- 2026-05-21: `a1b1050cba` 完成 stable2 全量回归，`20260521_162724` 8/8 passed；OOM 数保持 `50 / 61 / 38 / 4483 / 4359`，`post_cleanup=0`，默认日志无 `force-release` / BackingMap error。说明 snapshot validation 作为只读校验能力未改变现有 allocator 行为。
- 2026-05-21: 新增 `CollectMappedPagesFullyCoveredBy`，支持非 page-aligned FREE ranges，只返回完全落在 FREE range 内的 mapped pages；compactor Phase 1 前在 `VLOG(4)` 下执行 BackingMap pre-scan 并校验快照。默认 `GLOG_v=0` 不执行该扫描，仍不改变现有 compact 主路径。
- 2026-05-21: `e946af32c0` 完成 stable2 全量回归，`20260521_171756` 8/8 passed；OOM 数保持 `50 / 61 / 38 / 4483 / 4359`，`post_cleanup=0`，默认日志无 `force-release` / BackingMap error。说明 compact pre-scan 旁路在默认配置下未改变 allocator 行为。
- 2026-05-21: 补充 target-side `UnmappedPage {va, epoch}` 快照、`CollectUnmappedPagesFullyCoveredBy` 和 `ValidateUnmappedPages`；compactor `VLOG(4)` pre-scan 同时统计 source mapped pages 与 target unmapped pages。该能力为后续 `MovePage` 事务同时校验 source/target 做准备，默认配置不改变 compact 主路径。
- 2026-05-21: `d43de6a32b` 完成 stable2 全量回归，`20260521_191532` 8/8 passed；OOM 数保持 `50 / 61 / 38 / 4483 / 4359`，`post_cleanup=0`，默认日志无 `force-release` / BackingMap error。说明 target-side unmapped page collection / validation 未改变 allocator 行为。
- 2026-05-22: 补充 `CollectCompactCandidates` 作为 Phase 1b 的统一语义接口，一次返回 source mapped pages 和 target unmapped pages；compactor `VLOG(4)` pre-scan 改为依赖该接口。旧 parts 驱动 remap 主路径暂不切换。
- 2026-05-22: `601df8b0be` 完成 stable2 全量回归，`20260522_132942` 8/8 passed；OOM 数保持 `50 / 61 / 38 / 4483 / 4359`，`post_cleanup=0`，默认日志无 `force-release` / BackingMap error。说明 Phase 1b 的统一 candidate collection 在默认配置下未改变 allocator 行为，Phase 1b 只读候选收集阶段可以视为完成。

### Phase 1c: 移除 parts_ 和 remapped 机制

**目标**: `AllocationBlock` 替代 `BlockV2`

| 改动 | 文件 |
|------|------|
| `BlockV2` 移除 `parts_` 字段 | `vmm_allocator_v2_types.h` |
| 移除 `VmmHandleMeta::remapped` | `vmm_allocator_v2_types.h` |
| 移除 `CreateSyntheticAllocation` | `cuda_virtual_mem_allocator_v2.cc` |
| 移除 `underlying_allocations_` (BackingMap 接管) | `vmm_auto_growth_best_fit_allocator_v2.cc` |
| 移除 `RegisterHandleLayout` / `UnregisterHandleLayout` | `cuda_virtual_mem_allocator_v2.cc` |
| `FreeIdleChunks` 改用 `BackingMap::ReleaseBacking` | `vmm_auto_growth_best_fit_allocator_v2.cc` |

**验证**: 回归 8/8

### Phase 2a: 引入 RemapTransaction

**目标**: 事务化 remap 替代手动 rollback

| 改动 | 文件 |
|------|------|
| 新增 `RemapTransaction` 类 | `remap_transaction.h`, `remap_transaction.cc` |
| `Compact` 使用 `RemapTransaction` | `free_block_remap_compactor.cc` |

**验证**: 回归 8/8

**进展**:

- 2026-05-22: 引入 `RemapTransaction` 骨架，先承接 compact 里的事务性状态，而不改变 remap 主算法。当前 `RemapTransaction` 负责三件事：保存 `CollectCompactCandidates` 返回的 source/target snapshot、提供 `ValidateSourcePages` / `ValidateTargetPages` 只读校验接口、统一记录并回滚 `pending mapped ranges`。
- 2026-05-22: `free_block_remap_compactor.cc` 已接入 `RemapTransaction`，替代原本散落在 compactor 内部的 `PendingMappedRange` 状态收集与 `UnmapPendingMappedRanges` 回滚逻辑；`RollbackToOriginalVA`、parts 驱动 source 扫描、synthetic allocation 语义暂时保留，因此这一步仍属于 skeleton 接线，不是主路径切换。
- 2026-05-22: 这一阶段的完成标准不是“删掉旧 rollback”，而是把事务上下文先收口到一个独立对象中，为下一步 `MovePage` / `Commit` / `Rollback` 和 compact 主路径切换提供稳定落点。当前状态可视为 `Phase 2a` 已起步，待全量回归确认。
- 2026-05-22: `6fda10be2b` 引入事务骨架，`4853f28957` 将 candidate 准备与 source/target snapshot 校验收口到事务入口，`963ff5c94f` 增加显式 `Commit()` / `Rollback()` 并让 compactor 的成功/失败路径统一经由事务对象处理 pending destination mappings。
- 2026-05-24: `963ff5c94f` 完成 stable2 全量回归，`20260524_230331` 8/8 passed；OOM 数保持 `50 / 61 / 38 / 4483 / 4359`，`post_cleanup=0`，默认日志无 `force-release` / BackingMap error。`dsv3_45g_bounded` / `compact_all` elapsed 为 `659.15s / 644.75s`，说明此前两轮的时间上浮属于运行波动，当前事务接口收口未引入性能或行为回退。
- 2026-05-24: 将旧 `RollbackToOriginalVA` 作为 source restore action 注册到 `RemapTransaction`。所有 Phase 2 / gap-scatter / catch 失败分支现在只调用 `transaction.Rollback()`；destination range 在 `MapHandlesToVA` 前登记，事务可覆盖 map 成功后 bookkeeping 抛异常的清理空窗。事务先解除 pending destination mappings，再调用 source restore action。该适配保留旧 block/parts 恢复实现，避免 `RemapTransaction` 直接依赖 `BlockV2`。
- 2026-05-25: `62837cd47c` 完成 stable2 全量回归，`20260525_002339` 8/8 passed；OOM 数保持 `50 / 61 / 38 / 4483 / 4359`，`post_cleanup=0`，默认日志无 `force-release` / BackingMap error，`compact_no_grow` 为 SUCCESS。说明由事务统一调度 source restore 不改变现有 compact 行为，Phase 2a 的 rollback 控制流收口已验证通过。
- 2026-05-25: 新增 `RemapTransaction::MapHandlesToDestination(...)`，将 destination intent 登记和底层 `MapHandlesToVA` 调用收口为一个事务原语；tail、single-gap 与 gap-scatter 三条 placement 路径不再直接调用 allocator map API。该步骤不改变 source 选取或映射策略，作为后续 `MovePage` 接口的过渡实现，待全量回归验证。
- 2026-05-25: 新增 `RemapTransaction::MapHandleRangeToDestination(...)`，让 gap-scatter 直接传入全局 `remapped_handles/remapped_metas` 加 `[start, count)` 子区间；compactor 不再手工组装 `chunk/chunk_metas` 临时切片。该步骤继续收缩 compactor 的事务编排职责，待全量回归验证。
- 2026-05-26: 新增 `VmmHandleMeta::IsRemapped()/MarkRemapped()/ClearRemapped()`，并把 transaction、free path、compaction pre-check、BackingMap layout validate 的 `remapped` 直接访问改为走成员接口；同时给 `BlockV2` 增加 `IsActive()/IsFree()/IsGap()`，先替换 remap/gap 主路径上的显式 `BlockType` 判定。该步骤继续收口旧表示访问面，为后续真正删除 `remapped` / `kGap` 做准备。
- 2026-05-25: `8892814d52` 完成 stable2 全量回归，`20260525_111316` 8/8 passed；随后 `b767c46ee9` 完成 stable2 全量回归，`20260525_134224` 8/8 passed。两版 OOM 数均保持 `50 / 61 / 38 / 4483 / 4359`，`post_cleanup=0`，默认日志无 `force-release` / BackingMap error，说明 destination map 事务原语与区间子视图接口都未引入行为回退。
- 2026-05-25: 新增 `RemapTransaction::BuildDestinationLayout(...)`，将 tail、single-gap、gap-scatter 成功路径中重复的目标 `HandleLayout` 构造收口到事务对象；compactor 保留 synthetic allocation 注册与 block splice，但不再逐项构造 `VmmHandleMeta{dst + i * handle_size, ...}`。该步骤继续压缩 compactor 的目标布局细节，待全量回归验证。
- 2026-05-25: 新增 `RemapTransaction::MaterializeMappedRange(...)`，将目标 `HandleLayout`、synthetic allocation 和 `BlockV2(parts_)` 物化进一步统一收口到事务对象；compactor 的 tail、single-gap、gap-scatter 三条 success path 都改为直接消费事务返回的 `MaterializedRange`。`153eb50843` 完成 stable2 全量回归，`20260525_154510` 8/8 passed；OOM 数保持 `50 / 61 / 38 / 4483 / 4359`，`post_cleanup=0`，默认日志无 `force-release` / BackingMap error，说明 Phase 2a 的 destination success path 物化收口已验证通过。到此为止，Phase 2a 可以视为完成。

### Phase 2b: 删除旧 rollback 机制

**目标**: 清理所有补丁式 rollback 代码

| 删除 | 文件 |
|------|------|
| `RollbackToOriginalVA` | `free_block_remap_compactor.cc` |
| `RestoreGapToFree` | `free_block_remap_compactor.cc` |
| `UnmapPartialDestination` | `free_block_remap_compactor.cc` |
| `PendingMappedRange` / `UnmapPendingMappedRanges` | `free_block_remap_compactor.cc` |
| `MergeAdjacentGaps` (GAP 不再存在) | `free_block_remap_compactor.cc` |

**验证**: 回归 8/8

**进展**:

- 2026-05-25: `ce31b89025` 将 rollback action 从单一 source restore 回调扩展为通用 hook 栈，并把 source restore / gap restore / force-release 迁入 `remap_transaction.cc`；`db133af12a` 修复 `MaterializeMappedRange(...)` 的 `const` 接口一致性。两者在 `20260525_182802` stable2 全量回归中联合验证通过，8/8 passed；OOM 数保持 `50 / 61 / 38 / 4483 / 4359`，`post_cleanup=0`，默认日志无 `force-release` / BackingMap error。
- 2026-05-25: `23e8dccf36` 将 tail / single-gap / gap-scatter 的 block install 与 free/gap normalize 收口到事务对象。`20260525_190859` stable2 全量回归 8/8 passed；说明 success path 的 block 落盘从 compactor 挪到事务后未引入行为回退。
- 2026-05-25: `bab68ef167` 继续将 tail probe、single-gap 选择、gap-scatter placement 规划与执行收口到事务对象。`20260525_193644` stable2 全量回归 8/8 passed；OOM 数保持 `50 / 61 / 38 / 4483 / 4359`，`post_cleanup=0`，默认日志无 `force-release` / BackingMap error，说明 destination 侧 placement 规划事务化未引入行为回退。
- 2026-05-25: `cee4b699bb` 将 tail / single-gap / gap-scatter 三条 destination 路径的高层策略入口继续收口成 `ExecutePlacementStrategy(...)`。`20260525_200743` stable2 全量回归 8/8 passed；OOM 数保持 `50 / 61 / 38 / 4483 / 4359`，`post_cleanup=0`，默认日志无 `force-release` / BackingMap error，说明 destination 侧高层 placement 策略统一入口未引入行为回退。
- 2026-05-26: `c6fbcad8b3` 将 destination rollback 的记录从“计划 map 的区间”收紧为“实际 map 成功的区间”，并让 `MapHandlesToVA(...)` 在部分 `cuMemMap` / `cuMemSetAccess` 失败时自行回滚已落地的前缀映射。配套全量回归与单 case 复验表明 OOM、cleanup 与高压 replay elapsed 保持在既有波动区间内，说明 destination 侧失败清理边界收口未引入行为回退。
- 2026-05-26: 实际验证后，直接把 synthetic allocation 的 layout 注册推迟到 `Commit()` 会改变现有兼容层的异常落点，在 `dsv3_45g_compact_all` 中触发 source/destination rollback 缺口。当前采取的过渡方案更保守：保持 `CreateSyntheticAllocation(...)` 的 eager register 行为不变，但 `Rollback()` 会在丢弃 staged synthetic allocations 前显式 `UnregisterHandleLayout(...)`。这样可以清理未提交对象留下的脏 `allocation_layout_map_` 状态，同时不改变成功路径的 ownership / deleter 时序。该方案不是最终架构，而是 synthetic / GAP / parts_ 仍在场时的阶段性收口。
- 2026-05-26: 在上述过渡方案上继续收紧 synthetic 生命周期边界：`RemapTransaction` 不再暂存 `DecoratedAllocationPtr`，而是只暂存裸 `Allocation*` staged 对象；`Commit()` 时再显式包装成带 `FreeImpl()` deleter 的 `DecoratedAllocationPtr` 交给 `underlying_allocations_`，`Rollback()` 则调用底层 allocator 的 `DestroySyntheticAllocation(...)` 做 `UnregisterHandleLayout + delete`。这样 staged synthetic 与 committed synthetic 的销毁路径彻底分离，事务层不再手工 `release()+delete`。本地验证：`20260526_171103` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`；`20260526_172245` `probe_standard` PASS，`success=1`。
- 2026-05-26: 继续把 `underlying_allocations_` 的容器细节从 compactor/transaction 边界移除。`RemapTransaction` 不再持有 `std::list<DecoratedAllocationPtr>*`，改为接收显式的 synthetic commit hook；`FreeBlockRemapCompactor` 只负责把该 hook 透传给事务，真正的 `underlying_allocations_.emplace_back(...)` 收回到 `VMMAutoGrowthBestFitAllocatorV2::CompactImpl()`。这样 transaction/compactor 的职责收敛到“何时提交 staged synthetic”，而不是“提交到哪个容器里”。本地验证：`20260526_184235` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`；`20260526_185345` `probe_standard` PASS，`success=1`。
- 2026-05-26: 将 source rollback 状态继续收口成事务内部 callback。`RemapTransaction` 不再保留专门的 `source_blocks_` 指针成员，而是在 `CompactFreeBlocks(...)` 开始时安装 `rollback_source_mappings_` 闭包，统一由 `Rollback()` 调用。这样 source restore 的“做什么”仍然保留，但事务状态从“持有一组 block 指针字段”进一步收敛为“持有一个 rollback 动作”。本地验证：`20260526_190139` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`。
- 2026-05-26: 把 destination rollback 的 handle-range unmap 细节统一沉到底层 allocator。新增 `CUDAVirtualMemAllocatorV2::RollbackMappedHandleRange(...)`，同时复用到 `MapHandlesToVA(...)` 的 `cuMemMap` / `cuMemSetAccess` 失败前缀回滚和 `RemapTransaction::RollbackMappedDestinations()`。这样 mapped destination 的失败清理路径不再复制三份 loop，transaction 也不再自己按 `handle_size_` 逐段 unmap。合并验证：`20260526_191558` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`；`20260526_192708` `probe_standard` PASS，`success=1`。
- 2026-05-26: 继续把 destination rollback 从“数据记录”收口成“动作记录”。`RemapTransaction` 不再保留 `MappedDestinationRange` 结构和 `mapped_destination_ranges_` 向量，改为在每次 destination map 成功后直接登记 rollback callback，逆序执行时只关心动作顺序，不再关心 transaction 自己持有的 `{dst, handle_count}` 明细。该步骤是表达层面的收口，行为不变。
- 2026-05-26: 进一步清理空事务状态：只有在 `CollectRemapSources(...)` 真正收集到 source handles 后，才安装 `rollback_source_mappings_`。空事务不再携带未使用的 source rollback callback。合并验证：`20260526_193226` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`；`20260526_194335` `probe_standard` PASS，`success=1`。
- 2026-05-26: `underlying_allocations_` 继续收口到 allocator 内部固定入口。`TrackUnderlyingAllocation(...)` / `CommitSyntheticUnderlyingAllocation(...)` 收口普通与 synthetic allocation 的接管，`TryReleaseIdleUnderlyingAllocation(...)` 收口 `FreeIdleChunks()` 的释放循环。这样 `underlying_allocations_` 的直接读写点继续减少，但行为保持不变。合并验证：`20260526_195326` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`；`20260526_200436` `probe_standard` PASS，`success=1`。
- 2026-05-26: 开始把 `kGap` 相关旧 helper 从文件级散落入口收回到 transaction。`RestoreRemappedSourcesToFreeBlocks(...)` 改为经 `RestoreGapRangeToFreeBlock(...)` 成员入口恢复 source gap；随后 source gap restore 逻辑本体和 gap block 构造也迁入 `RemapTransaction`，旧 `RestoreGapToFree(...)` 退出主路径。到这一步，`kGap` 仍然存在，但“如何恢复 source gap”已收敛到 transaction 内部，后续替换 `kGap -> FREE + UNMAPPED` 只需改 transaction/block edit 入口。合并验证：`20260526_224102` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`；`20260526_225213` `probe_standard` PASS，`success=1`。
- 2026-05-26: 继续把 gap normalize 收回到 transaction。`MergeAdjacentGapBlocks(...)` 和 `MergeAdjacentFreeBlocks(...)` 都迁入 `RemapTransaction`，`NormalizeBlocks()` 不再依赖文件级 merge helper；source restore 在恢复 gap 后也通过 transaction 提供的 free-merge callback 收口 block normalize。这样 `kGap` 相关的 “create / restore / normalize” 三类 block 编辑都已经集中到 transaction 内部。合并验证：`20260526_230629` `dsv3_45g_compact_all` PASS，`ooms=4359`，`post_cleanup_reserved_gib=0.0`；`20260526_231741` `probe_standard` PASS，`success=1`。

---

## 十、代码量预估

| 组件 | 新增 | 删除 | 净变化 |
|------|------|------|--------|
| `BackingMap` | ~300 行 | — | +300 |
| `RemapTransaction` | ~120 行 | — | +120 |
| `AllocationView` (从 BlockV2 简化) | ~50 行改动 | ~200 行 parts 逻辑 | -150 |
| `Compactor` 重写 | ~150 行 | ~500 行 (旧逻辑) | -350 |
| Rollback 删除 | — | ~200 行 | -200 |
| Synthetic/HandleLayout 删除 | — | ~150 行 | -150 |
| **合计** | ~620 行 | ~1050 行 | **-430 行** |

净减少 ~430 行，同时消除 6 类已知 bug 面。

---

## 十一、风险与缓解

| 风险 | 缓解 |
|------|------|
| 两层一致性 bug | Phase 1a 双写 + DCHECK 验证；每步回归 8/8 |
| BackingMap 内存开销 | 80G/2MB × 80B ≈ 3.2MB，忽略不计 |
| Event 粒度从 block → page 的行为变化 | page event = 多 stream 独立跟踪；GC 防 event 积累 |
| 迁移期兼容性 | 可配 flag 切换新旧路径；默认旧路径直到全部验证通过 |
| Compact 行为差异 | 新逻辑更严格（§12 安全性证明）；不会错搬，可能暂时少搬 |
| FreeIdleChunks 后 VA 不收缩 | 设计选择: VA reservation 保留 (不占物理资源); 对进程虚拟地址空间足够大的场景无影响 |

---

## 十二、安全性条件证明

> **注**: 本节不是"等价性证明"——新旧逻辑的判定条件并非严格等价，也不存在简单的超集/子集关系。
> 这里证明的是: 新逻辑满足 Compact 的安全性条件（不会搬迁正在使用的 page、不会丢失数据）。

### 12.1 安全性条件定义

Compact 搬迁一个 page 是安全的，当且仅当:
1. **无活跃引用**: 该 page 的 VA 范围内没有任何 ACTIVE allocation（用户未持有指向此区域的指针）
2. **无 GPU 访问**: 没有任何 stream 正在或即将访问该 VA 的数据
3. **无 IPC 共享**: 该 page 的 handle 未被其他进程引用
4. **有效 backing**: 该 page 当前确实 mapped（有物理内存可搬）

### 12.2 新逻辑如何满足安全性条件

```
page.mapped                        → 条件 4 (有效 backing)
!page.ipc_exported                 → 条件 3 (无 IPC 共享)
AllEventsComplete(page.va)         → 条件 2 (无 GPU 访问)
alloc_view.IsRangeFree(page.va, page_size) → 条件 1 (无活跃引用)
```

**逐条论证**:

| 安全条件 | 检查方法 | 正确性 |
|----------|----------|--------|
| 无活跃引用 | `IsRangeFree(va, page_size)` | AllocationView 持有全局锁时查询; FREE 意味着无用户指针指向此区域 |
| 无 GPU 访问 | `AllEventsComplete(va)` | 遍历所有 stream 的 pending events; event 完成 ⊃ GPU 操作完成 |
| 无 IPC 共享 | `!page.ipc_exported` | IPC export 时设置; handle 被其他进程引用时不可 unmap |
| 有效 backing | `page.mapped` | BackingMap 与 driver 状态一致 (由 EnsureBacking/ReleaseBacking 维护) |

### 12.3 与旧逻辑的关系

旧逻辑 (`IsFullyCoveredHandle`):
```
part.handle_rel_off == 0 && part.len == handle_size && !meta->remapped
```

新旧逻辑的关系:
- 旧逻辑通过 `handle_rel_off == 0 && len == handle_size` 证明“历史 handle part 完整属于 FREE block”。
- 新逻辑不再依赖历史 part 身份，而是直接检查 `alloc_view.IsRangeFree(page.va, page_size)`，证明“当前 VA page 全部 FREE”。
- 新逻辑增加 `!ipc_exported` 和 multi-stream `AllEventsComplete`，这些是安全性约束，不是与旧逻辑的集合等价关系。
- 因为新模型摆脱历史 part 绑定，它可能接受旧模型因 partial/stale parts 拒绝的 page；同时它也会因 IPC 或 pending events 拒绝旧模型可能接受的 page。

**结论**: 新逻辑不宣称与旧逻辑等价，也不依赖“所有旧逻辑可搬 page 都可搬”的假设。只要四个安全条件同时成立，搬迁就是安全的；否则拒绝搬迁。∎

### 12.4 不变量

| 不变量 | 维护方 | 验证方法 |
|--------|--------|----------|
| `page.mapped` ↔ driver 中该 VA 确实有映射 | `EnsureBacking` / `ReleaseBacking` / `MovePage` | DCHECK on driver API return |
| `page.handle` 有效 ↔ `page.mapped == true` | `MovePage` (源清零) / `ReleaseBacking` (清零) | Rollback 中 map 失败时 release |
| `IsRangeFree(va, size)` → 无用户指针指向 [va, va+size) | AllocationView (ACTIVE→FREE 在 Free 时原子转换) | 全局 mutex 保证 |
| `AllEventsComplete` → 所有 GPU 操作已完成 | cudaEventQuery 是 driver 保证 | CUDA 语义 |

---

## 十三、总结

| 维度 | 当前 (PR4) | 升级后 |
|------|-----------|--------|
| 核心模型 | Block = VA + 物理信息耦合 | 三态: AllocationState × BackingState |
| Compact 代码量 | ~800 行 | ~300 行 |
| Compact 语义 | 修改 block list + ownership 转移 | **只改 BackingMap，不动 AllocationView** |
| Rollback 复杂度 | 5 个函数, ~200 行 | 1 个 `Rollback()`, ~40 行 |
| Remap 原语 | 分离 Unmap + Map + 中间状态 | `MovePage` 原子操作 |
| 已知 bug 面 | 6 类 | 设计层面消除 |
| Handle ownership | 分散 (original + synthetic + remapped flag) | 统一 (BackingMap) |
| GAP block | 需要维护 + 合并 | 不存在 (FREE + UNMAPPED) |
| IPC 支持 | 需侵入 parts_ | BackingMap 内闭环 |
| Event 语义 | 单 event, block level | **多 stream events, page level** |
| empty_cache 后 VA | 收缩 (不可恢复) | **保留 (可复用)** |
| 迁移风险 | — | 渐进, 每步可回滚 |
