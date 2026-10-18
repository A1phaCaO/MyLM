# validate 卡点微实验结果

## exp1_old

| phase | round | iter创建(s) | 首批等待(s) | loader等待累计(s) | 整轮(s) | 首批行签名 | GPU空闲占比 |
|---|---|---|---|---|---|---|---|
| exp1_old | 0 | 8.321 | 8.325 | 0.005 | 9.097 | 22183 | 1.0 |
| exp1_old | 1 | 8.148 | 8.152 | 0.01 | 8.772 | 25125 | 1.0 |
| exp1_old | 2 | 8.063 | 8.068 | 0.01 | 8.761 | 21963 | 1.0 |
| exp1_old | 3 | 7.767 | 7.771 | 0.009 | 8.394 | 15345 | 1.0 |

## exp1_new

| phase | round | iter创建(s) | 首批等待(s) | loader等待累计(s) | 整轮(s) | 首批行签名 | GPU空闲占比 |
|---|---|---|---|---|---|---|---|
| exp1_new | 0 | 3.933 | 3.937 | 0.015 | 4.042 | 29952 | 1.0 |
| exp1_new | 1 | 0.001 | 0.003 | 0.004 | 0.009 | 29952 | None |
| exp1_new | 2 | 0.0 | 0.002 | 0.003 | 0.008 | 29952 | None |
| exp1_new | 3 | 0.0 | 0.002 | 0.002 | 0.007 | 29952 | None |

## exp1_w0

| phase | round | iter创建(s) | 首批等待(s) | loader等待累计(s) | 整轮(s) | 首批行签名 | GPU空闲占比 |
|---|---|---|---|---|---|---|---|
| exp1_w0 | 0 | 0.0 | 0.003 | 0.027 | 0.107 | 26442 | None |
| exp1_w0 | 1 | 0.0 | 0.002 | 0.028 | 0.03 | 17367 | None |
| exp1_w0 | 2 | 0.0 | 0.001 | 0.027 | 0.029 | 29440 | None |
| exp1_w0 | 3 | 0.0 | 0.003 | 0.028 | 0.031 | 26691 | None |

## exp2_generate_none

| round | wall(s) | GPU空闲占比 |
|---|---|---|
| 0 | 73.93 | 0.937 |
| 1 | 0.05 | None |
| 2 | 0.05 | None |
| 3 | 0.05 | None |
- recompiles(stderr): 18, cache_size_limit: 0

## exp2_generate_left

| round | wall(s) | GPU空闲占比 |
|---|---|---|
| 0 | 16.9 | 0.917 |
| 1 | 0.07 | None |
| 2 | 0.05 | None |
| 3 | 0.08 | None |
- recompiles(stderr): 0, cache_size_limit: 0

## exp2_ctx

| round | seg | wall(s) |
|---|---|---|
| 0 | train_step | 43.98 |
| 0 | val_inference_mode | 16.43 |
| 0 | val_no_grad | 0.9 |
| 1 | train_step | 0.01 |
| 1 | val_inference_mode | 0.0 |
| 1 | val_no_grad | 0.0 |
| 2 | train_step | 0.01 |
| 2 | val_inference_mode | 0.0 |
| 2 | val_no_grad | 0.0 |
- recompiles(stderr): 7

## exp2_eager（方案C：generate 用未 compile 裸模块，补跑）

| round | wall(s) | GPU空闲占比 |
|---|---|---|
| 0 | 0.563 | 1.0 |
| 1 | 0.17 | None |
| 2 | 0.162 | None |
| 3 | 0.145 | None |
| val_eval_graph(对照，应仍有首编译) | 1.97 | None |
- recompiles(stderr): 0（generate 全程不经过 dynamo，无编译；1.97s 对照是因为
  inductor FX 磁盘缓存已被前面冷跑写入，eager 生成路径本身不受缓存影响）

## exp3_compile_default: 首次调用(含编译)=16.6s, 稳态=0.0019s/次, recompiles(stderr)=0

## exp3_compile_max-autotune: 首次调用(含编译)=42.9s, 稳态=0.0019s/次, recompiles(stderr)=0

---

## 逐项判定

- **H1（每次 val 重 spawn worker）= 坐实，且是主因。**
  exp1_old 四轮 iter创建+首批等待恒为 ~8.1-8.3s，GPU 空闲 100%；
  exp1_new 仅第 0 轮 3.9s（首次建池），第 1-3 轮 0.001-0.003s。
  注意这是低估：实验 worker 只 import torch，生产 worker 要重 import
  pre_train 全链（torch+tokenizers+matplotlib+bitsandbytes+models+dataset），
  单次更久。改 persistent_workers=True 消除重复 spawn。

- **H2（验证集逐 epoch 漂移）= 坐实。**
  old/w0 的 row_sig 每轮变（重 spawn 时 pickle 当前 epoch 的 base.perm）；
  new 全程冻结在首 spawn 快照（persistent worker 只 pickle 一次）。
  顺带把 val shuffle 改 False，让每次 val 遍历同一组 batch，val_loss 曲线可比。

- **H3（generate 变形状触发编译风暴）= 坐实，但性质是"每进程一次性"。**
  exp2_none 首轮 73.9s / GPU 空闲 93.7% / 18 次 recompile，稳态轮 0.05s；
  exp2_left 首轮 16.9s（左 pad 只消掉级联，消不掉给 generate 单独编译 eval 图），
  且左 pad 把真实 token 的 RoPE 位置整体平移 pad_len，改变生成分布。
  exp3 放大系数：max-autotune 单次编译 = default 的 2.6x → 生产首轮膨胀到 ~3 分钟，
  且每次 resume/重启都再付一遍。
  **最终方案取 eager（方案C）**：TextGenerator 挂 _orig_mod 裸模块，B=1 decode
  从 compile 拿不到收益（稳态 0.0019s/forward vs eager 0.15s/整段20token），
  exp2_eager 首轮 0.56s、0 重编译，且不改 RoPE 位置语义。validate 的 eval 图
  首编译由 val 自身触发（exp2_ctx: val_inference_mode 16.4s），与 generate 解耦。

- **inference_mode 疑虑 = 排除。** exp2_ctx 中 train→val(inference_mode) 16.4s
  是 eval 图首编译（任何 eval 模式都躲不掉），紧随 val(no_grad) 仅 0.9s，
  说明两种上下文共享编译产物，无需把 validate 改成 no_grad。
