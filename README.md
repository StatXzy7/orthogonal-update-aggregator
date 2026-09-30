# orthogonal-update-aggregator

Linux CPU 上运行的低秩客户端增量合并器。所有客户端在同一轮共享行正交输入基 `B`（`rank × inputDim`），只提交输出侧因子 `U`（`outputDim × rank`）和一个正样本权重 `w`。

服务器不会分别平均输入、输出因子。它先构造真实完整增量：

```text
Δ = Σ_i (w_i / Σ_j w_j) U_i B
```

成功验证整批提交后，对 `Δ` 做 SVD 重参数化，生成下一轮公共行正交基 `B'` 与输出因子 `C'`，使：

```text
C' B' = Δ
```

`targetRank=0` 表示保持当前秩。降低秩时只保留顶部右奇异向量，`MergeResult.Truncated=true`，并通过 `TruncationError` / `RelativeTruncationError` 报告被丢弃部分的 Frobenius 误差；此时 `Equivalent=false`。

## 使用

```go
agg, err := aggregator.New(outputDim, inputDim, rank, initialBasis)

err = agg.Submit(aggregator.Submission{
    Round:        agg.Round(),
    BasisVersion: agg.BasisVersion(),
    ClientID:     "client-a",
    OutputFactor: clientAOutputFactor,
    Weight:       positiveWeight,
})

result, err := agg.CommitRound(0) // 0 保持当前秩；低秩可传更小的正整数
```

查询和持久化：

- `MergedIncrement()`：上一轮权重归一化后的完整增量。
- `CurrentIncrement()`：当前输出因子与当前输入基的乘积。
- `InputBasis()` / `OutputFactor()`：下一轮公共基和输出因子的防御性副本。
- `Save(path)` / `Load(path)`：原子 JSON 保存恢复；待处理提交也会保留，恢复后可继续当前轮。
- `Round()` / `BasisVersion()` / `PendingClients()`：轮次、基版本和提交状态。

## 规则

- 输出和输入维度均不超过 12。
- 权重必须是有限正数；总权重按提交权重归一化。
- 同一客户端在同一轮不能重复计权。
- 轮次或基版本不匹配会被拒绝。
- 无客户端、维度冲突、非有限数、超范围目标秩都会返回明确错误。
- 只有整批提交验证和数值分解都成功才推进轮次与基版本。
- 合并失败会保留原基、原因子、待处理提交和上一轮查询状态。
- 等价性通过 `C'B'` 与 `Δ` 的乘积残差以及 `B'B'^T=I` 检查，不依赖奇异向量符号。
- 零增量会用正交补全构造有限的新基；重复奇异值不会触发符号比较。

唯一线性代数依赖是本地 Go 依赖 [Gonum](https://gonum.org/)，不下载模型，也不调用网络服务。

## 测试

```sh
go test ./...
```
