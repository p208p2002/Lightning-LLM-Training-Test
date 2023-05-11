# Lightning LLM Training Test
LLM硬體負載與吞吐量測試。

顯存使用量、記憶體使用量、實際算力(FLOP/s)與每秒鐘Token處理量。

## 計算訓練時間
$$D = Token總數$$

$$N = 模型參數量$$

$$C = 6DN$$

$$T_{days} = C/(FLOPS \times 86400 \times 10^{12})$$

## 最佳化模型參數量與資料集大小
參見: [p208p2002/Compute-Optimal-Model-Estimator](https://github.com/p208p2002/Compute-Optimal-Model-Estimator)
> 若已知實際算力，則`utilization=1`，或使用經驗值0.2~0.3。
