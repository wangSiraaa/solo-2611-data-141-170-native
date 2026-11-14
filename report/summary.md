# MCMC 分片样本诊断报告

## 链恢复记录

| 链 | 状态 | 片段数 | 读入行 | 去重 | 去warmup | 保留 | 共同长度L | 丢弃尾部 | 问题 |
|---|---|---|---|---|---|---|---|---|---|
| c1 | valid | 3 | 720 | 0 | 120 | 600 | 540 | 60 | - |
| c2 | valid | 3 | 720 | 0 | 140 | 580 | 540 | 40 | - |
| c3 | valid | 3 | 720 | 0 | 160 | 560 | 540 | 20 | - |
| c4 | valid | 3 | 720 | 0 | 180 | 540 | 540 | 0 | - |

- 有效链: ['c1', 'c2', 'c3', 'c4']；共同保留长度 L = 540
- 共同保留长度 L=540；各链丢弃尾部样本数: {'c1': 60, 'c2': 40, 'c3': 20, 'c4': 0}。

## 参数级结论

| 参数 | 状态 | 未通过项 | mean | q05 | q50 | q95 | MCSE(mean) | R-hat(rank) | R-hat(folded) | ESS bulk | ESS tail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| location | OK_NO_ISSUE_DETECTED | - | -0.03423 | -1.557 | -0.05929 | 1.484 | 0.03721 | 1.001 | 1.001 | 636 | 1161 |
| sticky | CHAINS_DISAGREE | rhat,bulk_ess,tail_ess | -0.33 | -1.757 | -0.3383 | 1.108 | 0.1861 | 1.193 | 1.067 | 22.2 | 83.4 |
| separated_modes | CHAINS_DISAGREE | rhat,bulk_ess,tail_ess | 0.005901 | -3.251 | 0.03247 | 3.258 | 2.122 | 1.733 | 1.009 | 2.9 | 53.9 |
| heavy_tail | OK_NO_ISSUE_DETECTED | - | -0.07084 | -2.411 | -0.01193 | 2.236 | 0.03433 | 0.999 | 1.001 | 2260 | 2180 |
| constant | CONSTANT_ALL_CHAINS | - | 5 | 5 | 5 | 5 | 0 | NaN | NaN | NaN | NaN |

### 逐项说明

- **location** [OK_NO_ISSUE_DETECTED / 未发现问题] R-hat 与 bulk/tail ESS 均通过复核阈值：这些有限诊断未发现问题（不证明收敛或后验模型正确）。
- **sticky** [CHAINS_DISAGREE / 链间不一致] 链间不一致：rhat_max 超过阈值，各链可能卡在不同模式或尚未平稳，链间合并估计尚不可信。同时 bulk/tail ESS 低于阈值：有效信息量不足；强自相关也会抬高 R-hat，需延长抽样后复查才能区分“混合慢”与“多模式”。
- **separated_modes** [CHAINS_DISAGREE / 链间不一致] 链间不一致：rhat_max 超过阈值，各链可能卡在不同模式或尚未平稳，链间合并估计尚不可信。rank R-hat 明显高于 folded R-hat，呈链间位置/模式漂移特征（各链各自平稳却卡在不同位置，而非尺度差异）。同时 bulk/tail ESS 低于阈值：有效信息量不足；强自相关也会抬高 R-hat，需延长抽样后复查才能区分“混合慢”与“多模式”。
- **heavy_tail** [OK_NO_ISSUE_DETECTED / 未发现问题] R-hat 与 bulk/tail ESS 均通过复核阈值：这些有限诊断未发现问题（不证明收敛或后验模型正确）。
- **constant** [CONSTANT_ALL_CHAINS / 全链同一常量] 参数在全部链中为同一常量：R-hat/ESS 无定义（非“通过”），均值即常量、MCSE 为 0。

## 口径与免责声明

- 秩变换：平均秩 + Blom 变换 Φ⁻¹((r−3/8)/(S+1/4))；split 奇数长度丢弃末样本。
- ESS 截断：Geyer 初始正序列（成对和首个负值处截断）+ 单调化；ESS 上限 m·n·log10(m·n)。
- tail ESS：5%/95% 经验分位指示序列 ESS 的较小者；MCSE(mean)=sd/√ESS(原始值)。
- 阈值来自 manifest decision_limits，仅为复核辅助；全部通过只说明这些有限诊断未发现问题，不能证明收敛或后验模型正确。
- 图文件：traces_<参数>.png（原链轨迹，灰底为已去除 warmup，虚线为被丢弃的尾部样本），overview.png。
