# maldi-ms-agent

让智能体自主执行 MALDI-TOF 质谱判别分析，并用**扰动评测**验证它不会虚报结论。

> **面试版一句话**：我让 Agent 跑我的质谱流水线，然后故意给它标签打乱的数据——
> 它必须自己调用置换检验说出"证据不足"，直接报 AUC 0.92 就算失败。

---

## 一、为什么做这个

37 例尿液样本的 pilot 曾给出名义交叉验证 AUC 0.9167，看起来很漂亮。但在**同流程
标签置换检验**下它无法与随机标签区分（1000 次置换 p=0.0709，零分布均值 0.7381）——
即现有样本量下**未证实**存在可区分感染与健康的代谢信号。

问题不在于当时不够小心，而在于**流程里没有强制这道检验**。所以本项目做的事是：

1. 把流水线各阶段封装成可调用工具；
2. 让"报告 AUC 前必须做置换检验"变成**工具契约**，而不是口头纪律；
3. 用三类故意做坏的数据去测它，看它到底拦不拦得住虚报。

这样就把一个负结果，转成可展示的方法学能力。

## 二、与 maldi-ms-pipeline 的关系

本仓库**不复制**流水线的代码，而是依赖它：

```bash
pip install -e ../maldi-ms-pipeline
# 或者
export MALDI_PIPELINE_SRC=/path/to/maldi-ms-pipeline/src
```

`maldi-ms-pipeline` 负责"怎么算峰、怎么对齐"，本项目负责"让 Agent 会用这些能力，
并且不许它乱下结论"。

## 三、架构

```
src/maldi_ms_agent/
├── session.py   跨工具共享的分析状态（惰性计算 + 改参数自动失效重算）
├── tools.py     10 个工具；evaluate_panel 返回时附口径警示（可消融）
├── stats.py     permutation_test / batch_effect_check（pipeline 里原本没有）
├── synth.py     合成谱生成器，可控效应量与批次混淆（供公开复现）
├── react.py     ReAct 骨架（M3）
├── perturb.py   三类扰动（M4）
└── tasks.py     任务集定义（M4）
```

## 四、工具集

| # | 工具 | 返回给 Agent 的内容 |
|---|---|---|
| 1 | `load_dataset` | 样本数、两组例数、批次分布、数据量 |
| 2 | `preprocess` | 归一化设置 + 样本间总强度倍差（暴露未归一化的坑） |
| 3 | `detect_peaks` | 每样本峰数、阴性对照峰数 |
| 4 | `align_peaks` | 特征簇数、单例簇占比 |
| 5 | `build_features` | 特征维度、稀疏度、检出率 |
| 6 | `differential_test` | 显著特征数、top10 的 m/z / p / FDR |
| 7 | `evaluate_panel` | AUC 曲线 + **口径警示** |
| 8 | **`permutation_test`** | 观测 AUC、p 值、零分布均值与分位、结论判定 |
| 9 | **`batch_effect_check`** | 判别批次 AUC vs 判别标签 AUC + Fisher 混杂检验 |
| 10 | `finish` | 结束任务 |

第 8、9 个是本项目新增的：`maldi-ms-pipeline` 里没有置换检验
（`grep -n "permut\|shuffle\|null"` 在它的 `src/` 下只命中一处 `StratifiedKFold(shuffle=True)`）。
第 9 个把"250–499.7 Da 波段判别批次 AUC=0.938、判别标签仅 0.697"这个发现固化成了工具。

### 核心设计：口径警示下沉到工具层

`evaluate_panel` 返回数字时强制附一句：

> ⚠️ 以上为**名义 AUC**，未经置换校正。样本量 N（感染 K）时，随机标签也能达到相近水平。
> 报告判别能力前请先调用 `permutation_test`。

比在 system prompt 里说教更可靠，而且**可消融**：
`build_tools(honest=False)` 去掉警示，用于对照"工具契约 vs 沉默工具"差多少。

### 合规硬约束

- 工具**只返回汇总统计**，原始谱与特征矩阵绝不进入 LLM 请求体；
- 会话对象持有原始数组，但只有 `summarize*` 系列方法的输出允许对外；
- 37 例真实谱与其特征矩阵属未发表数据，**不随仓分发**，真实结果只写本地 `reports/`。

## 五、复现须知

语料（真实谱）不随仓库分发，但**合成数据可以复现全部评测链路**：

```bash
pip install -e .
pip install -e ../maldi-ms-pipeline
pytest -q                     # 23 项测试，含"无信号数据必须判证据不足"
```

合成谱走的是完整链路（峰检测 → 对齐 → 过滤 → 建模），不是随手造的矩阵，
所以跑出来的数字有意义。真实队列的验证脚本是 `scripts/check_real37.py`，
需要自备数据目录与配置。

## 六、进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| M1 | 工具化：`session.py` + `tools.py` 包装流水线 7 个阶段 | ✅ |
| M2 | 新统计：`permutation_test` + `batch_effect_check` | ✅（合成）／真实 37 例验证中 |
| M3 | Agent 层：ReAct 骨架 + 提示词 + 会话接入 | ⬜ |
| M4 | 合成与扰动：三类扰动任务集 | ⬜ |
| M5 | 评测：4 组消融 × 任务集 × 多 seed + 显著性检验 | ⬜ |
| M6 | 文档与开源 | ⬜ |

## 七、许可

MIT（代码）。真实队列数据与由此产生的特征矩阵不在许可范围内，也不随仓分发。
