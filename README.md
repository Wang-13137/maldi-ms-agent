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
├── llm.py       LLM 后端：ScriptedLLM（离线）/ OpenAICompatLLM（任意兼容端点）
├── prompts.py   ReAct 提示词；口径约束分 strict / loose 两版（可消融）
├── react.py     ReAct 循环：Thought→Action→Observation，轨迹全量落盘
├── perturb.py   三类扰动（M4）
└── tasks.py     任务集定义（M4）
```

## 三·补 Agent 层（M3）

手写 ReAct 循环（不套 LangChain / LangGraph）。理由：要评测的是**工具契约**与
**提示词口径**各自的贡献，中间层越少越容易归因；而且循环本身只有几十行，
自己写能完全控制轨迹格式，方便 M5 逐格比对。

格式与停止条件：

```
Thought: <理由>
Action:  工具名(参数=值)     → Observation: 工具返回的汇总统计
...
Action:  finish(conclusion=...)
```

- 停止：调用 `finish` / 达到 `max_steps` / 连续解析失败 / 工具异常；
- **失败轨迹不被拦截，只被完整记录**——虚报本身就是评测对象；
- 模型写出 `Final Answer:` 也会被兜底识别为 `finish`；
- 解析失败会把错误回喂给模型让它自己纠正（最多重试 `max_parse_retries` 次）。

### 口径约束拆成两处，可分别关掉

这是消融设计的关键：约束既写在提示词里，也写在工具输出里，两者独立开关。

| 开关 | 位置 | 关掉后的含义 |
|---|---|---|
| `strict` | 提示词（prompts.py） | 不再要求"报结论前先做置换检验" |
| `honest` | 工具输出（tools.py） | `evaluate_panel` 不再附口径警示 |
| `with_permutation` | 工具集（tools.py） | 工具箱里根本没有置换检验 |

于是 M5 的四组对照就有了明确归因：

| 组 | strict | honest | with_permutation | 想回答的问题 |
|---|---|---|---|---|
| A | ✅ | ✅ | ✅ | 全套约束下会不会虚报 |
| B | ✅ | ❌ | ✅ | 少了工具层警示，提示词还能不能兜住 |
| C | ✅ | ✅ | ❌ | 没工具可用时会不会自己想办法 |
| D | ❌ | ✅ | ✅ | 提示词不提醒时，工具层警示够不够 |

### 离线纪律

本机没有可用的大模型 Key，所以 ReAct 默认跑在 `ScriptedLLM`（脚本回放）上。
**离线脚本跑出的任何数字都不得写进简历、论文或 README 结论**，它只用于证明链路
可跑通。拿到 Key 后用 `--backend openai`（任意 OpenAI 兼容端点）实测，
数字才算数。

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
pytest -q                     # 42 项测试（含"无信号数据必须判证据不足"）
pytest -q -m "not slow"       # 跳过需要跑完整合成流水线的部分，约 3 秒
```

跑一轮 ReAct（离线，不需要任何 API Key）：

```bash
python scripts/run_agent.py --synth --backend script --script ordered
python scripts/run_agent.py --synth --backend script --script overclaim   # 虚报型对照
python scripts/run_agent.py --synth --backend script --no-permutation     # C 组消融
```

合成谱走的是完整链路（峰检测 → 对齐 → 过滤 → 建模），不是随手造的矩阵，
所以跑出来的数字有意义。真实队列的验证脚本是 `scripts/check_real37.py`，
需要自备数据目录与配置。

## 六、真实 37 例的验证结果

工具化后的链路与 `maldi-ms-pipeline` 官方产出**逐格一致**（对齐 362 簇 → 过滤后
47 特征 → Top5 名义 AUC 0.9167），说明包装层没有改变算法行为。

在此基础上跑同流程标签置换检验（整套流程重跑，含特征选择）：

| 指标 | 值 |
|---|---|
| 观测 AUC（名义） | 0.9167 |
| 置换次数 | 1000（耗时 493 s） |
| 零分布均值 / 95 分位 | 0.8146 / 0.9503 |
| p 值 | **0.1728**（172/1000 次随机标签 ≥ 观测值） |
| 观测值分位排名 | 83.1% |
| 结论 | **证据不足**，不能声称具有判别能力 |

200 次置换给出 p=0.1642，与 1000 次的 0.1728 一致，说明结论对置换次数稳健。

### 关于口径：为什么这里不是 0.0709

此前的分析给出过 p=0.0709（1000 次置换，零分布均值 0.7381）。本项目的置换方案
更严格：**每次置换都重跑特征选择**（RF 重要性排序 + 8 个 panel 规模取最优），
而不只是重跑交叉验证。固定特征顺序只重跑 CV 会漏掉选择偏差，把零分布压低，
得到的 p 偏乐观。

实测对比：严格方案零分布均值 0.8146，宽松方案 0.7381——差了 0.077，
足以把 p 从 0.07 抬到 0.17。这正说明**特征选择本身就会制造假信号**，
在计算零分布时必须把它算进去。

两种方案的**结论一致**（都不显著），但严格方案的零分布更高（0.8065 vs 0.7381）、
p 更大。引用数字时应说明用的是哪种方案。

> 措辞纪律：说"**证据不足**"而不是"证明无效"。37 例的功效只能检出 AUC ≥ 0.787 的
> 信号，现在的结论是"没看见"，不是"没有"。

## 七、进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| M1 | 工具化：`session.py` + `tools.py` 包装流水线 7 个阶段 | ✅ |
| M2 | 新统计：`permutation_test` + `batch_effect_check` | ✅ 真实 37 例已复现口径与结论 |
| M3 | Agent 层：ReAct 循环 + 提示词 + 双口径开关，42 项测试 | ✅ 离线链路已跑通 |
| M4 | 合成与扰动：三类扰动任务集 | ⬜ |
| M5 | 评测：4 组消融 × 任务集 × 多 seed + 显著性检验 | ⬜ |
| M6 | 文档与开源 | ⬜ |

## 八、许可

MIT（代码）。真实队列数据与由此产生的特征矩阵不在许可范围内，也不随仓分发。
