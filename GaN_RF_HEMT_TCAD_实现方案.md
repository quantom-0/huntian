# GaN 射频 HEMT 轻量化 TCAD 软件 —— 完整实现方案

> 版本：v1.0
> 作者：器件建模工程师
> 定位：专用、轻量、可插拔、AI 增强的 GaN RF HEMT 器件仿真与协同优化平台

---

## 目录

1. [项目概述](#1-项目概述)
2. [设计目标与原则](#2-设计目标与原则)
3. [总体架构](#3-总体架构)
4. [模块化与即插即用设计](#4-模块化与即插即用设计)
5. [物理模型内核](#5-物理模型内核)
6. [数值求解与高性能计算](#6-数值求解与高性能计算)
7. [HFSS / EM 联合仿真接口](#7-hfss--em-联合仿真接口)
8. [LLM 智能层](#8-llm-智能层)
9. [LLM 参数自动拟合流程](#9-llm-参数自动拟合流程)
10. [数据 / 模型 / 材料库](#10-数据--模型--材料库)
11. [精度保障体系](#11-精度保障体系)
12. [技术栈选型](#12-技术栈选型)
13. [代码仓库目录结构](#13-代码仓库目录结构)
14. [分阶段实施路线图](#14-分阶段实施路线图)
15. [风险与对策](#15-风险与对策)
16. [验收标准](#16-验收标准)

---

## 1. 项目概述

本项目构建一套**专用于 GaN 射频 HEMT** 的轻量化 TCAD（Technology Computer-Aided Design）软件，具备以下核心能力：

- **器件级仿真**：极化 2DEG、量子限制、陷阱 / 电流坍塌、自热、直流 / 小信号 / 大信号射频特性提取。
- **HFSS 联合仿真**：与 Ansys HFSS（或其他 EM 工具）通过标准化接口协同，实现**版图（无源网络）与有源区（HEMT）联合仿真**。
- **模块化即插即用**：物理模型、求解器、计算后端、EM 适配器、智能模块均以插件形式组织，可独立更新、运行时热插拔。
- **LLM 智能增强**：仿真代码生成、**基于实验数据的参数自动拟合**、一键调参、结果诊断。
- **资源最大化**：充分调度多核 CPU 与 GPU，批量偏置 / 参数扫描与优化种群并行。

### 1.1 术语约定

| 术语 | 含义 |
|---|---|
| **有源区 / 本征器件** | 器件本征部分，用 TCAD / 紧凑模型描述 |
| **无源网络 / 版图** | 传输线、焊盘、匹配网络、栅/漏极集流条等，用 EM 工具描述 |
| **协同仿真（Co-Simulation）** | 无源网络 S 参数 + 有源器件模型在同一电路环境中联合求解 |
| **插件 / 模块（Module）** | 实现统一接口、可独立替换的软件单元 |
| **Deck** | 描述一次仿真设置的输入文件（结构化 DSL） |

---

## 2. 设计目标与原则

### 2.1 量化目标

| 指标 | 目标值 |
|---|---|
| 2DEG 面密度 n_s 精度 | ±5%（vs 霍尔测量） |
| 阈值电压 Vth | ±0.15 V |
| 饱和电流 Idmax / 跨导 gm | ±10% |
| fT / fmax | ±10~15% |
| 电流坍塌幅度 | ±20% |
| Pout / PAE（大信号） | ±10~15% |
| 单偏置点求解时间（2D, ~2 万节点） | < 1 s（CPU 单核），批量模式 GPU 加速 10×+ |
| 模块热插拔时间 | < 1 s，无需重启进程 |

### 2.2 设计原则

1. **专用优于通用**：只做 GaN RF HEMT，只做单极 n 型电子输运，2D 截面为主。
2. **模型即插件**：一切物理模型 / 求解器 / 后端 / 适配器都走统一接口，可插拔、可版本化、可独立发布。
3. **数字只能来自求解器**：LLM 仅负责翻译、规划、诊断、解释，绝不直接输出物理数值。
4. **误差可追溯**：每个结果携带参数快照、代码版本、网格、求解器设置，完全可复现。
5. **渐进增强**：先 Python 原型验证物理，再 C++/Kokkos 重写热路径，性能与正确性分离推进。

---

## 3. 总体架构

采用**分层 + 插件总线**架构：上层编排（Python）通过统一的**模块接口**驱动下层高性能内核（C++），EM 与 LLM 均作为一等公民插件接入。

```
┌──────────────────────────────────────────────────────────────────────┐
│                        接入层 (Access Layer)                         │
│    Web GUI (React/Vue) · Python API (FastAPI) · CLI                  │
│    Deck 编辑 · 参数面板 · 结果可视化 (Plotly/pyvista)                 │
├──────────────────────────────────────────────────────────────────────┤
│                    编排 / 智能层 (Orchestration, Python)             │
│   ┌────────────┐  ┌─────────────────┐  ┌────────────────────────┐   │
│   │ Agent 循环  │  │ 优化器 Optuna   │  │ RAG 知识库 (向量检索)   │   │
│   │ 任务规划    │  │ DE / 贝叶斯      │  │ 材料/模板/错误码/文献   │   │
│   └─────┬──────┘  └────────┬────────┘  └────────────┬───────────┘   │
│         └──────────────────┼─────────────────────────┘               │
│                    ┌───────▼────────┐                                │
│                    │ LLM Provider 层 │ 本地 vLLM/Ollama + 外部 API    │
│                    └────────────────┘  (OpenAI 兼容统一抽象)          │
├──────────────────────────────────────────────────────────────────────┤
│               模块总线 / 注册中心 (Module Bus & Registry)             │
│  统一接口 · 依赖解析 · 版本校验 · 生命周期管理 · 热插拔 · 事件总线    │
├───────────────┬──────────────────────────┬───────────────────────────┤
│ 仿真核心层     │  EM 协同层                │  数据 / 模型层            │
│ (C++20 内核)  │  (EM Co-Sim Layer)       │  (Data & Model Layer)     │
│ 网格 · 离散   │  HFSS 适配器 (pyAEDT)     │  材料参数库               │
│ Newton 求解  │  Touchstone .sNp I/O     │  器件模板库               │
│ 物理模型      │  版图 GDSII I/O          │  紧凑模型库 (ASM-HEMT 等) │
│ 后处理提取    │  轻量 2.5D/准静态回退     │  Benchmark / Golden 库    │
│              │  网表生成 / 去嵌          │  结果库 (SQLite/Parquet)  │
├───────────────┴──────────────────────────┴───────────────────────────┤
│                   计算后端层 (Compute Backend, 可插拔)               │
│   CPU: OpenMP/TBB + oneMKL/Eigen                                     │
│   GPU: CUDA + cuSPARSE/cuSOLVERSp/AmgX · Kokkos 性能可移植            │
│   批量: 偏置扫描 · 参数扫描 · 优化种群 数据并行                        │
└──────────────────────────────────────────────────────────────────────┘
```

数据流核心是**统一的仿真上下文（SimulationContext）**：一次协同仿真的所有输入（结构、物理参数、版图、EM 结果、实验数据、优化目标）都挂在这个上下文对象上，各模块按需读写，模块之间不直接耦合。

---

## 4. 模块化与即插即用设计

这是本方案的**地基**。目标是：新增一个迁移率模型、替换一个线性求解器、接入一个新的 EM 工具、升级一个 LLM 模型，都不需要改动其他任何模块。

### 4.1 统一模块接口

所有模块（物理、求解、后端、EM、智能）实现同一套契约：

```python
class IModule(Protocol):
    # ---- 元数据 ----
    module_id: str                    # 唯一标识，如 "mobility.farahmand"
    api_version: str                  # 接口契约版本，如 "1.0"
    module_version: str               # 语义化版本，如 "2.3.1"
    category: str                     # physics | solver | backend | em | ai | io
    capabilities: list[str]           # 声明的能力标签
    dependencies: list[str]           # 依赖的其他模块 id + 版本范围

    # ---- 生命周期 ----
    def initialize(self, config: dict) -> None: ...   # 读取配置、加载资源
    def validate(self, ctx: SimContext) -> list[str]: ...  # 自检，返回问题清单
    def run(self, ctx: SimContext) -> SimContext: ...      # 执行，读写上下文
    def terminate(self) -> None: ...                   # 释放资源

    # ---- 元数据自述 ----
    def describe(self) -> ModuleManifest: ...          # 参数 schema、默认值、单位、出处
```

要点：

- **上下文即接口**：模块之间通过 `SimContext` 交换数据，不直接 import 彼此，天然解耦。
- **参数 schema 自述**：每个模块暴露 Pydantic JSON Schema，说明可调参数、类型、单位、合法范围、文献出处。这同时是 **LLM 生成 Deck 和调参的可靠依据**（见 §9）。
- **能力声明**：`capabilities` 让注册中心能做"我需要一个 `transport.dd` + `quantum.sp` + `trap.surface`"的自动装配。

### 4.2 模块注册中心（Module Registry）

- 启动时扫描 `plugins/` 目录，发现并加载模块清单。
- **依赖解析**：拓扑排序 + 版本范围校验（如 `mobility` 依赖 `material.gan >= 3.0`）。
- **版本兼容**：`api_version` 不匹配时拒绝加载并给出明确错误，避免静默破坏。
- **热插拔**：模块以文件/包形式放在指定目录，替换后注册中心重扫描，正在运行的任务使用已加载实例，新任务使用新版本 —— 无需重启进程。
- **多版本共存**：同 `module_id` 可保留多个 `module_version`，Deck 中显式指定用哪个版本，保证旧结果可复现。

### 4.3 模块分类与清单（v1 范围）

| 类别 | 模块 | 说明 |
|---|---|---|
| **结构/网格** | `mesh.structured2d` | 层状规则/准结构化 2D 网格 |
| **物理** | `polarization.ambacher` | 自发+压电极化与屏蔽因子 |
| | `quantum.sp1d` | 1D Schrödinger–Poisson 量子修正 |
| | `mobility.farahmand` | GaN 场依赖迁移率 |
| | `mobility.caughty_thomas` | 通用迁移率（备选） |
| | `trap.surface` / `trap.buffer` | 表面态 / buffer 深能级陷阱 |
| | `thermal.electrothermal` | 自热电热耦合 |
| **输运** | `transport.dd` | 漂移扩散（主） |
| | `transport.hydro` | 能量平衡（可选，后期） |
| **求解** | `nonlinear.newton` | Newton–Raphson + 阻尼/伪瞬态延拓 |
| | `linear.gmres` / `linear.direct` | Krylov 迭代 / 直接 LU |
| **后端** | `backend.cpu_openmp` / `backend.gpu_cuda` | 计算后端 |
| **后处理** | `extract.iv` / `extract.sparam` / `extract.ft_fmax` / `extract.largesignal` | 特性提取 |
| **紧凑模型** | `compact.asm_hemt` / `compact.mvsg` / `compact.angelov` | 电路级模型（导出用） |
| **EM** | `em.hfss_adapter` / `em.touchstone_io` / `em.gdsii_io` / `em.lightweight_2p5d` | EM 协同（见 §7） |
| **智能** | `ai.deck_gen` / `ai.fit` / `ai.tune` / `ai.diagnose` | LLM 能力（见 §8/§9） |
| **优化** | `optimize.optuna` / `optimize.de` | 优化器后端 |

### 4.4 顶层应用也可插拔

"顶层"（应用外壳、工作流编排）同样模块化：

- 工作流定义为一个**有向无环图（DAG）**，节点是模块实例，边是上下文数据依赖。
- 顶层只负责"装配 DAG + 调度 + 事件分发"，不含业务逻辑。
- 用户/开发者可自定义 Pipeline 配置文件，注入自定义模块，扩展工作流而不改核心代码。

---

## 5. 物理模型内核

### 5.1 必选物理（少了就不叫 GaN RF HEMT）

| 物理 | 模型要点 |
|---|---|
| **极化与 2DEG** | AlGaN/GaN 界面的自发极化 P_sp + 压电极化 P_pz；2DEG 由极化束缚电荷诱导（非掺杂）；引入表面态屏蔽因子（有效极化 30%~70%） |
| **量子限制** | 1D Schrödinger–Poisson 自洽求三角阱子带；有效质量近似；Fermi–Dirac 统计（2DEG 简并） |
| **陷阱 / 电流坍塌** | 表面 donor 态（虚栅 → gate-lag）+ buffer 深能级（drain-lag / knee walkout）；非准静态占据率方程 + 瞬态 |
| **自热** | 晶格温度方程；GaN 热导率随温度下降；考虑 GaN/SiC 界面热阻 TBR |
| **输运** | 漂移扩散（DD）+ 场依赖迁移率（峰值/饱和速度）；短栅可选能量平衡（Hydrodynamic）插件 |

### 5.2 控制方程（DD 主路径）

```
∇·(ε∇ψ) = −q(p − n + N_D⁺ − N_A⁻ + ρ_pol + ρ_trap)     # 泊松（含极化/陷阱电荷）
∂n/∂t = (1/q)∇·J_n − R_n                                # 电子连续性
J_n = q n μ_n(E,T) E + q D_n ∇n                          # 漂移扩散通量
Schrödinger–Poisson:  −ℏ²/2m* · d²ψ_k/dz² + V(z)ψ_k = E_k ψ_k   # 量子修正
陷阱: ∂f_t/∂t = c_n n (1−f_t) − e_n f_t                  # 占据率动力学
热: C ∂T/∂t = ∇·(κ(T)∇T) + H_J                          # 晶格温度
```

- 空间离散：**有限体积 + Schafetter–Gummel 通量**（保证稳定性与局部守恒）。
- 非线性：Newton–Raphson + 阻尼 + 伪瞬态延拓（PTC）。
- 线性：GMRES/BiCGSTAB + 预条件（或小规模直接 LU）。

### 5.3 输出提取

- **直流**：Id–Vd、Id–Vg、gm、Vth、C–V。
- **小信号**：AC 分析 → Y 参数 → S 参数 → fT / fmax / MSG/MAG。
- **大信号**：load-pull → Pout / PAE / gain / OIP3。
- **瞬态**：pulsed IV、gate-lag、drain-lag。

---

## 6. 数值求解与高性能计算

### 6.1 CPU 侧

- 单次求解内部：OpenMP/TBB 并行装配矩阵 + oneMKL/Eigen 稀疏运算。
- **偏置点级并行**：IV/S 参数/频率扫描天然 embarrassingly parallel，多核各跑一点。

### 6.2 GPU 侧（按性价比排序）

| 策略 | 落地难度 | 收益 |
|---|---|---|
| 批量偏置/参数扫描（每实例一 CUDA block） | ⭐ 低 | 🔥 最高 |
| 优化/拟合种群并行 | ⭐⭐ 中 | 高 |
| Schrödinger–Poisson 多 slice 并行 | ⭐⭐ 中 | 中 |
| 稀疏线性求解器 GPU 化（AmgX/cuDSS/cuSOLVERSp） | ⭐⭐⭐ 高 | 单点加速 |

### 6.3 性能可移植

内核用 **Kokkos**（或 CUDA + OpenMP 双后端）编写，一份代码 CPU/GPU 都跑。`backend.*` 模块切换后端，业务代码不变。

### 6.4 求解精度控制（避免数值污染物理）

- 有限体积通量形式保证局部电荷守恒。
- **电流连续性门禁**：各端电流之和 ≈ 0（相对误差 < 1e-6），作为最灵敏的求解器健康检查。
- Inexact Newton：线性容差与 Newton 残差联动，避免过度迭代。
- 求解容差压到误差预算的 1/10 以下。

---

## 7. HFSS / EM 联合仿真接口

**目标**：让"版图（无源网络）+ 有源区（HEMT）"能在同一仿真环境中协同，实现器件-电路-版图一体化。

### 7.1 协同仿真架构

```
┌─────────────────────┐         ┌─────────────────────────────┐
│  版图 / 无源网络      │         │  有源区 / 本征 HEMT          │
│  GDSII + 层叠结构     │         │  TCAD → 紧凑模型 / S参数     │
└──────────┬──────────┘         └──────────────┬──────────────┘
           │   EM 求解 (HFSS)                   │  TCAD 求解
           ▼                                    ▼
   ┌─────────────────────────────────────────────────────┐
   │           EM 协同层 (Co-Sim Layer)                   │
   │  · 去嵌 (de-embedding)                              │
   │  · 网表生成：无源 .sNp + 有源子电路 合并              │
   │  · 电路联合求解 (Nexxim / 内置谐波平衡)              │
   └─────────────────────────────────────────────────────┘
           │
           ▼
   整器件响应（S 参数 / 大信号 / 稳定性 / 热+EM 联合）
```

### 7.2 标准数据契约（接口的核心）

所有 EM 工具通过**统一数据契约**对接，而非私有格式：

| 对象 | 契约格式 | 说明 |
|---|---|---|
| 版图 | **GDSII / OASIS** | 器件版图（gdstk / klayout 读写） |
| 层叠结构 | **JSON Schema** | 金属层、介质、厚度、电导率 |
| EM 结果（无源网络） | **Touchstone .sNp** | S 参数标准格式（scikit-rf 读写） |
| 有源器件模型 | **SPICE 网表 + 参数文件** 或 **.s2p** | 紧凑模型（ASM-HEMT/MVSG/Angelov）或小信号 S 参数 |
| 大信号行为 | **X-参数 / DynaFET 查找表** | 非线性大信号模型（可选） |
| 联合网表 | **SPICE 网表** | 无源 + 有源合并后的顶层网表 |

### 7.3 HFSS 适配器（Ansys 官方 Python API）

通过 **pyAEDT** 驱动 HFSS，实现自动化闭环：

```
pyAEDT 控制流程：
1. 导入 GDSII + 层叠结构 → 建 HFSS 工程
2. 设置激励端口 / 辐射边界 → 提交 EM 求解
3. 导出无源网络 S 参数 (.sNp)
4. 与 TCAD 导出的有源模型合并 → 生成联合网表
5. 在 Circuit (Nexxim) 中协同求解
6. 回读整器件 S 参数 / 大信号结果 → 写入 SimContext
```

适配器提供两种模式：

- **在线模式**：直接驱动本地 HFSS（需要 Ansys License）。
- **离线模式**：只消费已有 `.sNp` / 网表文件，无 License 也能跑（便于 CI 与协作）。

### 7.4 轻量化回退（无 HFSS 时）

为保持"轻量"，提供内置**准静态 / 2.5D 快速提取**作为回退：

- 从 GDSII + 层叠结构做准静态 **RLGC 提取 / PEEC**，得到互连寄生。
- 输出粗糙但快速的 S 参数，用于快速迭代；精度不足时再上 HFSS。
- 该模块同样实现 `em.*` 接口，与 HFSS 适配器**即插即用互换**。

### 7.5 去嵌（De-embedding）

联合仿真的精度高度依赖本征器件与寄生网络的正确分离：

- 支持标准 **open / short / through** 去嵌校准结构。
- 从整器件测量/仿真 S 参数中剥离焊盘、集流条寄生，得到本征器件 —— 这是**参数拟合（§9）和模型对标（§11）的前提**。

### 7.6 有源模型导出（TCAD → 电路）

TCAD 结果导出为电路可用的有源模型，三种保真度：

| 层级 | 模型 | 用途 |
|---|---|---|
| 高保真 | 紧凑模型（ASM-HEMT/MVSG）拟合到 TCAD | 电路级协同仿真 |
| 中保真 | 小信号 .s2p（各偏置点） | 线性 / 稳定性分析 |
| 大信号 | X-参数 / DynaFET 查找表 | 非线性 load-pull |

---

## 8. LLM 智能层

### 8.1 统一 Provider 抽象

```python
class ILLMProvider(Protocol):
    def chat(self, messages, tools=None) -> str: ...     # 通用对话
    def embed(self, texts) -> list[float]: ...           # 向量化（RAG 用）
```

- **本地**：vLLM / Ollama / llama.cpp，跑 Qwen2.5-Coder、DeepSeek-Coder 等量化模型（吃 GPU）。
- **外部**：OpenAI 兼容 API，与本地实现同一接口，热切换。
- **RAG**：向量库（Chroma/FAISS）存材料参数、Deck 模板、器件结构、错误码手册、文献要点，检索增强生成。

### 8.2 四大能力

| 能力 | 模块 | 职责 |
|---|---|---|
| 仿真代码生成 | `ai.deck_gen` | 自然语言 → 结构化 Deck → schema 校验 → 试运行闭环 |
| 参数自动拟合 | `ai.fit` | 实验数据 → 目标/参数空间 → 优化 → 诊断（见 §9） |
| 一键调参 | `ai.tune` | 读结果+日志 → 诊断 → 提出参数修改 → 验证 |
| 结果诊断 | `ai.diagnose` | 收敛失败/物理异常原因分析 + 建议 |

### 8.3 Deck DSL（LLM 与求解器之间的桥梁）

LLM 不直接写代码，而是**填充结构化 DSL**（YAML/JSON），由 parser + Pydantic 严格校验：

```yaml
device:
  structure:
    barrier: {material: AlGaN, x_al: 0.25, thickness_nm: 20}
    channel: {material: GaN, thickness_nm: 100}
    buffer:  {material: GaN_C_doped, thickness_nm: 1500}
    gate:    {length_um: 0.15, type: schottky}
  physics:
    polarization: {model: ambacher, screening: 0.5}
    quantum:      {model: sp1d}
    traps:        {surface: {Nt: 1e13, E: 1.0, sigma: 1e-15}}
    thermal:      {model: electrothermal, tbr: 10e-9}
simulation:
  sweep: {type: iv, vgs: [-6, 2], vds: [0, 20]}
  output: [id_vd, id_vg, sparam, ft_fmax]
```

每个字段的合法范围、单位、默认值都来自模块的 `describe()` manifest，LLM 据此生成，parser 据此校验，**越界即拒绝并回退重试**。

---

## 9. LLM 参数自动拟合流程

**目标**：给定实验数据（IV / C-V / S 参数 / pulsed IV / load-pull / Hall），自动标定器件模型参数，输出带不确定度的拟合结果。

### 9.1 完整闭环流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 数据接入      CSV / Touchstone / 实验数据库                │
│      │ 校验：物理合理性（单调性、量级、噪声）、去嵌           │
│      ▼                                                      │
│ 2. 目标提取      计算 FoM：Vth、gm、Idmax、n_s、fT、fmax、PAE │
│      ▼                                                      │
│ 3. LLM 目标翻译  自然语言 → 加权多目标损失 + 约束            │
│      │   "DC 与 S 参数在 5% 内，优先 fT"                     │
│      ▼                                                      │
│ 4. LLM 参数空间  从模块 manifest + 材料库先验，提出可调参数  │
│      │   白名单 + 物理边界（如 0 < x_al < 1）                │
│      ▼                                                      │
│ 5. 优化求解      Optuna (TPE/贝叶斯) / 差分进化              │
│      │   GPU 批量并行多个候选参数 → 每个跑一次 TCAD          │
│      ▼                                                      │
│ 6. 残差分析      LLM 读残差曲线 → 诊断 → 提出下一步           │
│      │   "低 Vg 区偏高 → 查表面陷阱密度"                     │
│      ▼                                                      │
│ 7. 验证          留出集盲测 + 交叉验证 + 过拟合守卫          │
│      ▼                                                      │
│ 8. 持久化        参数 + 出处 + 不确定度 写入模型库            │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 LLM 的角色边界（红线）

| 动作 | 谁来做 |
|---|---|
| 计算任何物理量 / 拟合值 | **求解器 / 优化器**（LLM 绝不碰） |
| 翻译目标、提出假设、诊断残差、解释结果 | LLM |
| 判定"拟合是否达标" | **独立验证集指标**（LLM 不裁决） |
| 参数最终验收 | 人在回路审批关键参数 |

### 9.3 防过拟合机制

- **参数白名单**：只允许拟合 manifest 中标记为 `calibratable` 的参数。
- **参数数 << 数据点数**：用 AIC/BIC 或交叉验证约束模型复杂度。
- **分层标定**（固定顺序，避免耦合漂移）：

```
结构层（Al 组分/厚度/极化） → 输运层（低场→峰值→饱和迁移率）
      → 陷阱层（表面态/buffer） → 热层（热导率/TBR）
```

- **独立留出集**：标定数据与报精度数据严格分离（留 20~30% 盲测）。
- **UQ 输出**：Sobol 敏感性 + 蒙特卡洛/PCE，输出"Vth = 0.42 ± 0.08 V"而非单点。

---

## 10. 数据 / 模型 / 材料库

| 库 | 内容 | 存储 |
|---|---|---|
| 材料参数库 | GaN/AlGaN 极化、能带、迁移率、热、陷阱参数（带出处+不确定度） | 版本化 JSON/YAML |
| 器件模板库 | 标准 HEMT 结构模板（栅长、势垒组分、场板配置） | YAML |
| 紧凑模型库 | ASM-HEMT / MVSG / Angelov 实现与参数集 | Python/C++ |
| Benchmark 库 | L0~L7 基准 deck + Golden 参考结果 + 容差阈值 | 目录 + Parquet |
| 实验数据库 | 原始测量数据（IV/C-V/S/pulsed/load-pull/Hall） | HDF5/Parquet |
| 结果库 | 每次仿真结果 + 完整元数据（参数快照/代码版本/网格/设置） | SQLite/Parquet |
| RAG 语料库 | 材料说明、模板、错误码、文献要点 | 向量库 (Chroma/FAISS) |

---

## 11. 精度保障体系

沿用 §2 的量化目标，落地为四层 V&V + 工程化闭环。

### 11.1 四层验证

| 层 | 内容 | 通过判据 |
|---|---|---|
| **Code Verification** | MMS 制造解验证收敛阶；平衡态费米平坦；解析极化式 | 达到格式理论阶；< 1e-6 |
| **Solution Verification** | 网格收敛 GCI；时间步/容差扫描 | 离散误差 < 预算 |
| **Validation** | L0~L7 基准阶梯（解析→量子→交叉TCAD→直流→小信号→陷阱→大信号） | 见 §2 精度表 |
| **UQ** | Sobol 敏感性 + 蒙特卡洛/PCE | 输出置信区间 |

### 11.2 基准阶梯

| 级别 | 基准 | 判据 |
|---|---|---|
| L0 | 平衡态、1D 泊松、FD 积分、极化解析式 | < 1e-6 |
| L1 | 1D Schrödinger–Poisson 子带 | 子带能量差 < 5 meV |
| L2 | 标准异质结 n_s、C-V | n_s 差 < 5% |
| L3 | 与 Sentaurus/Silvaco 交叉对比 IV/C-V | < 5% |
| L4 | 直流输出/转移特性 | Idmax/gm < 10%，Vth < 0.15 V |
| L5 | S 参数 → fT/fmax | < 10~15% |
| L6 | pulsed IV、gate/drain-lag | 坍塌幅度 < 20% |
| L7 | load-pull | Pout/PAE < 10~15% |

### 11.3 工程化闭环

- **Golden Data + CI**：每个基准存参考结果与容差，内核改动触发 CI 全跑，越界阻断合并。
- **验证状态标签**：每模块标 `verified / partially-verified / unverified`，报告透出。
- **可复现性**：结果携带完整元数据，可一键重跑。

---

## 12. 技术栈选型

| 层 | 选型 | 说明 |
|---|---|---|
| 内核 | C++20 + Eigen + Kokkos | 性能可移植；稀疏可接 oneMKL/cuSOLVERSp/AmgX |
| Python 绑定 | pybind11 / nanobind | nanobind 更轻 |
| 编排 | Python 3.11+ + FastAPI + Pydantic | 服务、schema 校验 |
| 原型验证 | DevSim / FiPy + scipy | 先验证物理，再重写热路径 |
| GPU 原型 | JAX / CuPy | 验证批量并行思路 |
| LLM | vLLM / Ollama（本地）+ OpenAI 兼容 API | 双通道 |
| RAG | Chroma / FAISS | 向量检索 |
| 优化 | Optuna / scipy.optimize / DEAP | 贝叶斯 + 差分进化 |
| EM 接口 | pyAEDT（在线）+ scikit-rf（Touchstone）+ gdstk/klayout（GDSII） | 协同仿真 |
| 可视化 | Plotly / matplotlib / pyvista | 结果展示 |
| 数据 | SQLite/Parquet + HDF5 | 结果与实验数据 |

---

## 13. 代码仓库目录结构

```
Huntian/
├── core/                       # C++ 内核（Kokkos）
│   ├── mesh/                   # 结构化网格
│   ├── discretize/             # FV + Schafetter-Gummel
│   ├── physics/                # 极化/量子/陷阱/热/迁移率
│   ├── solve/                  # Newton + 线性求解器
│   └── bind/                   # pybind11 绑定
├── python/                     # Python 编排层
│   ├── huntian/
│   │   ├── registry/           # 模块注册中心 + 接口定义
│   │   ├── context/            # SimContext
│   │   ├── deck/               # DSL parser + schema 校验
│   │   ├── pipeline/           # 工作流 DAG 装配
│   │   ├── backend/            # CPU/GPU 后端适配
│   │   ├── em/                 # EM 协同层（HFSS/Touchstone/GDSII）
│   │   ├── ai/                 # LLM Provider + RAG + 智能模块
│   │   ├── fit/                # 自动拟合 + UQ
│   │   ├── extract/            # 特性提取
│   │   └── db/                 # 材料/模型/结果库
├── plugins/                    # 可插拔模块（用户可扩展）
│   ├── physics/  ├── solver/  ├── em/  ├── ai/  └── ...
├── benchmarks/                 # L0~L7 基准 + golden/
│   ├── L0_analytic/  ├── L1_quantum/  ...  └── golden/
├── examples/                   # 示例 deck 与协同仿真案例
├── data/                       # 实验数据 / 材料库 / 模板
├── tests/                      # 单元 + 集成 + MMS 测试
├── docs/                       # 本文档与设计文档
└── pyproject.toml / CMakeLists.txt
```

---

## 14. 分阶段实施路线图

| 阶段 | 内容 | 交付物 | 周期 |
|---|---|---|---|
| **P0 骨架** | 模块注册中心、SimContext、Deck schema、事件总线、CI | 可插拔框架 + 空跑 Demo | 1~2 周 |
| **P1 物理 MVP** | Python 原型：极化 + SP + DD，2D 规则网格 | 复现标准 HEMT IV/C-V；通过 L0~L2 | 3~4 周 |
| **P2 内核提速** | C++20 + Kokkos 重写热路径；pybind11；CPU/GPU 后端 | 单点 < 1s，批量 GPU 加速 | 3~4 周 |
| **P3 GaN 特性** | 陷阱/瞬态、自热、S 参数/fT/fmax、load-pull | 通过 L4~L7 | 3~6 周 |
| **P4 EM 协同** | Touchstone/GDSII I/O、HFSS 适配器、轻量回退、去嵌、有源模型导出 | 版图+有源区联合仿真 Demo | 3~4 周 |
| **P5 LLM 层** | Deck 生成、自动拟合闭环、一键调参、RAG | AI 三能力可用 | 3~6 周 |
| **P6 工程化** | UQ、Golden CI、GUI、文档、打包发布 | 可交付 v1.0 | 持续 |

---

## 15. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| Newton 收敛困难（强极化+量子+陷阱） | 高 | 参考电位归一化、伪瞬态延拓、阻尼；作为核心护城河重点投入 |
| DD 在短栅大信号下系统性偏差 | 中 | 饱和速度模型 + Hydrodynamic 插件 + 经验修正因子，报告明确标注 |
| 极化/陷阱参数不确定度大 | 高 | 作为带不确定度参数；分层标定 + UQ；精度目标按"同数量级"设定 |
| GPU 单点加速收益低 | 中 | 工程重点放在批量扫描/拟合种群，而非单点线性求解 |
| LLM 输出不可信 / 过拟合式调参 | 高 | 红线：数字只来自求解器；参数白名单；独立验证集；人在回路 |
| HFSS License 依赖 | 中 | 离线模式 + 轻量 2.5D 回退，CI 无 License 可跑 |
| 模块接口演进破坏兼容 | 中 | api_version 契约 + 多版本共存 + 语义化版本 |

---

## 16. 验收标准

**功能验收：**

- [ ] 模块热插拔 Demo：运行中替换一个迁移率模块，新任务生效、旧任务不受影响，无需重启。
- [ ] HFSS 协同 Demo：GDSII 版图 + TCAD 有源模型 → 联合网表 → 整器件 S 参数回读，结果可复现。
- [ ] 无 License 环境下，轻量回退模块能产出粗糙 S 参数。
- [ ] LLM 生成 Deck 通过 schema 校验并可成功运行。
- [ ] 给定一组实验 IV/S 数据，自动拟合闭环能输出带不确定度的参数集，且在留出集上达标。

**精度验收（vs §2 目标）：**

- [ ] 通过 L0~L2 全部基准（数值正确性钉死）。
- [ ] L4 直流：Idmax/gm < 10%，Vth < 0.15 V。
- [ ] L5 小信号：fT/fmax < 10~15%。
- [ ] L6 陷阱：坍塌幅度 < 20%，时间常数同数量级。
- [ ] L7 大信号：Pout/PAE < 10~15%。

**性能验收：**

- [ ] 2D ~2 万节点单偏置点 < 1 s（CPU 单核）。
- [ ] 批量偏置扫描 GPU 加速 ≥ 10×。
- [ ] 模块加载 / 热插拔 < 1 s。

---

*本方案 v1.0 完整覆盖：物理内核、模块化即插即用、HFSS/EM 协同、LLM 自动拟合、精度保障与分阶段落地，可作为项目启动的蓝图。*
