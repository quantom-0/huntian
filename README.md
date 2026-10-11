# Huntian

> 面向 GaN 射频 HEMT 的**轻量化、模块化 TCAD** 内核。
> 当前状态：**P0 骨架**（注册中心 + SimContext + Deck schema）+ **P1 全部完成**（自洽 Schrödinger–Poisson 求解器 + Newton 耦合求解器 + 严格精确 Jacobian）+ **材料验证第一阶段**（`materials.dft` · `table` 后端：文献 DFT 查找表 + 覆盖审计）+ **漂移扩散输运**（`transport.dd`）。
> 待实现的设计稿：`materials.dft` 的 `qe_*` 后端、`transport.dd` 的高偏压收敛 —— 见 §11 设计文档索引；进度与欠账见 `工作日志.md`。

---

## 1. 这是什么

一套**核心零第三方依赖**（纯标准库）的 Python 平台，用来证明并落地整个软件架构的地基假设：

| 地基 | 实现 | 验证方式 |
|---|---|---|
| **模块即插件** | `ModuleRegistry` + `IModule`/`ModuleManifest` 契约 | 9 个真实插件从磁盘发现、注册、解析、执行；**解析量子模型被自洽 SP 求解器整体替换、材料库被 DFT 表后端整体替换、输运由 `transport.dd` 整体提供，下游均一行未改** |
| **服务即插件** | `ServiceRegistry` + Deck 的 `nonlinear`/`linear`/`backend` 槽 | 求解器由 Deck 配置、被物理模块按需解析；**换迭代策略、换线性代数同样只改一行绑定** |
| **数据靠上下文流动** | `SimContext` 命名 artifact + 溯源 + 指纹 | 模块之间零 import，全靠 artifact 键交换数据 |
| **文档可机械校验** | `Schema` + `Deck` + JSON Schema 输出 | 结构校验（抛异常）+ 上下文校验（返回 issue 列表，供 LLM 修复循环消费） |
| **物理可真验证** | 闭式解基准 + 收敛阶断言 | Poisson 到机器精度、Schrödinger 二阶、**耦合系统二阶（p ≈ 2.08）**、**耦合 Newton 局部阶 1.99（Jacobian 严格精确）**、输运 T0 闭式解 < 1e-6 且 T2 MMS 二阶 |

**核心不依赖 numpy；只有 `physics.quantum.sp1d`、`nonlinear.newton`、`linear.blocktridiag`、`transport.dd` 这四个插件需要它**（`materials.dft` 的 `table` 后端连 numpy 都不要）。这正是插件架构该有的样子：核心保持极简，插件自带依赖。

---

## 2. 快速开始

不需要安装任何东西，用系统 Python 3.10+ 即可。但**全量测试套件需要 numpy**：`physics.quantum.sp1d`、`nonlinear.newton`、`linear.blocktridiag`、`transport.dd` 四个插件在导入期就依赖它，缺了会产生 error 而非 skip（实测只剩 152 个用例被发现）。

```powershell
# 跑测试（426 个用例，纯 unittest）
python run_tests.py

# 跑端到端演示（含解析 vs 自洽对比、热插拔、可复现性）
python demo.py

# 命令行
$env:PYTHONPATH = "python"
python -m huntian.cli list-plugins --plugin-dir plugins
python -m huntian.cli validate decks/demo_hemt.json --plugin-dir plugins
python -m huntian.cli run decks/demo_hemt_sp1d.json --plugin-dir plugins --show ns
python -m huntian.cli schema --what deck      # 给 LLM 用的 JSON Schema
```

可选：`pip install -e ".[sp1d,yaml,dev]"`。
JSON Deck 在**任何环境**都能跑，YAML 只是便利。

---

## 3. 已实现的能力

### 3.1 模块注册中心（`huntian/registry.py`）

- **发现**：扫描 `plugins/` 目录（支持**多文件插件包**，`__init__.py` 可用相对导入）或按点分导入路径加载；单个插件损坏**不会**中断整轮发现。
- **契约校验**：插件的 `api_version` 主版本不匹配时拒绝加载。
- **多版本共存**：同一 `module_id` 可同时注册多个版本，旧结果永远可复现。
- **依赖解析**：`>=1.0,<2.0`、`^1.2.3`、`~1.2`、`a|b`、逗号 AND；拓扑排序 + 环检测 + 版本冲突检测。
- **参数校验**：未知参数、越界、类型错误一律拒绝 —— **LLM 生成配置的安全闸门**。
- **热插拔**：运行中注册新版本；**已创建的实例继续使用旧版本**。

### 3.2 SimContext（`huntian/context.py`）

模块之间**从不互相 import**，只通过命名 artifact 交换数据：

```python
with ctx.stage("quantum", module_id, "1.0.0",
               inputs=("polarization", "materials"),
               outputs=("quantum", "ns")) as scope:
    ctx.put("quantum", result, producer=module_id)
    scope.declare_output("quantum")
```

同时提供**溯源**（模块/版本/状态/耗时/输入输出）和**可复现指纹**（参数 + 模块版本 + 材料库的稳定哈希；**故意不序列化 artifact 数值**）。

### 3.3 Deck schema（`huntian/deck.py`）

两级校验，专为 LLM 闭环设计：

1. **结构校验**（构造时抛 `SchemaError`）：类型、范围、必填、跨字段规则。未知字段一律拒绝并给拼写提示。
2. **上下文校验**（返回 `(errors, warnings)`，不抛异常）：插件是否存在、版本是否可满足、参数是否为该插件已知、材料名是否可解析、**artifact 是否可达**。

第 2 步返回 issue 列表而不是抛异常 —— 这正是"**LLM 生成 → 校验 → 带着错误信息重试**"循环需要的接口。

### 3.4 Pipeline 与执行器（`huntian/pipeline.py`）

**执行顺序由 manifest 推导**，而不是硬编码：若模块 A `produces` 的 artifact 是模块 B `requires` 的，A 必在 B 之前。槽位顺序只做平局裁决 —— 所以**换一个数据依赖不同的模块，不需要改 pipeline**。

### 3.5 九个插件

| 模块 | 版本 | 类别 | 验证状态 | 说明 |
|---|---|---|---|---|
| `mesh.structured2d` | 0.1.0 | mesh | partially-verified | 层状堆栈 → 渐变竖直网格 + 以栅为中心的横向网格 |
| `physics.polarization.ambacher` | 0.1.0 | physics | partially-verified | AlGaN 自发+压电极化、应变、屏蔽因子 → 界面束缚面电荷 |
| `physics.quantum.analytic_tw` | 0.1.0 | physics | partially-verified | 三角阱解析电荷控制方程自洽迭代（**基线**） |
| **`physics.quantum.sp1d`** | **1.1.1** | physics | partially-verified | **自洽 1D Schrödinger–Poisson（面基界面 + Newton + 严格精确 Jacobian）** |
| **`nonlinear.newton`** | 1.0.0 | solver | partially-verified | **通用稠密 Newton–Raphson + 线搜索 + PTC（`nonlinear` 服务）** |
| **`linear.blocktridiag`** | 0.1.0 | solver | partially-verified | **纯 numpy 块三对角直接解（`linear` 服务，替代环境中缺失的 SciPy）** |
| **`transport.dd`** | 0.1.0 | transport | partially-verified | **稳态 2D 漂移扩散 + 偏压扫描（SG 通量、块三对角 Jacobian、PTC）；T0–T5 均有基准测试，绝对数值未经标定** |
| `materials.dft` | 0.1.0 | physics | partially-verified | 文献 DFT 查找表覆盖通道（`table` 后端，带溯源与不确定度） |
| `extract.fom` | 0.1.0 | extract | unverified | 电荷控制阈值估计 + FoM 打包 |

### 3.6 服务槽：求解器也是插件

`nonlinear` / `linear` / `backend` 这三个槽里的模块**不是流水线阶段**：它们不读 artifact、不写 artifact，而是在另一个模块**内部**被调用。它们通过上下文上的 `ServiceRegistry` 按需解析，参数由 Deck 配置：

```json
"modules": {
  "nonlinear": {
    "module": "nonlinear.newton",
    "version": "^1.0.0",
    "params": { "max_iterations": 40, "line_search": true, "armijo": 1e-4 }
  }
}
```

物理模块这样拿到它：

```python
solver = ctx.require_services().get("nonlinear.newton")
outcome = solver.solve(system.residual, system.jacobian, x0, options={"tolerance": 1e-6})
```

**换迭代策略和换物理模型是同一种操作：改一行绑定。** 一个未来的 Krylov 或信赖域求解器只要声明 `nonlinear` 能力、暴露同样的 `solve()` 就能顶上，物理模块一行不用改。

### 3.7 漂移扩散输运（`transport.dd`）

P1 的另一半：极化/量子回答"电荷从哪来"，`transport.dd` 回答"电荷怎么流"。稳态、等温、单极（电子）的 2D 漂移扩散，有限体积 + Scharfetter–Gummel 通量，未知量是 `(ψ, n)`，Jacobian 解析给出并组装成块三对角，线性步经 Deck 的 `linear` 槽解析（`linear.blocktridiag`），外层是阻尼 Newton + 回溯线搜索 + 伪瞬态延拓，最外面套偏压扫描。

```jsonc
// decks/demo_hemt_dd.json（片段）
"physics": { "transport": { "module": "transport.dd", "params": {
  "vds_sweep_V": "0.0,0.25,0.5,0.75,1.0", "vgs_sweep_V": "-2.0,0.0",
  "max_step_V": 0.05, "require_continuity": true, "continuity_tol": 1e-06 } } },
"modules": { "linear": { "module": "linear.blocktridiag" } }
```

产出 `electrostatics` / `carriers` / `current` / `iv` 四个 artifact（外加标量 `ns`），每个偏置点记录收敛状态、迭代数、残差与**端点电流连续性误差**。

**验证到什么程度**（细节见 `transport_dd_插件设计.md` §12.3）：

| 检查 | 实测 |
|---|---|
| T0 均匀电阻闭式解 `J = q N_D μ V/L` | 相对误差 **< 1e-6**（设计判据 0.1%），两端一致 < 1e-9，电流与偏压严格正比 |
| 平衡态（0 V）端点电流 | **2.7e-15 A/m**（机器零） |
| 连续性门禁（Vds ≤ 1 V） | 相对误差 **1e-12 或更好**，全部 10 个偏置点通过 |
| 耦合 Jacobian 对中心差分 | 沿 Newton 路径 **~1e-10**（**逐行**相对判据） |
| T2 Poisson 子问题 MMS | 加密 4 档，L∞ 误差 1.3e-2 → 2.0e-4，**观测阶 2.008 / 2.002 / 2.001** |
| demo deck | `status: success`，10 个偏置点约 50 s（用 `max_step_V = 0.05` 的子步长阶梯；0.25 V 阶梯约 9 s，但末端点不收敛） |

**为什么扫描止于 1 V**：Caughey–Thomas 场依赖迁移率（含速度饱和）已默认启用，但 Vds ≥ 2 V 时 Newton 仍不收敛（欠账）；短栅 150 nm 的速度过冲与弹道成分也不在模型里。这是诚实边界，不是调参问题。

### 3.8 材料验证：`materials.dft`（`table` 后端）

内置材料库的常数转录自文献，标记 `unverified`。`materials.dft` 提供一条**带不确定度与完整溯源**的第一性原理覆盖通道：

- **drop-in 替换 `materials` 槽**：加载内置库 → 叠加 DFT 常数 → 重新发布 `materials` artifact，`physics.polarization.ambacher` 与 `physics.quantum.sp1d` **一行未改**；
- **三种后端共用同一 artifact 契约**：`table`（默认，零第三方依赖，读 `data/dft_lookup/algan_gan.json`）/ `qe_local` / `qe_slurm`（后两者惰性导入 ASE + Quantum ESPRESSO，**本阶段未实现**）；
- **精度红线写进代码而不是文档**：`eg_policy=reject_pbe`（PBE 带隙系统性偏低，绝不允许直接喂给器件模型，除非显式 `scissor_to_reference` 并记录平移量）；`ecut`/`kspacing` 与表记录不符即视为 **stale**，`allow_stale_cache=False` 时明确报错而非静默使用；无收敛证据的条目标 `converged=false`；
- **产出 `dft_provenance`** 审计 artifact，逐条记录来源、泛函、赝势、精度设置与不确定度；`override_policy` 三态（`dft_wins` / `reference_wins` / `dft_only_gapless`）决定合并方向。

D0–D3 对照实验基准 + Deck 一行替换验收（`decks/demo_hemt_dft.json`）。

---

## 4. 自洽 Schrödinger–Poisson 求解器

这是 P1 第一步的成果：把解析三角阱估计换成**真正解方程**，并且**在不改动任何其他文件的前提下**替换掉它。

### 4.1 它解什么

```
z = 0        金属 / AlGaN 界面     Schottky 边界条件 E_c(0) = q(φ_b − V_g)
0 < z < d    AlGaN 势垒            ε_b, m*_b
z = d        极化面电荷 + 导带带阶台阶 ΔE_c
d < z < L    GaN 沟道/缓冲          ε_c, m*_c
z = L        背面零场（绝缘缓冲）
```

耦合方程组：

```
d/dz(ε ε0 dφ/dz) = q n(z) − σ δ(z−d)                    Poisson（有限体积）
−ħ²/2 d/dz(1/m* dψ_i/dz) + E_c(z) ψ_i = E_i ψ_i          Schrödinger（BenDaniel–Duke）
n(z) = Σ_i n_i |ψ_i(z)|²,  n_i = g_v m*_i kT/(πħ²) ln(1+e^{(E_F−E_i)/kT})
```

**费米能级是能量零点**，由栅极肖特基势垒钉住界面导带；自洽解出亚带能级、占据数和二维面密度。

### 4.2 数值方法

**异质界面是一个"面"，不是节点。** 节点落在 `d ± h/2`，于是带阶跃变、介电常数跃变、有效质量跃变和面电荷**全部落在单元格边界上**。把它们压在节点上会以 O(h) 的分辨率模糊掉这四件事，直接吃掉耦合求解的一阶收敛。

- **Poisson**：非均匀网格上的有限体积。面介电常数取两侧的长度加权串联平均；面电荷通过消去界面电位、代入两个单侧通量关系处理，得到**精确的**介电常数权重 `ε_b/(ε_b+ε_c)` 与 `ε_c/(ε_b+ε_c)`（和为 1）。对分片线性解，这个格式是**精确的**（见 4.3）。
- **Schrödinger**：BenDaniel–Duke 有限差分。用控制体宽度加权使算子**对称化**，转成标准对称特征值问题（`numpy.linalg.eigh`）。
- **耦合（Newton，默认）**：组装耦合残差的 Jacobian 并交给 `nonlinear.newton` 服务求解。
  Jacobian **严格精确**，含三项：**亚带能量响应**（rank-n 更新，承载主导物理）、**波函数响应**（`δψᵢ = Σ_{j≠i} ψⱼ⟨ψⱼ|δE_c|ψᵢ⟩/(Eᵢ−Eⱼ)`，求和**遍历全谱、不截断**）与**态密度质量响应**（`m*ᵢ = 1/Σ|ψᵢ|²w/m` 也依赖 ψᵢ，漏掉它是一个**缺项**，特征是"误差不随差分步长下降"的 1.3e-5 地板）。
  **做法**：波函数响应的求和就是三对角 Hamilton 的正交 resolvent，`Σ_{k≠i} ψₖψₖᵀw/(Eᵢ−Eₖ) = −(A−Eᵢ)†`；用**秩-1（Woodbury/Sherman–Morrison）正则化 + 固定规范的三对角消元**把 `(A−Eᵢ)†` 一次分解、对 N 个右端批量求出，代价 O(N²)/亚带而不是本征矢求和的 O(N³)。工程上 Thomas 求解器扩展成**块右端**（一次消元、N 列）。
  **验证**：对中心差分相对误差 ~1e-11（纯差分公式噪声）；模块内保留一条独立实现 `quantum_response_jacobian_reference`（全谱本征矢求和），两条路径一致到 4e-13。耦合 Newton 的局部收敛阶回到 **1.99**，残差落到机器精度（1.3e-16）。
  **代价（如实）**：装配比截断版贵一个数量级 —— 119 节点 0.6 → 5.1 ms，208 节点 1.3 → 16.5 ms（同机交错计时取 7 次最小值）；整解约 **1.5–1.7 倍**，**迭代数与线性求解次数不变**。买到的是一致性（transport.dd 误差预算的前提），不是更少的迭代。
- **耦合（Anderson，回退）**：阻尼不动点 + Anderson (Pulay) 加速。裸映射的线性增益约 **−30**（面密度涨 1% → 势阱深到让密度再变 ~30%），纯阻尼要求 β < 0.06 且需几百次迭代。保留它作为独立交叉验证。
- **热启动**：先用耗尽近似（二分法解标量电荷控制方程）给出接近解的起点。**只影响收敛速度，不影响收敛结果**（有测试证明）。
- **密度钳制**：可选，默认**关闭**。超过 `σ_eff/e` 势垒场会反向、物理上不可能有 2DEG，但把密度**正好钉在**界上会制造一个伪不动点，两种迭代都可能卡死在那里 —— 实测确认过，所以它只作为最后手段。

### 4.3 验证：子求解器对闭式解，耦合系统对收敛阶

| 基准 | 闭式解 | 实测 |
|---|---|---|
| 双介质平行板（无面电荷） | 分片线性精确解 | **< 1e-10 V** |
| 双介质平行板 + 界面面电荷 | 分片线性精确解 | **< 1e-10 V（精确）** |
| 无限深方阱 | E_n = n²π²ħ²/2m*L² | **二阶收敛** |
| 谐振子 | E_n = ħω(n+½) | 误差 < 1e-3 |
| **三角阱** | Airy 零点 E_n = \|ζ_n\|(ħ²F²/2m*)^{1/3} | **二阶收敛**，5.5e-5 → 3.4e-6 |
| E_c 算子 vs φ 形式 | 同一离散化的独立实现 | **2.9e-13 eV** |
| 耦合 Jacobian | 中心有限差分 | **相对误差 ~1e-11**（差分公式噪声；补上质量响应前恒有 1.3e-5 地板） |
| Jacobian 两条独立路径 | 秩-1 亏格消元 vs 全谱本征矢求和 | **4.0e-13** |
| Newton | f(x)=x²−2 | 观测阶 > 1.7；扩张线性映射 2 步内收敛 |
| 耦合 Newton 收敛阶 | 相邻三个残差估局部阶 | **1.99**（截断 Jacobian 时只能到 ~1.5） |
| Anderson | 增益 −5/−3/−8 的线性映射 | 收敛到 1e-10；纯混合发散 > 1 |

耦合系统的**不变量与无关性**（全部有测试）：

| 检查 | 结果 |
|---|---|
| **竖直间距收敛阶** | **p ≈ 2.08**（面基界面之前是 ~1.0） |
| **耦合 Newton 收敛阶** | **p ≈ 1.99**（截断 Jacobian 时只能到 ~1.5），末位残差 1.3e-16 |
| 沟道域尺寸 30 → 240 nm | n_s 只变 0.0008%（准连续态是合法求积，不是盒子假象） |
| 面密度低于极化上界 σ_eff/e | ✓（占比 0.83） |
| ∫n dz = Σᵢ nᵢ | 1e-8 相对一致 |
| 热启动 0.35σ → 0.95σ | < 0.5%（收敛结果与初值无关） |
| Newton vs Anderson | 同一答案，Newton 少一个数量级迭代 |

### 4.4 与解析模型的对比

同一个器件（Al₀.₂₅Ga₀.₇₅N/GaN，20 nm 势垒，Lg = 150 nm，screening 0.8）：

| 量 | 解析三角阱 | 自洽 SP | 差 |
|---|---|---|---|
| n_s (cm⁻²) | 7.58287e12 | 7.95848e12 | **+4.95%** |
| σ_eff/e 占据率 | 0.7886 | 0.8276 | +4.95% |
| E_F 距界面带边 (eV) | 0.461018 | 0.308840 | −33.0% |
| E₀ 距界面带边 (eV) | 0.380340 | 0.229336 | −39.7% |
| **E_F − E₀ (eV)** | 0.0806778 | 0.0795039 | **−1.46%** |

以及速度（同一器件、同一离散化）：

| 算法 | 迭代 | 函数求值 | 线性求解 | 阶段耗时 | n_s |
|---|---|---|---|---|---|
| **Newton** | **10** | 10 | 9 | **~140 ms** | 7.958480e12 |
| Anderson | 60 | 61 | — | 341 ms | 7.958479e12 |

Newton 用约 1/6 的迭代和约 **1/3** 的时间，落到同一位数上。

> **计时口径**：上表绝对值来自一次空闲机器的运行。精确 Jacobian 把装配从 0.6/1.3 ms 提到 5.1/16.5 ms（119/208 节点），整解实测 **1.5–1.7 倍**（119 节点 85 → 145 ms，208 节点 300 → 460 ms，同进程交错、各取 7 次最小值），迭代数与线性求解次数**不变**，所以 Newton 一格按 90 ms × 1.5 标成 ~140 ms。测量期间本机另有重计算任务在跑，绝对值仅供参考，**比值**才是可靠的那部分。

**这个差异的形态本身就是物理结论**：自洽解的势阱更浅，所以 E_F 和 E₀ 都掉了约三分之一；但**决定 n_s 的亚带间距 E_F − E₀ 几乎没变**。两个模型对**势**的看法不同，对**统计**的看法收敛到同一处。

自洽解还给出解析模型根本没有的东西：多亚带阶梯、波函数穿透、真实的有效质量、束缚态数目、波函数质心与展宽。

### 4.5 即插即用：这就是验收测试

整个替换**只改 Deck 里的一行绑定**：

```diff
   "quantum": {
-    "module": "physics.quantum.analytic_tw",  "version": "^0.1.0",
+    "module": "physics.quantum.sp1d",         "version": "^1.0.0",
     "params": { "delta_ec_ratio": 0.7 }
   }
```

- `extract.fom` **一行未改**（它只认 artifact 键 `quantum` / `ns`，不认实现）
- Pipeline 的槽位、顺序、参数传递全部不变
- 版本钉住保证：解析 Deck 写的 `^0.1.0`，**永远不会**被悄悄升级到 sp1d

---

## 5. 怎么写一个新插件

一个插件就是一个目录，`__init__.py` 暴露两个名字：`MANIFEST` 和 `MODULE`。多文件插件可以用相对导入（`from .solver import ...`）。

```python
from huntian.registry import BaseModule, ModuleCategory, ModuleManifest, ParamSpec

MANIFEST = ModuleManifest(
    module_id="physics.mobility.my_model",       # 点分小写，至少两段
    module_version="0.1.0",
    category=ModuleCategory.PHYSICS,
    capabilities=("mobility",),                  # 用于槽位契约检查
    requires=("structure", "materials"),         # 需要的 artifact
    produces=("mobility",),                      # 产出的 artifact
    parameters=(
        ParamSpec.float("mu0_cm2_Vs", 1800.0, minimum=0.0,
                        calibratable=True, source="Hall measurement"),
    ),
    verification="unverified",
)

class MyMobility(BaseModule):
    def execute(self, ctx):
        structure = ctx.require("structure")
        ctx.put("mobility", {...}, producer=MANIFEST.module_id)
        return ctx

MODULE = MyMobility
```

丢进 `plugins/` 即可，**不需要修改任何其他文件**。

---

## 6. 可复现性与热插拔

`demo.py` 第 20 与 23 节完整演示：

```
-- 20. Same inputs, same fingerprint          -- 23. Version pinning protects the validated Deck
  analytic        run A : bcfbb07632737b44      the sp1d Deck pins '^1.0.0' -> v1.1.1
  analytic        run B : bcfbb07632737b44      fingerprint unchanged : True
  identical             : True
  self-consistent run A : 9d716641ad967853      versions available now : ['2.0.0', '1.1.1']
  self-consistent run B : 9d716641ad967853
  identical             : True                  Both versions coexist; nothing was reloaded.
```

注意语义细节：`^1.0.0` **允许** 1.x 的小版本升级（所以钉住的 sp1d 会解析到 1.1.1），只挡主版本 —— 这是标准 semver 行为，也是"钉住"到底钉住了什么。**反过来说**：指纹只哈希输入与解析出的模块版本，数值解本身不进指纹，所以数值行为一变就必须靠版本号表达 —— 本次 Jacobian 精确化把 sp1d 从 1.1.0 提到 **1.1.1**（self-consistent 指纹随之从 `c26cb881f33c000c` 变为 `9d716641ad967853`；解析模型未受影响，指纹不变）。

---

## 7. 诚实的边界

这部分同样重要，避免误用。

**已经验证的：**
- Poisson 有限体积格式：双介质平行板**有无面电荷都精确到 1e-10 V**
- E_c 算子与独立的 φ 形式离散化一致到 2.9e-13 eV
- Schrödinger 有限差分：三个闭式解，二阶收敛断言
- Newton 求解器：线性系统 2 步、标量非线性问题观测阶 > 1.7、线搜索在远起点下恢复
- **耦合 Jacobian 严格精确**：对中心差分 ~1e-11（三条独立 check：残差层、密度层、以及模块内全谱参考实现 4e-13）
- 耦合系统**竖直间距二阶收敛（p ≈ 2.08）**、**Newton 局部收敛阶 1.99**、物理边界、初值/域尺寸无关性

**已知局限（明确写出，没有粉饰）：**
- **精确 Jacobian 的代价是装配变贵**：比旧的截断版贵约一个数量级（119 节点 0.6 → 5.1 ms，208 节点 1.3 → 16.5 ms），整解贵 1.5–1.7 倍；迭代数与线性求解次数不变。这笔钱是为 transport.dd 的误差预算付的，不是为了省迭代。将来若嫌贵，两条正路都已看清：(a) 三对角逆的**半可分（semiseparable）Green 函数两步扫描**可把装配降到 O(N) 次的向量化运算；(b) 耦合 Newton 的线性步改成**矩阵无关（matrix-free）Krylov**（`工作日志.md` §6-7），响应就只需对向量作用，即每个亚带一次三对角求解 O(N)，根本不必组装稠密矩阵。
- **线性步仍是稠密的**（`nonlinear.newton` 用 `np.linalg.solve`）：它自己装不下 2 万节点的 DD。DD 因此**不走**这个服务，而是经 Deck 的 `linear` 槽解析 `linear.blocktridiag`（纯 numpy 块三对角直接解，O(nz·m³)）。把 `nonlinear.newton` 也迁到 `linear` 槽上仍是 `工作日志.md` §6-7 的欠账。
- **Anderson 回退比 Newton 脆弱**：它更慢，而且在某些网格上会因密度钳制而卡住。默认算法是 Newton。
- **材料常数未经 L2 独立验证**：`huntian/data/materials.json` 的数值转录自 Ambacher 2000/2002 与 Vurgaftman 2003，标记 `unverified`。`materials.dft`（`table` 后端）现已提供一个**带不确定度与完整溯源**的文献 DFT 覆盖通道（`data/dft_lookup/algan_gan.json` + `dft_provenance` 审计），但它是有意声明的 **0 K、理想晶体、文献转录**先验 —— 跑通 L2 基准（Hall 对比）之前**不要相信绝对数值**。
- **`materials.dft` 的默认表路径是相对当前工作目录的**：`table_path` 默认 `data/dft_lookup/algan_gan.json`。在仓库根运行时正常，从别处运行（例如 `pip install` 之后）会报 `ModuleConfigError: lookup table not found` —— 已实测确认。再者 `plugins/` 不在 `python/` 包内，安装后不随发行版分发，该查找表也不在任何 wheel / sdist 里。**PyPI 发布前必须一并解决**（改用 `importlib.resources` 定位，或让表随插件目录一起分发）。
- **屏蔽因子（screening）没有公认取值**，默认 0.8 只是起点，必须用 Hall 数据标定。
- **有效质量近似**，忽略非抛物性、应变对质量的修正。
- **势垒以上的态是有限域的盒子本征态，不是真正的束缚态**。它们是 3D 连续态的合法求积（面密度对域尺寸不敏感于 1e-5 以内），但**单个**波函数不是物理的。
- **只有 Γ 谷**，`valley_degeneracy` 只是缩放态密度。
- **无陷阱、无自热、无量热耦合**：所以电流坍塌、gate-lag、drain-lag 一概没有。
- **`extract.fom` 是占位提取器（`unverified`）**：它的 V_th 来自全耗尽电荷控制关系，只依赖势垒静电学，**与量子模型无关**；它报告的 n_s 才来自量子 artifact。
- **迁移率是占位参数**：`mu0_cm2_Vs`（1500 cm²/Vs）与 `vsat_cm_s`（2e7 cm/s）都未经标定，L2 之前绝对数值不可信。场依赖迁移率用 Caughey–Thomas（默认启用，`vsat_cm_s <= 0` 退回常数迁移率），导数进 Jacobian 保持精确。
- **2DEG 面密度比极化上限高约 1.7 倍**：来源于"层界落在节点上"导致的介电/带阶抹平（一格）。设计稿 §3 已预见这一代价，`transport_dd_插件设计.md` §12.2 首次给出量化。
- **漂移扩散只覆盖到约 1 V**：速度饱和已实现（Caughey–Thomas，默认启用），但 Vds ≥ 2 V 时 Newton 仍不收敛（欠账）；短栅（150 nm）的速度过冲与弹道成分缺席，方案 §15 的对策（hydro 插件）原样适用。
- **表面钉扎是工程近似**：无栅表面被钉在肖特基参考电位（真实表面由施主态钉在特定能级，属 `trap.surface` 的活）。它同时承担"给势垒电位一个锚"的数值职责。

一句话：**骨架可信，量子求解器与 Newton 求解器都验证到闭式解，网格收敛与 Newton 收敛阶都修到了二阶（Jacobian 严格精确）；漂移扩散已能跑出守恒到 1e-13 的 IV 曲线，通过 T0 闭式解（< 1e-6）、T2 二阶收敛与电流饱和判据，但绝对数值未标定、Vds ≥ 2 V 仍不收敛。**

---

## 8. 目录结构

```
Huntian/
├── python/huntian/
│   ├── schema.py          声明式 schema + 严格校验 + JSON Schema 输出
│   ├── semver.py          零依赖语义化版本与范围匹配
│   ├── events.py          同步事件总线
│   ├── context.py         SimContext / Artifact / 溯源 / 指纹
│   ├── materials.py       材料库（含出处与验证状态）
│   ├── registry.py        IModule / Manifest / 发现 / 解析 / 热插拔
│   ├── services.py        ServiceRegistry：按需解析协作者模块
│   ├── deck.py            Deck schema / 解析 / 两级校验 / 服务槽
│   ├── pipeline.py        Pipeline 装配 + Executor + 服务装配
│   ├── cli.py             命令行入口
│   └── data/materials.json
├── plugins/
│   ├── mesh_structured2d/
│   ├── polarization_ambacher/
│   ├── quantum_analytic_tw/      基线：解析三角阱
│   ├── quantum_sp1d/             自洽 Schrödinger–Poisson（多文件插件包）
│   │   ├── __init__.py           插件入口（MANIFEST + MODULE）
│   │   └── solver.py             纯数值：面基网格 / PoissonOperator / 精确 Jacobian / 驱动
│   ├── nonlinear_newton/         Newton 服务模块（多文件插件包）
│   │   ├── __init__.py           插件入口，暴露 solve() / verify_jacobian()
│   │   └── solver.py             通用稠密 Newton + 线搜索 + PTC + Jacobian 校验
│   ├── linear_blocktridiag/      linear 服务：纯 numpy 块三对角直接解
│   │   ├── __init__.py
│   │   └── solver.py
│   ├── transport_dd/             漂移扩散输运（多文件插件包）
│   │   ├── __init__.py           插件入口（MANIFEST + MODULE）
│   │   ├── plugin.py             扫描循环 / PTC 状态机 / artifact 写出
│   │   ├── physics.py            统计（Boltzmann/Joyce-Dixon）、场依赖迁移率、边界条件
│   │   ├── assemble.py           残差 / 块三对角 Jacobian 组装（FV + SG 通量）
│   │   └── newton.py             阻尼 Newton + 线搜索 + PTC
│   ├── materials_dft/            第一性原理材料库（多文件插件包，table 后端零依赖）
│   │   ├── __init__.py           插件入口（MANIFEST + MODULE）
│   │   ├── plugin.py             生命周期 / 精度红线 / 覆盖合并 / dft_provenance
│   │   ├── table_io.py           查找表读写、结构校验、stale 检测
│   │   ├── qe_driver.py          ASE/Quantum ESPRESSO 驱动（惰性导入，未实现）
│   │   └── slurm.py              集群作业（惰性导入，未实现）
│   └── extract_fom/
├── data/
│   └── dft_lookup/algan_gan.json 文献 DFT 查找表（版本化数据资产，进 golden CI）
├── decks/
│   ├── demo_hemt.json            解析量子模型
│   ├── demo_hemt_sp1d.json       同一器件：自洽 SP + Newton（改了一行绑定 + 服务槽）
│   ├── demo_hemt_dd.json         同一器件：transport.dd 输运阶段（+ linear 槽）
│   ├── demo_hemt_dft.json        同一器件：materials.dft 替换材料槽（一行绑定）
│   └── demo_hemt.yaml            YAML 示例
├── tests/                        426 个用例（含 golden/dft_benchmarks_d0_d3.json 对照）
├── docs/tex/                     模块化技术文档（LaTeX 源，一个插件一章）
│   └── build/                    构建产物（PDF 与中间文件，不入库）
├── tools/check_encoding.py       编码守卫：合法 UTF-8 / 无 BOM / 无 U+FFFD / 无孤立 CR
├── demo.py                       端到端演示
├── run_tests.py                  测试入口（无需安装；**全量套件需要 numpy**）
├── pyproject.toml                打包与元数据
├── README.md                     本文件（现状）
├── 工作日志.md                   每阶段做了什么、为什么、欠账、教训、版本历史
├── CONTRIBUTING.md               贡献指南（项目特有约定）
├── CODE_OF_CONDUCT.md            Contributor Covenant 2.1
├── LICENSE                       MIT
├── .github/                      CI 工作流 + issue / PR 模板
└── 命名*.md                      改名决策的证据链（三份，见 §11）
```

---

## 9. 测试覆盖

```
python run_tests.py    ->  426 tests, 0 failures, 0 errors   (~18 s)
```

| 文件 | 覆盖 |
|---|---|
| `test_semver.py` | 版本解析、排序、`^`/`~`/前缀/OR 语义、需求解析 |
| `test_schema.py` | 默认值、类型强制、未知字段拒绝与拼写提示、范围/枚举、嵌套路径、JSON Schema、语义钩子 |
| `test_context.py` | artifact 读写、溯源（成功/失败）、事件、指纹稳定性与敏感性、快照 |
| `test_registry.py` | 注册/选择/多版本、依赖拓扑、环检测、版本冲突、参数拒绝、热插拔、插件发现与损坏隔离、**多文件插件相对导入** |
| **`test_services.py`** | **ServiceRegistry 配置/缓存/能力解析/释放、Deck 服务槽与阶段槽分离、Executor 服务装配、缺少服务时的清晰报错** |
| `test_deck.py` | 加载、结构拒绝（含路径）、语义规则、参数/材料/模块校验、artifact 可达性 |
| `test_pipeline.py` | 拓扑排序、端到端执行、指纹可复现、严格/可选失败语义 |
| `test_plugins.py` | Ambacher 面电荷解析值、n_s 上界与单调趋势、网格不变量 |
| **`test_quantum_sp1d.py`** | **面基网格不变量、E_c 算子与 φ 形式一致性、平行板闭式解（含面电荷，精确）、Schrödinger 三个闭式解与收敛阶、精确 Jacobian（残差层/密度层中心差分 ~1e-11，秩-1 亏格 vs 全谱本征矢双路径互证、外来态拒绝）、块右端 Thomas、Newton 二次收敛（阶 1.99）与机器精度、Newton vs Anderson 一致性、耦合系统二阶收敛、初值/域尺寸/温度/栅压** |
| **`test_nonlinear_newton.py`** | **Newton 收敛阶、线搜索回溯与单调性、PTC、奇异 Jacobian 报告、Jacobian 校验器、模块契约与参数校验** |
| **`test_hotswap_quantum.py`** | **能力与 artifact 契约、下游零改动、两模型差异的物理合理性、版本钉住、第三方实现热插拔** |
| **`test_materials_dft*.py`** | **MANIFEST 与零 calibratable 红线、eg_policy（reject_pbe/scissor）、stale 拒绝、查找表 schema、覆盖策略三态、合并契约与 dft_ 前缀键、D0–D3 对实验基准（golden 对照）** |
| **`test_hotswap_materials.py`** | **materials 槽替换验收：流水线仅增一个 materials 阶段、polarization/sp1d/extract 零改动、override_report 全溯源、指纹两次运行一致** |
| **`test_linear_blocktridiag.py`** | **块三对角解对稠密 LU 逐例一致、正则化、奇异枢轴报错、`linear` 能力解析与缺少服务时的报错** |
| **`test_transport_dd.py`** | **Bernoulli 闭式解与反射恒等式、SG 通量四个解析偏导对中心差分、电势偏导反对称、`∂J/∂n` 与 n 无关、异质界面零通量不变量；网格掩码互斥、DOS 折进带边、非线性 Poisson 起点、耦合 Jacobian（逐行相对 FD）、T2 Poisson 子问题 MMS 二阶收敛（观测阶 2.008/2.002/2.001）、平衡态端点电流 ≈ 0、偏压解电流守恒（T3 门禁）、2DEG 面密度对极化上限、T0 均匀电阻闭式解与偏压正比、延拓无关性（T4）、栅压控制趋势（T5）、Deck 集成与参数热插拔** |

物理测试刻意断言**不变量、趋势与收敛阶**，而不是魔法数字 —— 这样参数演化时它们依然有效。

---

## 10. 下一步

按优先级排列；标 📝 的已有独立设计稿（见 §11）：

1. ✅ **把 Jacobian 做到严格精确（已完成）** —— 波函数响应的态求和的**不再截断**：用秩-1 Woodbury/Sherman–Morrison 正则化 + 固定规范的三对角消元得到 `(A−Eᵢ)†`，块右端 Thomas 一次分解求 N 列，代价 O(N²)/亚带（替代全谱本征矢求和的 O(N³)）；顺带发现并补上了**态密度质量响应**这一缺项。结果：对中心差分 ~1e-11、与模块内全谱参考实现一致到 4e-13、耦合 Newton 局部阶 **1.99**。代价（装配贵一个数量级、整解 1.5–1.7 倍）与两条后续提速路线已写进 §7。
2. ✅ **`transport.dd`（已完成）** 📝 —— 漂移扩散 + 连续性方程，把平衡态求解扩展成带偏压求解。落地：`linear.blocktridiag` 服务（纯 numpy 块三对角直接解，替代环境里缺失的 SciPy）+ `plugins/transport_dd/`（physics/assemble/newton/plugin）+ `decks/demo_hemt_dd.json`（端到端 `status: success`）+ 31 项测试。实测：T0 均匀电阻闭式解相对误差 < 1e-6、T2 MMS 观测阶 2.008/2.002/2.001、平衡态端点电流 2.7e-15 A/m、Vds ≤ 1 V 连续性 1e-12 或更好、耦合 Jacobian 沿路径 ~1e-10、Caughey–Thomas 迁移率带来电流饱和。**局限**：Vds ≥ 2 V 的 Newton 收敛、T1 与 1D SP 的带边对比（界面抹平）、2DEG 面密度比极化上限高约 1.7 倍，均在 `transport_dd_插件设计.md` §12 中说明。
3. **L2 基准 + golden CI** —— 用 Hall 测量对比 n_s，把测试套件接入回归门禁。
4. **陷阱与自热** —— 电流坍塌、gate/drain-lag、晶格温度（P3）。
5. ✅ **`materials.dft` 第一阶段（`table` 后端，已完成）** —— 插件落地：文献 DFT 查找表 + 覆盖合并 + `dft_provenance` 审计 + 精度红线（`reject_pbe` / scissor / stale 拒绝 / `converged` 标记）+ D0–D3 golden 基准 + Deck 一行替换验收（`decks/demo_hemt_dft.json`）。**第二阶段（`qe_local` 驱动 + 重算表格 + 真收敛记录）与第三阶段（`qe_slurm` + SQS 合金扫描，D4）未开工**，见 `工作日志.md` §6。设计稿 `DFT材料计算模块_插件设计.md`。

---

## 11. 设计文档索引

| 文档 | 状态 | 对应内容 |
|---|---|---|
| `GaN_RF_HEMT_TCAD_实现方案.md` | v1.0 总纲 | 完整方案的物理范围、架构、路线图（§14 的 P0/P1/P2…） |
| `transport_dd_插件设计.md` | v0.1.2 已落地（含 §12 实现偏差记录） | `transport.dd`：漂移扩散输运，README §10 第 2 项 |
| `DFT材料计算模块_插件设计.md` | v0.2 部分实现（`table` 后端已落地，`qe_*` 未实现） | `materials.dft`：材料库的 DFT 验证通道，README §10 第 5 项 |
| `工作日志.md` | 持续更新 | 每阶段做了什么、为什么、欠账、教训、版本历史 |

> **文档一致性**：两份设计稿的路线图编号均已对齐到本节（`transport.dd` = 第 2 项、`materials.dft` = 第 5 项）。`transport_dd_插件设计.md` 已升 v0.1.2：§1–§11 保持设计原样，**§12 为实现偏差记录**（架构偏差 D1–D5、六项模型决策、T0–T5 各级基准的兑现情况、两条验证纪律）。`nonlinear.newton` v1.0.0 已落地（含阻尼线搜索 + PTC），对本模块唯一的缺口是稠密线性步 —— DD 因此改经 `linear` 槽解析 `linear.blocktridiag`，把 `nonlinear.newton` 也迁到该槽仍是欠账。

---

*对应《GaN 射频 HEMT 轻量化 TCAD 软件 —— 完整实现方案》§14 的 P0 阶段与 P1 全部步骤。*
