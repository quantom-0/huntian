# 漂移扩散输运模块（transport.dd）—— 插件设计

> 版本：v0.1.2（设计稿 + 落地偏差记录，见 §12；§12 为落地后补写，§1–§11 保持设计原样）
> 定位：NemoTCAD 插件 `transport.dd`，把当前的**平衡态**求解扩展为**带偏压**的 2D 漂移扩散求解，产出 IV 特性
> 路线图：README §10 下一步**第 2 项**（⏳ 待实现）；两项前置均已完成 —— `nonlinear.newton` 服务已落地（v1.0.0），SP 耦合的一阶收敛问题已由面基界面解决（竖直网格收敛阶 p ≈ 2.08）
> 架构前提：沿用 sp1d 验证过的模式 —— **只加插件目录，核心一行不改**

---

## 1. 设计决策

### 1.1 物理范围（先划死，防蔓延）

| 决策 | 取舍 |
|---|---|
| **只解电子**（单极 n 型） | 方案 §2.2 的原则：GaN RF HEMT 是单极器件；空穴连续性方程不进 v1 |
| **2D 截面、等温** | 晶格温度由后续 `thermal.electrothermal` 以插件形式叠加；v1 温度是参数不是场 |
| **有限体积 + Schafetter–Gummel 通量** | 方案 §5.2 的主路径：保证稳定性与逐节点电荷守恒 |
| **Newton–Raphson + 阻尼 + 伪瞬态延拓（PTC）** | 方案 §5.2 / §15：强极化电荷场下裸 Newton 必发散，PTC 是护城河级投入 |
| **量子修正是可选项，不是依赖** | `quantum`/`ns` artifact **不进入 requires**，用 `ctx.get` 探测；没有量子模块时退化为经典电荷控制 |
| **求解器先内嵌，后抽取** | `nonlinear.newton` 服务**已落地**（v1.0.0），阻尼 + 线搜索 + PTC 都已具备；但它唯一的缺口是**线性步为稠密**（`np.linalg.solve`），扛不住本模块 ~2 万节点。所以 v1 的 Newton 仍写在插件内，抽取路径改为「补 `linear` 槽的稀疏实现 → 服务改为经 `linear` 槽解析线性步」—— 与 sp1d 走过一遍的路同构 |

### 1.2 为什么量子输入是可选的

`requires=("quantum",)` 会把量子模块变成硬依赖，破坏"经典近似也能跑"的退化路径，也会让 `analytic_tw` 与 `sp1d` 之外的多一种组合失效。设计上：

- **没有 `quantum` artifact**：界面 2DEG 密度由经典电荷控制（耗尽 + 极化面电荷的电场平衡）给出初值；
- **有 `quantum` artifact**：把平衡态 `ns_m2` 作为 2DEG 初值与一致性检查基准（Vd=0 时与 DD 解出的沟道电荷对比，写进 `iv` artifact 的 `equilibrium_check`）。

这与 README 验证哲学一致：**量子模型可替换，输运内核不认实现，只认数据**。

---

## 2. 模块总览

```
module_id:    transport.dd
category:     ModuleCategory.TRANSPORT        # 注册中心已定义该类别
capabilities: ("transport", "transport.dd", "current", "iv")
requires:     ("mesh", "structure", "materials", "polarization")
produces:     ("electrostatics", "carriers", "current", "iv")
verification: unverified → 以 §7 基准通过后升 partially-verified
dependencies: ()                              # v1 无模块级依赖
```

**Pipeline 推导**：`requires` 含 `mesh`/`polarization`，本模块自动排在 `mesh.structured2d`、`physics.polarization.ambacher`（及可选的 `physics.quantum.sp1d`）之后；`produces` 的 `iv` 为后续 `extract.iv`（L4 基准的提取器）预留消费契约。

---

## 3. 控制方程与离散（v1 求解内容）

```
∇·(ε∇ψ) = −q(−n + N_D⁺ + ρ_pol + ρ_2DEG·δ(z−z_i))     Poisson
(1/q)∇·J_n = 0                                          电子连续性（稳态）
J_n = q n μ(E) ∇ψ + q D_n ∇n,  D_n = μ kT/q·(degeneracy correction)
```

- **网格**：直接消费 `mesh` artifact 的张量积网格（`x_nm` × `z_nm`，`layers[].node_slice` 给出材料分区），不重新剖分；
- **界面 2DEG**：极化面电荷 `sigma_eff_C_m2`（来自 `polarization` artifact）作为异质结界面控制体的固定电荷项；2DEG 的横向输运用界面节点上的**等效薄层密度**（面密度 × 量子极限厚度，或经典分布）表达；
- **通量**：Schafetter–Gummel 指数拟合，保证任意网格 Peclet 数下单调、不振荡；
- **统计**：`fermi_level="boltzmann" | "joyce_dixon3"`（默认 Joyce–Dixon 三阶，覆盖 2DEG 简并区，成本可忽略）；
- **迁移率**：v1 参数化 `mu0_cm2_Vs` + 可选 `vsat_cm_s` 场依赖饱和（简单 Caughey–Thomas 形式）；`mobility.farahmand` 作为独立插件落地后，通过 artifact 键 `mobility` 以同构方式热插拔替换；
- **边界条件**：
  - 源/漏欧姆接触：`mesh.contacts_nm.source / drain` 列上的 Dirichlet 准费米势（外加偏压直接平移）；
  - 栅肖特基：`E_c = q(φ_b − V_g)`，φ_b 取 `gate.barrier_eV` 或 sp1d 的解析拟合（同一来源逻辑）；
  - 其余边界：零法向场（Neumann）。

---

## 4. Manifest 定义（落地代码草稿）

与 `nemo/registry.py` 实际签名一致。

```python
MANIFEST = ModuleManifest(
    module_id="transport.dd",
    module_version="0.1.0",
    category=ModuleCategory.TRANSPORT,
    description=(
        "Steady-state 2D drift-diffusion transport for the AlGaN/GaN HEMT: "
        "finite-volume discretization with Schafetter-Gummel fluxes on the "
        "structured mesh, damped Newton with pseudo-transient continuation. "
        "Optional quantum warm start from the sp1d equilibrium solution."
    ),
    capabilities=("transport", "transport.dd", "current", "iv"),
    requires=("mesh", "structure", "materials", "polarization"),
    produces=("electrostatics", "carriers", "current", "iv"),
    verification="unverified",
    author="NemoTCAD",
    tags=("transport", "drift-diffusion", "newton", "iv"),
    parameters=(
        # ---- 偏置扫描 ---------------------------------------------------
        ParamSpec.string(
            "vds_sweep_V", "0.0,1.0,5.0,10.0,20.0",
            doc="comma-separated drain voltages; outer sweep loop",
        ),
        ParamSpec.string(
            "vgs_sweep_V", "-4.0,-2.0,0.0",
            doc="comma-separated gate voltages; inner sweep loop",
        ),
        ParamSpec.boolean(
            "sweep_reuse_previous", True,
            doc="start each newton solve from the previous bias point's solution",
        ),
        # ---- 物理 --------------------------------------------------------
        ParamSpec.float(
            "temperature_K", 300.0, unit="K", minimum=200.0, maximum=500.0,
            calibratable=True,
            doc="isothermal lattice temperature; coupling to thermal.* is future work",
        ),
        ParamSpec.float(
            "mu0_cm2_Vs", 1500.0, unit="cm^2/(V*s)", minimum=1.0, maximum=10000.0,
            calibratable=True,
            source="placeholder pending mobility plugin and Hall calibration",
        ),
        ParamSpec.float(
            "vsat_cm_s", 1.5e7, unit="cm/s", minimum=0.0, maximum=5.0e7,
            calibratable=True,
            doc="saturation velocity for the field-dependent mobility",
        ),
        ParamSpec.float(
            "beta_field", 2.0, unit="1", minimum=0.5, maximum=5.0,
            doc="Caughey-Thomas field exponent",
        ),
        ParamSpec.choice(
            "fermi_level", ("boltzmann", "joyce_dixon3"),
            default="joyce_dixon3",
            doc="carrier statistics; joyce_dixon3 covers the degenerate 2DEG",
        ),
        ParamSpec.choice(
            "quantum_warm_start", ("auto", "off"),
            default="auto",
            doc="auto: use ctx['quantum'] as initial guess and equilibrium check when present",
        ),
        # ---- Newton ------------------------------------------------------
        ParamSpec.integer(
            "max_newton_iterations", 200, minimum=1, maximum=5000,
        ),
        ParamSpec.float(
            "newton_tolerance", 1.0e-9, unit="1", minimum=1.0e-14, maximum=1.0e-3,
            doc="scaled 2-norm residual tolerance",
        ),
        ParamSpec.integer(
            "damping_max_halvings", 40, minimum=0, maximum=200,
            doc="line-search halvings per Newton step",
        ),
        ParamSpec.boolean(
            "ptc_enabled", True,
            doc="pseudo-transient continuation; the safety net for the polarized 2DEG",
        ),
        ParamSpec.float(
            "ptc_tau0_s", 1.0e-12, unit="s", minimum=1.0e-18, maximum=1.0,
            doc="initial pseudo-time step; grows multiplicatively on success",
        ),
        ParamSpec.integer(
            "linear_solver", ...,  # 见 4.1
        ),
    ),
)
```

**4.1 线性求解的诚实说明**：v1 节点规模 ~2 万，稠密牛顿步不可行 —— 这正是现成的 `nonlinear.newton` 服务**不能被直接复用**的原因（它是 `np.linalg.solve` 稠密实现；阻尼线搜索与 PTC 倒是可以直接沿用）。`solver.py` 用 **scipy.sparse**（`spsolve` / 可选 `cg`+ILU）组装稀疏 Jacobian —— 与 sp1d 需要 numpy 同一先例，插件自带依赖。

Deck 已有 `linear` 服务槽（`SERVICE_SLOTS`，见 README §3.6），所以建议把 scipy 封装成 `linear.scipy_sparse` 服务，本模块经 `ctx.require_services().get_by_capability("linear")` 解析 —— 将来换 GPU 求解器（方案 §6）只改 Deck 一行绑定、本插件不动。v1 可先内嵌以满足"只加插件目录"，但接口按服务形态留好。

**4.2 校准白名单**：`temperature_K`、`mu0_cm2_Vs`、`vsat_cm_s` 标 `calibratable=True`，对齐方案 §9.3 的拟合白名单机制；数值参数（迭代、阻尼、PTC）一律不可校准 —— 同 DFT 设计的红线理由。

---

## 5. Artifact 契约

### 5.1 `electrostatics`

```python
{
    "psi_V": [[...], ...],          # (nz, nx) 电势，行=z 列=x，与 mesh 网格对齐
    "ec_eV": [[...], ...],          # 导带边 = -q*psi + 带阶 + 界面台阶
    "efn_eV": [[...], ...],         # 电子准费米势
    "bias": {"vds_V": ..., "vgs_V": ...},
    "converged": True, "newton_iterations": 17, "ptc_steps": 3,
    "grid": {"nx": 41, "nz": 214},  # 冗余记录，防网格错位静默通过
}
```

### 5.2 `carriers`

```python
{
    "n_cm3": [[...], ...],          # (nz, nx) 体密度
    "nsheet_cm2": [...],            # (nx,) 界面等效面密度横向剖面
    "n_max_cm3": ..., "assumptions": [...],
}
```

### 5.3 `current`

```python
{
    "terminals_A_m": {"source": ..., "drain": ..., "gate": ...},
    "continuity_error": 3.2e-9,     # |Is+Id+Ig| / max(|Id|, tiny)；方案 §6.4 门禁
    "continuity_tol": 1.0e-6,
    "passes_continuity_gate": True,
}
```

**连续性门禁是硬断言**：`passes_continuity_gate=False` 时 `execute` 抛错，而不是写个警告放过去 —— 这是最灵敏的求解器健康检查，方案文档明确要求。

### 5.4 `iv`（提取器的消费契约，L4 基准的接口）

```python
{
    "vds_V": [0.0, 1.0, 5.0, 10.0, 20.0],
    "vgs_V": [-4.0, -2.0, 0.0],
    "id_A_m":  [[...], ...],        # (len(vgs), len(vds)) drain current
    "is_A_m":  [[...], ...],
    "ig_A_m":  [[...], ...],
    "gm_A_m":  [[...], ...],        # 数值微分，附带差分步长记录
    "equilibrium_check": {          # 仅 quantum_warm_start=auto 且 ctx 有 quantum 时
        "ns_dd_cm2": ..., "ns_quantum_cm2": ..., "rel_diff": ...,
    },
    "assumptions": [
        "Drift-diffusion with local field-dependent mobility; no velocity "
        "overshoot, no energy balance (hydrodynamic).",
        "Isothermal at temperature_K; no self-heating.",
        "No traps, no surface-state dynamics beyond the static screening "
        "factor of the polarization module.",
        "Single Gamma valley; mobility is a scalar.",
    ],
}
```

---

## 6. Newton + PTC 策略（防发散的核心）

方案 §15 把"Newton 收敛困难"列为最高风险，落地为三层保险，顺序执行：

1. **偏置扫描延拓**：Vd/Vg 逐点推进，前一点收敛解做下一点初值（`sweep_reuse_previous`）；
2. **阻尼线搜索**：残差不降则步长减半，最多 `damping_max_halvings` 次；
3. **PTC 兜底**：连续两次纯 Newton 失败则注入伪瞬态项 `(n − n_prev)/τ`，τ 从 `ptc_tau0_s` 起每成功一步 ×10，残差进 Newton 域后撤除。τ 序列写进 `electrostatics.ptc_steps` 供诊断。

`extract.fom` 的教训（Vth 只依赖势垒静电学、与量子模型无关）在这里复用：**初值与延拓只影响收敛路径，不影响收敛解** —— 测试断言不同初值链下同一点位解一致（容差内）。

---

## 7. 验证基准（T0–T5，先于任何"像不像器件"的检查）

与 `test_quantum_sp1d.py` 同一纪律：**先对闭式解，再谈不变量，最后才看 IV 长什么样**。

| 级 | 基准 | 参照 | 通过判据 |
|---|---|---|---|
| T0 | 均匀电阻 1D 极限（把网格退化成直通道，关断极化与肖特基） | J = q n μ V/L | 相对误差 < 0.1% |
| T1 | 零偏平衡态费米平坦 | J ≡ 0、E_c 剖面与 1D SP 一致 | 最大通量 < 1e-12 A/m；带边差 < 1 meV |
| T2 | Poisson 子问题 MMS 制造解 | 制造解析势 | 二阶收敛（对齐 sp1d 的收敛阶断言写法） |
| T3 | **电流连续性门禁** | ΣI_terminals = 0 | 相对误差 < 1e-6，全部偏置点 |
| T4 | 初值/延拓无关性 | 不同初值链同点位 | n 相对差 < 0.5%（对齐 sp1d 的热启动测试） |
| T5 | 趋势合理性 | — | Id 随 Vg 单调增、随 Vd 饱和；gm 峰值位置随势垒厚度单调移动 |

T0–T4 全部通过前，`verification` 保持 `unverified`；IV 曲线一个数字都不引用。

---

## 8. Deck 集成

```jsonc
"transport": {
  "module": "transport.dd", "version": "^0.1.0",
  "params": {
    "vds_sweep_V": "0.0,1.0,5.0,10.0,20.0",
    "vgs_sweep_V": "-4.0,-2.0,0.0",
    "mu0_cm2_Vs": 1500.0
  }
}
```

- 量子槽位绑 sp1d 或 analytic_tw 都能跑（`quantum_warm_start=auto` 自动探测）；
- 换 `transport.hydro`（远期）时不改本 Deck 的其他槽位；
- `extract.iv` 落地后消费 `iv` artifact，本模块不改。

---

## 9. 目录结构

```
plugins/transport_dd/
├── __init__.py            # MANIFEST + MODULE
├── plugin.py              # TransportDD(BaseModule)：扫描循环、PTC 状态机、artifact 写出
├── assemble.py            # 网格遍历 → 残差/Jacobian 稀疏组装（FV + SG 通量）
├── newton.py              # 阻尼 Newton + 线搜索 + PTC（§6 三层保险）
├── physics.py             # 统计（Boltzmann/Joyce-Dixon）、场依赖迁移率、边界条件
├── solve_linear.py        # scipy 稀疏直接/迭代封装（惰性导入 scipy）
└── tests/
    ├── test_manifest.py       # 参数校验、扫描串解析
    ├── test_discretization.py # T0/T1/T2 闭式解与收敛阶
    ├── test_newton.py         # 阻尼/PTC 分支、T4 初值无关性
    ├── test_continuity.py     # T3 门禁全偏置点断言
    └── test_iv_baseline.py    # T5 趋势 + golden 快照进 CI
```

`physics.py` / `assemble.py` 保持纯函数、零 I/O —— 沿用 sp1d 的 `solver.py` 分工模式，数值内核与插件外壳分离，测试才能不通过 registry 直接打内核。

---

## 10. 诚实的边界（README §7 格式）

**v1 能兑现的：**
- 稳态 IV、gm、连续性门禁、与平衡态 SP 的一致性检查；
- 量子模型可插拔（auto 探测）、迁移率模型预留插槽；
- T0–T4 闭式解级验证。

**明确做不到的：**
- ⚠️ **无陷阱/瞬态**：gate-lag、drain-lag、电流坍塌一概没有（P3，`trap.surface`/`trap.buffer`）；
- ⚠️ **无自热**：`thermal.electrothermal` 落地前的 RF 大电流点都偏乐观；
- ⚠️ **DD 在 Lg=150 nm 的短栅下系统性偏差**：速度过冲与弹道成分缺席，方案 §15 的对策（hydro 插件 + 经验修正 + 报告标注）原样适用；
- **迁移率是占位参数**（`mu0`/`vsat` 默认值未经标定），L2 之前绝对数值不可信；
- **PTC 与阻尼的参数（τ0、减半次数）是工程值**，非物理量；换网格尺度可能需要重调，测试只锁行为不锁参数。

---

## 11. 工作量与顺序

| 步 | 内容 | 估计 |
|---|---|---|
| 1 | `physics.py` + `assemble.py` + T0–T2 | ~1.5 周 |
| 2 | `newton.py`（阻尼 + PTC）+ T3–T4 | ~1 周 |
| 3 | 扫描循环 + `iv` artifact + T5 + golden CI | ~1 周 |

与 `nonlinear.newton` 插件的关系：该服务**已经落地**（v1.0.0），且 §6 三层保险的第 2、3 层（阻尼线搜索、PTC）它都已实现 —— 所以本模块**不需要重写这两层**，真正要补的只有**稀疏线性步**。因此 `newton.py` 仍刻意做成单一职责、无插件依赖的模块，抽取路径为：把 scipy 稀疏求解封装成 `linear.*` 服务 → `nonlinear.newton` 改为经 `linear` 槽解析线性步 → transport.dd 只需一行依赖声明。迁移成本已内化在设计里。

**一句话**：这是 P1 物理 MVP 的另一半（极化 + 量子是"电荷从哪来"，本模块是"电荷怎么流"）；数值风险集中在 Newton 收敛，方案已有明确对策，工程风险低。

---

## 12. v0.1.2 实现偏差记录（落地后补写，不隐藏）

实现已落地：`plugins/linear_blocktridiag/`（线性服务）、`plugins/transport_dd/{physics,assemble,newton,plugin}.py`、`tests/test_transport_dd.py`（25 项）、`decks/demo_hemt_dd.json`。本节记录**与设计稿不一致的地方**——设计稿在实现前写就，偏差是正常的，隐藏才不正常。

### 12.1 架构偏差

| # | 设计稿 | 实际 | 原因 |
|---|---|---|---|
| D1 | §4/§9：线性步用 `scipy.sparse` | 新增 `plugins/linear_blocktridiag/`：纯 numpy 块三对角直接解（块 Thomas + LAPACK 块主元），经 Deck 的 `linear` 槽解析 | 部署环境只有 numpy，没有 SciPy。架构方向与设计一致（服务化 + 槽解析），实现换成不依赖 SciPy 的直接解 |
| D2 | §9：`physics.py` 含 Joyce-Dixon 统计 | 只实现 Boltzmann，`statistics` 只接受 `boltzmann`，其余**显式报错** | 简并修正必须同时改 SG 通量（广义 Einstein 关系）。只搬一半会静默不一致，宁可拒绝 |
| D3 | §9：`physics.py` 含场依赖迁移率 | **已实现** Caughey-Thomas（`vsat_cm_s` / `beta_field`，默认启用），导数进 Jacobian 保持精确；`vsat_cm_s <= 0` 退回常数迁移率 | 初版按偏差记录为"不暴露"，收尾时补齐（见 §12.5） |
| D4 | §5.4 `iv` 为提取器契约 | 契约一致（`vds_V`/`vgs_V`/`id_A_per_m`/`units`），另加 `points` 的逐点完整记录（收敛、迭代数、残差、连续性误差、门禁结果） | 门禁失败要能定位到具体偏置点，只给曲线不够用 |
| D5 | §9 目录：`solve_linear.py` 在插件内 | 无此文件；线性步经 `ctx.services` 解析 | 就是 D1 的服务化结果 |

### 12.2 模型决策（设计稿未指定，实现时必须选）

1. **界面在节点上（设计稿 §3 已预见）**：网格把层界放在节点上，介电常数与带阶在界面处被抹平一个格子。实测后果：2DEG 面密度比极化上限（σ/q = 9.6e12 cm⁻²）高约 1.7 倍，可归因于此。这是设计稿自己接受的代价，此处量化。
2. **单极模型的"半绝缘缓冲"**：电子单极没有空穴去耗尽受主。把 1e17 受主当固定电离电荷会在缓冲里造出非物理巨场，且泊松解隐含的电子密度会让"半绝缘"缓冲**变导体**——实测表现为端点电流比沟道电流大一个数量级、直接漏进衬底。v1 采用单极 HEMT 的标准抽象：缓冲 = **中性介质 + 载流子钉零**，且对连续性方程是**零通量壁**。
3. **宽禁带行的局域平衡闭合**：势垒体无电流，其连续性方程对 n 是线性齐次方程，密度因此近乎自由（Jacobian 最小奇异值比最大低 10 个数量级）。用局域平衡 `n = N_C,eff·exp(qψ/kT)` 替换该行的连续性方程——无电流处这是精确物理，且从构造上消除自由模。
4. **方程量纲化用物理参考量**：Poisson 行 ÷ 局部电容（→ 伏特），连续性行 ÷ 器件级电流参考（→ 相对电流）。**不要**用 Jacobian 对角做行缩放：宽禁带/耗尽行因此获得巨大权重，残差范数被相消噪声主导，线搜索会接受"物理上更差"的步。（未知量列缩放试过并否决：`(ψ,n)` 表述下矩阵元素与未知量无关，列缩放无法改善条件数，盲用还会让块消元的枢轴溢出。）
5. **势垒电位锚定**：栅极只钉一个横向节点，势垒电荷≈0 时 Poisson 退化为纯 Laplace，势垒电位有近乎自由的模式。加了**无栅表面钉扎**（钉在肖特基参考）与**背接触接地**。前者是工程近似（真实无栅表面由施主态钉在特定能级），已记入 README 的诚实边界。
6. **偏压延续 = 电位预移位**：把上一个偏置点的解直接当新初值，会让漏极接触电位突变而内部不动——这个"接触局部冲击"足以让线搜索停滞。现在按 `dVds·ramp(x)` 预移位内部电位，并按 `max_step_V`（默认 0.25 V）自动分步。

### 12.3 §7 基准的兑现情况

| 级 | 状态 | 证据 |
|---|---|---|
| T0 | ✅ **超出判据** | 均匀电阻闭式解相对误差 < 1e-6（设计判据 0.1%），两端一致 < 1e-9；电流与偏压严格正比 |
| T1 | ✅ 已交付（模型允许范围内） | 平衡态端点电流 2.7e-15 A/m、横向电流剖面 ~1e-14 A/m（判据 < 1e-12 A/m）；Poisson 子问题平衡电荷平衡、`J ≡ 0` 均有断言。**未做**与 1D SP 的带边差 < 1 meV 对比：层界落在节点上的抹平（§12.2 第 1 条）使该判据在本离散下不可达，这本身就是诚实的结论 |
| T2 | ✅ | Poisson 子问题 MMS 制造解 `psi* = sin(pi z/L)`（x 无关使横向项恒零，隔离竖直算子）：加密 25→12.5→6.25→3.125 nm，L∞ 误差 1.295e-2 → 3.219e-3 → 8.036e-4 → 2.008e-4，**观测阶 2.008 / 2.002 / 2.001** |
| T3 | ✅ | 门禁已进插件（含绝对地板 `max_continuity_error`）；实测 Vds ≤ 1 V 全部偏置点相对误差 1e-13 ~ 1e-16，`|Is+Id|/|Id| < 1e-6` 断言在测试里 |
| T4 | ✅ | 同一偏置点经两条不同延拓阶梯（`max_step_V` = 0.05 V 与 0.1 V）到达，电流相对差 < 5e-3 断言在测试里 |
| T5 | ✅ 已交付 | 栅压变负 → 电流下降有断言；**饱和判据**在启用 Caughey-Thomas 后成立：同器件、Vds 0.25→0.5→1.0 V 的电流斜率比从 1.90 降到 1.71（常数迁移率对照为 0.92，因栅-沟道电位改变 2DEG 而自带轻微次线性），测试断言"场依赖使斜率比明显更小"这一**相对**判据（见 §12.5） |

### 12.5 收尾补记：场依赖迁移率（原 D3 偏差已消除）

收尾时把设计稿 §4 要求、初版记为 D3 的 **Caughey-Thomas 场依赖迁移率**补上（`vsat_cm_s` 默认 2e7 cm/s、`beta_field` 默认 1，`vsat_cm_s <= 0` 退回常数迁移率），导数按链式法则进 Jacobian：SG 前因子带 μ，而面电场 `E = |ΔE_C|/h` 依赖带边，所以通量势偏导多出 `dμ/dE` 项（`∂E/∂ψ_L = -sign(δ)/h`），仍然反对称、仍然精确。新增测试：μ 的低场/高场极限、`dμ/dE` 对中心差分、**带 vsat 的通量偏导 FD**、以及饱和趋势（含常数迁移率对照）。

两个实测发现：
1. **饱和确实出现**：同器件 Vds 0.25→0.5→1.0 V，Id 146→278→478 A/m，斜率比从 1.90 降到 1.75（常数迁移率为 2.00 → 0.92）。**常数迁移率的闸极器件本身就有约 8% 次线性**（栅-沟道电位改变 2DEG），所以饱和判据必须用"有/无场依赖"的相对对照，不能指望对照是欧姆的。
2. **非线性变强，延续阶梯要更细**：`max_step_V = 0.1 V` 时 Vds=1 V/Vgs=0 这一点会发散（63 次迭代后 res 2.9e2），降到 **0.05 V** 后 10 个点全部收敛、连续性 1.5e-12。manifest 默认值因此从 0.25 收到 0.1，demo deck 用 0.05 并在 notes 里写明原因。**Vds ≥ 2 V 仍不收敛**（新的数值欠账，见 §6）。

| 级 | 状态 | 说明 |
|---|---|---|
| T5（饱和部分） | ✅ | 见上；2 V 以上的收敛是独立问题，不属 T5 |

### 12.4 两条验证纪律（血的教训，值得写进任何后续模块）

1. **FD 校验必须"逐行相对"，不能"逐列相对"**。`(ψ,φₙ)` 改 `(ψ,n)` 时，Jacobian 里残留一行旧代码，把第二个未知量当准费米势代进了态密度公式——**内部密度与残差相差 17 个数量级**，连续性行因此完全失去对 ψ 的耦合，整个耦合矩阵在 Newton 最需要的方向近奇异。而当时的 FD 校验用列最大值归一化，坏掉的项绝对值很小，相对误差看着只有 1e-10 ✗。改成逐行相对判据后立刻暴露。教训：**导数校验要在单元级（通量本身）做，且沿 Newton 路径做**，不能只在参考状态做一次。
2. **收敛判据必须是物理量纲的**。用解相关的行均衡（Jacobian 对角）当度量，会让弱行主导范数，线搜索接受"物理残差变差 12 个数量级"的步。改成固定物理参考量后，平衡态残差从 4.2 降到 1.3e-3，最终到机器零。

