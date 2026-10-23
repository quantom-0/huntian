# DFT 材料计算模块 —— 插件设计

> 版本：v0.2（第一阶段 **table 后端已落地**；`qe_*` 后端未实现）
> 定位：Huntian 插件 `materials.dft`，为标记 `unverified` 的材料库提供一条**自主、可溯源、带不确定度**的第一性原理验证与替代通道
> 路线图：README §10 下一步**第 5 项**（第一阶段 ✅；第二阶段 `qe_local` + D0–D3 重算、第三阶段 `qe_slurm` + SQS 扫描 ⏳）
> 架构前提：沿用 README §3 已验证的替换路径（`physics.quantum.sp1d` 模式）——物理模块一行不改；例外：Deck 需新增 `materials` 槽（见 §7 注）

## 0. 实现状态（v0.2 新增）

**已落地（2026-10-05，385 测试全绿）**：

- `table` 后端完整实现：`huntian/plugins/materials_dft/`（`plugin.py` / `table_io.py` / `qe_driver.py` / `slurm.py`），qe_* 为惰性导入桩，table 模式第三方依赖为零（子进程级测试保证）；
- 查找表资产 `huntian/plugins/materials_dft/data/algan_gan.json`（v1.0.0）：GaN/AlN 端元共 10 项常数 + PBE 原始带隙，逐值标注文献出处与不确定度；**文献转录、非本所计算**，全部条目 `convergence: null`（诚实声明，不伪造收敛记录）；
- 精度红线全部在代码里：`eg_policy=reject_pbe` 默认硬拒绝（validate 返回 issue、阶段拒跑）；scissor 采用参考值并把修正量写入 `dft_eg_shift_eV`；数值设置与表记录不符默认拒绝（`stale-cache`），`allow_stale_cache=true` 才放行并写入 assumptions；无收敛记录的条目仅在显式 `override_policy` 下采用、逐行标注 `converged: false`；
- `dft_provenance` artifact：backend/functional/numerics（含表内容哈希）/cache 计数/逐条目溯源/override_report/assumptions；
- D0–D3 基准进 golden CI（`tests/golden/dft_benchmarks_d0_d3.json` 检出比对 + 容差带断言）；当前实测：晶格 ≤0.35%、弹性 ≤2.8%、极化 0%、带隙 ≤3.1%；
- Deck 验收：`decks/demo_hemt_dft.json` 一行绑定，`test_hotswap_materials.py` 断言 polarization/sp1d/extract 零改动、流水线仅增一个 materials 阶段、两次运行指纹一致。

**与 v0.1 设计稿的偏差（有意为之）**：

1. **Deck 需新增 `materials` 槽**（`deck.py` ModulesSpec + SLOT_ORDER 首位的 4 行小改）：v0.1 声称"只加插件目录，核心一行不改"，但现有 Deck 没有 materials 绑定面；与 P1-2 为 sp1d 新增 ServiceRegistry + 服务槽同理，属于"为插件开槽"的一次性核心加法。物理模块（polarization/sp1d）确实一行未改。
2. **测试放 `tests/` 而非插件目录内**：`run_tests.py` 只发现 `tests/`，且全部测试同处一处才能进统一的 golden CI。
3. **`override_policy` 语义澄清**：`reference_wins` / `dft_only_gapless` 对**带隙同样生效**（设计稿只说"每个材料属性"，实现把带隙纳入统一策略，eg_policy 只管"采用哪个泛函的隙"）。
4. **convergence 缺失的"降权"实现为**：显式 opt-in 采用 + `converged: false` 逐行标注 + assumptions 声明——而不是静默采用，也不是直接拒用（表资产里一条收敛记录都没有时直接拒用会使通道空转）。

**未变**：qe_local / qe_slurm 驱动、SQS 合金扫描（D4）、缺陷/无序修正、热学参数不做 DFT 的边界声明。

---

## 1. 设计决策（为什么是这个形态）

### 1.1 三层结论

| 问题 | 决策 | 理由 |
|---|---|---|
| 手写 DFT 内核？ | **否** | 平面波 SCF + 赝势 + 应力/声子是数月级工程，且验证成本极高；与本项目"器件仿真"主目标无关 |
| 计算引擎？ | **ASE + Quantum ESPRESSO**（驱动适配器，非数值内核） | 插件自带依赖，符合"核心零依赖、插件自带依赖"哲学（`physics.quantum.sp1d` 需要 numpy 即先例） |
| 运行时形态？ | **离线计算 + 查表复用** | DFT 单点成本（HSE 超胞小时级）与器件仿真（亚秒级）差 5 个数量级以上，不可能内嵌每次仿真；SimContext 指纹**故意不序列化 artifact 数值**，所以缓存必须在插件层显式实现（见 §6） |

### 1.2 与材料库的关系

`huntian/data/materials.json` 的 `provenance.verification_status` 当前为 `unverified`。本模块的产出**不改变该文件**，而是：

- 运行时把 DFT 计算结果作为**顶层覆盖层**合并进 `MaterialDB`（`MaterialDB.merge(override=True)` 机制已存在）；
- 每个被覆盖的常数携带 `source = "DFT (functional, pseudopotential, date)"` 与独立的验证状态，下游溯源完整可见。

这是"物理模型为形、数据可换可审"这一既有原则的延伸（见 `huntian/materials.py` 模块 docstring）。

---

## 2. 模块总览

**一个插件、三种运行模式**，由参数 `backend` 切换：

| backend | 行为 | 依赖 | 用途 |
|---|---|---|---|
| `table`（默认） | 读取已算好的查找表 JSON，零外部依赖 | 无 | CI、无 DFT 环境、生产仿真 |
| `qe_local` | 本机驱动 Quantum ESPRESSO（经 ASE） | ase、pw.x | 开发与小体系重算 |
| `qe_slurm` | 生成作业并轮询队列（SQS 超胞等重任务） | ase + 集群 | 合金与应变扫描 |

三种模式产出**完全相同的 artifact 结构**，Deck 与下游模块无感 —— 这正是"计算后端可插拔"概念（方案 §6）在插件内部的最小复刻。

```
module_id:    materials.dft
category:     ModuleCategory.PHYSICS
capabilities: ("materials", "materials.dft", "materials.validation")
requires:     ("structure",)
produces:     ("materials", "dft_provenance")
verification: partially-verified（上线时的自我声明；以 §8 基准通过情况为准）
```

> **槽位语义**：`materials` artifact 在现有 demo Deck 中由核心装配提供；本模块是 `materials` 槽位的**替换实现**——与 sp1d 替换 `analytic_tw` 同构，Deck 里改一行绑定、钉住版本范围即可（§7）。Pipeline 执行顺序由 manifest 数据依赖自动推导，无需改动。

---

## 3. Manifest 定义（落地代码草稿）

与 `huntian/registry.py` 实际签名一致；`ParamSpec` 的 `source` / `calibratable` 字段照常携带出处，供 LLM 层与拟合白名单消费。

```python
MANIFEST = ModuleManifest(
    module_id="materials.dft",
    module_version="0.1.0",
    category=ModuleCategory.PHYSICS,
    description=(
        "First-principles (DFT) material parameter provider: computes or looks up "
        "AlGaN/GaN material constants (polarization, elasticity, band edges, effective "
        "mass) from plane-wave DFT via an ASE/Quantum ESPRESSO driver, or serves them "
        "from a precomputed table. Drop-in replacement for the builtin materials slot."
    ),
    capabilities=("materials", "materials.dft", "materials.validation"),
    requires=("structure",),
    produces=("materials", "dft_provenance"),
    verification="partially-verified",
    author="Huntian",
    tags=("dft", "first-principles", "materials", "validation"),
    parameters=(
        # ---- 运行模式 ---------------------------------------------------
        ParamSpec.choice(
            "backend", ("table", "qe_local", "qe_slurm"),
            default="table",
            doc="table: serve precomputed JSON; qe_*: drive Quantum ESPRESSO",
        ),
        ParamSpec.string(
            "table_path", "",
            doc="lookup-table path; empty string selects the bundled table",
        ),
        # ---- 泛函与精度分级（红线见 §5）--------------------------------
        ParamSpec.choice(
            "functional", ("pbe", "hse06"),
            default="hse06",
            doc="exchange-correlation functional; PBE gaps are systematically "
                "underestimated and must not feed Eg directly (see eg_policy)",
        ),
        ParamSpec.choice(
            "eg_policy", ("reject_pbe", "scissor_to_reference"),
            default="reject_pbe",
            doc="reject_pbe: fail validation if functional=pbe and band gaps are "
                "requested; scissor_to_reference: shift PBE gap to reference Eg "
                "and record the shift in dft_provenance.assumptions",
        ),
        ParamSpec.string(
            "reference_eg_source", "experiment",
            doc="where scissor reference gaps come from: 'experiment' or a "
                "materials.json property key",
        ),
        # ---- 计算设置（qe_* 模式）----------------------------------------
        ParamSpec.float(
            "ecut_wfc_ry", 60.0, unit="Ry", minimum=20.0, maximum=150.0,
            doc="plane-wave kinetic-energy cutoff; convergence-checked in D-benchmarks",
        ),
        ParamSpec.float(
            "kspacing_1_A", 0.20, unit="1/angstrom", minimum=0.05, maximum=0.5,
            doc="k-point density (Monkhorst spacing); convergence-checked",
        ),
        ParamSpec.string(
            "pseudopotential_family", "SSSP/PBEsol/efficiency",
            doc="pseudopotential set with version, recorded in provenance",
        ),
        ParamSpec.float(
            "strain_step", 0.01, unit="1", minimum=0.001, maximum=0.05,
            doc="finite-strain step for piezoelectric coefficient extraction",
        ),
        ParamSpec.boolean(
            "spin_orbit", False,
            doc="include SOC in band-structure runs (raises cost ~4x)",
        ),
        # ---- 合金建模 ----------------------------------------------------
        ParamSpec.integer(
            "sqs_size", 64, minimum=8, maximum=512,
            doc="supercell size for special-quasirandom-structure AlGaN runs",
        ),
        ParamSpec.integer(
            "sqs_seed", 1, minimum=0,
            doc="SQS generation seed; enters the cache key, so tables stay reproducible",
        ),
        # ---- 缓存 --------------------------------------------------------
        ParamSpec.string(
            "cache_dir", "dft_cache",
            doc="result cache directory; key = hash(structure, functional, "
                "pseudopotential, ecut, kspacing, sqs_seed)",
        ),
        ParamSpec.boolean(
            "allow_stale_cache", False,
            doc="if True, serve cache entries computed with different numerical "
                "settings (recorded in provenance); False refuses with a clear error",
        ),
        # ---- 产出控制 ----------------------------------------------------
        ParamSpec.choice(
            "override_policy", ("dft_wins", "reference_wins", "dft_only_gapless"),
            default="dft_wins",
            doc="which source supplies each material property in the merged DB",
        ),
    ),
)
```

**刻意不设 `calibratable=True` 的任何参数**：DFT 设置是数值方法选择，不是器件拟合量；若允许拟合回路调整 `ecut` 之类，等于让优化器改网格换精度，违背"数字只能来自求解器"的红线（方案 §9.2）。

---

## 4. Artifact 契约

### 4.1 `materials`（下游消费，契约不变）

与现有 `MaterialDB` 完全同构 —— 本模块加载内置库后叠加 DFT 覆盖层，`polarization.ambacher`、`physics.quantum.sp1d` **零改动**（它们只调 `materials.get(name).numeric(key)`）。

覆盖层条目的 `props` 键与 `materials.json` 对齐，新增键以 `dft_` 前缀隔离：

| 键 | 单位 | 说明 | 状态 |
|---|---|---|---|
| `a_lattice_A` / `c_lattice_A` | Å | 平衡晶格常数（弛豫后） | 可覆盖 |
| `C13_GPa` / `C33_GPa` | GPa | 弹性常数（应力-应变拟合） | 可覆盖 |
| `e31_C_m2` / `e33_C_m2` | C/m² | 压电系数（有限应变 × Born 有效电荷） | 可覆盖 |
| `p_sp_C_m2` | C/m² | 自发极化（Berry 相位） | 可覆盖，标注方法 |
| `m_eff` | m₀ | Γ 谷有效质量（能带拟合） | 可覆盖 |
| `Eg_eV` | eV | 带隙 | **受 `eg_policy` 约束**（§5） |
| `dft_eg_uncorrected_eV` | eV | 泛函原始带隙（无论是否采用） | 只附加，不消费 |
| `dft_eg_shift_eV` | eV | scissor 修正量（若启用） | 只附加 |
| `dft_uncertainty` | — | 每项的 `{value, lo, hi, kind}`（见 4.2） | 只附加 |

### 4.2 `dft_provenance`（新 artifact，溯源与验证用）

```python
{
    "backend": "table" | "qe_local" | "qe_slurm",
    "functional": "pbe" | "hse06",
    "pseudopotential_family": "SSSP/PBEsol/efficiency@1.3.0",
    "numerics": {"ecut_wfc_ry": 60.0, "kspacing_1_A": 0.20, "sqs_seed": 1},
    "cache_hits": 12, "cache_misses": 3,
    "entries": {
        "GaN": {
            "source": "DFT/PBE/SSSP@1.3.0/2026-10",
            "verification": "partially-verified",
            "uncertainty": {
                "p_sp_C_m2": {"kind": "convergence", "lo": -0.041, "hi": -0.036},
                "Eg_eV":     {"kind": "systematic", "note": "PBE; not adopted"}
            },
            "assumptions": [...],   # 与 sp1d 风格一致，见下
        },
        ...
    },
    "override_report": [
        {"property": "p_sp_C_m2", "material": "GaN",
         "builtin": -0.034, "dft": -0.0385, "adopted": "dft", "delta_pct": 13.2},
        ...
    ],
}
```

`assumptions` 沿用 sp1d 的诚实边界风格，至少包含：

- 计算泛函及其对带隙的系统性影响（PBE 低估 GaN 带隙 ~1.7 eV vs 实验 3.4 eV 是**预期行为**，不是 bug）；
- 合金值来自 SQS 超胞 + Vegard 参考，存在统计不确定度；
- 未含温度修正（DFT 是 0 K；`thermal_exponent` 等仍来自内置库）；
- 未含缺陷/无序对极化的修正。

---

## 5. 精度红线（必须在代码里，不是文档里）

1. **`eg_policy = "reject_pbe"`（默认）**：`functional=pbe` 且需要带隙时，`validate()` 返回硬 issue，Deck 上下文校验拒绝运行。带隙被 sp1d 的 `delta_ec_ratio × ΔEg` 和极化模块消费，静默喂入低估带隙是**物理错误**。
2. **`scissor_to_reference`**：允许 PBE + 刚性平移到参考带隙，修正量必须写入 `dft_eg_shift_eV` 与 assumptions，且 `verification` 至多标 `partially-verified`。
3. **收敛性证据**：`validate()` 检查查找表（或缓存）是否带收敛测试记录（ecut/k 点双点外推）；缺失时降权处理并在 `override_report` 标注 `converged: false`，不静默使用。

---

## 6. 缓存与可复现

SimContext 指纹只覆盖"参数 + 模块版本"，**故意不含 artifact 数值**（README §3.2），因此 DFT 结果的复用由插件自己的缓存负责：

```
cache_key = sha256(
    structure(组分x, 应变状态) + functional + pseudopotential_family
    + ecut_wfc_ry + kspacing_1_A + sqs_seed + spin_orbit
)[:16]

dft_cache/<cache_key>.json   # 单点结果 + numerics + convergence record
```

- `allow_stale_cache=False`（默认）：缓存键不匹配即重算或报错（table 模式下报错并提示重算路径）——**数值设置不同而结果"碰巧在库里"是最危险的复用**。
- 查找表（随插件分发的 `huntian/plugins/materials_dft/data/*.json`）本身视为**版本化数据资产**：进入 golden CI（方案 §11.3），表内容变更触发回归。

---

## 7. Deck 集成（验收形态）

复刻 sp1d 的"一行替换"验收（README §4.5）：

```diff
   "materials": {
-    "source": "builtin"
+    "module": "materials.dft", "version": "^0.1.0",
+    "params": { "backend": "table", "eg_policy": "reject_pbe" }
   }
```

验收断言（对齐 `test_hotswap_quantum.py` 的写法）：

- `polarization.ambacher` 与 `physics.quantum.sp1d` **一行未改**；
- pipeline 槽位与顺序不变；
- `dft_provenance.override_report` 非空且每个覆盖项可追溯；
- 同一 Deck 两次运行的 SimContext 指纹一致。

---

## 8. 验证基准（DFT 侧自己的 V&V，L 阶梯的支线）

DFT 模块验证的是**它与实验/文献的一致性**，不替代 L0–L2（那是求解器对解析/测量的验证）。定义 D0–D3 四级：

| 级 | 基准 | 闭式/参照 | 通过判据 |
|---|---|---|---|
| D0 | 端元晶格常数 | 实验值（GaN a=3.189 Å, c=5.185 Å） | 相对误差 < 1% |
| D1 | 弹性常数 C13/C33 | 实验/Raman 文献 | < 10% |
| D2 | 自发极化 p_sp | Ambacher 值（GaN −0.034, AlN −0.090 C/m²） | < 15%，并记录方法差异（Berry 相位 vs 经验式） |
| D3 | 带隙 | 实验（GaN 3.42 eV） | HSE 修正后 < 5%；**PBE 原始值只记录不采用** |
| D4 | AlGaN bowing（Eg, p_sp） | SQS 超胞 vs Vegard+bowing 文献 | 趋势一致，偏差 < 文献散布 |

原则与 `test_quantum_sp1d.py` 相同：**断言不变量、趋势与误差界，不钉魔法数字**；每个 D 级基准输出 golden 结果进 CI。

---

## 9. 目录结构

```
huntian/plugins/materials_dft/
├── __init__.py           # MANIFEST + MODULE（入口）
├── plugin.py             # DftMaterials(BaseModule)：生命周期 / 缓存 / 合并
├── table_io.py           # 查找表读写与结构校验
├── qe_driver.py          # ASE/Quantum ESPRESSO 驱动（backend=qe_* 时才导入）
│                         #   导入失败 → 该 backend 报清晰错误，table 模式不受影响
├── slurm.py              # 作业生成与轮询（可选，qe_slurm 专用）
├── postprocess.py        # 应力/介电/Born 电荷 → 材料常数 + 不确定度
└── tests/
    ├── test_manifest.py      # 参数校验、eg_policy 红线、calibratable 为空
    ├── test_table_backend.py # 查表、缓存键、stale 拒绝
    ├── test_overrides.py     # MaterialDB 合并、dft_ 前缀键、override_report
    └── test_benchmarks_d0_d3.py  # §8 基准（golden 对照）
```

`qe_driver.py` / `slurm.py` 的**惰性导入**是硬要求：保证 `table` 模式下插件的第三方依赖为零，与"JSON Deck 在任何环境都能跑"（README §2）一致。

---

## 10. 诚实的边界（照 README §7 的格式写）

**能兑现的：**
- 端元与合金材料常数的 DFT 重算、带不确定度的覆盖、完整溯源 —— 直接攻击"材料常数未经独立验证"这个已声明的最大软肋；
- 三种 backend 同一 artifact 契约，CI 可无 DFT 环境运行；
- 对下游完全透明（sp1d 已验证过的替换路径）。

**明确做不到的（写进 assumptions，不粉饰）：**
- ⚠️ **DFT 是 0 K、理想晶体、无缺陷**：真实 MOCVD 材料的位错、杂质、组分涨落造成的参数离散，DFT 给不出，只能以不确定度形式吸收；
- ⚠️ **PBE 带隙不可直接用于器件**（见 §5 红线）；HSE/GW 成本高 2–3 个数量级，合金扫描基本只能 PBE + 修正；
- ⚠️ **有效质量拟合依赖 k·p/有限差分能带处理**，Γ 谷附近尚可，远离 Γ 点不做承诺；
- **热学参数（热导率、热指数）不做 DFT**：需要声子/BTE，属于另一个量级的工作，继续用内置库；
- **不解决 L2**：Hall 对标仍是最终裁决，DFT 只是给 L2 提供一个比"转录文献"更硬的先验。

---

## 11. 路线图位置与工作量

| 项 | 估计 |
|---|---|
| 阶段 | P5 之后的支线；或更早以 `backend=table` 先行（查表本身就是无依赖插件） |
| 顺序 | 先 `table` 后端 + 一份手工整理的 AlGaN/GaN 查找表（~1 周）→ 再 `qe_local` 驱动 + D0–D3 基准（~2 周）→ `qe_slurm` 与 SQS 合金扫描（~1 周） |
| 依赖 | ase、quantum-espresso（仅 qe_* 模式）；CI 只需 table 模式 |

**一句话**：这个模块的全部新意只在"数据从哪来"，架构上它是对已验证替换路径的第三次复用 —— 风险集中在 §5 的精度红线与 §8 的基准纪律，而不是软件工程。
