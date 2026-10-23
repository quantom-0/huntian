# Huntian 模块化技术文档（LaTeX 源）

> 本目录是《Huntian：GaN 射频 HEMT 轻量化 TCAD 内核 —— 模块化技术文档》的
> LaTeX 源码。**一个插件对应一个模块章**，架构与代码仓库 `huntian/plugins/` 同构。

---

## 1. 构建

需要 TeX Live（本仓库验证于 TeX Live 2025）与 XeLaTeX。

```powershell
cd docs/tex
.\build.ps1              # 完整构建 + 日志体检
.\build.ps1 -Clean       # 彻底清理 build/
.\build.ps1 -Quiet       # 不打印 latexmk 的过程输出
```

等价的直接命令：

```powershell
latexmk -pdfxe           # 读 latexmkrc，自动跑 xelatex -> biber -> 索引 -> 收敛
```

产物：`build/main.pdf`。中间文件全部落在 `build/`，源码目录保持干净。

`build.ps1` 在构建后从 `build/main.log` 里数出错误、未定义引用、未定义引文、
重复标签、溢出盒，并据此给出"干净 / 不干净"的结论与退出码 ——
**结论由日志计数得出，不靠人眼判断**。

---

## 2. 目录结构

```
docs/tex/
├── main.tex                    根文件：只装宏包、按序 \input、印全局部件
├── latexmkrc                   构建配置（xelatex + biber + 索引）
├── build.ps1                   一键构建 + 日志体检
├── refs.bib                    全局唯一参考文献库
├── preamble/
│   ├── packages.tex            宏包加载与引擎设定
│   ├── style.tex               版式、字体、页眉页脚、题注、断行宽容度
│   ├── math.tex                全局数学宏（符号宏的唯一登记处）
│   ├── refs.tex                交叉引用宏（替代 cleveref，见 §4）
│   ├── modulecard.tex          模块卡片 + 可标定参数表
│   └── verify.tex              验证基准表 + 边界清单
├── frontmatter/
│   ├── abstract.tex            摘要
│   ├── notation.tex            全局符号表（唯一真源）
│   ├── conventions.tex         全局约定（单位 / 坐标 / 验证表述纪律）
│   └── outline.tex             文档结构总览（模块 ↔ 章号对照表）
├── overview/                   ★ 总论 5 章（第 1–5 章）
│   ├── intro.tex               引言与文档范围
│   ├── architecture.tex        系统架构：模块即插件、服务即插件
│   ├── dataflow.tex            数据流与 artifact 契约
│   ├── verification.tex        验证方法学
│   └── module_index.tex        模块索引与依赖关系
├── modules/                    ★ 九个模块章（第 6–14 章），一个插件一个文件
│   ├── mesh_structured2d.tex         第 6 章
│   ├── materials_dft.tex             第 7 章
│   ├── polarization_ambacher.tex     第 8 章
│   ├── quantum_analytic_tw.tex       第 9 章
│   ├── quantum_sp1d.tex              第 10 章（样板章，P0）
│   ├── transport_dd.tex              第 11 章
│   ├── extract_fom.tex               第 12 章
│   ├── nonlinear_newton.tex          第 13 章（服务模块变体）
│   └── linear_blocktridiag.tex       第 14 章（服务模块变体）
├── appendix/                   ★ 附录 A–D
│   ├── benchmarks.tex          闭式解基准汇总
│   ├── parameters.tex          可标定参数总表
│   ├── contracts.tex           版本、指纹与 artifact 键矩阵
│   └── debts.tex               已知边界与欠账清单
└── build/                      构建产物（已 gitignore）
```

---

## 3. 模块化约定（改文档前请先读这一节）

1. **一个插件 = 一个文件 = 一章。** 增删模块只需在 `main.tex` 里增删一行
   `\input{modules/...}`，不要在别处写正文。

2. **章号是冻结契约。** 第 6–14 章的编号不因成文先后而改变，以免交叉引用
   失效。尚未成文的章节由 `frontmatter/future_labels.tex` 声明前向标签，
   它只推进章计数器、不排版内容、不写目录。**写成一章时，必须同时做两件事**：
   在 `main.tex` 放入真正的 `\input`，并从 `future_labels.tex` 删除对应的
   `\futurechapter` 行 —— 否则章号会重复计数。构建日志里每一条
   `[huntian-doc] forward-label: ...` 都对应一个"已编号但尚未成文"的章。

3. **每章固定七节骨架**（服务模块用变体）：

   | 节 | 内容 |
   |---|---|
   | x.1 | 模块契约（`modulecard` 开头，字段顺序固定） |
   | x.2 | 控制方程与物理假设 |
   | x.3 | 离散化与数值算法 |
   | x.4 | 耦合求解（仅耦合型模块有） |
   | x.5 | 验证与基准（`veriftable`：基准 \| 判据 \| 实测） |
   | x.6 | 误差来源与适用边界（`bounds`） |
   | x.7 | 接口契约与小结 |

4. **文件头带机器可读标识**：每个模块章开头以注释给出 `module_id`、
   `module_version`、`plugin_dir`、`source_files`、`tests`。
   这是后续"文档 ↔ MANIFEST 机械比对"的锚点。

5. **验证数字的表述纪律**：`判据`列写测试断言的门槛，`实测`列写文档编写时
   实际观测到的值；仅有趋势正确或无独立基准者，一律进 §x.6，不得写入
   `实测`列。详见 `frontmatter/conventions.tex`。

---

## 4. 已知的排版约束（踩过的坑，勿回退）

这些都不是风格偏好，而是会让构建**硬失败**的坑。每条都在源码注释里复述过。

### 4.1 `cleveref` 与 `ctex` 不兼容 → 用自带引用宏

最小复现：

```latex
\documentclass{ctexbook}
\usepackage{cleveref}
\begin{document}\cref{s1}\par\section{X}\label{s1}\end{document}
```

第二遍编译必报 `Extra \endcsname`。已确认与 `hyperref` 无关
（`book` + `hyperref` + `cleveref` 正常，`ctexbook` + `cleveref` 失败）。
`zref-clever` 与 ctex 兼容，但不提供中文引用名。

因此本目录用 `preamble/refs.tex` 里一组\emph{按类型区分}的宏，建立在
`\hyperref` + `\ref*` 之上：

| 宏 | 输出 |
|---|---|
| `\crefchap{...}` | 第 10 章 |
| `\crefsec{...}` | 第 10.2 节 |
| `\crefeq{...}` | 式 (3) |
| `\creftab{...}` | 表 2 |
| `\creffig{...}` | 图 1 |
| `\crefapp{...}` | 附录 A |

正文中禁止写死"见 §4.2"这类字面编号。

### 4.2 自定义环境里不能用 `\begin{tabularx}` / `\begin{table}`

环境的 begin 代码与 end 代码跨越 `\newenvironment` 边界，凡是"配对敏感"的
构造都会踩到 LaTeX 的 `\@currenvir` 记账：

* `\begin{tabularx}` —— `tabularx.sty:53` 是
  `\def\tabularx#1{\edef\TX@{\@currenvir}...}`，它把**当前环境名**存进
  `\TX@`，再一路扫描到 `\end{<那个名字>}`。在自定义环境里当前环境名是
  `modulecard`/`veriftable`，于是它去吞 `\end{modulecard}`，把整章内容
  吸进表格。**改用 `\tabular` ... `\endtabular` 原语**，列宽用 `\dimexpr`
  精确算到 `\textwidth`（因此不需要 tabularx 的自适应 `X` 列）。
* `\begin{table}` / `\@float` —— 同样的边界问题，且症状更隐蔽
  （报 `Illegal pream-token (\bottomrule)`）。**改用非浮动表格 +
  `\captionof`**（`caption` 宏包）。附带好处：表就地排版，不会漂到别页。
* `\begin{center}` —— 改用 `\centering` 声明。

### 4.3 `\tabularx` 与 `\multicolumn` 不兼容

`\multicolumn` 会破坏 tabularx 的 `X` 列宽度记账，症状是**很后面的下一个
表格**报 `Illegal pream-token (\csname)`，定位极难。本目录已不使用 tabularx，
普通 `tabular` 里 `\multicolumn` 是安全的（`\vnote` 就靠它排表注）。

### 4.4 表注必须自成一行

在 `\tabular` 里，`\\` 之后紧跟的下一个记号若不是 `\noalign`/`\omit`，
LaTeX 就认为新行已经开始，随后的 `\bottomrule` 便报 `Misplaced \noalign`。
因此 `\vnote` 自己排成一行、并以 `\\` 收尾。

### 4.5 `\detokenize` 里不要写 `\_`

`\pid{}`/`\akey{}`/`\fpath{}` 用 `\texttt{\detokenize{...}}` 处理含下划线的
标识符，**直接写裸下划线即可**。若写成 `\_`，`\detokenize` 会把它还原成
字面的"反斜杠 + 下划线"印出来（`physics.quantum.analytic\_tw`）。

### 4.6 自带上下标的宏不能再加角标

`\Ec`=`E_{\mathrm c}`、`\epsr`=`,`、`\meff`=`m^{*}` 这类宏本身已带角标，
再写 `\Ec_i` 或 `\meff^{b}` 会触发 `Double subscript`/`Double superscript`，
而 `\meff_i^{2}` 同样报 `Double superscript`（同一基上两个上标）。
网格节点索引用 `\Eci{i}`、`\epsri{i}`、`\Ecifl{i}`（逗号下标形式）。
新增带角标的物理量宏时，请在 `preamble/math.tex` 一并登记。

### 4.7 构建脚本需要 UTF-8 BOM

本机默认 shell 是 Windows PowerShell 5.1，它把无 BOM 的 UTF-8 文件按 ANSI
读取，会串掉脚本里的中文并导致解析错误。`build.ps1` 因此带 UTF-8 BOM。
另外脚本内**不要**设 `$ErrorActionPreference='Stop'`：latexmk / xelatex /
biber 都往 stderr 写正常进度信息，在 Stop 模式下会被当成致命错误抛出。

---

## 5. 参考文献

`refs.bib` 是全局唯一来源，正文 `\cite{}` 全局累积，`\printbibliography`
只出现一次。样式为 `gb7714-2015`（GB/T 7714-2015 国标），后端 `biber`。

**收录纪律**（写在 `refs.bib` 文件头，请勿放宽）：

1. 每条必须给出 DOI 或 ISBN，或明确标注其可核验载体；
2. 条目元数据一律经 Crossref / 出版商记录核对，**不凭记忆书写**；
3. 每条的核验方式与日期记在 `note` 字段，使读者可复查。

未核验的条目放在 `refs.bib` 末尾的"待核验队列"注释里，不得进入正文。

> **已发现的上游引用错误。** 仓库多处把 Ambacher 等 2002 年的工作记为
> *J. Appl. Phys.* **92**, 3202 (2002)。该卷页无法在 Crossref 的 JAP 2002 年
> 记录中复现；该文实际发表于 *J. Phys.: Condens. Matter* **14**, 3399 (2002)，
> DOI `10.1088/0953-8984/14/13/302`。本文档统一采用核验后的出处，
> 并在 `refs.bib` 的 `ambacher2002` 条目 `note` 中记录此事。

---

## 6. 索引

`imakeidx` + `makeindex`（latexmk 自动驱动），双栏。

中文条目一律带 ASCII 排序键，因此排序完全由文档控制、与索引引擎无关：

```latex
\index{3method@方法!Newton--Raphson}
\index{1concept@概念!2DEG@二维电子气}
```

排序键前缀决定**一级分类的出现次序**：

| 前缀 | 分类 |
|---|---|
| `1concept@` | 概念 |
| `2qty@` | 物理量 |
| `3method@` | 方法 |
| `4module@` | 模块 |
| `5artifact@` | 数据通道 |
| `6param@` | 参数 |

新增条目请沿用这六类；确需新分类时，在 `1`–`6` 之外另取前缀并在本表登记。

> **与 P0 计划的偏差（如实记录）。** 原计划用 `xindy` + 中文排序规则
> （`index.xdy`）。实际落地时改为 `makeindex`：由于每条索引都带 ASCII
> 排序键，两个引擎给出的**次序完全相同**，而 xindy 需要额外维护
> `.xdy` 样式文件、且中文仍需自定义 sort-rule 才有拼音序。
> 若将来要做真正的拼音排序，`imakeidx` 的 `program=texindy` 选项与
> `index.xdy` 是现成的接入点。

---

## 7. 当前进度

| 部分 | 章 | 状态 |
|---|---|---|
| **总论** | **1 引言与文档范围** | **已成文** |
| **总论** | **2 系统架构：模块即插件、服务即插件** | **已成文** |
| **总论** | **3 数据流与 artifact 契约** | **已成文** |
| **总论** | **4 验证方法学** | **已成文** |
| **总论** | **5 模块索引与依赖关系** | **已成文** |
| 网格与材料 | **6 `mesh.structured2d` v0.1.0** | **已成文** |
| 网格与材料 | **7 `materials.dft` v0.1.0** | **已成文** |
| 极化与量子 | **8 `physics.polarization.ambacher` v0.1.0** | **已成文** |
| 极化与量子 | **9 `physics.quantum.analytic_tw` v0.1.0** | **已成文** |
| 极化与量子 | **10 `physics.quantum.sp1d` v1.1.1** | **已成文（样板章，P0）** |
| 输运与提取 | **11 `transport.dd` v0.1.0** | **已成文** |
| 输运与提取 | **12 `extract.fom` v0.1.0** | **已成文** |
| 数值服务 | **13 `nonlinear.newton` v1.0.0** | **已成文** |
| 数值服务 | **14 `linear.blocktridiag` v0.1.0** | **已成文** |
| **附录** | **A 闭式解基准汇总** | **已成文** |
| **附录** | **B 可标定参数总表** | **已成文** |
| **附录** | **C 版本、指纹与 artifact 键矩阵** | **已成文** |
| **附录** | **D 已知边界与欠账清单** | **已成文** |
| 参考文献 / 索引 | — | 已可用（24 条库，24 条被引；索引 122 条） |

**全文 19 章 + 4 附录全部成文**，前向标签机制已删除（`frontmatter/future_labels.tex`
不再存在）——所有章号由真实章节产生。

正文中的数字全部可在仓库的测试套件或基准记录中回查；本文档当前版本对应的
仓库测试规模为 **426 个用例，全部通过**（`python run_tests.py`）。

### 每阶段的构建体检记录

| 阶段 | 内容 | 页数 | 错误 | 未定义引用 | 未定义引文 | 溢出盒 |
|---|---|---|---|---|---|---|
| P0 | 骨架 + 第 10 章样板章 | 32 | 0 | 0 | 0 | 0 |
| P1 | 第 6–9 章 | 76 | 0 | 0 | 0 | 0 |
| P2 | 第 11–12 章 | 96 | 0 | 0 | 0 | 0 |
| P3 | 第 13–14 章（九个模块章齐） | 116 | 0 | 0 | 0 | 0 |
| **P4** | **总论 1–5 章 + 附录 A–D（全文齐）** | **164** | **0** | **0** | **0** | **0** |

构建结论由 `build.ps1` 从 `build/main.log` 计数得出，不靠人眼判断。

---

## 8. 后续阶段的注意事项（写给下一阶段的自己）

1. **`materials.dft` 的 qe_* 后端未实现**，第 7 章已如实标注。若将来落地，
   第 7 章需要新增"相 2"小节并更新 §7.6 的边界清单。
2. **界面落位这条线索已闭环**：第 6 章（落节点）→ 第 10 章（落面）→
   第 11 章（量化 1.7 倍代价）。第 3 章（数据流）成文时应把它作为
   "接口取舍的代价"的例证再提一次。
3. **服务槽的两处描述必须与代码一致**：第 13 章记 `nonlinear` 槽、
   第 14 章记 `linear` 槽，第 11 章引用了后者。三处的调用签名
   （`solve(residual, jacobian, x0, options=)` 与
   `solve(lower, diagonal, upper, rhs, regularization=)`）已逐字对齐，
   改动其一时必须同步其余两处。
4. **第 12 章显式记录了"$V_{\text{th}}$ 与量子模型无关"**（§12.2）。
   第 10 章的即插即用验收与它互为依据。
5. **`\textmu` 不可用**：本机 Latin Modern 下没有该符号，单位微米请写
   `\U{\mu m}`。
6. **写新章节时最容易犯的四类错**（P1–P3 都踩过，改法已固化，
   **交稿前请用脚本扫一遍**，不要靠肉眼读缩略图）：
   - 引用宏必须用类型化的 `\crefchap{ch:...}`（标签用**裸下划线**）、
     `\crefsec`、`\crefeq`、`\creftab`。泛型 `\cref` 与句首大写的
     `\Cref*` **都不存在**；标签里写 `ch:transport\_dd` 会报
     `Missing \endcsname`。
   - **`\akey{}` / `\pid{}` / `\fpath{}` 的内容一律写"裸"字符**：
     裸下划线 `_`、裸花括号 `{`、裸百分号 `%`。它们经 `\detokenize`
     会原样印出，**写成 `\_`、`\{` 反而会印出一个多余的反斜杠**。
     这是本项目最隐蔽的一类排版错 —— 它不报错、不影响编译，
     只能靠看渲染结果或脚本扫描发现。
   - 自带角标的宏（`\Ec`/`\epsr`/`\NC`/`\sigpol`/`\meff`）后面不能再直接
     写下标；节点索引用 `\Eci{}`/`\epsri{}`/`\NCeff{}`/`\Ecifl{}`。
   - `\mid` 是数学专用（`\midrule` 不是），文本模式下要用 `$|$`。

   对应的扫描命令（应全部为空）：
   ```powershell
   Select-String -Path modules\*.tex,frontmatter\*.tex -CaseSensitive `
     -Pattern '\\cref\{','\\Cref','\\(pid|akey|fpath)\{[^}]*\\[_&%{}]',
              '\\(Ec|epsr|ns|NC|mO|Ei|Ej)_'
   ```
7. **验证表列宽是 $\approx$ 78\,mm / 34\,mm / 38\,mm**。往"判据"或"实测"
   列塞长句会很挤；长内容请移进首列或 `\vnote`（表注，通栏）。
8. **模块卡片首列 30\,mm**：低于此值"依赖 artifact"会折行。



