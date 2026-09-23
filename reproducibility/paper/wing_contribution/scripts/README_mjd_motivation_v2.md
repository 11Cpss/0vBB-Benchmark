# Figure 2：论文版 motivation 三联图

运行（工作目录不限）：

```bash
/home/wenyu/summer/.venv/bin/python "/home/wenyu/iclr final paper/overleaf/wing_contribution/scripts/plot_mjd_motivation_v2.py"
```

当前机器已有上述 Python 环境。其他机器可使用装有 NumPy（2.0 或更新）和 Matplotlib 的 Python 3 环境；字体读取项目内的 `fonts/Nimbus*.otf`，无需联网。不要使用当前系统的 `/usr/bin/python`（Python 2）。

默认输出到 `figures/mjd_motivation_main_v2_generated`：

- `.pdf`：嵌入字体的矢量图，适用于 Overleaf。
- `.svg`：矢量曲线和文本，可继续编辑（编辑器需安装对应字体）。
- `.png`：600 dpi 预览。
- `.json`：输入文件 SHA-256、事件编号、核对后的 AUC、阴影范围。

原始 `mjd_motivation_main.pdf` 和教授参考 `mjd_motivation_main_v2.pdf` 均保留。绘图过程直接读取 CSV，不嵌入或裁切参考 PDF。新图采用三个 90 × 120 pt 的等大绘图区，面板标题统一放在绘图区上方。绘图区上下边界和横轴标题对齐。图例放在各自绘图区内部：(a) 左上方为 Low / High，(b) 左上方为 Signal-like / Background-like 和保留分箱阴影，(c) 右下方为 ROC 图例，AUC 单独成列、右对齐。(b) 的 Background-like 分两行显示；(c) 的图例避开曲线，保留完整的匹配后对角线。

样式参考 [XENONnT 的样式文件](https://github.com/XENONnT/xenon_plot_style/blob/master/xenonnt_plot_style/styles/xenonnt.mplstyle) 与 [示例 notebook](https://github.com/XENONnT/xenon_plot_style/blob/master/xenonnt_plot_style/examples/XENON_Plotting.ipynb)：使用蓝色 `#4067B1`、红色 `#B9123E`、淡青色 `#6CCEF5`，1 pt 数据线、向内主次刻度、无边框图例、透明区域标记、600 dpi PNG 与固定尺寸矢量输出。完整边框、左右和下侧刻度、内置图例和当前面板布局是本图的设计选择，不是该仓库强制规则；顶部刻度关闭。绘图无需安装 XENON 包，也不包含 XENON 标识。

字体继续复用 `evaluation/paper_style.py`：正文同款 Nimbus Roman（Times 兼容字体），数学文本为 Computer Modern。所有轴标题、刻度、图例、注释和面板标题统一为 9 pt。画布为 5.5 × 2.55 英寸，直接使用论文版心宽度；不使用 `bbox_inches="tight"`，以免改变插入正文后的物理字号。坐标轴和主刻度为 0.8 pt，次刻度为 0.6 pt。

## 图中数据

| 面板 | 数据来源 | 绘图方式 |
| --- | --- | --- |
| (a) | `evaluation/mjd_style/data/mjd_example_waveforms.csv` | 两条 MJD 基线扣除波形，每条 3800 点；事件 2407207 / 2585508，约 158 / 1132 keV |
| (b) | `figures/data/supernemo_matching_50keV.csv` | SuperNEMO 原始类别分布，50 keV 分箱，每类积分为 1；密度乘以 1000 |
| (c) | `figures/data/supernemo_matching_roc.csv` | 原始、保留和匹配后的完整 ROC 曲线，未抽样；AUC 从曲线积分核对 |

(b,c) 是项目已有 **SuperNEMO 0ν/Bi214 测试集能量诊断**；它们不是 MJD 数据，也不是训练后的模型分数。图中使用 `Signal-like` / `Background-like` 标签。Figure 2 的 caption 已同步明确测试集、能量分数和匹配条件，纠正原 caption 中的 training samples / trained classifier 描述；正文其他段落未修改。

三个 AUC 分别为 0.9706126082261715、0.9297152925868035、0.5000239466037062，图例显示四位小数。2200 keV 标记来自保存的真实工作点，FPR = 0.10220718383789062、TPR = 0.9395701090494791。

## 蓝色阴影

默认 `--shade-mode retained`，与图注中的 retained bins 一致。数据记录的 common support 是 1169.39–2734.195 keV；进一步剔除稀疏分箱后，实际保留 283 个 5 keV 分箱。精确分箱由保留集 ROC 中的所有有限能量阈值恢复；已与原始 `illustration_weights.npz` 的保留分箱逐一核对。保留低能端的缺口，最右端裁剪到 common-support 上界。

如需单独复现教授参考图的约 **1100–3000 keV** 示意范围：

```bash
/home/wenyu/summer/.venv/bin/python "/home/wenyu/iclr final paper/overleaf/wing_contribution/scripts/plot_mjd_motivation_v2.py" \
  --shade-mode reference \
  --output-stem "/home/wenyu/iclr final paper/overleaf/wing_contribution/figures/mjd_motivation_main_v2_reference_band"
```

此模式的图例会自动标为 `Reference band`，不会称为 retained bins。其他参数：`--shade-mode common-support` 仅绘制 common-support 连续区间；`--shade-mode none` 关闭阴影；`--shade-range LOW HIGH` 自定义阴影端点；`--dpi` 控制 PNG 分辨率。

论文的 `sections/mjd_motivation_figure.tex` 已使用新生成图：

```latex
\includegraphics[width=\linewidth]{wing_contribution/figures/mjd_motivation_main_v2_generated.pdf}
```
