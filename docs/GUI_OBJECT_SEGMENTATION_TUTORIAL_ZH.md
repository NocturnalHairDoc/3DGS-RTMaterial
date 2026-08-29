# 在 GUI 中分割自己的物体：逐步复现教程

本文面向第一次打开本项目的用户。目标是：加载一个已经训练好的 3DGS
场景，在 GUI 中选出一个真实物体，检查不同视角下的结果，保存分割，并在
下一次启动时恢复它。

如果只想尽快得到一个物体，先按“路线 A：手动点选”操作。若边界不稳定且
已有多张训练照片的 SAM 掩码，使用“路线 C：多照片 SAM”，它通常更容易复现。

## 0. 准备场景

需要一个已经训练好的 3DGS 模型。`-m` 接受以下任一种路径：

- 模型目录，例如 `/data/models/my_scene`
- 模型中的 `point_cloud` 目录
- 一个训练完成的 `point_cloud.ply`

具有 SAGA 对比特征和 scale gate 的模型最适合交互式语义分割。普通 3DGS
也能打开，但程序使用几何和外观代理特征，结果更接近颜色/空间聚类，不一定
能理解完整物体。

在项目根目录执行：

```bash
conda activate gaussian_splatting_v2
cd /absolute/path/to/3DGS-RTMaterial

# 只检查模型和分割资源，不启动 CUDA 或窗口
python rt_gs_gui_v3.py -m /absolute/path/to/my_model --dry-run

# 检查成功后启动 GUI
python rt_gs_gui_v3.py -m /absolute/path/to/my_model --scale 1.5
```

启动时建议明确给出 `-m`，不要在第一次操作时使用 `--one-click`。后者会自动
执行分割，适合快速预览，但不利于严格复现手动步骤。

如果物体没有出现在合适的位置，可重新启动并加入 `--fit-camera always`：

```bash
python rt_gs_gui_v3.py -m /absolute/path/to/my_model \
  --scale 1.5 --fit-camera always
```

主视窗操作：左键拖动旋转，鼠标中键拖动平移，滚轮缩放。启用分割点选后，
右键单击用于添加提示点，不要把右键单击和左键旋转混淆。

## 1. 先选择合适的方法

| 目标 | 方法 | 前置数据 | 建议 |
|---|---|---|---|
| 快速选择一个清楚可见的物体 | Manual | SAGA 特征通常更适合；普通 3DGS 也可尝试 | 首选入门路线 |
| 自动把整个场景拆成若干区域 | Auto | 不需要 SAM 掩码 | 适合探索，不保证每组对应语义物体 |
| 从多个训练视角稳定选择同一物体 | SAM-driven | COLMAP 场景及预计算 `sam_masks/*.pt` | 对遮挡和相似背景通常更可靠 |

三种方法都在右侧 **Segmentation** 面板中选择。最后的分割结果都可以在
**View → Segmentation** 中观察，并用相同方式保存。

## 2. 路线 A：在主 GUI 中手动点选一个物体

### 2.1 第一次分割

1. 在右侧 **View** 选择 **RGB (Baseline)**，旋转相机，找到目标物体轮廓最
   完整、遮挡最少的视角。
2. 展开 **Segmentation**，模式选择 **Manual**。
3. 将 **Scale** 先设为 `0.5`，**Threshold** 先设为 `0.65`。这是可用的
   起点，不是所有场景的固定最优值。
4. 勾选 **Click mode** 和 **Preview heatmap**，暂时不要勾选
   **Multi-click**。
5. 在目标物体内部、远离边缘的位置右键单击一次。热力图应突出目标区域。
6. 如果热力图合理，单击 **Segment 3D**。
7. 在 **View** 中切换到 **Segmentation**。目标物体应显示为一种分割颜色，
   未选择区域保持暗色。
8. 左键拖动旋转模型，至少检查正面、侧面和背面。不要只根据提示点所在视角
   判断成功。

### 2.2 调整边界

如果包含过多背景，单击 **Roll back**，提高 **Threshold** 后重新执行。例如
依次尝试 `0.70`、`0.75`、`0.80`。阈值越高，结果通常越严格，但也更容易漏掉
物体边缘。

如果目标缺失较多，单击 **Roll back**，降低阈值，或改变 **Scale** 后重新
点选。Scale 控制使用 SAGA 多尺度特征的语义粒度，物体大小不同时需要单独
调整；没有一个数值适用于所有场景。

当一次提示只能覆盖物体的一部分时：

1. **Roll back** 当前结果。
2. 勾选 **Multi-click**。
3. 在同一个物体的不同可见部件内部右键点选 2–4 次。
4. 观察热力图后再按 **Segment 3D**。

多点在当前实现中采用并集：任意一个提示匹配的 Gaussian 都可能被选中。因此
不要点击背景作为“负点”；错误点击会扩大泄漏范围。要删除错误提示，取消
Multi-click 后重新点击，或执行 **Clear all** 后重做。

### 2.3 继续分割第二个物体

1. 先按第 4 节保存当前物体。
2. 展开 **Material**，单击 **Confirm & hide**。当前物体会暂时隐藏，同时提示
   状态被清空。
3. 回到 **View → RGB (Baseline)**，对下一个物体重复上述步骤。

**Roll back** 只撤销最近一次分割；**Clear all** 会清空当前场景中的全部分割、
隐藏状态和材料分配。

## 3. 路线 B：自动聚类整个场景

1. 在 **Segmentation** 中选择 **Auto**。
2. 若环境安装了 HDBSCAN，优先选择 **HDBSCAN (auto K)**。从
   **Min cluster size = 50** 开始；提高该值会减少小碎片。
3. 若只有 KMeans，选择 **KMeans (fixed K)**，将 **K** 设为预计的大区域数量，
   例如先尝试 `8`。
4. 单击 **Auto Segment**，等待终端打印完成信息。
5. 切换到 **View → Segmentation**，旋转场景检查每种颜色是否形成连续对象。
6. 不满意时按 **Clear all**，修改参数并重新执行。

自动聚类的用途是快速获得全场景候选区域。它不会询问“我要哪个物体”，也不
保证一个语义物体恰好对应一个标签。若目标与背景颜色或空间位置相近，应改用
路线 A 或路线 C。

也可以在启动时复现自动聚类：

```bash
# 自动选择 HDBSCAN；不可用时回退到 KMeans
python rt_gs_gui_v3.py -m /absolute/path/to/my_model --auto-segment auto

# 明确使用固定 8 类的 KMeans
python rt_gs_gui_v3.py -m /absolute/path/to/my_model \
  --auto-segment kmeans --clusters 8
```

## 4. 保存、关闭和重新加载

### 4.1 保存一个或多个掩码

展开 **Save / Load**：

1. 在上方名称框输入不带扩展名的名字，例如 `red_chair`。
2. 只有一个物体，或只想保存最近完成的物体时，按 **Save segment**。文件为
   `./segmentation_res/red_chair.pt`。
3. 已经分割多个物体并希望保留全部标签时，按 **Save all**。文件为
   `./segmentation_res/red_chair_full.pt`。

相对路径以启动命令所在目录为基准。因此应从项目根目录启动 GUI，避免把结果
保存到意外位置。

### 4.2 保存完整 GUI 工程（推荐）

如果还需要恢复材料、隐藏物体、相机姿态和相关界面设置：

1. 展开 **Project & Export**。
2. 在路径框中保留默认路径或输入 `./segmentation_res/my_scene_project.npz`。
3. 单击 **Save project**。

`.pt` 是便于其他脚本使用的 Gaussian 掩码；`.npz` 是继续 GUI 编辑时更完整的
工程状态。建议两者都保存。

### 4.3 验证重新加载

关闭后用同一个模型重新启动 GUI：

```bash
python rt_gs_gui_v3.py -m /absolute/path/to/my_model --scale 1.5
```

只加载掩码时，在 **Save / Load** 的下方路径框输入
`./segmentation_res/red_chair.pt`，保持 **Confidence-aware preview** 未勾选，
然后按 **Load segmentation**。

加载完整工程时，在 **Project & Export** 输入 `.npz` 路径，第一次单击
**Load project** 查看摘要，确认模型名和 Gaussian 数量正确后，再单击已变成
**Confirm load** 的按钮。工程状态只能加载到对应的同一模型/点云版本。

## 5. 路线 C：在多张照片中选择，再融合到 3D

该路线一次建立一个物体的 selection JSON。需要：

- 数据集目录下有 `images/` 和 COLMAP `sparse/`
- 已经预计算的 SAM proposal tensor，例如 `sam_masks/<image_stem>.pt`
- 模型和数据集来自同一次 COLMAP 重建，图像名称能够对应

先打开独立的多照片选择 GUI：

```bash
python -m segmentation.multiview_selection \
  --source /absolute/path/to/colmap_dataset \
  --masks /absolute/path/to/colmap_dataset/sam_masks \
  --output segmentation_res/my_object.json
```

在弹出的选择器中：

1. 用 **Previous photo** / **Next photo** 浏览训练照片。
2. 先选择 **Whole-object brush**，在目标内部按住左键涂抹；它会优先选择笔迹下
   较大的有效 SAM proposal。如果覆盖仍不完整，再切换 **Fine-parts brush**，
   补选主体、壶嘴、把手等遗漏部件。绿色覆盖层表示当前 proposals 的并集。
3. 如果选择错误，按住右键涂过错误区域以删除对应 proposal，或按
   **Clear this photo** 清空当前照片。
4. 在至少两张照片中选择同一个物理物体。建议选择 3–6 个差异明显且遮挡较少
   的视角。每张图可以包含多个物体部件 proposal，但融合前会先合成一个掩码，
   因此该照片仍然只贡献一次跨视角投票。
   不建议把几十张相近视角全部选中；保存超过 12 个视角时选择器会要求再次确认。
5. 如果只出现细碎绿色区域，检查启动参数 `--max-area`。大型前景物体可使用
`--max-area 0.50`；过小的上限会在选择前直接排除完整物体 proposal。
6. 单击 **Save selection**。少于两个有效视角时程序会拒绝保存。
7. 单击 **Close**。

然后把 selection JSON 融合进主 GUI：

```bash
python rt_gs_gui_v3.py -m /absolute/path/to/my_model \
  --multiview-selection segmentation_res/my_object.json
```

窗口打开后切换到 **View → Segmentation**，旋转一周检查结果，再按第 4 节保存。

也可以先正常打开主 GUI，在 **Segmentation** 中选择 **SAM-driven**，把 JSON
路径填入 **Selection JSON**，再按 **Run SAM Instance Graph**。两种入口执行的
是同一类多视图融合流程。

若要生成可用于批处理的 `.pt`，而不打开主 GUI：

```bash
python -m segmentation.sam_driven \
  -m /absolute/path/to/my_model \
  --selection segmentation_res/my_object.json \
  -o segmentation_res/my_object_mask.pt \
  --diagnostics segmentation_res/my_object_diagnostics.json
```

对于第二个物体，创建新的 JSON，例如 `second_object.json`，不要在
`my_object.json` 中混合两个物体的照片选择。

## 6. 一个完整操作示例

下面以“从场景中分割椅子”为例。替换两条绝对路径即可：

```bash
cd /absolute/path/to/3DGS-RTMaterial
conda activate gaussian_splatting_v2

export SEG_TUTORIAL_MODEL=/absolute/path/to/trained/chair_scene
export SEG_TUTORIAL_DATA=/absolute/path/to/colmap/chair_scene

python rt_gs_gui_v3.py -m "$SEG_TUTORIAL_MODEL" --dry-run
python rt_gs_gui_v3.py -m "$SEG_TUTORIAL_MODEL" --scale 1.5
```

在主 GUI 中依次执行：

1. **View → RGB (Baseline)**。
2. **Segmentation → Manual**。
3. Scale `0.5`，Threshold `0.65`。
4. 勾选 **Click mode**、**Preview heatmap**。
5. 在椅子座面内部右键单击。
6. 单击 **Segment 3D**。
7. **View → Segmentation**，旋转 360° 检查。
8. 若背景泄漏：**Roll back**，Threshold 改为 `0.75`，重新右键点选并分割。
9. **Save / Load** 名称输入 `chair_manual`，按 **Save segment**。
10. **Project & Export** 保存 `./segmentation_res/chair_project.npz`。
11. 重启同一个模型并重新加载两个文件，确认颜色区域和相机/工程状态可恢复。

如果手动结果跨视角不完整，再执行：

```bash
python -m segmentation.multiview_selection \
  --source "$SEG_TUTORIAL_DATA" \
  --masks "$SEG_TUTORIAL_DATA/sam_masks" \
  --output segmentation_res/chair_multiview.json

python rt_gs_gui_v3.py -m "$SEG_TUTORIAL_MODEL" \
  --multiview-selection segmentation_res/chair_multiview.json
```

按路线 C 在 3–6 张照片里选择同一把椅子，保存后再次进行 360° 检查。保留
`chair_manual.pt`、多视图结果、selection JSON 和 diagnostics，便于比较和复现。

## 7. 视觉验收清单

不要只记录“程序运行成功”。一个可交付的对象分割至少应通过以下检查：

- 正面、左右侧和背面的主体连续，没有只分出提示点附近的一小块
- 桌面、墙、地面等相邻背景没有形成明显的大面积泄漏
- 细结构（椅腿、把手、叶片等）是否保留，应根据任务要求明确记录
- 遮挡边界没有把前后两个物体大面积粘连
- 重启并加载后，Gaussian 数量匹配且结果不变
- 保存一张 RGB 截图和至少三张不同视角的 Segmentation 截图

人工掩码只在确认可靠时才可作为定量真值。来源不明、明显错位或不完整的人工
掩码应忽略；它们不影响上述 GUI 视觉验收，也不应被用于报告 IoU。

## 8. 常见问题

**右键没有生成热力图**：确认已勾选 **Click mode**，点击位置在主图像内，且
窗口焦点位于主窗口。

**几乎整个场景都被选中**：默认 Threshold 可能过低。Roll back 后从 `0.65`
开始逐步提高，不要直接保存第一次结果。

**多点后结果反而变差**：多点是并集，不支持背景负点。清空后只在同一物体的
可靠内部区域点选。

**普通 PLY 能打开但选不出完整语义物体**：这是代理特征的预期限制。训练
SAGA 对比特征，或使用多照片 SAM 路线。

**SAM 选择器找不到照片**：`sam_masks` 中 `.pt` 的文件 stem 必须能在
`source/images/` 中找到同名图像。

**SAM JSON 保存失败**：同一物体必须至少在两张照片中有有效 proposal。

**加载掩码提示数量不匹配**：掩码来自不同训练迭代、不同 densification 结果或
不同模型。只能加载 Gaussian 顺序和数量一致的结果。

**Confidence-aware preview 加载失败**：普通 GUI 保存的 `.pt` 没有配套置信度
文件，应保持该选项关闭。它只用于带置信度文件的离线结果。

## 9. 提交可复现结果时应包含什么

建议为每个物体保存以下内容：

```text
segmentation_res/
├── my_object.json                  # 多视图方法使用的照片/proposal 选择
├── my_object.pt                    # 单物体 Gaussian 掩码
├── my_scene_full.pt                # 可选：所有物体的整数标签
├── my_scene_project.npz            # 完整 GUI 工程状态
└── my_object_diagnostics.json      # 多视图离线方法的诊断信息
```

同时记录模型路径或模型版本、启动命令、方法、Scale、Threshold 或 K、所选照片
数量，以及视觉验收截图。这样其他用户无需猜测参数，就能按相同步骤复现。
