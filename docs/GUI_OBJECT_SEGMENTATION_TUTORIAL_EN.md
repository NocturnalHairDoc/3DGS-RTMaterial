# uired
 | Useful for exploration; clusters are not guaranteed to be semantic objects |
| Select the same object from several training views | SAM-driven | COLMAP scene and precomputed `sam_masks/*.pt` | Usually more stable under occlusion or similar backgrounds |

All three methods are selected from the **Segmentation** panel. Their results
can be inspected under **View → Segmentation** and saved with the same controls.

## 2. Route A: manually select one object in the main GUI

### 2.1 Create the first segmentation

1. Select **RGB (Baseline)** under **View**. Orbit the camera until the target
   has a clear outline and little occlusion.
2. Expand **Segmentation** and select **Manual**.
3. Start with **Scale = 0.5** and **Threshold = 0.65**. These are starting
   values, not universal optimums.
4. Enable **Click mode** and **Preview heatmap**. Leave **Multi-click** disabled
   for the first attempt.
5. Right-click once well inside the object and away from its boundary. The
   heatmap should emphasize the target region.
6. If the heatmap looks reasonable, click **Segment 3D**.
7. Switch to **View → Segmentation**. The selected object should use one segment
   color while unselected Gaussians remain dark.
8. Orbit around the scene and check at least the front, sides, and back. Do not
   judge the result only from the prompt viewpoint.

### 2.2 Adjust the boundary

If too much background is included, click **Roll back**, raise **Threshold**,
and run the selection again. For example, try `0.70`, `0.75`, and `0.80` in
order. A higher threshold is usually stricter, but it can remove valid boundary
regions.

If much of the target is missing, click **Roll back**, lower the threshold, or
change **Scale** before selecting again. Scale controls the semantic granularity
of SAGA's multi-scale features, so objects of different sizes may need different
values.

If one prompt covers only part of the object:

1. **Roll back** the current result.
2. Enable **Multi-click**.
3. Right-click inside two to four visible parts of the same object.
4. Inspect the heatmap and then click **Segment 3D**.

Multi-click currently takes the union of prompt matches: a Gaussian can be
selected if it matches any prompt. Do not click the background as a negative
prompt. To remove an incorrect prompt, disable Multi-click and start the prompt
sequence again, or use **Clear all** and repeat the selection.

### 2.3 Continue with a second object

1. Save the current object using Section 4.
2. Expand **Material** and click **Confirm & hide**. The current object is
   temporarily hidden and the prompt state is cleared.
3. Return to **View → RGB (Baseline)** and repeat the procedure for the next
   object.

**Roll back** only undoes the most recent segmentation. **Clear all** removes
all current segment labels, hidden states, and material assignments.

## 3. Route B: automatically cluster the scene

1. Select **Auto** in the **Segmentation** panel.
2. If HDBSCAN is installed, select **HDBSCAN (auto K)**. Start with
   **Min cluster size = 50**; increasing it reduces small fragments.
3. If only KMeans is available, select **KMeans (fixed K)** and set **K** to an
   estimate of the number of large regions, such as `8`.
4. Click **Auto Segment** and wait for completion in the terminal.
5. Switch to **View → Segmentation** and orbit around the scene to check whether
   each color forms a coherent region.
6. If the result is unsuitable, click **Clear all**, adjust the parameters, and
   run it again.

Automatic clustering is useful for obtaining candidate regions across a whole
scene. It does not ask which object you want, and one semantic object is not
guaranteed to map to one label. Use Route A or Route C when the target and its
background have similar color or spatial features.

You can also reproduce automatic clustering at startup:

```bash
# Prefer HDBSCAN and fall back to KMeans if HDBSCAN is unavailable.
python rt_gs_gui_v3.py -m /absolute/path/to/my_model --auto-segment auto

# Explicitly use KMeans with eight clusters.
python rt_gs_gui_v3.py -m /absolute/path/to/my_model \
  --auto-segment kmeans --clusters 8
```

## 4. Save, close, and reload

### 4.1 Save one or more masks

Expand **Save / Load**:

1. Enter a name without an extension, such as `red_chair`.
2. For one object, or only the most recently completed object, click
   **Save segment**. The output is `./segmentation_res/red_chair.pt`.
3. If the scene contains several segmented objects and all labels should be
   retained, click **Save all**. The output is
   `./segmentation_res/red_chair_full.pt`.

Relative paths are resolved from the directory in which the GUI was launched.
Start the GUI from the repository root to avoid saving files in an unexpected
location.

### 4.2 Save the complete GUI project

To restore materials, hidden objects, camera pose, and relevant interface
settings:

1. Expand **Project & Export**.
2. Keep the default path or enter
   `./segmentation_res/my_scene_project.npz`.
3. Click **Save project**.

A `.pt` file is a Gaussian mask that can be used by other scripts. An `.npz`
file stores a more complete GUI editing state. Saving both is recommended.

### 4.3 Verify reloading

Close the GUI and restart it with the same model:

```bash
python rt_gs_gui_v3.py -m /absolute/path/to/my_model --scale 1.5
```

To load only a mask, enter `./segmentation_res/red_chair.pt` in the lower path
field under **Save / Load**, leave **Confidence-aware preview** disabled, and
click **Load segmentation**.

To load a complete project, enter its `.npz` path under **Project & Export**.
Click **Load project** once to review the summary. After checking the model name
and Gaussian count, click the button again when it changes to **Confirm load**.
A project state can only be loaded into the matching model and point-cloud
version.

## 5. Route C: select an object in several photographs and fuse it into 3D

This route creates one selection JSON for one physical object. It requires:

- `images/` and COLMAP `sparse/` directories in the dataset
- Precomputed SAM proposal tensors such as
  `sam_masks/<image_stem>.pt`
- A model and dataset produced from the same COLMAP reconstruction, with
  matching image names

Open the separate multi-photo selector:

```bash
python -m segmentation.multiview_selection \
  --source /absolute/path/to/colmap_dataset \
  --masks /absolute/path/to/colmap_dataset/sam_masks \
  --output segmentation_res/my_object.json
```

In the selector:

1. Use **Previous photo** and **Next photo** to browse the training images.
2. Start with **Whole-object brush** and left-drag inside the target. It prefers
   larger eligible SAM proposals under the stroke. If coverage is incomplete,
   switch to **Fine-parts brush** and add missing regions such as the body,
   spout, handle, or other thin parts. The green overlay is the union of the
   selected proposals in the current photograph.
3. To correct a mistake, right-drag across the unwanted proposal or click
   **Clear this photo** to reset only the current photograph.
4. Select the same physical object in at least two photographs. Three to six
   diverse, lightly occluded views are recommended. A photograph may contain
   several selected part proposals, but they are unioned before fusion, so that
   photograph still contributes at most one cross-view vote.
5. Avoid selecting dozens of nearly identical views. The selector asks for
   confirmation when more than 12 views are selected.
6. If only small green fragments appear, check `--max-area`. A large foreground
   object may require `--max-area 0.50`; a lower limit can exclude its complete
   proposal before selection.
7. Click **Save selection**. The program refuses to save fewer than two valid
   selected views.
8. Click **Close**.

Photographs with no selected proposal are ignored during 3D fusion. They are
not treated as evidence that the object is absent.

Fuse the selection JSON in the main GUI:

```bash
python rt_gs_gui_v3.py -m /absolute/path/to/my_model \
  --multiview-selection segmentation_res/my_object.json
```

After the window opens, switch to **View → Segmentation**, inspect the object
through a full orbit, and save it using Section 4.

Alternatively, open the main GUI normally, select **SAM-driven** under
**Segmentation**, enter the JSON path in **Selection JSON**, and click
**Run SAM Instance Graph**. Both entry points use the same type of multi-view
fusion.

To create a reusable `.pt` mask without opening the main GUI:

```bash
python -m segmentation.sam_driven \
  -m /absolute/path/to/my_model \
  --selection segmentation_res/my_object.json \
  -o segmentation_res/my_object_mask.pt \
  --diagnostics segmentation_res/my_object_diagnostics.json
```

For a second object, create a separate JSON such as `second_object.json`. Do not
mix photographs of two physical objects in `my_object.json`.

## 6. Complete example

This example segments a chair. Replace the two absolute paths:

```bash
cd /absolute/path/to/3DGS-RTMaterial
conda activate gaussian_splatting_v2

export SEG_TUTORIAL_MODEL=/absolute/path/to/trained/chair_scene
export SEG_TUTORIAL_DATA=/absolute/path/to/colmap/chair_scene

python rt_gs_gui_v3.py -m "$SEG_TUTORIAL_MODEL" --dry-run
python rt_gs_gui_v3.py -m "$SEG_TUTORIAL_MODEL" --scale 1.5
```

In the main GUI:

1. Select **View → RGB (Baseline)**.
2. Select **Segmentation → Manual**.
3. Set Scale to `0.5` and Threshold to `0.65`.
4. Enable **Click mode** and **Preview heatmap**.
5. Right-click inside the chair seat.
6. Click **Segment 3D**.
7. Select **View → Segmentation** and inspect the result through a 360-degree
   orbit.
8. If the background leaks into the result, click **Roll back**, change
   Threshold to `0.75`, and repeat the prompt and segmentation.
9. Enter `chair_manual` under **Save / Load** and click **Save segment**.
10. Save `./segmentation_res/chair_project.npz` under **Project & Export**.
11. Restart with the same model and reload both files to confirm that the
    segmentation and project state can be restored.

If the manual result is incomplete across viewpoints, run:

```bash
python -m segmentation.multiview_selection \
  --source "$SEG_TUTORIAL_DATA" \
  --masks "$SEG_TUTORIAL_DATA/sam_masks" \
  --output segmentation_res/chair_multiview.json

python rt_gs_gui_v3.py -m "$SEG_TUTORIAL_MODEL" \
  --multiview-selection segmentation_res/chair_multiview.json
```

Select the chair in three to six photographs, save the selection, and perform
another 360-degree inspection. Keep `chair_manual.pt`, the multi-view result,
the selection JSON, and the diagnostics file when comparing the two methods.

## 7. Visual acceptance checklist

Do not record only that the program completed. Check the following before
accepting an object segmentation:

- The main body is continuous from the front, both sides, and the back.
- Adjacent surfaces such as the table, wall, or floor do not form a large leak.
- The treatment of thin structures such as legs, handles, or leaves is recorded
  according to the task requirements.
- Occlusion boundaries do not merge large parts of foreground and background
  objects.
- The Gaussian count matches after restarting and reloading, and the result is
  unchanged.
- Save one RGB screenshot and at least three segmentation screenshots from
  different viewpoints.

Use a manual mask as quantitative ground truth only after it has been checked.
Ignore masks that have unknown provenance, obvious misalignment, or incomplete
coverage. They should not be used to report IoU.

## 8. Troubleshooting

**Right-click does not produce a heatmap:** Make sure **Click mode** is enabled,
the click is inside the main image, and the main window has focus.

**Almost the entire scene is selected:** The threshold may be too low. Click
**Roll back** and increase it gradually from `0.65` instead of saving the first
result.

**Multi-click makes the result worse:** Multi-click is a union and does not
support background negative points. Clear the prompts and click only reliable
interior regions of the same object.

**A plain PLY opens but does not produce a complete semantic object:** This is
an expected limitation of proxy features. Train SAGA contrastive features or
use Route C with multi-photo SAM masks.

**The SAM selector cannot find a photograph:** The stem of each `.pt` file in
`sam_masks` must match an image under `source/images/`.

**The SAM JSON cannot be saved:** The same object must have a valid proposal in
at least two photographs.

**A loaded mask has the wrong number of Gaussians:** The mask came from a
different training iteration, densification result, or model. The Gaussian
order and count must match.

**Confidence-aware preview fails to load:** A normal `.pt` saved by the GUI has
no accompanying confidence file. Leave this option disabled unless loading an
offline result with a matching confidence file.

## 9. Files to include with a reproducible result

For each object, retain the relevant files:

```text
segmentation_res/
├── my_object.json                  # Photo/proposal selection for Route C
├── my_object.pt                    # Single-object Gaussian mask
├── my_scene_full.pt                # Optional integer labels for all objects
├── my_scene_project.npz            # Complete GUI project state
└── my_object_diagnostics.json      # Multi-view offline diagnostics
```

Also record the model path or version, launch command, segmentation method,
Scale, Threshold or K, number of selected photographs, and visual-inspection
screenshots. This allows another user to repeat the same procedure without
guessing the parameters.
