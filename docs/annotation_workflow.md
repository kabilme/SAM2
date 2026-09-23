# Annotation & Tracking Workflow

This guide details the interactive annotation lifecycle, prompt modes, polygon conversion, and multi-frame propagation in the **SAM3 Video Polygon Annotator**.

---

## 1. The Annotation Lifecycle

```
Select Frame → Select Class → Prompt SAM 3 → Generate Mask → Convert to Polygon → Refine Vertices → Propagate / Review
```

### Step 1: Frame Navigation
- Browse frames on the bottom **Frame Timeline** or use **Left / Right Arrow** keys.
- Filter frames by review status: *All*, *Unreviewed*, *Reviewed*, or *Negative*.

### Step 2: Selecting Class & Mode
- In the **Class Panel** on the left dock, select the active object category (e.g., `scooter`).
- In the toolbar, select your prompting method:
  - **Positive Point (`+`)**: Marks a foreground point on the target object.
  - **Negative Point (`-`)**: Marks a background point to exclude non-target regions.
  - **Bounding Box (`Box`)**: Click and drag a tight box encompassing the target object.
  - **Manual Polygon (`Poly`)**: Manually click arbitrary boundary points.

---

## 2. Interactive SAM Prompting & Refinement

1. **Initial Point**: Click inside the desired object on the canvas. SAM immediately predicts the instance mask.
2. **Refinement Points**:
   - If parts of the object are missing, add another **Positive Point** on the omitted region.
   - If background or adjacent objects are falsely included, add a **Negative Point** on the incorrect area.
3. The adapter merges existing prompts and recalculates the mask in real-time.

---

## 3. Mask-to-Polygon Conversion

Once segmentation succeeds, the binary mask is automatically vectorized:
- **Contour Finding**: Detects closed exterior boundaries using topological hierarchy analysis.
- **Ramer-Douglas-Peucker (RDP) Simplification**: Reduces noisy pixel steps while preserving sharp corners and curve fidelity.
  - Controlled by `tolerance_ratio` (default: `0.004` of the object diagonal).
  - Minimum area filter rejects spurious noise fragments (< 20 px).
- The resulting polygon has at least 3 vertices and coordinates normalized between `0.0` and `1.0`.

---

## 4. Manual Polygon Editing

Switch to **Edit Mode** (`E`) or double-click any polygon to adjust vertices:
- **Move Vertex**: Click and drag any existing vertex handle.
- **Insert Vertex**: Click anywhere along an edge segment to split it and create a new control point.
- **Delete Vertex**: Right-click on a vertex handle and choose **Delete Vertex**. Polygons must maintain at least 3 vertices.
- **Delete Polygon**: Select the polygon and press `Delete` or `Backspace`.

---

## 5. Multi-Frame Propagation (Tracking)

When an object is annotated in frame $t$:
1. Select the annotation on the canvas or in the **Objects List**.
2. Click **Propagate Forward** (`Ctrl+P` or toolbar icon).
3. Specify how many frames to propagate (e.g. next 10, 30, or all remaining frames).
4. The background propagation worker:
   - Uses SAM video memory features or optical frame tracking to locate the object in each subsequent frame.
   - Generates matching polygon instances assigned with the same object ID and class.
   - Shows progress in a non-blocking dialog that can be paused or cancelled at any time.

---

## 6. Frame Review Statuses

Every frame has one of three statuses:
- **Unreviewed** (Default): Frame has not been validated by the user.
- **Reviewed** (`R`): Frame has verified annotations ready for training export.
- **Negative / Background** (`N`): Frame contains zero targets of interest (acts as background negative sample in training to minimize false positives).
