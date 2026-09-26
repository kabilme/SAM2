You are a senior Python computer-vision engineer, ML engineer, and UI/UX developer.

Your task is to design and implement a complete production-quality Python application that performs SAM 2 assisted polygon annotation on frames extracted from an input video, similar in workflow to modern browser-based annotation platforms such as Roboflow.

The application must be LOCAL-FIRST. It must run on the user's computer, use the local GPU when available, process video files locally, perform SAM 2 inference locally, allow interactive polygon editing, and export a complete YOLOv8-compatible instance-segmentation dataset.

Do not create a conceptual prototype. Implement a functional application with proper error handling, persistent project files, a usable graphical interface, and a complete dataset export pipeline.

IMPORTANT:
- Do not implement a fake or placeholder segmentation algorithm.
- Do not replace SAM 2 with a different segmentation model unless explicitly required as a fallback.
- Do not assume a fixed SAM 2 API if the installed SAM 2 version exposes a different API.
- Inspect the installed SAM 2 package/repository and use the currently available public APIs.
- Keep all SAM 2-specific integration isolated inside a dedicated adapter/service layer so that future SAM 2 API changes do not require rewriting the entire application.
- The application must continue to work without internet access after the required Python dependencies and model weights have been installed.
- Never upload the user's video, frames, masks, or annotations to an external service unless an explicit cloud integration is later requested.
- Preserve original video frames and project data. Never destructively modify source files.

==================================================
1. PRIMARY OBJECTIVE
==================================================

Build a Python desktop application named:

SAM2 Video Polygon Annotator

The application workflow must be:

INPUT VIDEO
    ↓
Video metadata extraction
    ↓
Frame extraction / sampling
    ↓
Frame browser
    ↓
Select frame
    ↓
SAM 2 prompt-based segmentation
    ↓
Generate object mask
    ↓
Convert mask to polygon
    ↓
Assign class
    ↓
Display editable polygon
    ↓
Manual polygon refinement
    ↓
Propagate/tracking across subsequent frames
    ↓
Review generated annotations
    ↓
Correct rejected/incorrect masks
    ↓
Save annotations
    ↓
Train/validation/test split
    ↓
Export YOLOv8 segmentation dataset
    ↓
Validate exported dataset
    ↓
Create ZIP archive

The final output should be directly usable for training an Ultralytics YOLO segmentation model.

==================================================
2. RECOMMENDED TECHNOLOGY STACK
==================================================

Use Python 3.11 or a currently supported Python version compatible with the installed SAM 2 environment.

Preferred components:

Core:
- Python
- OpenCV
- NumPy
- PyTorch
- PIL/Pillow

SAM:
- Meta SAM 2
- Official SAM 2 Python implementation
- Official SAM 2 checkpoint/model loading mechanisms

GUI:
- PySide6

Annotation canvas:
- QGraphicsView / QGraphicsScene OR an equivalent high-performance custom annotation canvas

Video:
- OpenCV VideoCapture
- FFmpeg only where needed for robust codec support

Geometry:
- OpenCV contours
- Shapely where useful
- NumPy

Serialization:
- JSON
- YAML

Packaging:
- ZIP
- pathlib
- tempfile
- logging

Testing:
- pytest

Use asynchronous/background execution for expensive operations so the GUI never freezes during:
- video frame extraction
- SAM inference
- mask generation
- mask propagation
- video processing
- dataset export
- dataset validation

Use QThread, QThreadPool/QRunnable, or another appropriate Qt-compatible worker architecture.

==================================================
3. APPLICATION ARCHITECTURE
==================================================

Use a modular architecture.

Suggested structure:

sam2_annotator/
│
├── app.py
├── main.py
│
├── config/
│   ├── config.py
│   └── defaults.yaml
│
├── models/
│   ├── sam2_adapter.py
│   ├── sam2_image_service.py
│   └── sam2_video_service.py
│
├── video/
│   ├── video_reader.py
│   ├── frame_extractor.py
│   └── frame_cache.py
│
├── annotation/
│   ├── annotation_manager.py
│   ├── polygon.py
│   ├── mask_to_polygon.py
│   ├── polygon_editor.py
│   ├── object_tracker.py
│   └── annotation_serializer.py
│
├── dataset/
│   ├── dataset_builder.py
│   ├── yolo_exporter.py
│   ├── dataset_validator.py
│   └── split_manager.py
│
├── ui/
│   ├── main_window.py
│   ├── video_panel.py
│   ├── frame_timeline.py
│   ├── annotation_canvas.py
│   ├── class_panel.py
│   ├── properties_panel.py
│   ├── toolbar.py
│   ├── export_dialog.py
│   └── progress_dialog.py
│
├── project/
│   ├── project_manager.py
│   └── project_schema.py
│
├── utils/
│   ├── geometry.py
│   ├── image_utils.py
│   ├── logging_utils.py
│   └── device_utils.py
│
├── tests/
│   ├── test_video.py
│   ├── test_polygon.py
│   ├── test_yolo_export.py
│   ├── test_dataset_validation.py
│   └── test_project.py
│
├── requirements.txt
├── README.md
└── pyproject.toml

Keep model-specific functionality independent from UI code.

The GUI must never directly invoke low-level SAM 2 inference code.

==================================================
4. APPLICATION STARTUP
==================================================

When the application starts:

1. Detect:
   - Python version
   - PyTorch version
   - CUDA availability
   - CUDA device count
   - GPU name
   - VRAM where available
   - CPU
   - RAM

2. Display a startup diagnostic panel.

Example:

Device:
GPU: NVIDIA RTX XXXXX
CUDA: Available
PyTorch: XXXXX
SAM 2: Loaded / Not Loaded

3. Allow the user to select:
   - CUDA
   - CPU

4. Default to CUDA if available.

5. If CUDA is unavailable:
   - display an informational message
   - allow CPU execution
   - do not crash

6. Provide configuration for:
   - model checkpoint path
   - device
   - precision
   - frame cache location
   - project directory
   - maximum cache size

==================================================
5. PROJECT CREATION
==================================================

The user must be able to create a new annotation project.

Project creation dialog:

Project Name
Video File
Output Directory
Class Names
Frame Sampling Method
Train/Validation/Test Split
SAM 2 Model

Allow users to define multiple classes.

Example:

Class ID | Class Name
0        | scooter
1        | person
2        | helmet
3        | car

The application must automatically assign class IDs sequentially beginning at 0.

Allow the user to add, rename, reorder, and remove classes.

Do not silently change class IDs after annotations already exist.

==================================================
6. VIDEO IMPORT
==================================================

Accept common video formats:

.mp4
.avi
.mov
.mkv
.webm

When importing a video, extract:

- filename
- absolute path
- frame count
- FPS
- duration
- width
- height
- codec if available

Display:

Video:
example.mp4

Resolution:
1920 × 1080

FPS:
30.0

Frames:
18,420

Duration:
10:14

==================================================
7. FRAME EXTRACTION
==================================================

Implement a robust frame extraction module.

The user must be able to choose:

A. Extract every frame
B. Extract every Nth frame
C. Extract a fixed number of frames
D. Extract at a specified interval in seconds
E. Extract a specific frame range
F. Extract keyframes manually
G. Extract frames around selected timestamps

Example:

Video FPS = 30

Sampling:
Every 10 frames

Extracted:
frame_000001
frame_000011
frame_000021
...

Support configurable frame naming.

Preferred naming:

frame_00000001.jpg
frame_00000002.jpg
frame_00000003.jpg

Store the original source video frame number in metadata.

Do NOT lose the relationship between:
- extracted frame
- original frame number
- timestamp
- annotation

For every extracted frame create metadata such as:

{
    "frame_id": 1,
    "source_frame_index": 31,
    "timestamp_seconds": 1.033,
    "filename": "frame_00000001.jpg",
    "width": 1920,
    "height": 1080
}

==================================================
8. FRAME CACHE
==================================================

Do not load the entire video into RAM.

Implement lazy loading.

Only load:
- current frame
- nearby frames
- thumbnails

Maintain an LRU or configurable frame cache.

Allow cache cleanup.

Keep source frames immutable.

==================================================
9. FRAME BROWSER UI
==================================================

Create a frame browser similar to a professional labeling application.

Main layout:

---------------------------------------------------------
| File | Project | Annotation | Dataset | Export | Help |
---------------------------------------------------------
| Toolbar                                                |
---------------------------------------------------------
|       |                                               |
| Frames|            Annotation Canvas                  |
|       |                                               |
|       |                                               |
---------------------------------------------------------
|       |                                               |
|       |                                               |
---------------------------------------------------------
|       | Timeline                                      |
---------------------------------------------------------

Left panel:
Frame thumbnails

Center:
Annotation canvas

Right panel:
Classes / Objects / Properties

Bottom:
Video/frame timeline

==================================================
10. FRAME TIMELINE
==================================================

Provide:

- current frame number
- original frame number
- timestamp
- previous frame
- next frame
- play
- pause
- jump to frame
- jump to timestamp
- frame stepping
- keyframe markers
- annotated-frame markers
- propagation markers

Keyboard shortcuts:

Left Arrow:
Previous frame

Right Arrow:
Next frame

Space:
Play/Pause

Ctrl+S:
Save

Ctrl+Z:
Undo

Ctrl+Shift+Z:
Redo

Delete:
Delete selected object

N:
Next frame

P:
Previous frame

M:
Manual polygon mode

S:
SAM 2 mode

==================================================
11. ANNOTATION MODES
==================================================

Implement at least these annotation modes.

MODE A:
SAM 2 Text Prompt

MODE B:
SAM 2 Positive Click

MODE C:
SAM 2 Negative Click

MODE D:
SAM 2 Bounding Box / Exemplar Prompt

MODE E:
Manual Polygon

MODE F:
Edit Existing Polygon

MODE G:
Object Tracking / Propagation

==================================================
12. SAM 2 TEXT PROMPT
==================================================

Provide a text prompt field.

Example:

"scooter"

When user enters the prompt and runs segmentation:

1. Send the current image/frame to SAM 2.
2. Submit the text prompt through the official SAM 2 API.
3. Retrieve segmentation results.
4. Retrieve masks and associated detections.
5. Display masks over the image.
6. Allow user to select one or multiple detected objects.
7. Convert the selected mask to polygon.
8. Assign the corresponding class.

Do not hard-code the exact SAM 2 inference call.

Instead:

Create a SAM2Adapter abstraction:

class SAM2Adapter:
    load_model()
    segment_image()
    segment_with_text()
    segment_with_points()
    segment_with_box()
    initialize_video()
    propagate_video()
    refine_mask()
    release()

Implement the adapter based on the currently installed official SAM 2 API.

==================================================
13. POSITIVE CLICK PROMPT
==================================================

Allow the user to click on an object.

A positive click is displayed as:

+ point

When executed:

1. Send point coordinates to SAM 2.
2. Generate candidate mask(s).
3. Display candidate masks.
4. Allow candidate selection.
5. Save selected mask.

Coordinate transformation is critical.

The canvas may be zoomed and panned, but SAM 2 must receive coordinates relative to the ORIGINAL frame.

Implement:

screen → canvas → image → original-image coordinates

and the inverse transform:

original-image → canvas → screen

Never use displayed/zoomed coordinates directly for model inference.

==================================================
14. NEGATIVE CLICK PROMPT
==================================================

Allow the user to add negative clicks.

Display:

- Positive clicks using one visual marker
- Negative clicks using another visual marker

When the user executes/refines segmentation:

Send:

positive_points
negative_points

to SAM 2 using the currently supported visual-prompt API.

Allow multiple positive and negative points.

Example:

Positive:
(812, 422)
(835, 455)

Negative:
(750, 410)
(920, 490)

The user must be able to add, remove, and move prompt points before rerunning inference.

==================================================
15. BOUNDING BOX / EXEMPLAR PROMPT
==================================================

Allow the user to draw a rectangle around an object.

Workflow:

Mouse Down
↓
Drag
↓
Mouse Up
↓
Show rectangle
↓
Send rectangle/exemplar prompt to SAM 2
↓
Generate mask
↓
Display mask
↓
Accept / refine / reject

The original image coordinates must be used for model inference.

==================================================
16. MASK DISPLAY
==================================================

Display generated masks as a semi-transparent overlay.

Requirements:

- Adjustable mask opacity
- Object-specific visibility
- Toggle all masks
- Hide/show individual objects
- Different visual appearance for selected object
- Polygon outline
- Vertices
- Bounding box
- Object ID
- Class name

Provide:

[Show Masks]
[Show Polygons]
[Show Bounding Boxes]
[Show Vertices]
[Show Object IDs]

==================================================
17. MASK TO POLYGON CONVERSION
==================================================

This is one of the most important parts.

SAM 2 produces masks.

YOLO segmentation requires polygons.

Implement a reliable mask-to-polygon conversion pipeline.

Algorithm:

1. Receive binary mask.
2. Remove invalid pixels.
3. Optionally apply configurable morphological cleanup.
4. Find contours.
5. Select appropriate object contour(s).
6. Simplify contour using configurable polygon approximation.
7. Remove duplicate vertices.
8. Remove very short edges.
9. Ensure at least 3 valid points.
10. Validate polygon.
11. Normalize coordinates only during YOLO export.

Use OpenCV contours.

Allow a configurable:

Polygon Simplification Tolerance

Example:

0.1%
0.25%
0.5%
1.0%
2.0%

Do not overly simplify masks by default.

Store the high-quality original mask as well as the editable polygon whenever practical.

The application must support polygon regeneration from the mask.

==================================================
18. HOLES AND COMPLEX MASKS
==================================================

Handle masks with holes carefully.

The YOLO polygon representation should not be assumed to preserve arbitrary mask topology perfectly.

When a mask contains complex internal holes:

1. Detect the topology.
2. Preserve the source binary mask internally.
3. Generate the closest valid YOLO polygon representation supported by the selected export format.
4. Warn the user when topology cannot be represented exactly.
5. Never silently claim pixel-perfect equivalence when it is not possible.

==================================================
19. MANUAL POLYGON EDITOR
==================================================

Implement a professional polygon editing interface.

The user must be able to:

- drag vertices
- add vertices
- delete vertices
- move the entire polygon
- split polygon
- merge polygons where practical
- undo
- redo
- redraw polygon
- delete polygon
- duplicate annotation
- change class
- change object ID

Mouse interaction:

Left-click:
Select vertex/object

Double-click:
Add vertex

Right-click:
Context menu

Delete:
Delete selected vertex/object

Shift + click:
Add point where appropriate

Ctrl + Z:
Undo

Ctrl + Shift + Z:
Redo

==================================================
20. POLYGON VALIDATION
==================================================

Validate polygons before saving.

Check:

- minimum 3 vertices
- finite coordinates
- coordinates within image boundaries
- no invalid NaN
- no duplicated consecutive points
- no zero-area polygon
- no catastrophic self-intersection
- valid object association
- valid class ID

If invalid:

Do not export it.

Highlight the annotation in the UI.

Display an actionable error.

==================================================
21. OBJECT MANAGEMENT
==================================================

Each annotated object should have:

object_id
class_id
class_name
frame_id
source_frame_index
mask
polygon
bounding_box
confidence
source
tracking_status
modified
visibility
created_at
updated_at

Example:

{
    "object_id": "obj_00012",
    "class_id": 0,
    "class_name": "scooter",
    "frame_id": 105,
    "source_frame_index": 2210,
    "confidence": 0.91,
    "source": "sam2",
    "tracking_status": "tracked",
    "modified": true
}

==================================================
22. MULTIPLE OBJECTS
==================================================

The application must support multiple instances of the same class.

Example:

Frame:
3 scooters

Annotations:

Object 1:
class = scooter
object_id = obj_001

Object 2:
class = scooter
object_id = obj_002

Object 3:
class = scooter
object_id = obj_003

Do not merge them simply because they have the same class.

==================================================
23. OBJECT TRACKING / PROPAGATION
==================================================

Implement video propagation using the official SAM 2 video capabilities where supported.

Primary workflow:

1. User annotates an object on frame N.
2. User selects:
   "Propagate Forward"
3. SAM 2 video/session logic propagates the object through subsequent frames.
4. Generate masks for each propagated frame.
5. Convert masks to polygons.
6. Display generated annotations.
7. Mark propagated annotations as:
   source = "sam2_track"

Also allow:

Propagate Backward

Propagate Forward N Frames

Propagate to End

Propagate Between Keyframes

IMPORTANT:

Do not automatically accept every propagated annotation.

Implement confidence/review status:

Auto
Needs Review
Accepted
Rejected
Manually Corrected

==================================================
24. KEYFRAME SYSTEM
==================================================

Implement keyframes.

A keyframe is a frame on which the user manually confirms or edits annotations.

Example:

Frame 100:
Manual annotation

Frames 101–130:
SAM propagation

Frame 131:
Manual correction

Frames 132–180:
Propagation

This should create:

Keyframe 100
Propagation 101–130
Keyframe 131
Propagation 132–180

Allow the user to define propagation boundaries.

==================================================
25. TEMPORAL INTERPOLATION
==================================================

Where appropriate, support interpolation between confirmed keyframes.

Do not invent object movement when tracking is unreliable.

If interpolation is used:

- preserve polygon vertex correspondence where possible
- otherwise use mask-based or tracking-based propagation
- mark generated annotations as automatically generated
- require review

==================================================
26. ANNOTATION CONFIDENCE
==================================================

Store confidence where SAM 2 exposes a meaningful confidence/quality metric.

Display:

High
Medium
Low

or numeric confidence.

Never manufacture confidence values.

If the model does not provide a meaningful confidence score for a specific operation, use:

confidence = null

and display:

N/A

==================================================
27. REVIEW WORKFLOW
==================================================

Provide a dedicated Review Mode.

Review Mode must allow:

Previous Issue
Next Issue
Accept
Reject
Edit
Regenerate
Re-run SAM
Delete

Identify potential issues:

- missing annotation
- mask too small
- mask too large
- polygon invalid
- polygon outside image
- tracking discontinuity
- sudden area change
- class mismatch
- duplicate object
- missing object
- annotation crossing frame boundaries unexpectedly

Use heuristics to flag suspicious results, but do not automatically delete them.

==================================================
28. FRAME DIFFERENCING
==================================================

Optionally calculate basic temporal diagnostics:

- object area variation
- centroid movement
- bounding box movement
- IoU between consecutive masks
- polygon area change

Flag sudden changes.

Example:

Frame 120 area:
10,230 px

Frame 121 area:
10,450 px

Frame 122 area:
45,800 px

Flag:

"Sudden mask area change detected."

==================================================
29. SAVE PROJECT
==================================================

The project must be persistent.

Use a project directory such as:

my_project/
│
├── project.json
│
├── source/
│   └── source_video.mp4
│
├── frames/
│   ├── frame_00000001.jpg
│   ├── frame_00000002.jpg
│   └── ...
│
├── thumbnails/
│   └── ...
│
├── masks/
│   ├── frame_00000001/
│   │   ├── obj_001.png
│   │   └── obj_002.png
│   └── ...
│
├── annotations/
│   ├── frame_00000001.json
│   └── ...
│
├── previews/
│   └── ...
│
└── export/
    └── ...

project.json must include:

- project name
- creation time
- source video metadata
- frame extraction settings
- classes
- annotation metadata
- SAM 2 configuration
- dataset split configuration
- application version

==================================================
30. AUTOSAVE
==================================================

Implement autosave.

Default:

Every 30 seconds

Also save after:

- annotation creation
- annotation deletion
- polygon modification
- class modification
- propagation
- frame extraction settings change

Autosave must be atomic.

Write to temporary file and replace the original project file safely.

==================================================
31. UNDO / REDO
==================================================

Implement command-based or snapshot-based undo/redo.

Undo must support:

- create polygon
- delete polygon
- move vertex
- add vertex
- delete vertex
- change class
- delete object
- mask replacement
- propagation
- manual corrections

==================================================
32. DATASET SPLITTING
==================================================

Before export allow:

Train %
Validation %
Test %

Default:

Train = 70%
Validation = 20%
Test = 10%

Allow:

80 / 10 / 10
70 / 20 / 10
90 / 10 / 0

IMPORTANT:

For video-derived datasets, provide an option:

"Split by source frame sequence"

to reduce temporal leakage.

Prefer grouping nearby frames or sequences rather than randomly placing adjacent video frames into train and validation.

Provide an option for:

Random
Sequential
Grouped-by-video-segment

The chosen strategy must be recorded in project metadata.

==================================================
33. DATASET EXPORT
==================================================

Export to:

YOLOv8 Instance Segmentation format

Target structure:

dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml

Use matching filenames.

Example:

images/train/frame_00001025.jpg

labels/train/frame_00001025.txt

==================================================
34. YOLO LABEL FORMAT
==================================================

For each object write one row:

<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

Coordinates MUST be normalized to the range:

0.0 to 1.0

Normalization:

x_normalized = x_pixel / image_width
y_normalized = y_pixel / image_height

Class ID starts at 0.

Example:

0 0.3125 0.4213 0.3191 0.4170 0.3268 0.4217 ...

Do not export pixel coordinates.

Do not export bounding-box format.

This project is specifically exporting segmentation polygons.

==================================================
35. YOLO POLYGON VALIDATION
==================================================

Before writing each YOLO label:

1. Verify class ID exists.
2. Verify polygon has at least 3 points.
3. Verify all coordinates are finite.
4. Clamp tiny floating-point deviations into [0,1].
5. Reject severely invalid coordinates.
6. Remove duplicate points.
7. Ensure polygon area > minimum threshold.
8. Ensure the output row has an even number of coordinate values after class ID.
9. Verify the number of polygon points is >= 3.

If invalid:

Report:

Frame
Object
Class
Error

Do not silently skip the annotation.

Provide:

"Export with errors"
and
"Cancel export"

Default should be Cancel export when errors are found.

==================================================
36. DATA.YAML
==================================================

Generate:

data.yaml

Example:

path: /absolute/or/relative/dataset/path

train: images/train
val: images/val
test: images/test

names:
  0: scooter
  1: person
  2: helmet

Do not include a "background" class merely because unannotated image areas exist.

==================================================
37. NEGATIVE / EMPTY FRAMES
==================================================

Support frames containing zero objects.

For an empty frame:

- Export the image.
- Create an empty `.txt` label file OR handle it according to the chosen dataset-validation convention.
- Ensure the dataset remains loadable by Ultralytics.

The UI should display:

No Objects

and allow the frame to be intentionally marked as a negative example.

Do not confuse:
"No annotation"
with
"annotation processing failed."

Store explicit frame status:

unreviewed
negative
annotated
reviewed
rejected

==================================================
38. EXPORT MASKS
==================================================

In addition to YOLO polygons, optionally export original binary masks.

Example:

masks/
├── train/
├── val/
└── test/

Use PNG.

Allow user to choose:

[x] Export YOLO labels
[x] Export source images
[x] Export binary masks
[x] Export visualization previews
[x] Export project JSON
[x] Generate ZIP

==================================================
39. VISUALIZATION PREVIEWS
==================================================

Generate optional preview images.

For example:

preview/frame_000001.jpg

containing:

- source image
- colored polygon overlay
- class name
- object ID

Also optionally generate:

mask-only image

polygon-only image

Use separate output folders.

==================================================
40. EXPORT VALIDATOR
==================================================

After export automatically validate the dataset.

Check:

Dataset root exists

images/train exists

images/val exists

labels/train exists

labels/val exists

data.yaml exists

Every image has a corresponding label file

Every label file parses correctly

Every class ID is valid

Every polygon has >= 3 points

All coordinates are 0–1

No NaN

No infinity

Image dimensions are readable

Class names match IDs

No missing image/label pairs

Generate a validation report.

Example:

Dataset Validation

Images:
Train: 1,240
Val: 320
Test: 160

Objects:
Train: 1,982
Val: 507
Test: 239

Classes:
scooter: 1,850
person: 720
helmet: 158

Errors:
0

Warnings:
3

==================================================
41. DATASET STATISTICS
==================================================

Display:

Total Images
Total Annotated Images
Total Negative Images
Total Objects
Objects per Class
Images per Split
Average Polygon Points
Minimum Polygon Area
Maximum Polygon Area

Optional charts:

Class distribution
Split distribution
Object count per image

==================================================
42. EXPORT ZIP
==================================================

Provide:

Export Dataset

Then:

Create ZIP

Example:

scooter_dataset_yolov8.zip

ZIP should contain:

dataset/
├── images/
├── labels/
└── data.yaml

Do not include enormous temporary cache files.

==================================================
43. FILE NAMING
==================================================

Use deterministic names.

Recommended:

videoName_frame_00000001.jpg

Example:

scooter_video_frame_00000001.jpg

Corresponding annotation:

scooter_video_frame_00000001.txt

This prevents collisions when combining datasets.

Sanitize file names.

Never use characters unsupported by common operating systems.

==================================================
44. IMAGE FORMAT
==================================================

Default:

JPEG

Allow:

PNG

JPEG quality should be configurable.

The original image dimensions must not change merely because the canvas uses a scaled rendering.

Do not resize exported images unless the user explicitly selects an export resolution.

==================================================
45. COORDINATE SYSTEM
==================================================

This is critical.

Maintain three coordinate spaces:

1. Video/source coordinates
2. Image coordinates
3. GUI/canvas coordinates

Use a dedicated coordinate transformation class.

Example:

CoordinateTransformer

Methods:

image_to_canvas()
canvas_to_image()
image_to_screen()
screen_to_image()

Ensure annotations remain accurate under:

- zoom
- pan
- resize
- high-DPI displays
- different aspect ratios

==================================================
46. ZOOM AND PAN
==================================================

Annotation canvas must support:

Zoom In
Zoom Out
Fit Image
100%
200%
400%
800%

Mouse-wheel zoom

Middle mouse or space-drag pan

When zooming, annotations must remain accurately aligned.

==================================================
47. CLASS PANEL
==================================================

Right-side class panel:

Classes

0 scooter
1 person
2 helmet
3 car

Each class should support:

Select
Rename
Add
Delete
Reorder where safe
Set active class

Keyboard shortcut:

Number keys can select classes where practical.

==================================================
48. OBJECT PANEL
==================================================

Show all objects in current frame.

Example:

Objects

☑ obj_001 — scooter
☑ obj_002 — person
☑ obj_003 — scooter

Clicking an object selects it on the canvas.

Allow:

Show/Hide
Select
Rename ID
Change class
Delete
Lock

Locked objects cannot accidentally be modified.

==================================================
49. ANNOTATION SOURCES
==================================================

Store how every annotation was created:

manual
sam2_text
sam2_point
sam2_box
sam2_track
interpolated
imported

Display this information in the properties panel.

==================================================
50. MANUAL ANNOTATION
==================================================

Implement a freehand/manual polygon tool.

Workflow:

Select class
Click points
Double-click to close

or:

Click starting point again

Allow cancelling the polygon before completion.

Do not create a polygon with fewer than three points.

==================================================
51. MASK REFINEMENT WORKFLOW
==================================================

Typical user workflow:

1. Select frame.
2. Select class "scooter".
3. Select SAM 2.
4. Click scooter.
5. SAM 2 generates mask.
6. User sees mask.
7. User adds positive point if required.
8. User adds negative point on background.
9. Re-run refinement.
10. Accept mask.
11. Convert mask to polygon.
12. Edit polygon.
13. Mark frame reviewed.
14. Propagate to next frames.

This workflow must be extremely fast.

Optimize for repeated annotation operations.

==================================================
52. SAM 2 VIDEO WORKFLOW
==================================================

Use SAM 2's video segmentation/tracking capability when available.

Design:

SAM2VideoSession

Responsibilities:

initialize(video/frame sequence)
add_prompt(frame_index, prompt)
propagate_forward(...)
propagate_backward(...)
refine(...)
get_masks(...)
release()

Do not mix UI logic into the video session.

The adapter must translate application-level annotation requests into the currently supported SAM 2 API.

==================================================
53. TRACKING FAILURE HANDLING
==================================================

If tracking loses an object:

Do not continue blindly.

Show:

Tracking issue detected around frame N.

Allow:

Re-prompt
Add positive point
Add negative point
Redraw box
Manually edit
Stop propagation

When the user corrects the object:

resume propagation from the corrected frame.

==================================================
54. PERFORMANCE REQUIREMENTS
==================================================

The application should:

- avoid blocking the GUI
- cache model components
- avoid repeatedly loading the SAM model
- process batches where appropriate
- release GPU memory when necessary
- use inference mode for inference
- use half precision where supported and safe
- avoid storing unnecessary duplicate images in memory

Use:

torch.inference_mode()

where appropriate.

Make precision configurable:

FP32
FP16
BF16

Only expose modes supported by the detected hardware/model.

==================================================
55. THREADING
==================================================

Never perform long-running SAM inference directly on the UI thread.

Create workers for:

Video extraction
SAM inference
Propagation
Mask-to-polygon conversion
Dataset export
Dataset validation
ZIP creation

Workers must emit progress signals.

Example:

0%
25%
50%
75%
100%

==================================================
56. ERROR HANDLING
==================================================

Handle:

Missing video
Unreadable video
Unsupported codec
Corrupt frame
Missing model
Invalid checkpoint
CUDA unavailable
GPU out of memory
SAM 2 initialization failure
SAM inference failure
Invalid prompt
Invalid polygon
Disk full
Permission error
Project corruption
Invalid class
Export failure

Do not crash the application.

Show user-friendly error messages.

Also write technical diagnostics to:

logs/app.log

==================================================
57. GPU OUT-OF-MEMORY RECOVERY
==================================================

If CUDA OOM occurs:

1. Catch exception.
2. Clear unused CUDA memory where safe.
3. Release temporary tensors.
4. Suggest lowering resolution/precision where appropriate.
5. Allow retry.
6. Never corrupt the project.

Do not automatically destroy the user's annotation state.

==================================================
58. SETTINGS
==================================================

Provide a Settings screen.

Options:

Model checkpoint
Device
Precision
Frame cache location
Cache size
JPEG quality
Polygon simplification
Minimum polygon area
Autosave interval
Default train/val/test split
Auto-preview
Mask opacity
Default propagation interval

Persist settings.

==================================================
59. PROJECT IMPORT / RESUME
==================================================

The user must be able to close the application and reopen the project.

On reopening:

- restore classes
- restore frame metadata
- restore annotations
- restore selected frame if stored
- restore dataset configuration
- restore keyframes
- restore tracking information

Do not rerun SAM unnecessarily.

==================================================
60. RECOVERY
==================================================

Maintain:

project.json
project.backup.json

Before overwriting project.json:

1. Write temp file.
2. Validate JSON.
3. Backup current project.
4. Atomically replace.

==================================================
61. DATA INTEGRITY
==================================================

Every annotation must reference valid:

frame_id
class_id
object_id

Do not leave orphaned objects.

When deleting a frame annotation, remove all dependent annotation objects.

When deleting a class:

Do NOT automatically delete annotations.

Instead require the user to:

reassign class
or explicitly delete affected annotations.

==================================================
62. YOLO EXPORT OPTIONS
==================================================

Export dialog:

Dataset Name
Output Folder

Split:
Train __ %
Val __ %
Test __ %

Sampling:
All annotated frames
Reviewed frames only
Accepted only

Image format:
JPEG
PNG

Polygon Simplification:
None
Low
Medium
High
Custom

Export:
[x] Images
[x] YOLO labels
[x] data.yaml
[x] Masks
[x] Preview images
[x] Validation report
[x] Project metadata

ZIP:
[x] Create ZIP

==================================================
63. YOLO DATASET COMPATIBILITY
==================================================

The exported dataset must follow the standard Ultralytics YOLO instance-segmentation directory and label conventions.

Each image must have a matching `.txt` file.

Each label row must contain:

class_id + normalized polygon coordinates

Do not generate detection-format rows.

Do not mix bounding-box annotations and polygon annotations inside segmentation label files.

==================================================
64. DATASET VALIDATION REPORT
==================================================

Generate:

validation_report.json

and:

validation_report.txt

Example:

{
    "valid": true,
    "images": 1720,
    "labels": 1720,
    "objects": 3012,
    "classes": {
        "scooter": 2498,
        "person": 401,
        "helmet": 113
    },
    "errors": [],
    "warnings": []
}

==================================================
65. OPTIONAL YOLO TRAINING TEST
==================================================

Provide an optional menu item:

"Verify with Ultralytics"

If Ultralytics is installed, perform a lightweight dataset loading test.

Do not start a complete model training run automatically.

At minimum verify that:

- data.yaml parses
- images are readable
- segmentation labels parse
- classes are valid

If supported, run a small dataset sanity check.

==================================================
66. UI/UX DESIGN
==================================================

The interface should resemble a modern professional computer-vision annotation tool.

Use:

- dark theme by default
- clear toolbar
- left thumbnail panel
- central canvas
- right annotation panel
- bottom timeline
- status bar
- progress indicators

Avoid clutter.

The most common operations should be one or two clicks away.

Primary toolbar:

Open Video
Save
Undo
Redo

Select
SAM 2
Point+
Point-
Box
Polygon
Edit
Delete

Track
Propagate

Review
Export

==================================================
67. STATUS BAR
==================================================

Display:

Project
Frame
Original Video Frame
Timestamp
Objects
Selected Object
SAM Status
GPU
Save Status

Example:

Frame 125 / 1842
Source Frame 3740
00:02:04.666
Objects: 3
GPU: RTX XXXX
Saved

==================================================
68. ANNOTATION COLORING
==================================================

Use deterministic colors for object visualization.

The actual exported dataset must not depend on display colors.

Display color must be UI-only.

==================================================
69. LOGGING
==================================================

Use Python logging.

Levels:

DEBUG
INFO
WARNING
ERROR
CRITICAL

Write:

logs/app.log

Log:

- video loading
- frame extraction
- model loading
- inference duration
- propagation
- export
- validation
- exceptions

Never log sensitive credentials.

==================================================
70. CONFIGURATION
==================================================

Create:

config/defaults.yaml

Example:

device: cuda
precision: fp16

frame:
  default_sampling: every_n
  every_n: 10

polygon:
  simplify_tolerance: 0.005
  min_area: 20

dataset:
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1

ui:
  mask_opacity: 0.45

==================================================
71. COMMAND LINE INTERFACE
==================================================

Although the main application is graphical, implement basic CLI commands.

Examples:

python main.py

python main.py --project project_directory

python main.py --extract-frames input.mp4 --every-n 10

python main.py --validate-dataset dataset/

python main.py --export project_directory --output dataset/

Keep CLI functionality independent from GUI functionality.

==================================================
72. AUTOMATIC FRAME EXTRACTION CLI
==================================================

Implement:

python -m sam2_annotator.extract \
    --video input.mp4 \
    --output frames/ \
    --every-n 10

Output metadata:

frames.json

==================================================
73. EXPORT CLI
==================================================

Implement:

python -m sam2_annotator.export \
    --project project/ \
    --output dataset/

This must produce:

dataset/
images/
labels/
data.yaml

==================================================
74. PROJECT SCHEMA
==================================================

Define a documented JSON schema.

At minimum:

{
    "schema_version": 1,
    "application_version": "...",
    "project_name": "...",
    "video": {},
    "classes": [],
    "frames": [],
    "annotations": [],
    "settings": {},
    "dataset": {}
}

Make schema versioned.

==================================================
75. TESTING
==================================================

Create unit tests for:

- video metadata
- frame extraction
- frame naming
- coordinate transformation
- polygon validation
- mask-to-polygon conversion
- normalization
- YOLO label generation
- data.yaml generation
- dataset validation
- project serialization
- project recovery

Create integration tests using a tiny sample video and mockable SAM 2 adapter.

The SAM2 adapter must be injectable so that tests do not require downloading or loading the full model.

==================================================
76. MOCK SAM2 ADAPTER
==================================================

Implement:

MockSAM2Adapter

for automated tests.

It should return deterministic synthetic masks.

This is only for tests.

Do NOT use it in production mode.

Production must use the real SAM 2 implementation.

==================================================
77. MODEL ADAPTER DESIGN
==================================================

Do not hard-code SAM 2 implementation throughout the project.

Use:

SAM2AdapterInterface

SAM2LocalAdapter

MockSAM2Adapter

If the actual SAM 2 API changes:

only SAM2LocalAdapter should require modification.

Document precisely which official SAM 2 APIs are being used.

Before implementation, inspect the installed SAM 2 repository/package to determine:

- model initialization
- checkpoint loading
- image predictor
- video predictor/session
- prompt syntax
- returned mask format
- returned scores
- video propagation API

Do not invent API names.

==================================================
78. SAM 2 PROMPT ABSTRACTION
==================================================

Create generic application-level prompt classes.

Example:

TextPrompt
PointPrompt
BoxPrompt
MaskRefinementPrompt

Example:

PointPrompt(
    positive_points=[...],
    negative_points=[...]
)

Then translate these into the installed SAM 2 API inside the adapter.

==================================================
79. MASK FORMAT
==================================================

Internally use:

numpy.ndarray

with a binary mask representation.

Preferred:

dtype = uint8
0 = background
1 = object

Where model output contains logits/probabilities:

convert explicitly to the correct binary representation.

Never confuse:

probability
logit
binary mask

==================================================
80. IMAGE/MASK DIMENSION VALIDATION
==================================================

Before polygon conversion:

assert:

mask_height == image_height
mask_width == image_width

If not:

resize the mask appropriately using nearest-neighbor interpolation only when required.

Never use bilinear interpolation for binary segmentation masks unless explicitly justified.

==================================================
81. BOUNDING BOX GENERATION
==================================================

Automatically derive bounding boxes from masks.

Compute:

x_min
y_min
x_max
y_max

Use bounding boxes for display and diagnostics.

Do not export these instead of polygons.

==================================================
82. POLYGON SIMPLIFICATION
==================================================

Polygon simplification must be configurable.

Use OpenCV approximation or another robust geometry algorithm.

Never simplify below 3 valid points.

Keep a copy of the original high-resolution contour.

Allow:

Original
Simplified

representation.

==================================================
83. LARGE VIDEO SUPPORT
==================================================

Do not require the complete extracted frame set to remain in RAM.

Support large videos by using:

- disk cache
- lazy image loading
- thumbnails
- background extraction
- optional frame prefetching

Show disk usage estimates.

==================================================
84. LONG VIDEO SUPPORT
==================================================

For very large videos:

Allow extraction in chunks.

Example:

Extract frames 0–10,000
Extract frames 10,001–20,000
etc.

Do not crash because a video contains millions of frames.

==================================================
85. THUMBNAILS
==================================================

Generate thumbnails asynchronously.

Use smaller images for the frame browser.

Never perform SAM 2 inference on thumbnails.

SAM 2 inference should operate on the original-resolution frame or a deliberate model-input representation supported by the SAM 2 implementation.

Maintain coordinate mapping back to original image coordinates.

==================================================
86. IMAGE RESOLUTION
==================================================

Do not permanently resize the source frame.

If model inference requires a transformed input:

1. record original dimensions
2. apply model preprocessing
3. run inference
4. map output mask back to original image coordinates

All saved annotations must ultimately correspond to original frame coordinates.

==================================================
87. MULTI-MONITOR / HIGH-DPI SUPPORT
==================================================

Ensure the GUI works on:

- 1080p
- 1440p
- 4K
- Windows scaling above 100%

Avoid hard-coded pixel dimensions wherever possible.

==================================================
88. CONTEXT MENUS
==================================================

Right-click on object:

Edit
Refine with SAM 2
Change Class
Hide
Lock
Delete
Track
Propagate

Right-click on frame:

Annotate
Mark Negative
Clear Annotations
Review
Export Frame

==================================================
89. AUTO-ANNOTATION MODE
==================================================

Provide:

"Auto Annotate"

Workflow:

1. User enters class prompt.
2. Application runs SAM 2 on selected frames.
3. Candidate masks are generated.
4. Masks are converted into polygons.
5. Results are displayed for review.
6. User accepts/rejects them.

Never automatically export unreviewed annotations unless the user explicitly chooses:

"Export All Generated"

==================================================
90. BATCH ANNOTATION
==================================================

Allow the user to select:

- frame range
- every Nth frame
- specific frames
- entire extracted sequence

Then execute SAM-based annotation in the background.

Show:

Processed
Successful
Rejected
Failed
Needs Review

==================================================
91. REVIEW QUEUE
==================================================

Create a review queue containing:

- low-confidence annotations
- invalid polygons
- tracking failures
- unusual mask changes
- unmatched class
- frames with missing expected objects

Provide:

Next Review Item

==================================================
92. EXPORT PREVIEW
==================================================

Before actual export display:

Images:
1720

Objects:
3012

Train:
1204 images

Val:
344 images

Test:
172 images

Classes:
3

Potential Issues:
2 warnings

Then allow:

Cancel
Export

==================================================
93. DATASET MANIFEST
==================================================

Generate:

dataset_manifest.json

Containing:

- source project
- source video
- generation timestamp
- split strategy
- class mapping
- frame extraction settings
- polygon simplification settings
- application version

==================================================
94. README GENERATION
==================================================

Generate a dataset README.

It must describe:

Dataset name
Source video
Class names
Number of images
Number of objects
Train/val/test split
Annotation type
Export format
Frame sampling method

Do not make claims about annotation accuracy that have not been measured.

==================================================
95. LICENSE / MODEL NOTICE
==================================================

Because the application integrates Meta SAM 2:

- retain any required SAM 2 license notices
- clearly separate application code from the model package
- do not redistribute model weights unless permitted
- consult the currently installed SAM 2 repository/license before packaging or redistribution
- document model licensing requirements in README.md

==================================================
96. SECURITY
==================================================

The application should not require API keys.

Do not transmit video data externally.

Do not automatically connect to cloud services.

Any future cloud integrations must be explicitly configured.

==================================================
97. DEPENDENCY MANAGEMENT
==================================================

Create:

requirements.txt

and preferably:

pyproject.toml

Pin or constrain versions where necessary for compatibility.

Do not blindly install conflicting versions of:

torch
torchvision
CUDA-specific packages
SAM 2 dependencies

Clearly document SAM 2 environment requirements separately if the official project requires a specialized environment.

==================================================
98. WINDOWS SUPPORT
==================================================

The primary target should be Windows because the application is intended for local desktop use.

Ensure:

- Windows path handling
- UTF-8 filenames
- no Linux-only assumptions
- correct subprocess handling
- correct multiprocessing behavior

Use:

if __name__ == "__main__":

where multiprocessing requires it.

==================================================
99. APPLICATION STARTUP CHECKLIST
==================================================

On startup verify:

[✓] Python
[✓] PyTorch
[✓] CUDA
[✓] GPU
[✓] OpenCV
[✓] PySide6
[✓] SAM 2 package
[✓] SAM 2 checkpoint
[✓] Writable project directory

If a requirement is missing, show exactly what is missing.

Example:

SAM 2 model checkpoint not configured.

Please select:
Settings → SAM 2 Model → Checkpoint

==================================================
100. FIRST-RUN EXPERIENCE
==================================================

On first launch:

Show:

"Create Project"

Then:

1. Select video.
2. Enter project name.
3. Define classes.
4. Select frame sampling.
5. Extract frames.
6. Open annotation workspace.
7. Explain the basic SAM 2 workflow.

Provide a small guided help overlay.

==================================================
101. REQUIRED END-TO-END WORKFLOW
==================================================

The application MUST support the following exact sequence without manual file manipulation:

User selects:

input.mp4

↓

Application reads video metadata

↓

User selects:

Every 10 frames

↓

Application extracts frames

↓

Frame thumbnails appear

↓

User selects frame

↓

User selects class:

scooter

↓

User activates SAM 2

↓

User clicks object

↓

SAM 2 produces mask

↓

Mask displayed

↓

User adds negative point

↓

SAM 2 refines mask

↓

User accepts

↓

Mask converted to polygon

↓

Polygon displayed with vertices

↓

User corrects one vertex

↓

Annotation is saved

↓

User selects:

Propagate Forward

↓

SAM 2 propagates object

↓

Generated masks/polygons appear

↓

User reviews frames

↓

User corrects problematic frame

↓

Propagation resumes from corrected keyframe

↓

User selects:

Export Dataset

↓

Application creates:

images/
labels/
data.yaml

↓

Application validates dataset

↓

Application generates:

validation_report.json

↓

Application optionally creates:

dataset.zip

This exact workflow must be tested.

==================================================
102. ACCEPTANCE CRITERIA
==================================================

The project is considered complete only when all of the following are true.

A. VIDEO

The application can load a real video.

B. FRAME EXTRACTION

The application can extract selected frames without loading the entire video into RAM.

C. SAM 2

A real SAM 2 model is loaded and performs actual segmentation.

D. PROMPTS

The user can perform at least:

text prompt
positive point
negative point
box/exemplar prompt

E. POLYGON

SAM 2 masks can be converted to editable polygons.

F. EDITING

The user can manually edit polygon vertices.

G. MULTI-OBJECT

Multiple instances per frame are supported.

H. TRACKING

The application supports SAM 2 video propagation where the installed SAM 2 API supports it.

I. PERSISTENCE

Projects can be saved and reopened.

J. YOLO EXPORT

The application generates valid YOLO segmentation labels.

K. YAML

A valid data.yaml is generated.

L. VALIDATION

The exported dataset is automatically checked.

M. GUI

The interface remains responsive during long-running operations.

N. ERROR HANDLING

Invalid videos/models/annotations/export failures do not crash the application.

O. TESTS

Automated tests exist for major non-model components.

==================================================
103. IMPLEMENTATION PRIORITY
==================================================

Develop in phases.

PHASE 1:
Application shell
- PySide6
- Main window
- Project manager
- Settings
- logging

PHASE 2:
Video
- video loading
- metadata
- frame extraction
- thumbnails
- timeline

PHASE 3:
Annotation
- canvas
- point prompts
- box prompts
- manual polygons
- object manager

PHASE 4:
SAM 2
- adapter
- image segmentation
- prompt processing
- mask visualization

PHASE 5:
Polygon
- mask conversion
- polygon editing
- validation

PHASE 6:
Video propagation
- SAM 2 video session
- tracking
- keyframes
- correction workflow

PHASE 7:
Persistence
- project JSON
- autosave
- recovery
- undo/redo

PHASE 8:
Dataset
- train/val/test
- YOLO export
- data.yaml
- validation

PHASE 9:
Packaging
- documentation
- testing
- build instructions

==================================================
104. CODING RULES
==================================================

Write clean, maintainable Python.

Use:

- type hints
- dataclasses where appropriate
- descriptive function names
- docstrings for public classes/functions
- small testable functions
- centralized configuration
- structured logging

Avoid:

- massive single-file applications
- global mutable state
- hard-coded paths
- hard-coded GPU names
- hard-coded SAM 2 APIs
- blocking UI calls
- silent exception handling
- duplicated coordinate-transform logic

==================================================
105. DO NOT USE PLACEHOLDERS IN CORE FEATURES
==================================================

Do not implement:

pass

TODO

"implement later"

mock inference

fake masks

fake tracking

for the production workflow.

A mocked SAM 2 adapter is permitted ONLY for unit testing.

==================================================
106. README REQUIREMENTS
==================================================

README.md must include:

1. Project overview
2. Features
3. System requirements
4. Python version
5. GPU requirements
6. SAM 2 installation/setup
7. Checkpoint setup
8. Installation
9. Running the application
10. Creating a project
11. Extracting video frames
12. SAM 2 annotation workflow
13. Tracking workflow
14. Polygon editing
15. YOLO export
16. Dataset validation
17. CLI usage
18. Troubleshooting
19. Project directory structure
20. License notices

==================================================
107. USER DOCUMENTATION
==================================================

Create:

docs/
├── installation.md
├── user_guide.md
├── annotation_workflow.md
├── sam2_setup.md
├── yolo_export.md
└── troubleshooting.md

==================================================
108. FINAL DELIVERABLES
==================================================

The final implementation must produce:

1. Complete source code
2. requirements.txt
3. pyproject.toml
4. README.md
5. Documentation
6. Tests
7. Example configuration
8. Project schema
9. CLI
10. GUI application
11. SAM 2 adapter
12. Video extraction pipeline
13. Polygon editor
14. Dataset exporter
15. Dataset validator

==================================================
109. IMPORTANT IMPLEMENTATION DETAIL
==================================================

Before writing the SAM 2 integration:

INSPECT THE ACTUAL INSTALLED SAM 2 REPOSITORY AND DETERMINE THE CURRENT PUBLIC PYTHON API.

Do not rely on outdated examples from memory.

Create a small isolated integration test:

tests/test_sam2_integration.py

which verifies:

1. Model can initialize.
2. Model can load a test image/frame.
3. A simple prompt can be submitted.
4. A mask can be returned.
5. Mask dimensions can be retrieved.
6. Mask can be converted to NumPy.
7. Polygon extraction succeeds.

Only after this test succeeds should the production UI invoke the SAM 2 adapter.

==================================================
110. IMPORTANT EXPORT DETAIL
==================================================

The target dataset is YOLOv8 INSTANCE SEGMENTATION format.

Example:

dataset/
├── images/
│   ├── train/
│   │   ├── video_frame_000001.jpg
│   │   └── video_frame_000002.jpg
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   │   ├── video_frame_000001.txt
│   │   └── video_frame_000002.txt
│   ├── val/
│   └── test/
│
└── data.yaml

Example label:

0 0.120000 0.280000 0.145000 0.271000 0.172000 0.265000 0.201000 0.270000

The first value is the class ID.

All remaining values are polygon coordinates:

x1 y1 x2 y2 x3 y3 ...

Coordinates are normalized.

==================================================
111. EXAMPLE COMPLETE PROJECT
==================================================

For testing, create a small example project using:

example_video.mp4

Classes:

0 = scooter

Frame sampling:

every 5 frames

Then demonstrate:

Frame extraction
SAM 2 annotation
Polygon editing
Propagation
Export

The example should produce:

example_project/
...
example_dataset/
...

==================================================
112. QUALITY STANDARD
==================================================

Think like the developer of a commercial annotation tool.

The most important properties are:

Accuracy
Responsiveness
Data integrity
Recoverability
Reproducibility
Ease of use

Never sacrifice annotation coordinate accuracy for UI convenience.

Never silently lose annotations.

Never silently overwrite project data.

Never silently discard invalid annotations.

==================================================
113. FINAL DEVELOPMENT INSTRUCTION
==================================================

Start implementation now.

First inspect the installed Python/SAM 2 environment and determine the actual SAM 2 APIs available.

Then create the project structure.

Then implement the video pipeline.

Then implement the annotation canvas.

Then implement the real SAM 2 adapter.

Then implement mask-to-polygon conversion.

Then implement tracking/propagation.

Then implement project persistence.

Then implement YOLOv8 segmentation export.

Then implement validation.

Then write automated tests.

After implementation, run the application through an end-to-end test using a sample video.

Fix all errors found during testing.

Do not merely provide code snippets or an architecture document.

Produce the actual runnable application.

The application must ultimately allow a user to take:

INPUT VIDEO

and, without external annotation software, produce:

YOLOv8 INSTANCE-SEGMENTATION DATASET

with:

images/
labels/
data.yaml

and correctly normalized polygon annotations.