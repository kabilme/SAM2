"""Main application window for SAM2 Video Polygon Annotator."""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QFileDialog, QMessageBox, QLabel,
    QStatusBar, QApplication
)

from sam2_annotator.config.config import AppConfig
from sam2_annotator.project.project_manager import ProjectManager
from sam2_annotator.annotation.annotation_manager import AnnotationManager
from sam2_annotator.annotation.polygon import PolygonAnnotation
from sam2_annotator.video.frame_cache import FrameCache
from sam2_annotator.video.video_reader import VideoReader, VideoMetadata
from sam2_annotator.video.frame_extractor import FrameExtractor, FrameMetadata
from sam2_annotator.models.sam2_adapter import SAM2AdapterInterface, SAM2LocalAdapter, MockSAM2Adapter
from sam2_annotator.models.sam2_image_service import SAM2ImageService
from sam2_annotator.models.sam2_video_service import SAM2VideoService
from sam2_annotator.dataset import (
    DatasetSplitter, DatasetValidator, YOLOExporter,
    BaseDatasetExporter, create_exporter, EXPORT_FORMATS,
)

from sam2_annotator.ui.annotation_canvas import (
    AnnotationCanvas, MODE_SELECT, MODE_POINT_POS, MODE_POINT_NEG, MODE_BOX, MODE_POLYGON, MODE_EDIT
)
from sam2_annotator.ui.toolbar import MainToolBar
from sam2_annotator.ui.video_panel import VideoPanel
from sam2_annotator.ui.frame_timeline import FrameTimeline
from sam2_annotator.ui.class_panel import ClassPanel
from sam2_annotator.ui.properties_panel import PropertiesPanel
from sam2_annotator.ui.startup_dialog import StartupDialog
from sam2_annotator.ui.project_dialog import ProjectDialog
from sam2_annotator.ui.export_dialog import ExportDialog
from sam2_annotator.ui.settings_dialog import SettingsDialog
from sam2_annotator.ui.progress_dialog import ProgressDialog
from sam2_annotator.ui.delete_frame_dialog import DeleteFrameDialog
from sam2_annotator.utils.device_utils import get_system_diagnostics
from sam2_annotator.utils.logging_utils import logger


# Worker for background frame extraction
class FrameExtractionWorker(QThread):
    progress = Signal(int, int, str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(
        self,
        video_paths: List[Path],
        frames_dir: Path,
        thumbnails_dir: Path,
        params: Dict[str, Any],
        start_frame_id: int = 1,
    ):
        super().__init__()
        self.video_paths = [Path(p) for p in video_paths]
        self.frames_dir = Path(frames_dir)
        self.thumbnails_dir = Path(thumbnails_dir)
        self.params = params
        self.start_frame_id = start_frame_id
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            all_frames: List[FrameMetadata] = []
            curr_id = self.start_frame_id
            total_videos = len(self.video_paths)

            for v_idx, v_path in enumerate(self.video_paths):
                if self._cancelled:
                    break

                extractor = FrameExtractor(v_path, self.frames_dir, self.thumbnails_dir)

                def prog_cb(cur, tot, msg, v_i=v_idx, vp=v_path):
                    prefix = f"[Video {v_i + 1}/{total_videos}: {vp.name}] "
                    self.progress.emit(cur, tot, prefix + msg)

                v_frames = extractor.extract_frames(
                    strategy=self.params.get("sampling_strategy", "every_n"),
                    every_n=self.params.get("every_n", 10),
                    interval_seconds=self.params.get("interval_seconds", 1.0),
                    fixed_count=self.params.get("fixed_count", 100),
                    start_frame_id=curr_id,
                    video_name=v_path.name,
                    progress_callback=prog_cb,
                    is_cancelled=lambda: self._cancelled,
                )
                all_frames.extend(v_frames)
                curr_id += len(v_frames)

            self.finished.emit(all_frames)
        except Exception as e:
            logger.error("Frame extraction worker failed: %s", e)
            self.error.emit(str(e))


# Worker for background video propagation
class PropagationWorker(QThread):
    progress = Signal(int, int, str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, service: SAM2VideoService, initial_anno: PolygonAnnotation, target_frames: List[FrameMetadata]):
        super().__init__()
        self.service = service
        self.initial_anno = initial_anno
        self.target_frames = target_frames
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            results = self.service.propagate_object(
                initial_annotation=self.initial_anno,
                target_frames=self.target_frames,
                progress_callback=lambda cur, tot, msg: self.progress.emit(cur, tot, msg),
                is_cancelled=lambda: self._cancelled,
            )
            self.finished.emit(results)
        except Exception as e:
            logger.error("Propagation worker failed: %s", e)
            self.error.emit(str(e))


# Worker for background dataset and video export with progress bar
class DatasetExportWorker(QThread):
    progress = Signal(int, int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        exporter: BaseDatasetExporter,
        split_dict: Dict[str, List[FrameMetadata]],
        annotations_by_frame: Dict[int, List[PolygonAnnotation]],
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.exporter = exporter
        self.split_dict = split_dict
        self.annotations_by_frame = annotations_by_frame
        merged_params = dict(params or {})
        merged_params.update(kwargs)
        self.params = merged_params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            result = self.exporter.export(
                split_dict=self.split_dict,
                annotations_by_frame=self.annotations_by_frame,
                export_masks=self.params.get("export_masks", True),
                export_previews=self.params.get("export_previews", True),
                create_zip=self.params.get("create_zip", True),
                include_null_frames=self.params.get("include_null_frames", True),
                fps=self.params.get("fps", 30.0),
                alpha=self.params.get("alpha", 0.45),
                draw_bbox=self.params.get("draw_bbox", True),
                progress_callback=lambda cur, tot, msg: self.progress.emit(cur, tot, msg),
                is_cancelled=lambda: self._cancelled,
            )
            self.finished.emit(result)
        except Exception as e:
            logger.error("Dataset export worker failed: %s", e)
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, project_path: Optional[str] = None, device_override: Optional[str] = None):
        super().__init__()
        self.resize(1380, 880)

        # Core State
        self.config = AppConfig.load()
        if device_override:
            self.config.model.device = device_override

        self.project_manager = ProjectManager()
        self.annotation_manager = AnnotationManager()
        self.frame_cache = FrameCache(frames_dir=Path("frames"), max_size=self.config.frame.cache_size_limit)

        # Initialize SAM Adapter (Production or Mock for fallback)
        self.sam_adapter: SAM2AdapterInterface = SAM2LocalAdapter()
        self._init_sam_model()
        self._update_window_title()

        self.image_service = SAM2ImageService(self.sam_adapter)
        self.video_service = SAM2VideoService(self.sam_adapter, self.frame_cache)

        # Workers
        self.current_worker: Optional[QThread] = None

        # Build UI Components
        self._setup_ui()
        self._setup_menus()
        self._setup_shortcuts()
        self._setup_autosave()

        # Open project if provided
        if project_path and Path(project_path).exists():
            self.open_project(Path(project_path))

    def get_model_display_name(self) -> str:
        """Return the active model name for window title and UI labels."""
        ckpt = (
            getattr(self.sam_adapter, "checkpoint_path", "")
            or self.config.model.checkpoint_path
            or self.config.model.default_model_type
            or "sam2.1_hiera_tiny.pt"
        )
        return Path(ckpt).name

    def _update_window_title(self) -> None:
        """Update window title bar to mention active model name and project."""
        model_name = self.get_model_display_name()
        if self.project_manager.data and self.project_manager.data.project_name:
            self.setWindowTitle(f"{model_name} Video Polygon Annotator - {self.project_manager.data.project_name}")
        else:
            self.setWindowTitle(f"{model_name} Video Polygon Annotator")

    def _init_sam_model(self) -> None:
        """Attempt to load SAM model; fallback safely to CPU or inform user."""
        try:
            success = self.sam_adapter.load_model(
                checkpoint_path=self.config.model.checkpoint_path or self.config.model.default_model_type,
                device=self.config.model.device,
                precision=self.config.model.precision,
            )
            if not success:
                logger.warning("Could not load production SAM weights, falling back to mock adapter for safety.")
                self.sam_adapter = MockSAM2Adapter()
                self.sam_adapter.load_model(device="cpu")
        except Exception as e:
            logger.error("Exception loading SAM model: %s", e)
            self.sam_adapter = MockSAM2Adapter()
            self.sam_adapter.load_model(device="cpu")

        if hasattr(self, "toolbar"):
            self.toolbar.set_model_name(self.get_model_display_name())
        if hasattr(self, "status_sam_label"):
            self.status_sam_label.setText(
                f"{self.get_model_display_name()}: {'Loaded' if self.sam_adapter.is_loaded() else 'Not Loaded'}"
            )

    def _setup_ui(self) -> None:
        # Central Canvas
        self.canvas = AnnotationCanvas(self)
        self.setCentralWidget(self.canvas)

        # Main Toolbar with dynamic model name
        self.toolbar = MainToolBar(self, model_name=self.get_model_display_name())
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        # Connect toolbar mode signals
        self.toolbar.mode_changed.connect(self.canvas.set_mode)
        self.toolbar.btn_zoom_fit.triggered.connect(self.canvas.fit_image)
        self.toolbar.btn_zoom_100.triggered.connect(lambda: self.canvas.zoom_to(1.0))
        self.toolbar.btn_zoom_200.triggered.connect(lambda: self.canvas.zoom_to(2.0))
        self.toolbar.text_prompt_submitted.connect(self._on_text_prompt_submitted)

        # Left Dock: Frame Thumbnails
        self.video_panel = VideoPanel(self.frame_cache, self)
        self.left_dock = QDockWidget("Frames", self)
        self.left_dock.setWidget(self.video_panel)
        self.left_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)

        # Right Dock: Classes and Properties
        self.class_panel = ClassPanel(self)
        self.properties_panel = PropertiesPanel(self)
        if hasattr(self.config.model, "propagation_frames"):
            self.properties_panel.set_default_propagation_frames(self.config.model.propagation_frames)

        self.right_dock_classes = QDockWidget("Classes", self)
        self.right_dock_classes.setWidget(self.class_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock_classes)

        self.right_dock_props = QDockWidget("Properties & Objects", self)
        self.right_dock_props.setWidget(self.properties_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock_props)

        # Bottom Dock: Timeline
        self.timeline = FrameTimeline(self)
        self.bottom_dock = QDockWidget("Timeline", self)
        self.bottom_dock.setWidget(self.timeline)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bottom_dock)

        # Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self.status_project_label = QLabel("No Project Loaded")
        self.status_frame_label = QLabel("Frame: - / -")
        self.status_objects_label = QLabel("Objects: 0")
        self.status_sam_label = QLabel(f"{self.get_model_display_name()}: {'Loaded' if self.sam_adapter.is_loaded() else 'Not Loaded'}")
        self.status_device_label = QLabel(f"Device: {self.config.model.device.upper()}")
        self.status_save_label = QLabel("Saved")

        self.status_bar.addWidget(self.status_project_label, 2)
        self.status_bar.addWidget(self.status_frame_label, 1)
        self.status_bar.addWidget(self.status_objects_label, 1)
        self.status_bar.addWidget(self.status_sam_label, 1)
        self.status_bar.addWidget(self.status_device_label, 1)
        self.status_bar.addWidget(self.status_save_label, 1)

        # Connect inter-widget signals
        self.video_panel.frame_selected.connect(self._on_frame_selected)
        self.video_panel.delete_requested.connect(self.delete_frames)
        self.video_panel.mark_null_requested.connect(self._on_mark_null_batch)
        self.video_panel.mark_reviewed_requested.connect(self._on_mark_reviewed_batch)
        self.video_panel.toggle_keyframe_requested.connect(self._on_batch_toggle_keyframes)
        self.timeline.frame_changed.connect(self._on_frame_selected)
        self.timeline.keyframe_toggled.connect(self._on_keyframe_toggled)
        self.timeline.null_frame_clicked.connect(self._mark_frame_negative)
        self.timeline.delete_frame_clicked.connect(self.delete_current_frame)

        self.class_panel.class_selected.connect(self._on_class_selected)
        self.class_panel.classes_modified.connect(self._on_classes_modified)

        self.properties_panel.object_selected.connect(self.canvas.select_object)
        self.properties_panel.object_deleted.connect(self._on_object_deleted)
        self.properties_panel.propagate_requested.connect(self._on_propagate_requested)

        self.canvas.object_selected.connect(self.properties_panel.select_object)
        self.canvas.annotation_changed.connect(self._on_annotation_canvas_changed)
        self.canvas.prompt_point_added.connect(self._on_prompt_point_added)
        self.canvas.prompt_box_completed.connect(self._on_prompt_box_completed)
        self.canvas.manual_polygon_completed.connect(self._on_manual_polygon_completed)

        self.annotation_manager.register_change_listener(self._on_annotations_updated)

    def _setup_menus(self) -> None:
        mb = self.menuBar()

        # File Menu
        file_menu = mb.addMenu("&File")
        new_proj_act = file_menu.addAction("&New Project...")
        new_proj_act.setShortcut(QKeySequence("Ctrl+N"))
        new_proj_act.triggered.connect(self.new_project_dialog)

        open_proj_act = file_menu.addAction("&Open Project...")
        open_proj_act.setShortcut(QKeySequence("Ctrl+O"))
        open_proj_act.triggered.connect(self.open_project_dialog)

        save_proj_act = file_menu.addAction("&Save Project")
        save_proj_act.setShortcut(QKeySequence("Ctrl+S"))
        save_proj_act.triggered.connect(self.save_project)

        add_vid_act = file_menu.addAction("&Add Video(s) to Project...")
        add_vid_act.setShortcut(QKeySequence("Ctrl+Shift+V"))
        add_vid_act.triggered.connect(self.add_videos_dialog)

        file_menu.addSeparator()
        exit_act = file_menu.addAction("E&xit")
        exit_act.setShortcut(QKeySequence("Ctrl+Q"))
        exit_act.triggered.connect(self.close)

        # Edit Menu
        edit_menu = mb.addMenu("&Edit")
        undo_act = edit_menu.addAction("&Undo")
        undo_act.setShortcut(QKeySequence("Ctrl+Z"))
        undo_act.triggered.connect(self.annotation_manager.undo)

        redo_act = edit_menu.addAction("&Redo")
        redo_act.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        redo_act.triggered.connect(self.annotation_manager.redo)

        edit_menu.addSeparator()
        del_act = edit_menu.addAction("&Delete Selected Object")
        del_act.setShortcut(QKeySequence("Delete"))
        del_act.triggered.connect(self._delete_active_object)

        del_frame_act = edit_menu.addAction("Delete Current &Frame...")
        del_frame_act.setShortcut(QKeySequence("Ctrl+Delete"))
        del_frame_act.triggered.connect(self.delete_current_frame)

        # Annotation Menu
        anno_menu = mb.addMenu("&Annotation")
        clear_frame_act = anno_menu.addAction("&Clear Frame Annotations")
        clear_frame_act.triggered.connect(self._clear_current_frame)

        mark_rev_act = anno_menu.addAction("Mark Frame &Reviewed")
        mark_rev_act.setShortcut(QKeySequence("R"))
        mark_rev_act.triggered.connect(self._mark_frame_reviewed)

        mark_neg_act = anno_menu.addAction("Mark Frame as &Null Frame (No Objects)")
        mark_neg_act.setShortcut(QKeySequence("N"))
        mark_neg_act.triggered.connect(self._mark_frame_negative)

        mark_all_null_act = anno_menu.addAction("Mark All Unannotated as Null &Frames")
        mark_all_null_act.triggered.connect(self._mark_all_unannotated_as_null)

        toggle_kf_act = anno_menu.addAction("&Toggle Keyframe (K)")
        toggle_kf_act.setShortcut(QKeySequence("K"))
        toggle_kf_act.triggered.connect(self._toggle_current_keyframe)

        propagate_act = anno_menu.addAction("&Propagate Selected Object Forward")
        propagate_act.setShortcut(QKeySequence("Ctrl+P"))
        propagate_act.triggered.connect(self._on_propagate_action_triggered)

        # Dataset Menu
        data_menu = mb.addMenu("&Dataset")
        export_act = data_menu.addAction("&Export...")
        export_act.setShortcut(QKeySequence("Ctrl+E"))
        export_act.triggered.connect(self.export_dataset_dialog)

        val_act = data_menu.addAction("&Validate Exported Dataset...")
        val_act.triggered.connect(self.validate_dataset_dialog)

        # Settings & Help
        settings_menu = mb.addMenu("&Settings")
        pref_act = settings_menu.addAction("&Preferences...")
        pref_act.triggered.connect(self.open_settings_dialog)

        help_menu = mb.addMenu("&Help")
        diag_act = help_menu.addAction("&System Diagnostics...")
        diag_act.triggered.connect(self.open_diagnostics_dialog)
        about_act = help_menu.addAction("&About SAM2 Annotator")
        about_act.triggered.connect(self._show_about)

    def _setup_shortcuts(self) -> None:
        """Register application-level keyboard shortcuts."""
        # Frame stepping: Left / Right arrow, N, P
        step_next_act = QAction(self)
        step_next_act.setShortcuts([QKeySequence("Right"), QKeySequence("N")])
        step_next_act.triggered.connect(self.timeline.step_next)
        self.addAction(step_next_act)

        step_prev_act = QAction(self)
        step_prev_act.setShortcuts([QKeySequence("Left"), QKeySequence("P")])
        step_prev_act.triggered.connect(self.timeline.step_prev)
        self.addAction(step_prev_act)

        # Play / Pause: Space
        space_act = QAction(self)
        space_act.setShortcut(QKeySequence("Space"))
        space_act.triggered.connect(self.timeline.toggle_play)
        self.addAction(space_act)

        # Mode Shortcuts: S (Select), M (Manual Polygon), B (Box)
        s_act = QAction(self)
        s_act.setShortcut(QKeySequence("S"))
        s_act.triggered.connect(lambda: self._set_mode_ui(MODE_SELECT))
        self.addAction(s_act)

        m_act = QAction(self)
        m_act.setShortcut(QKeySequence("M"))
        m_act.triggered.connect(lambda: self._set_mode_ui(MODE_POLYGON))
        self.addAction(m_act)

        b_act = QAction(self)
        b_act.setShortcut(QKeySequence("B"))
        b_act.triggered.connect(lambda: self._set_mode_ui(MODE_BOX))
        self.addAction(b_act)

    def _set_mode_ui(self, mode: str) -> None:
        self.toolbar.set_active_mode(mode)
        self.canvas.set_mode(mode)

    def _setup_autosave(self) -> None:
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self._on_autosave)
        self.autosave_timer.start(self.config.ui.autosave_interval_seconds * 1000)

    def _on_autosave(self) -> None:
        if self.project_manager.project_dir and self.project_manager.is_dirty:
            self.project_manager.save_project(self.annotation_manager)
            self.status_save_label.setText("Autosaved")

    # ---------------- Project Actions ----------------

    def new_project_dialog(self) -> None:
        diag = ProjectDialog(self)
        if diag.exec():
            params = diag.get_project_params()
            self._create_project(params, params.get("video_metadatas", [params.get("video_metadata")]))

    def _create_project(self, params: Dict[str, Any], video_metadatas: Any) -> None:
        p_dir = params["project_dir"]
        self.project_manager.create_project(
            project_dir=p_dir,
            project_name=params["name"],
            video_metadata=video_metadatas,
            class_names=params["classes"],
            settings={
                "sampling": params.get("sampling_strategy", "every_n"),
                "every_n": params.get("every_n", 10),
                "interval_seconds": params.get("interval_seconds", 1.0),
                "fixed_count": params.get("fixed_count", 100),
            },
        )
        self.class_panel.set_classes(self.project_manager.classes)
        self.properties_panel.set_classes(self.project_manager.classes)
        self.frame_cache = FrameCache(self.project_manager.frames_dir, self.project_manager.thumbnails_dir)
        self.video_panel.frame_cache = self.frame_cache
        self.video_service.frame_cache = self.frame_cache

        # Start multi-video frame extraction in background worker
        video_paths = params.get("video_paths", [params.get("video_path")])
        prog_diag = ProgressDialog("Extracting Video Frames...", self)
        worker = FrameExtractionWorker(
            video_paths=video_paths,
            frames_dir=self.project_manager.frames_dir,
            thumbnails_dir=self.project_manager.thumbnails_dir,
            params=params,
            start_frame_id=1,
        )
        self.current_worker = worker

        worker.progress.connect(prog_diag.set_progress)
        prog_diag.cancelled.connect(worker.cancel)

        def on_finished(frames: List[FrameMetadata]):
            prog_diag.accept()
            self.project_manager.set_frames(frames)
            self.project_manager.save_project()
            self.video_panel.set_frames(frames)
            self.timeline.set_frames(frames)
            self.status_project_label.setText(f"Project: {params['name']}")
            self._on_frame_selected(1)
            QMessageBox.information(
                self,
                "Success",
                f"Extracted {len(frames)} frames from {len(video_paths)} video(s) successfully!"
            )

        def on_error(err_msg: str):
            prog_diag.reject()
            QMessageBox.critical(self, "Extraction Error", f"Failed to extract frames: {err_msg}")

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.start()
        prog_diag.exec()

    def add_videos_dialog(self) -> None:
        """Add one or more additional video files to the currently open project."""
        if not self.project_manager.project_dir:
            QMessageBox.warning(self, "No Project Open", "Please open or create a project before adding videos.")
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video File(s) to Add to Project",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*.*)"
        )
        if not paths:
            return

        video_paths = [Path(p) for p in paths]
        new_metadatas = []
        for vp in video_paths:
            try:
                reader = VideoReader(vp)
                if reader.metadata:
                    new_metadatas.append(reader.metadata)
                reader.close()
            except Exception as e:
                logger.error("Could not read video metadata for %s: %s", vp, e)

        if not new_metadatas:
            QMessageBox.critical(self, "Error", "None of the selected video files could be read.")
            return

        start_id = len(self.project_manager.frames) + 1
        sampling_strategy = self.project_manager.data.settings.get("sampling", "every_n")
        params = {
            "sampling_strategy": sampling_strategy,
            "every_n": self.project_manager.data.settings.get("every_n", getattr(self.config.frame, "every_n", 10)),
            "interval_seconds": self.project_manager.data.settings.get("interval_seconds", getattr(self.config.frame, "interval_seconds", 1.0)),
            "fixed_count": self.project_manager.data.settings.get("fixed_count", getattr(self.config.frame, "fixed_count", 100)),
        }

        prog_diag = ProgressDialog("Adding Videos & Extracting Frames...", self)
        worker = FrameExtractionWorker(
            video_paths=video_paths,
            frames_dir=self.project_manager.frames_dir,
            thumbnails_dir=self.project_manager.thumbnails_dir,
            params=params,
            start_frame_id=start_id,
        )
        self.current_worker = worker

        worker.progress.connect(prog_diag.set_progress)
        prog_diag.cancelled.connect(worker.cancel)

        def on_finished(new_frames: List[FrameMetadata]):
            prog_diag.accept()
            self.project_manager.frames.extend(new_frames)
            self.project_manager.add_video_metadata(new_metadatas)
            self.project_manager.save_project()
            self.video_panel.set_frames(self.project_manager.frames)
            self.timeline.set_frames(self.project_manager.frames)
            QMessageBox.information(
                self,
                "Videos Added",
                f"Successfully extracted and added {len(new_frames)} frames from {len(video_paths)} video(s)!\n"
                f"Project now contains {len(self.project_manager.frames)} total frames."
            )

        def on_error(err_msg: str):
            prog_diag.reject()
            QMessageBox.critical(self, "Extraction Error", f"Failed to add video frames: {err_msg}")

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.start()
        prog_diag.exec()

    def open_project_dialog(self) -> None:
        p_dir = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if p_dir:
            self.open_project(Path(p_dir))

    def open_project(self, project_dir: Path) -> None:
        if self.project_manager.load_project(project_dir, self.annotation_manager):
            if hasattr(self.annotation_manager, "tracker") and self.annotation_manager.tracker:
                for f in self.project_manager.frames:
                    self.annotation_manager.tracker.mark_keyframe(f.frame_id, f.is_keyframe)
            self.class_panel.set_classes(self.project_manager.classes)
            self.properties_panel.set_classes(self.project_manager.classes)
            self.frame_cache = FrameCache(self.project_manager.frames_dir, self.project_manager.thumbnails_dir)
            self.video_panel.frame_cache = self.frame_cache
            self.video_service.frame_cache = self.frame_cache
            self.video_panel.set_frames(self.project_manager.frames)
            self.timeline.set_frames(self.project_manager.frames)
            self.status_project_label.setText(f"Project: {self.project_manager.data.project_name}")
            self._update_window_title()
            if self.project_manager.frames:
                self._on_frame_selected(1)
        else:
            QMessageBox.critical(self, "Error", f"Failed to load project from {project_dir}")

    def save_project(self) -> None:
        if self.project_manager.save_project(self.annotation_manager):
            self.status_save_label.setText("Saved")
        else:
            QMessageBox.warning(self, "Save Error", "Failed to save project.")

    # ---------------- Frame Navigation & Annotation ----------------

    def _on_frame_selected(self, frame_id: int) -> None:
        self.annotation_manager.active_frame_id = frame_id
        self.timeline.set_current_frame(frame_id)
        self.video_panel.select_frame(frame_id)

        if not self.project_manager.frames or frame_id > len(self.project_manager.frames):
            return

        current_frame = self.project_manager.frames[frame_id - 1]
        img = self.frame_cache.get_frame(current_frame.filename)
        self.canvas.set_image(img)

        annos = self.annotation_manager.get_annotations_for_frame(frame_id)
        self.canvas.set_annotations(annos)
        self.properties_panel.set_annotations(annos)

        self.status_frame_label.setText(f"Frame: {frame_id} / {len(self.project_manager.frames)}")
        if len(annos) == 0:
            self.status_objects_label.setText("Objects: 0 (Null Frame)")
        else:
            self.status_objects_label.setText(f"Objects: {len(annos)}")

    def _on_class_selected(self, class_id: int, class_name: str) -> None:
        self.annotation_manager.active_class_id = class_id
        self.annotation_manager.active_class_name = class_name

    def _on_classes_modified(self) -> None:
        self.project_manager.classes = self.class_panel.classes
        self.properties_panel.set_classes(self.class_panel.classes)
        self.project_manager.is_dirty = True

    def _on_keyframe_toggled(self, frame_id: int, is_keyframe: bool) -> None:
        if 0 < frame_id <= len(self.project_manager.frames):
            self.project_manager.frames[frame_id - 1].is_keyframe = is_keyframe
        if hasattr(self.annotation_manager, "tracker") and self.annotation_manager.tracker:
            self.annotation_manager.tracker.mark_keyframe(frame_id, is_keyframe)
        self.timeline.update_keyframe_marker(frame_id, is_keyframe)
        self.video_panel.apply_filter()
        self.video_panel.select_frame(frame_id)
        status_msg = f"Frame #{frame_id} marked as Keyframe ★" if is_keyframe else f"Frame #{frame_id} unmarked as Keyframe"
        self.status_bar.showMessage(status_msg, 3000)
        self.project_manager.is_dirty = True
        self.status_save_label.setText("Modified*")

    def _toggle_current_keyframe(self) -> None:
        """Toggle keyframe status on the active frame (Shortcut: K)."""
        fid = self.annotation_manager.active_frame_id
        if not self.project_manager.frames or fid < 1 or fid > len(self.project_manager.frames):
            return
        current_kf = self.project_manager.frames[fid - 1].is_keyframe
        new_kf = not current_kf
        self._on_keyframe_toggled(fid, new_kf)

    def _on_batch_toggle_keyframes(self, frame_ids: List[int]) -> None:
        """Toggle keyframe status across a list of selected frames."""
        if not self.project_manager.frames:
            return
        has_tracker = hasattr(self.annotation_manager, "tracker") and self.annotation_manager.tracker
        for fid in frame_ids:
            if 0 < fid <= len(self.project_manager.frames):
                new_state = not self.project_manager.frames[fid - 1].is_keyframe
                self.project_manager.frames[fid - 1].is_keyframe = new_state
                if has_tracker:
                    self.annotation_manager.tracker.mark_keyframe(fid, new_state)

        cur_fid = self.annotation_manager.active_frame_id
        self.timeline.set_frames(self.project_manager.frames)
        self.timeline.set_current_frame(cur_fid)
        self.video_panel.apply_filter()
        self.video_panel.select_frame(cur_fid)
        self.project_manager.is_dirty = True
        self.status_save_label.setText("Modified*")
        self.status_bar.showMessage(f"Updated keyframe state for {len(frame_ids)} frame(s)", 3000)

    def _on_annotations_updated(self) -> None:
        fid = self.annotation_manager.active_frame_id
        annos = self.annotation_manager.get_annotations_for_frame(fid)
        self.canvas.set_annotations(annos, self.annotation_manager.selected_object_id)
        self.properties_panel.set_annotations(annos, self.annotation_manager.selected_object_id)
        if len(annos) == 0:
            self.status_objects_label.setText("Objects: 0 (Null Frame)")
            if 0 < fid <= len(self.project_manager.frames):
                self.project_manager.frames[fid - 1].review_status = "negative"
        else:
            self.status_objects_label.setText(f"Objects: {len(annos)}")
            if 0 < fid <= len(self.project_manager.frames):
                if self.project_manager.frames[fid - 1].review_status == "negative":
                    self.project_manager.frames[fid - 1].review_status = "annotated"
        self.video_panel.apply_filter()
        self.timeline.set_current_frame(fid)
        self.status_save_label.setText("Modified*")
        self.project_manager.is_dirty = True

    def _on_annotation_canvas_changed(self, object_id: str) -> None:
        self._on_annotations_updated()

    def _on_prompt_point_added(self, x: float, y: float, is_positive: bool) -> None:
        fid = self.annotation_manager.active_frame_id
        if not self.project_manager.frames or fid > len(self.project_manager.frames):
            return

        frame_meta = self.project_manager.frames[fid - 1]
        img = self.frame_cache.get_frame(frame_meta.filename)
        if img is None:
            return

        # Positive and negative points from canvas
        pos_pts = [(px, py) for px, py, pos in self.canvas.active_prompt_points if pos]
        neg_pts = [(px, py) for px, py, pos in self.canvas.active_prompt_points if not pos]

        active_class = self.class_panel.get_active_class()
        cls_id = active_class.id if active_class else 0
        cls_name = active_class.name if active_class else "object"

        anno = self.image_service.segment_points_to_annotation(
            image_bgr=img,
            frame_id=fid,
            source_frame_index=frame_meta.source_frame_index,
            class_id=cls_id,
            class_name=cls_name,
            positive_points=pos_pts,
            negative_points=neg_pts,
            simplify_tolerance=self.config.polygon.simplify_tolerance,
            min_area=self.config.polygon.min_area,
        )
        if anno:
            self.annotation_manager.add_annotation(anno)
            self.canvas.clear_active_prompts()
            self._mark_frame_status(fid, "annotated")

    def _on_prompt_box_completed(self, x1: float, y1: float, x2: float, y2: float) -> None:
        fid = self.annotation_manager.active_frame_id
        if not self.project_manager.frames or fid > len(self.project_manager.frames):
            return

        frame_meta = self.project_manager.frames[fid - 1]
        img = self.frame_cache.get_frame(frame_meta.filename)
        if img is None:
            return

        active_class = self.class_panel.get_active_class()
        cls_id = active_class.id if active_class else 0
        cls_name = active_class.name if active_class else "object"

        anno = self.image_service.segment_box_to_annotation(
            image_bgr=img,
            frame_id=fid,
            source_frame_index=frame_meta.source_frame_index,
            class_id=cls_id,
            class_name=cls_name,
            box=(x1, y1, x2, y2),
            simplify_tolerance=self.config.polygon.simplify_tolerance,
            min_area=self.config.polygon.min_area,
        )
        if anno:
            self.annotation_manager.add_annotation(anno)
            self._mark_frame_status(fid, "annotated")

    def _on_manual_polygon_completed(self, points: List[tuple]) -> None:
        fid = self.annotation_manager.active_frame_id
        if not self.project_manager.frames or fid > len(self.project_manager.frames):
            return

        frame_meta = self.project_manager.frames[fid - 1]
        active_class = self.class_panel.get_active_class()
        cls_id = active_class.id if active_class else 0
        cls_name = active_class.name if active_class else "object"

        anno = PolygonAnnotation(
            object_id="",
            class_id=cls_id,
            class_name=cls_name,
            frame_id=fid,
            source_frame_index=frame_meta.source_frame_index,
            points=points,
            source="manual",
            tracking_status="confirmed",
        )
        self.annotation_manager.add_annotation(anno)
        self._mark_frame_status(fid, "annotated")

    def _on_text_prompt_submitted(self, text: str) -> None:
        fid = self.annotation_manager.active_frame_id
        if not self.project_manager.frames or fid > len(self.project_manager.frames):
            return
        frame_meta = self.project_manager.frames[fid - 1]
        img = self.frame_cache.get_frame(frame_meta.filename)
        if img is None:
            return

        masks_with_conf = self.sam_adapter.segment_with_text(img, text)
        if not masks_with_conf:
            QMessageBox.information(self, f"{self.get_model_display_name()} Text Prompt", f"No instances found for prompt: '{text}'")
            return

        active_class = self.class_panel.get_active_class()
        cls_id = active_class.id if active_class else 0
        cls_name = active_class.name if active_class else text

        from sam2_annotator.annotation.mask_to_polygon import mask_to_polygons
        count = 0
        for mask, conf in masks_with_conf:
            polys = mask_to_polygons(mask, tolerance_ratio=self.config.polygon.simplify_tolerance)
            for p in polys:
                anno = PolygonAnnotation(
                    object_id="",
                    class_id=cls_id,
                    class_name=cls_name,
                    frame_id=fid,
                    source_frame_index=frame_meta.source_frame_index,
                    points=p,
                    confidence=conf,
                    source="sam2_text",
                )
                self.annotation_manager.add_annotation(anno)
                count += 1

        if count > 0:
            self._mark_frame_status(fid, "annotated")
            QMessageBox.information(self, "SAM 2", f"Segmented {count} instances for '{text}'")

    def _on_propagate_action_triggered(self) -> None:
        """Trigger propagation from menu or shortcut for the currently selected object."""
        if self.annotation_manager.selected_object_id:
            self._on_propagate_requested(self.annotation_manager.selected_object_id)
        else:
            QMessageBox.information(self, "Propagate", "Please select an object in the active frame to propagate.")

    def _on_propagate_requested(
        self,
        object_id: str,
        frame_count: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> None:
        fid = self.annotation_manager.active_frame_id
        anno = self.annotation_manager.get_selected_annotation(fid)
        if not anno:
            return

        total_f = len(self.project_manager.frames)
        if fid >= total_f:
            QMessageBox.information(self, "Tracking", "Already on last frame.")
            return

        # Fallback to properties panel values if not passed
        if frame_count is None:
            frame_count = getattr(self.properties_panel, "propagation_frames", 30)
        if mode is None:
            mode = getattr(self.properties_panel, "propagation_mode", "fixed")

        if mode == "end_of_video":
            target_frames = self.project_manager.frames[fid : total_f]
        elif mode == "next_keyframe":
            # Search for the next keyframe after fid
            next_kf_idx = None
            for idx in range(fid, total_f):
                f_meta = self.project_manager.frames[idx]
                if f_meta.is_keyframe or (
                    hasattr(self.annotation_manager, "tracker")
                    and self.annotation_manager.tracker
                    and self.annotation_manager.tracker.is_keyframe(f_meta.frame_id)
                ):
                    next_kf_idx = idx
                    break

            if next_kf_idx is not None:
                # Target frames up to the next keyframe (inclusive)
                target_frames = self.project_manager.frames[fid : next_kf_idx + 1]
            else:
                target_frames = self.project_manager.frames[fid : total_f]
                self.status_bar.showMessage("No subsequent keyframe found; propagating to end of video.", 4000)
        else:  # "fixed"
            step_count = min(max(1, frame_count), total_f - fid)
            target_frames = self.project_manager.frames[fid : fid + step_count]

        if not target_frames:
            QMessageBox.information(self, "Tracking", "No subsequent frames to propagate to.")
            return

        prog_diag = ProgressDialog(f"Propagating Object Tracking ({len(target_frames)} frames)...", self)
        worker = PropagationWorker(self.video_service, anno, target_frames)
        self.current_worker = worker

        worker.progress.connect(prog_diag.set_progress)
        prog_diag.cancelled.connect(worker.cancel)

        def on_finished(new_annos: List[PolygonAnnotation]):
            prog_diag.accept()
            for a in new_annos:
                self.annotation_manager.add_annotation(a)
                self._mark_frame_status(a.frame_id, "annotated")
            self._on_annotations_updated()
            QMessageBox.information(self, "Propagation Complete", f"Successfully propagated through {len(new_annos)} frames.")

        def on_error(err: str):
            prog_diag.reject()
            QMessageBox.warning(self, "Propagation Issue", f"Tracking warning: {err}")

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.start()
        prog_diag.exec()

    def _delete_active_object(self) -> None:
        fid = self.annotation_manager.active_frame_id
        if self.annotation_manager.selected_object_id:
            self.annotation_manager.remove_annotation(fid, self.annotation_manager.selected_object_id)

    def _on_object_deleted(self, object_id: str) -> None:
        fid = self.annotation_manager.active_frame_id
        self.annotation_manager.remove_annotation(fid, object_id)

    def _clear_current_frame(self) -> None:
        fid = self.annotation_manager.active_frame_id
        self.annotation_manager.clear_frame_annotations(fid)

    def _mark_frame_reviewed(self) -> None:
        fid = self.annotation_manager.active_frame_id
        self._mark_frame_status(fid, "reviewed")

    def _mark_frame_negative(self) -> None:
        fid = self.annotation_manager.active_frame_id
        self.annotation_manager.clear_frame_annotations(fid)
        self._mark_frame_status(fid, "negative")
        self.status_objects_label.setText("Objects: 0 (Null Frame)")

    def _mark_all_unannotated_as_null(self) -> None:
        """Scan all frames; if a frame has 0 annotations, mark its status as negative (null frame)."""
        marked_count = 0
        for frame in self.project_manager.frames:
            annos = self.annotation_manager.get_annotations_for_frame(frame.frame_id)
            if len(annos) == 0:
                frame.review_status = "negative"
                marked_count += 1
        self.video_panel.apply_filter()
        self.timeline.set_current_frame(self.annotation_manager.active_frame_id)
        self.project_manager.is_dirty = True
        self.status_save_label.setText("Modified*")
        QMessageBox.information(
            self,
            "Null Frames",
            f"Marked {marked_count} unannotated frames as Null (negative background) frames."
        )

    def _mark_frame_status(self, frame_id: int, status: str) -> None:
        if 0 < frame_id <= len(self.project_manager.frames):
            self.project_manager.frames[frame_id - 1].review_status = status
            self.video_panel.apply_filter()
            self.timeline.set_current_frame(frame_id)
            self.project_manager.is_dirty = True

    def delete_current_frame(self) -> None:
        """Delete the currently active video frame from the project."""
        if not self.project_manager.frames:
            QMessageBox.information(self, "No Frames", "There are no video frames in the current project.")
            return
        fid = self.annotation_manager.active_frame_id
        self.delete_frames([fid])

    def delete_frames(self, frame_ids: Optional[List[int]] = None) -> None:
        """Prompt confirmation and delete specified frame(s) from project."""
        if not self.project_manager.project_dir or not self.project_manager.frames:
            QMessageBox.warning(self, "No Project", "No active project is open.")
            return

        if not frame_ids:
            frame_ids = [self.annotation_manager.active_frame_id]

        target_set = set(frame_ids)
        frames_to_del = [f for f in self.project_manager.frames if f.frame_id in target_set]
        if not frames_to_del:
            return

        # Gather annotation counts for warning dialog
        anno_counts = {
            f.frame_id: len(self.annotation_manager.get_annotations_for_frame(f.frame_id))
            for f in frames_to_del
        }

        # Show confirmation dialog
        dialog = DeleteFrameDialog(frames_to_del, anno_counts, parent=self)
        if not dialog.exec():
            return

        delete_files = dialog.should_delete_files()
        lowest_id = min(f.frame_id for f in frames_to_del)

        # Perform deletion in ProjectManager
        deleted = self.project_manager.delete_frames(
            frame_ids=[f.frame_id for f in frames_to_del],
            delete_files=delete_files,
            annotation_manager=self.annotation_manager,
            frame_cache=self.frame_cache,
        )

        # Save project state atomically
        self.project_manager.save_project(self.annotation_manager)

        # Update UI components
        remaining_count = len(self.project_manager.frames)
        self.video_panel.set_frames(self.project_manager.frames)
        self.timeline.set_frames(self.project_manager.frames)

        if remaining_count > 0:
            new_active_id = max(1, min(lowest_id, remaining_count))
            self._on_frame_selected(new_active_id)
        else:
            self.annotation_manager.active_frame_id = 1
            self.canvas.set_image(None)
            self.properties_panel.set_annotations([])
            self.status_frame_label.setText("Frame: 0 / 0")
            self.status_objects_label.setText("Objects: 0")

        self.status_save_label.setText("Saved")
        msg = f"Deleted {len(deleted)} frame(s) successfully."
        self.statusBar().showMessage(msg, 4000)
        logger.info(msg)

    def _on_mark_null_batch(self, frame_ids: List[int]) -> None:
        """Mark multiple frames as null (negative background) frames."""
        for fid in frame_ids:
            if 0 < fid <= len(self.project_manager.frames):
                self.project_manager.frames[fid - 1].review_status = "negative"
                self.annotation_manager.clear_frame_annotations(fid)
        self.video_panel.apply_filter()
        if self.annotation_manager.active_frame_id in frame_ids:
            self._on_frame_selected(self.annotation_manager.active_frame_id)
        self.project_manager.is_dirty = True
        self.project_manager.save_project(self.annotation_manager)

    def _on_mark_reviewed_batch(self, frame_ids: List[int]) -> None:
        """Mark multiple frames as reviewed."""
        for fid in frame_ids:
            if 0 < fid <= len(self.project_manager.frames):
                self.project_manager.frames[fid - 1].review_status = "reviewed"
        self.video_panel.apply_filter()
        if self.annotation_manager.active_frame_id in frame_ids:
            self._on_frame_selected(self.annotation_manager.active_frame_id)
        self.project_manager.is_dirty = True
        self.project_manager.save_project(self.annotation_manager)

    # ---------------- Dataset & Settings ----------------

    def export_dataset_dialog(self) -> None:
        if not self.project_manager.frames:
            QMessageBox.warning(self, "Export", "Project has no frames to export.")
            return

        def_out = self.project_manager.export_dir
        annotated_count = sum(
            1 for f in self.project_manager.frames
            if len(self.annotation_manager.get_annotations_for_frame(f.frame_id)) > 0
        )
        diag = ExportDialog(
            default_output_dir=def_out,
            total_frames=len(self.project_manager.frames),
            classes_count=len(self.project_manager.classes),
            annotated_frames_count=annotated_count,
            parent=self,
        )
        if diag.exec():
            params = diag.get_export_params()
            format_id = params.get("format", "yolo_segmentation")
            format_info = EXPORT_FORMATS.get(format_id, {})

            if format_info.get("supports_splits", False):
                split_dict = DatasetSplitter.split_frames(
                    frames=self.project_manager.frames,
                    train_ratio=params["train_ratio"],
                    val_ratio=params["val_ratio"],
                    test_ratio=params["test_ratio"],
                    strategy=params["split_strategy"],
                )
            else:
                split_dict = {"all": self.project_manager.frames}

            try:
                exporter = create_exporter(
                    format_id=format_id,
                    output_dir=params["output_dir"],
                    classes=self.project_manager.classes,
                    frames_dir=self.project_manager.frames_dir,
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Configuration Error", str(e))
                return

            total_frames = len(self.project_manager.frames)
            title = f"Exporting {format_info.get('name', 'Dataset')}..."
            prog_diag = ProgressDialog(title, self)
            prog_diag.set_progress(0, total_frames, "Initializing export...")

            worker = DatasetExportWorker(
                exporter=exporter,
                split_dict=split_dict,
                annotations_by_frame=self.annotation_manager.frame_annotations,
                params=params,
            )
            self.current_worker = worker

            worker.progress.connect(prog_diag.set_progress)
            prog_diag.cancelled.connect(worker.cancel)

            def on_finished(result: Dict[str, Any]):
                prog_diag.accept()
                if result.get("status") == "cancelled":
                    QMessageBox.information(self, "Export Cancelled", "Export was cancelled by user.")
                    return
                elif result.get("status") == "error":
                    QMessageBox.critical(self, "Export Failed", result.get("error", "An unknown error occurred."))
                    return

                if format_id == "rendered_video":
                    msg = (
                        f"Annotated video successfully exported!\n\n"
                        f"Video File: {result.get('video_path')}\n"
                        f"Total Frames: {result.get('total_frames')}\n"
                        f"Resolution: {result.get('resolution')}\n"
                        f"Playback FPS: {result.get('fps')}\n"
                        f"Total Objects Rendered: {result.get('total_objects')}"
                    )
                    QMessageBox.information(self, "Video Export Complete", msg)
                    return

                # Run automatic validator across all export formats
                valid_str = ""
                try:
                    validator = DatasetValidator(params["output_dir"], format_id=format_id)
                    report = validator.validate()
                    valid_str = f"\nValidation Status: {'PASSED' if report['valid'] else 'WARNINGS / ERRORS FOUND'}"
                except Exception as e:
                    logger.warning("Automatic validation encountered an issue: %s", e)

                zip_info = f"\nArchive: {result.get('zip_path')}\n" if result.get("zip_path") else ""
                format_display = format_info.get("name", format_id)
                msg = (
                    f"{format_display} successfully exported to:\n{params['output_dir']}\n{zip_info}\n"
                    f"Total Images: {result.get('total_images', total_frames)}\n"
                    f" - Annotated Images: {result.get('annotated_images', annotated_count)}\n"
                    f" - Null / Background Images: {result.get('null_images', 0)}\n"
                    f"Total Objects: {result.get('total_objects', 0)}{valid_str}"
                )
                QMessageBox.information(self, "Export Complete", msg)

            def on_error(err_msg: str):
                prog_diag.reject()
                QMessageBox.critical(self, "Export Error", f"Failed to export: {err_msg}")

            worker.finished.connect(on_finished)
            worker.error.connect(on_error)
            worker.start()
            prog_diag.exec()

    def validate_dataset_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Dataset or Artifact Directory to Validate")
        if path:
            validator = DatasetValidator(Path(path))
            report = validator.validate()
            status_str = "PASSED" if report["valid"] else "FAILED"
            fmt_str = f"Format: {report.get('format_name', 'Unknown')}\n"
            msg = (
                f"Validation Status: {status_str}\n\n"
                f"{fmt_str}"
                f"Images / Frames: {report['stats']['images_count']}\n"
                f"Labels / Files: {report['stats']['labels_count']}\n"
                f"Objects / Detections: {report['stats']['objects_count']}\n"
                f"Errors: {len(report['errors'])}\n"
                f"Warnings: {len(report['warnings'])}\n"
            )
            if not report["valid"] and report["errors"]:
                msg += "\nTop Errors:\n" + "\n".join(report["errors"][:5])
            elif report["warnings"]:
                msg += "\nTop Warnings:\n" + "\n".join(report["warnings"][:5])
            QMessageBox.information(self, "Dataset Validation", msg)

    def open_settings_dialog(self) -> None:
        diag = SettingsDialog(self.config, self)
        if diag.exec():
            # Save configuration
            cfg_path = Path("config") / "defaults.yaml"
            self.config.save(cfg_path)
            self.status_device_label.setText(f"Device: {self.config.model.device.upper()}")
            if hasattr(self.properties_panel, "set_default_propagation_frames"):
                self.properties_panel.set_default_propagation_frames(self.config.model.propagation_frames)
            QMessageBox.information(self, "Settings", "Settings saved successfully.")

    def open_diagnostics_dialog(self) -> None:
        diag = StartupDialog(self)
        if diag.exec():
            self.config.model.device = diag.selected_device
            self.status_device_label.setText(f"Device: {self.config.model.device.upper()}")

    def _show_about(self) -> None:
        model_name = self.get_model_display_name()
        QMessageBox.about(
            self,
            f"About {model_name} Video Polygon Annotator",
            f"<h3>{model_name} Video Polygon Annotator</h3>"
            f"<p>A local-first application for interactive {model_name} assisted polygon "
            "annotation on video frames and YOLOv8 instance segmentation export.</p>"
            "<p>Version: 1.0.0</p>"
        )
