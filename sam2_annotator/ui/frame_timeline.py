"""Bottom timeline panel with playback controls, frame scrubbing, and keyframe markers."""

from typing import List, Optional, Set
from PySide6.QtCore import Qt, Signal, QTimer, QPoint
from PySide6.QtGui import QPainter, QColor, QPolygon, QPen
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSlider,
    QLabel, QSpinBox, QStyle, QStyleOptionSlider
)

from sam2_annotator.video.frame_extractor import FrameMetadata


class KeyframeTimelineSlider(QSlider):
    """Horizontal slider that paints golden diamond markers at keyframe positions."""

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.keyframe_indices: Set[int] = set()

    def set_keyframes(self, keyframe_indices: Set[int]) -> None:
        self.keyframe_indices = set(keyframe_indices)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.keyframe_indices or self.maximum() <= self.minimum():
            return

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        handle = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
        half_handle = handle.width() // 2 if handle.isValid() else 8
        track_left = half_handle
        track_width = max(1, self.width() - 2 * half_handle)

        min_val = self.minimum()
        max_val = self.maximum()
        span = max_val - min_val
        if span <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor("#f59e0b"))
        painter.setPen(QPen(QColor("#ffffff"), 1))

        y = self.height() // 2

        for kf in self.keyframe_indices:
            if min_val <= kf <= max_val:
                x = int(track_left + (kf - min_val) / float(span) * track_width)
                diamond = QPolygon([
                    QPoint(x, y - 6),
                    QPoint(x + 5, y),
                    QPoint(x, y + 6),
                    QPoint(x - 5, y),
                ])
                painter.drawPolygon(diamond)


class FrameTimeline(QWidget):
    """Bottom playback bar for video timeline navigation and frame stepping."""

    frame_changed = Signal(int)  # frame_id
    keyframe_toggled = Signal(int, bool)  # frame_id, is_keyframe
    null_frame_clicked = Signal()
    delete_frame_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames: List[FrameMetadata] = []
        self.current_frame_id: int = 1
        self.is_playing: bool = False

        # Playback timer
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.step_next)

        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 4, 8, 4)
        main_layout.setSpacing(4)

        # 1. Slider row
        slider_layout = QHBoxLayout()
        self.time_label = QLabel("00:00.000")
        self.time_label.setStyleSheet("font-family: monospace; font-size: 12px; font-weight: bold;")
        slider_layout.addWidget(self.time_label)

        self.slider = KeyframeTimelineSlider(Qt.Horizontal)
        self.slider.setRange(1, 1)
        self.slider.setValue(1)
        self.slider.valueChanged.connect(self._on_slider_moved)
        slider_layout.addWidget(self.slider)

        self.total_time_label = QLabel("00:00.000")
        self.total_time_label.setStyleSheet("font-family: monospace; font-size: 12px; color: #888888;")
        slider_layout.addWidget(self.total_time_label)
        main_layout.addLayout(slider_layout)

        # 2. Controls row
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(6)

        self.prev_btn = QPushButton("◀ Prev (P)")
        self.prev_btn.setToolTip("Previous Frame (Left Arrow or P)")
        self.prev_btn.clicked.connect(self.step_prev)
        ctrl_layout.addWidget(self.prev_btn)

        self.play_btn = QPushButton("▶ Play (Space)")
        self.play_btn.setToolTip("Toggle Playback (Space)")
        self.play_btn.clicked.connect(self.toggle_play)
        ctrl_layout.addWidget(self.play_btn)

        self.next_btn = QPushButton("Next (N) ▶")
        self.next_btn.setToolTip("Next Frame (Right Arrow or N)")
        self.next_btn.clicked.connect(self.step_next)
        ctrl_layout.addWidget(self.next_btn)

        ctrl_layout.addSpacing(15)

        # Frame counter & jump spinbox
        ctrl_layout.addWidget(QLabel("Frame:"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 1)
        self.frame_spin.setValue(1)
        self.frame_spin.valueChanged.connect(self._on_spin_changed)
        ctrl_layout.addWidget(self.frame_spin)

        self.frame_count_label = QLabel("/ 0")
        ctrl_layout.addWidget(self.frame_count_label)

        ctrl_layout.addSpacing(15)

        self.keyframe_btn = QPushButton("☆ Keyframe")
        self.keyframe_btn.setCheckable(True)
        self.keyframe_btn.setToolTip("Mark/unmark current frame as Keyframe reference (Shortcut: K)")
        self._update_keyframe_btn_style(False)
        self.keyframe_btn.clicked.connect(self._on_keyframe_clicked)
        ctrl_layout.addWidget(self.keyframe_btn)

        self.null_btn = QPushButton("⚫ Null Frame")
        self.null_btn.setToolTip("Mark this frame as a Null Frame (Background / No Objects)")
        self.null_btn.clicked.connect(self.null_frame_clicked.emit)
        ctrl_layout.addWidget(self.null_btn)

        self.delete_frame_btn = QPushButton("🗑 Delete Frame")
        self.delete_frame_btn.setToolTip("Delete current video frame from project (Ctrl+Delete)")
        self.delete_frame_btn.setStyleSheet(
            "QPushButton:hover { background-color: #5c1d1d; border-color: #d32f2f; color: #ffcdd2; }"
        )
        self.delete_frame_btn.clicked.connect(self.delete_frame_clicked.emit)
        ctrl_layout.addWidget(self.delete_frame_btn)

        ctrl_layout.addStretch()

        self.status_badge = QLabel("Status: Unreviewed")
        self.status_badge.setStyleSheet("color: #aaaaaa; font-weight: bold;")
        ctrl_layout.addWidget(self.status_badge)

        main_layout.addLayout(ctrl_layout)

    def _update_keyframe_btn_style(self, is_checked: bool) -> None:
        """Update button text and stylesheet based on active keyframe state."""
        if is_checked:
            self.keyframe_btn.setText("★ Keyframe [ON]")
            self.keyframe_btn.setStyleSheet(
                "QPushButton {"
                "  background-color: #f59e0b;"
                "  color: #111827;"
                "  font-weight: bold;"
                "  border: 1px solid #d97706;"
                "  border-radius: 4px;"
                "  padding: 4px 10px;"
                "}"
                "QPushButton:hover {"
                "  background-color: #fbbf24;"
                "  color: #000000;"
                "}"
            )
        else:
            self.keyframe_btn.setText("☆ Keyframe")
            self.keyframe_btn.setStyleSheet(
                "QPushButton {"
                "  background-color: #2b2b2b;"
                "  color: #cccccc;"
                "  border: 1px solid #444444;"
                "  border-radius: 4px;"
                "  padding: 4px 10px;"
                "}"
                "QPushButton:hover {"
                "  background-color: #383838;"
                "  color: #ffffff;"
                "  border-color: #f59e0b;"
                "}"
            )

    def _update_slider_keyframes(self) -> None:
        """Refresh keyframe marker positions on the timeline slider."""
        kf_set = {f.frame_id for f in self.frames if f.is_keyframe}
        self.slider.set_keyframes(kf_set)

    def set_frames(self, frames: List[FrameMetadata]) -> None:
        self.frames = frames
        total = len(frames)
        self._update_slider_keyframes()
        if total > 0:
            self.slider.blockSignals(True)
            self.frame_spin.blockSignals(True)

            self.slider.setRange(1, total)
            self.frame_spin.setRange(1, total)
            self.frame_count_label.setText(f"/ {total}")

            self.total_time_label.setText(self._format_time(frames[-1].timestamp_seconds))

            self.slider.blockSignals(False)
            self.frame_spin.blockSignals(False)

            self.set_current_frame(1)
        else:
            self.slider.blockSignals(True)
            self.frame_spin.blockSignals(True)
            self.slider.setRange(1, 1)
            self.frame_spin.setRange(1, 1)
            self.frame_count_label.setText("/ 0")
            self.time_label.setText("00:00.000")
            self.total_time_label.setText("00:00.000")
            self.status_badge.setText("Status: No Frames")
            self.status_badge.setStyleSheet("color: #666666;")
            self.slider.blockSignals(False)
            self.frame_spin.blockSignals(False)

    def set_current_frame(self, frame_id: int) -> None:
        if not self.frames:
            return
        frame_id = max(1, min(len(self.frames), frame_id))
        self.current_frame_id = frame_id

        self.slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.slider.setValue(frame_id)
        self.frame_spin.setValue(frame_id)
        self.slider.blockSignals(False)
        self.frame_spin.blockSignals(False)

        current_meta = self.frames[frame_id - 1]
        self.time_label.setText(self._format_time(current_meta.timestamp_seconds))

        # Update keyframe toggle state & appearance
        self.keyframe_btn.blockSignals(True)
        self.keyframe_btn.setChecked(current_meta.is_keyframe)
        self._update_keyframe_btn_style(current_meta.is_keyframe)
        self.keyframe_btn.blockSignals(False)

        # Update status badge with keyframe indicator
        kf_suffix = " | ★ Keyframe" if current_meta.is_keyframe else ""
        if current_meta.review_status == "negative":
            self.status_badge.setText(f"Status: ⚫ Null Frame (No Objects){kf_suffix}")
            self.status_badge.setStyleSheet("color: #ffb74d; font-weight: bold;")
        elif current_meta.review_status in ["annotated", "reviewed"]:
            badge_color = "#f59e0b" if current_meta.is_keyframe else "#66bb6a"
            self.status_badge.setText(f"Status: 🟢 {current_meta.review_status.title()}{kf_suffix}")
            self.status_badge.setStyleSheet(f"color: {badge_color}; font-weight: bold;")
        else:
            badge_color = "#f59e0b" if current_meta.is_keyframe else "#aaaaaa"
            self.status_badge.setText(f"Status: ⚪ {current_meta.review_status.title()}{kf_suffix}")
            self.status_badge.setStyleSheet(f"color: {badge_color}; font-weight: bold;")

    def step_next(self) -> None:
        if not self.frames:
            return
        if self.current_frame_id < len(self.frames):
            self.set_current_frame(self.current_frame_id + 1)
            self.frame_changed.emit(self.current_frame_id)
        elif self.is_playing:
            self.toggle_play()

    def step_prev(self) -> None:
        if not self.frames:
            return
        if self.current_frame_id > 1:
            self.set_current_frame(self.current_frame_id - 1)
            self.frame_changed.emit(self.current_frame_id)

    def toggle_play(self) -> None:
        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶ Play (Space)")
        else:
            if not self.frames:
                return
            self.is_playing = True
            self.play_btn.setText("⏸ Pause (Space)")
            self.play_timer.start(100) # 10 fps preview step

    def _on_slider_moved(self, val: int) -> None:
        self.set_current_frame(val)
        self.frame_changed.emit(val)

    def _on_spin_changed(self, val: int) -> None:
        self.set_current_frame(val)
        self.frame_changed.emit(val)

    def _on_keyframe_clicked(self) -> None:
        if self.frames and 0 < self.current_frame_id <= len(self.frames):
            is_kf = self.keyframe_btn.isChecked()
            self.frames[self.current_frame_id - 1].is_keyframe = is_kf
            self._update_keyframe_btn_style(is_kf)
            self._update_slider_keyframes()
            self.set_current_frame(self.current_frame_id)
            self.keyframe_toggled.emit(self.current_frame_id, is_kf)

    @staticmethod
    def _format_time(seconds: float) -> str:
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{mins:02d}:{secs:02d}.{millis:03d}"

