"""Bottom timeline panel with playback controls, frame scrubbing, and keyframe markers."""

from typing import List, Optional
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSlider,
    QLabel, QSpinBox
)

from sam3_annotator.video.frame_extractor import FrameMetadata


class FrameTimeline(QWidget):
    """Bottom playback bar for video timeline navigation and frame stepping."""

    frame_changed = Signal(int)  # frame_id
    keyframe_toggled = Signal(int, bool)  # frame_id, is_keyframe

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

        self.slider = QSlider(Qt.Horizontal)
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

        self.keyframe_btn = QPushButton("★ Keyframe")
        self.keyframe_btn.setCheckable(True)
        self.keyframe_btn.clicked.connect(self._on_keyframe_clicked)
        ctrl_layout.addWidget(self.keyframe_btn)

        ctrl_layout.addStretch()

        self.status_badge = QLabel("Status: Unreviewed")
        self.status_badge.setStyleSheet("color: #aaaaaa; font-weight: bold;")
        ctrl_layout.addWidget(self.status_badge)

        main_layout.addLayout(ctrl_layout)

    def set_frames(self, frames: List[FrameMetadata]) -> None:
        self.frames = frames
        total = len(frames)
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
            self.slider.setRange(1, 1)
            self.frame_spin.setRange(1, 1)
            self.frame_count_label.setText("/ 0")

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
        self.keyframe_btn.setChecked(current_meta.is_keyframe)
        self.status_badge.setText(f"Status: {current_meta.review_status.title()}")

    def step_next(self) -> None:
        if self.current_frame_id < len(self.frames):
            self.set_current_frame(self.current_frame_id + 1)
            self.frame_changed.emit(self.current_frame_id)
        elif self.is_playing:
            self.toggle_play()

    def step_prev(self) -> None:
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
            self.keyframe_toggled.emit(self.current_frame_id, is_kf)

    @staticmethod
    def _format_time(seconds: float) -> str:
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{mins:02d}:{secs:02d}.{millis:03d}"
