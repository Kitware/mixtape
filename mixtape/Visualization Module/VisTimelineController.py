# timeline_controller.py - Unified timeline control for synchronization

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QSlider, QLabel, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal, QObject


class TimelineController(QWidget):
    """Controls for synchronized timeline between video and visualization"""
    
    # Signal emitted when position changes
    position_changed = pyqtSignal(float)  # 0-100 percentage
    
    def __init__(self):
        super().__init__()
        
        # Initialize UI
        self.init_ui()
        
        # Initialize variables
        self.total_steps = 0
        self.total_frames = 0
        self.frame_step_ratio = 1.0
    
    def init_ui(self):
        """Initialize the UI components"""
        
        # Create horizontal layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 0)
        
        # Create position slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 1000)  # Use 0-1000 for more precision
        self.position_slider.setValue(0)
        self.position_slider.valueChanged.connect(self.on_slider_value_changed)
        layout.addWidget(self.position_slider)
        
        # Position label
        self.position_label = QLabel("0%")
        layout.addWidget(self.position_label)
        
        # Set fixed height
        self.setFixedHeight(40)
    
    def setup(self, total_steps, total_frames):
        """Initialize with total steps and frames"""
        
        self.total_steps = max(1, total_steps)
        self.total_frames = max(1, total_frames)
        
        # Calculate ratio for mapping between frames and steps
        self.frame_step_ratio = self.total_frames / self.total_steps
        
        # Reset position
        self.set_position(0)
    
    def set_position(self, position, from_video=False, from_viz=False):
        """Set the position (0-100%)"""
        
        # Update slider (scaled to 0-1000 internally)
        slider_value = int(position * 10)
        
        # Only update if value has changed to avoid feedback loops
        if self.position_slider.value() != slider_value:
            # Temporarily block signals to avoid feedback
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(slider_value)
            self.position_slider.blockSignals(False)
        
        # Update label
        self.position_label.setText(f"{position:.1f}%")
        
        # Emit signal unless this change came from a component that already knows
        if not from_video and not from_viz:
            self.position_changed.emit(position)
    
    def on_slider_value_changed(self, value):
        """Handle slider value changes"""
        
        # Convert 0-1000 to 0-100%
        position = value / 10.0
        
        # Update label
        self.position_label.setText(f"{position:.1f}%")
        
        # Emit signal
        self.position_changed.emit(position)
    
    def frame_to_step(self, frame, offset=0):
        """Convert a frame number to a step index with offset"""
        
        if self.total_frames <= 0 or self.total_steps <= 0:
            return 0
        
        # Calculate step based on frame with offset
        step = int((frame + offset) / self.frame_step_ratio)
        
        # Add debugging to verify the mapping
        # print(f"Mapping frame {frame} to step {step} (with offset {offset})")
        
        # Ensure step is within valid range
        return max(0, min(step, self.total_steps - 1))




    
    def step_to_frame(self, step):
        """Convert a step index to a frame number"""
        
        if self.total_frames <= 0 or self.total_steps <= 0:
            return 0
        
        # Calculate frame based on step
        frame = int(step * self.frame_step_ratio)
        
        # Ensure within bounds
        return max(0, min(frame, self.total_frames - 1))

