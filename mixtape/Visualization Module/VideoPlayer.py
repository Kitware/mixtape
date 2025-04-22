# video_player.py - Qt-based video player component

import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QSlider, 
                           QHBoxLayout, QPushButton, QStyle, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize

class VideoPlayerWidget(QWidget):
    """Widget for playing and controlling video playback"""
    
    # Signal emitted when the current frame changes
    frame_changed = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        
        # Create the video display and controls layout
        self.init_ui()
        
        # Initialize video playback variables
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.playing = False
        
        # Setup timer for playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        
        # Initially, no video is loaded
        self.reset()
    
    def init_ui(self):
        """Initialize user interface components"""
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create video display label
        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.video_label)
        
        # Create controls layout
        controls_layout = QHBoxLayout()
        
        # Play/pause button
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)
        
        # Step backward button
        self.back_button = QPushButton()
        self.back_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.back_button.clicked.connect(lambda: self.seek_relative(-1))
        controls_layout.addWidget(self.back_button)
        
        # Step forward button
        self.forward_button = QPushButton()
        self.forward_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.forward_button.clicked.connect(lambda: self.seek_relative(1))
        controls_layout.addWidget(self.forward_button)
        
        # Add controls to main layout
        layout.addLayout(controls_layout)
        
        # Add info label for frame details
        self.info_label = QLabel("Frame: 0 / 0")
        layout.addWidget(self.info_label)
    
    def load_video(self, video_path):
        """Load a video file for playback"""
        
        # Close any previously opened video
        if self.cap is not None:
            self.cap.release()
        
        # Open the new video file
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            self.video_label.setText(f"Error opening video: {video_path}")
            return False
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # Default if unable to determine
        
        # Reset to first frame
        self.current_frame = 0
        self.show_frame(0)
        
        # Update UI to show video is loaded
        self.update_info_label()
        
        return True
    
    def reset(self):
        """Reset player state when no video is loaded"""
        
        # Clear the video display
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.update_frame_display(blank_image)
        
        # Reset playback state
        self.current_frame = 0
        self.total_frames = 0
        self.playing = False
        
        # Stop the playback timer
        self.timer.stop()
        
        # Update UI
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.info_label.setText("No video loaded")
    
    def update_frame_display(self, frame):
        """Update the video display with the given frame"""
        
        # Convert OpenCV's BGR format to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Create a scaled pixmap to fit the label
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), 
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Set the pixmap to the label
        self.video_label.setPixmap(pixmap)
    
    def show_frame(self, frame_idx):
        """Show a specific frame from the video"""
        
        if self.cap is None or not self.cap.isOpened():
            return False
        
        # Ensure frame index is within bounds
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        
        # Set the frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read the frame
        ret, frame = self.cap.read()
        
        if ret:
            # Update the display
            self.update_frame_display(frame)
            
            # Update current frame
            self.current_frame = frame_idx
            self.update_info_label()
            
            # Emit signal that frame has changed
            self.frame_changed.emit(frame_idx)
            
            return True
        else:
            print(f"Error reading frame {frame_idx}")
            return False
    
    def next_frame(self):
        """Display the next frame during playback"""
        
        if self.cap is None or not self.cap.isOpened():
            return
        
        # Check if we've reached the end
        if self.current_frame >= self.total_frames - 1:
            self.toggle_play()  # Stop playback
            return
        
        # Show the next frame
        self.show_frame(self.current_frame + 1)
    
    def toggle_play(self):
        """Toggle between play and pause"""
        
        if self.cap is None or not self.cap.isOpened():
            return
        
        self.playing = not self.playing
        
        if self.playing:
            # Start playback
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.timer.start(int(1000 / self.fps))
        else:
            # Pause playback
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stop()
    
    def seek_percent(self, percent):
        """Seek to position specified as percentage (0-100)"""
        
        if self.cap is None or not self.cap.isOpened():
            return
        
        # Calculate frame from percentage
        frame = int((percent / 100) * self.total_frames)
        
        # Show the frame
        self.show_frame(frame)
    
    def seek_relative(self, frames):
        """Seek a relative number of frames from current position"""
        
        if self.cap is None or not self.cap.isOpened():
            return
        
        # Calculate new frame position
        new_frame = self.current_frame + frames
        
        # Show the frame
        self.show_frame(new_frame)
    
    def update_info_label(self):
        """Update the information label with current frame details"""
        
        self.info_label.setText(f"Frame: {self.current_frame} / {self.total_frames}")
    
    def resizeEvent(self, event):
        """Handle resize events to scale the video display"""
        
        super().resizeEvent(event)
        
        # If video is loaded, update the displayed frame to match new size
        if self.cap is not None and self.cap.isOpened():
            # Save current position
            current = self.current_frame
            
            # Redisplay current frame at new size
            self.show_frame(current)
