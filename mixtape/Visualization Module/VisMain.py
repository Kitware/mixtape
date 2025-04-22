# main.py - Main application entry point for VisGUI system

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QSplitter, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

from VideoPlayer import VideoPlayerWidget
from Vis import VisualizationWidget, InfoPanel
from VisDataManager import DataManager
from VisTimelineController import TimelineController
from VisConfig import DEFAULT_LOG_DIR
import random


class MainWindow(QMainWindow):
    """Main application window containing video player and visualization panels"""
    
    def __init__(self):
        super().__init__()
        
        # Setup the main window properties
        self.setWindowTitle("Crafter Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)

        self.frame_offset = 1


        # Create the central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create a splitter to divide the window into two panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Create data manager to handle CSV data and synchronization
        self.data_manager = DataManager()
        
        # Create video player widget (left panel)
        self.video_player = VideoPlayerWidget()
        splitter.addWidget(self.video_player)
        
        # Create visualization widget (right panel)
        self.visualization = VisualizationWidget()
        splitter.addWidget(self.visualization)
        self.visualization.data_manager = self.data_manager
        self.visualization.info_panel.data_manager = self.data_manager

        # Create timeline controller at bottom
        self.timeline = TimelineController()
        main_layout.addWidget(self.timeline)
        
        # Set initial splitter sizes (50% each)
        splitter.setSizes([600, 600])
        
        # Connect signals between components
        self.timeline.position_changed.connect(self.on_timeline_position_changed)
        self.video_player.frame_changed.connect(self.on_video_frame_changed)

        # Setup menu actions
        self.setup_menu()
        
        # Load the most recent data by default
        # self.load_latest_data()
        self.open_random_files()


    def open_random_files(self):
        """Randomly select a log file and its corresponding video file."""
        # List all CSV log files in the logs directory
        csv_files = [f for f in os.listdir(DEFAULT_LOG_DIR) if f.endswith('.csv')]
        if not csv_files:
            print("No log files found.")
            return

        # Pick a random csv file
        random_csv = random.choice(csv_files)
        csv_path = os.path.join(DEFAULT_LOG_DIR, random_csv)

        # Try to find the corresponding video file (same base name, .mp4)
        base_name = os.path.splitext(random_csv)[0]
        video_path = os.path.join(DEFAULT_LOG_DIR, f"{base_name}.mp4")

        # If the video doesn't exist, pick a random video file
        if not os.path.exists(video_path):
            video_files = [f for f in os.listdir(DEFAULT_LOG_DIR) if f.endswith('.mp4')]
            if not video_files:
                print("No video files found.")
                return
            video_path = os.path.join(DEFAULT_LOG_DIR, random.choice(video_files))

        # Load the selected files
        self.load_data(csv_path, video_path)


    def on_timeline_position_changed(self, position):
        """Handle timeline position changes (0-100%)"""
        # Update video position
        self.video_player.seek_percent(position)
        
        # Calculate corresponding step for visualization
        frame = self.video_player.current_frame
        step = self.timeline.frame_to_step(frame)
        
        # Update visualization position without triggering back-propagation
        self.visualization.update_position(step, from_timeline=True)

    def setup_menu(self):
        """Create the application menu bar with actions"""
        
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')

        # Open random video and log 
        random_action = file_menu.addAction('Open Random Log and Video')
        random_action.triggered.connect(self.open_random_files)

        # # Open action
        # open_action = file_menu.addAction('Open Log and Video...')
        # open_action.triggered.connect(self.open_files)
        
        # Exit action
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        # Toggle visualization types
        show_cumulative = view_menu.addAction('Show Cumulative Rewards')
        show_cumulative.setCheckable(True)
        show_cumulative.setChecked(True)
        show_cumulative.triggered.connect(lambda checked: self.visualization.toggle_view('cumulative', checked))
        
        show_components = view_menu.addAction('Show Reward Components')
        show_components.setCheckable(True)
        show_components.setChecked(True)
        show_components.triggered.connect(lambda checked: self.visualization.toggle_view('components', checked))
    
    def open_files(self):
        """Open dialog to select log and video files"""
        
        log_file, _ = QFileDialog.getOpenFileName(
            self, "Select Event Log File", DEFAULT_LOG_DIR, "CSV Files (*.csv)"
        )
        
        if log_file:
            # Try to find corresponding video with same base name
            base_name = os.path.splitext(os.path.basename(log_file))[0]
            video_dir = os.path.dirname(log_file)
            possible_video = os.path.join(video_dir, f"{base_name}.mp4")
            
            video_file = possible_video if os.path.exists(possible_video) else None
            
            # If no matching video found, ask user to select
            if not video_file:
                video_file, _ = QFileDialog.getOpenFileName(
                    self, "Select Video File", video_dir, "Video Files (*.mp4)"
                )
            
            if video_file:
                self.load_data(log_file, video_file)
    
    def load_latest_data(self):
        """Find and load the most recent log and video files"""
        
        try:
            # Find CSV files in log directory
            csv_files = [f for f in os.listdir(DEFAULT_LOG_DIR) if f.endswith('.csv')]
            
            if not csv_files:
                return
            
            # Sort by modification time (newest first)
            csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(DEFAULT_LOG_DIR, x)), reverse=True)
            latest_csv = os.path.join(DEFAULT_LOG_DIR, csv_files[0])
            
            # Find video files in log directory
            video_files = [f for f in os.listdir(DEFAULT_LOG_DIR) if f.endswith('.mp4')]
            
            if not video_files:
                return
                
            # Sort by modification time (newest first)
            video_files.sort(key=lambda x: os.path.getmtime(os.path.join(DEFAULT_LOG_DIR, x)), reverse=True)
            latest_video = os.path.join(DEFAULT_LOG_DIR, video_files[0])
            
            self.load_data(latest_csv, latest_video)
            
        except Exception as e:
            print(f"Error loading latest data: {e}")
    
    def load_data(self, log_file, video_file):
        """Load data from log and video files"""

        if not self.data_manager.load_data(log_file):
            print(f"Failed to load data from {log_file}")
            return False
    
        # Load event data
        self.data_manager.load_data(log_file)
                
        # Pass data to visualization
        self.visualization.set_data(
            self.data_manager.time_steps,
            self.data_manager.reward_log,
            self.data_manager.action_log,  # Use action IDs directly
            self.data_manager.reward_components
        )
        
        # Pass video to player
        self.video_player.load_video(video_file)

        
        # Setup timeline controller
        total_steps = len(self.data_manager.time_steps)
        total_frames = self.video_player.total_frames
        self.timeline.setup(total_steps, total_frames)
        
        # Update window title
        self.setWindowTitle(f"Crafter Analysis - {os.path.basename(log_file)}")
    
    def on_timeline_position_changed(self, position):
        """Handle timeline position changes (0-100%)"""
        
        # Update video position
        self.video_player.seek_percent(position)
        
        # Calculate corresponding step for visualization
        frame = self.video_player.current_frame
        step = self.timeline.frame_to_step(frame)
        
        # Update visualization position without triggering back-propagation
        self.visualization.update_position(step, from_timeline=True)
    
        # Define an offset for synchronization 
        self.frame_offset = 1

    # Update the on_video_frame_changed method
    def on_video_frame_changed(self, frame):
        """Handle video frame changes"""

         # Avoid processing the same frame repeatedly at video end
        if hasattr(self, 'last_processed_frame') and self.last_processed_frame == frame:
            return
        self.last_processed_frame = frame

        # Update timeline position
        position = (frame / self.video_player.total_frames) * 100
        self.timeline.set_position(position, from_video=True)
        
        # Calculate corresponding step for visualization with offset
        step = self.timeline.frame_to_step(frame, offset=self.frame_offset)
        
        # Update visualization position
        self.visualization.update_position(step, from_video=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
