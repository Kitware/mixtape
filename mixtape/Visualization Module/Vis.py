# Vis.py - Updated with improved colors, cleaner annotations, and better styling

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                           QGraphicsDropShadowEffect, QSizePolicy, QGridLayout, QPushButton, 
                           QTabWidget, QListWidget, QStyle, QFrame, QGraphicsDropShadowEffect, QListWidgetItem,
                           QToolTip)
from PyQt5.QtCore import Qt, pyqtSlot, QRectF, QEvent
from PyQt5.QtGui import QColor, QPainter, QFont, QBrush, QPicture, QIcon, QCursor
import pyqtgraph as pg

class InfoPanel(QFrame):
    """Panel that displays achievement information in two tabs"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set up styling
        self.setFrameShape(QFrame.StyledPanel)
        self.data_manager = None
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(5)
        
        # Title with proper contrast
        self.title_label = QLabel("Agent Achievements")
        self.title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #111111;
            background-color: #f0f0f0;
            padding: 3px;
            border-radius: 3px;
        """)
        self.layout.addWidget(self.title_label)
        
        # Create tabbed container for achievements
        self.achievements_tabs = QTabWidget()
        self.achievements_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 5px 10px;
                margin-right: 2px;
                color: #666666;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
                color: #111111;
            }
        """)
        
        # Create tab for completed achievements
        self.completed_tab = QWidget()
        completed_layout = QVBoxLayout(self.completed_tab)
        completed_layout.setContentsMargins(5, 5, 5, 5)
        completed_layout.setSpacing(2)
        
        self.completed_list = QListWidget()
        self.completed_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.completed_list.setStyleSheet("""
            QListWidget {
                background-color: #ffffff;
                border: none;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #e0e0e0;
                color: #006600;
            }
            QListWidget::item:hover {
                background-color: #f0fff0;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #cccccc;
                min-height: 20px;
                border-radius: 5px;
            }
        """)
        completed_layout.addWidget(self.completed_list)
        
        # Create tab for available achievements
        self.available_tab = QWidget()
        available_layout = QVBoxLayout(self.available_tab)
        available_layout.setContentsMargins(5, 5, 5, 5)
        available_layout.setSpacing(2)
        
        self.available_list = QListWidget()
        self.available_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.available_list.setStyleSheet("""
            QListWidget {
                background-color: #ffffff;
                border: none;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #e0e0e0;
                color: #666666;
            }
            QListWidget::item:hover {
                background-color: #f0f0f0;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #cccccc;
                min-height: 20px;
                border-radius: 5px;
            }
        """)
        available_layout.addWidget(self.available_list)
        
        # Add tabs to container
        self.achievements_tabs.addTab(self.completed_tab, "Completed (0)")
        self.achievements_tabs.addTab(self.available_tab, "Available (22)")
        
        # Add the tabs widget to the main layout
        self.layout.addWidget(self.achievements_tabs)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)
        
        # Initialize achievement lists
        self.initialize_achievements()
    
    def initialize_achievements(self):
        """Initialize the achievement lists - all start in available tab"""
        self.achievement_list = [
            'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
            'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
            'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
            'place_plant', 'place_stone', 'place_table', 'wake_up'
        ]
        
        # Initially all achievements are in the available tab
        self.achievements_tabs.addTab(self.completed_tab, "Completed (0)")
        self.achievements_tabs.addTab(self.available_tab, f"Available ({len(self.achievement_list)})")

        
        # Add all achievements to the available list
        for ach in self.achievement_list:
            item = QListWidgetItem()
            name = ach.replace('_', ' ').title()
            
            # Check if dependencies exist to show additional info
            if self.data_manager:
                deps = self.data_manager.get_achievement_dependencies(ach)
                if deps:
                    deps_text = ", ".join(d.replace('_', ' ').title() for d in deps)
                    item.setText(f"◯ {name} (Needs: {deps_text})")
                else:
                    item.setText(f"◯ {name}")
            else:
                item.setText(f"◯ {name}")
                
            item.setData(Qt.UserRole, ach)  # Store achievement ID
            self.available_list.addItem(item)
    
    
    def update_state(self, step_data):
        """Update the achievements based on the current timeline position"""
        if not step_data or not self.data_manager:
            return
            
        # Get current step
        current_step = step_data.get('time_step', 0)
        
        # Get completed achievements UP TO the current step
        completed = self.data_manager.get_completed_achievements(step=current_step)
        step_achievements = self.data_manager.get_step_achievements(current_step)
        
        # Track which achievements need to move between tabs
        to_move_to_completed = []
        to_move_to_available = []
        
        # Check available achievements to see if any should move to completed
        for i in range(self.available_list.count()):
            item = self.available_list.item(i)
            ach_id = item.data(Qt.UserRole)
            
            if ach_id in completed:
                to_move_to_completed.append(ach_id)
        
        # Check completed achievements to see if any should move back to available
        for i in range(self.completed_list.count()):
            item = self.completed_list.item(i)
            ach_id = item.data(Qt.UserRole)
            
            if ach_id not in completed:
                to_move_to_available.append(ach_id)
        
        # Move items between lists
        for ach_id in to_move_to_completed:
            self._move_achievement_to_completed(ach_id, ach_id in step_achievements)
            
        for ach_id in to_move_to_available:
            self._move_achievement_to_available(ach_id)
        
        # Update available tab to show dependencies
        self._update_available_dependencies(completed)
        
        # Update tab titles with counts
        self.achievements_tabs.setTabText(0, f"Completed ({self.completed_list.count()})")
        self.achievements_tabs.setTabText(1, f"Available ({self.available_list.count()})")
        
        # Update main title with counts
        total = len(self.achievement_list)
        completed_count = self.completed_list.count()
        self.title_label.setText(f"Agent Achievements ({completed_count}/{total})")
    
    def _move_achievement_to_completed(self, ach_id, is_new=False):
        """Move an achievement from available to completed tab"""
        # Find and remove from available list
        for i in range(self.available_list.count()):
            item = self.available_list.item(i)
            if item.data(Qt.UserRole) == ach_id:
                item = self.available_list.takeItem(i)
                name = ach_id.replace('_', ' ').title()
                
                # Modify the item for completed list
                if is_new:
                    item.setBackground(QBrush(QColor('#7fff00')))  # Chartreuse green
                    item.setForeground(QBrush(QColor('#000000')))  # Black text for contrast
                    # Make font bold for new achievements
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                else:
                    item.setText(f"✓ {name}")
                    item.setBackground(QBrush(QColor('#b3e6cc')))  # Light green
                    item.setForeground(QBrush(QColor('#006600')))  # Dark green text
                
                # Add to completed list
                self.completed_list.addItem(item)
                break
    
    def _move_achievement_to_available(self, ach_id):
        """Move an achievement from completed back to available tab"""
        # Find and remove from completed list
        for i in range(self.completed_list.count()):
            item = self.completed_list.item(i)
            if item.data(Qt.UserRole) == ach_id:
                item = self.completed_list.takeItem(i)
                name = ach_id.replace('_', ' ').title()
                
                # Reset formatting
                item.setBackground(QBrush(QColor('#ffffff')))  # White
                item.setForeground(QBrush(QColor('#666666')))  # Gray text
                font = item.font()
                font.setBold(False)
                item.setFont(font)
                
                # Check if dependencies exist
                if self.data_manager:
                    deps = self.data_manager.get_achievement_dependencies(ach_id)
                    if deps:
                        deps_text = ", ".join(d.replace('_', ' ').title() for d in deps)
                        item.setText(f"◯ {name} (Needs: {deps_text})")
                    else:
                        item.setText(f"◯ {name}")
                else:
                    item.setText(f"◯ {name}")
                
                # Add to available list
                self.available_list.addItem(item)
                break
    
    def _update_available_dependencies(self, completed):
        """Update the status of available achievements based on dependencies"""
        for i in range(self.available_list.count()):
            item = self.available_list.item(i)
            ach_id = item.data(Qt.UserRole)
            name = ach_id.replace('_', ' ').title()
            
            if self.data_manager:
                deps = self.data_manager.get_achievement_dependencies(ach_id)
                if deps:
                    # Check if all dependencies are completed
                    deps_completed = all(d in completed for d in deps)
                    if deps_completed:
                        # Achievement is ready to be completed
                        item.setText(f"◯ {name} (Ready!)")
                        item.setBackground(QBrush(QColor('#fff8e1')))  # Light amber
                        item.setForeground(QBrush(QColor('#996600')))  # Dark amber text
                    else:
                        # Show which dependencies are needed
                        deps_text = ", ".join(d.replace('_', ' ').title() for d in deps)
                        item.setText(f"◯ {name} (Needs: {deps_text})")
                        item.setBackground(QBrush(QColor('#ffffff')))  # White
                        item.setForeground(QBrush(QColor('#666666')))  # Gray text


class CustomBarGraphItem(pg.GraphicsObject):
    """Custom bar graph with improved styling and tooltips"""
    
    def __init__(self, x, height, width=0.8, brushes=None, pens=None):
        pg.GraphicsObject.__init__(self)
        self.x = np.array(x)
        self.height = np.array(height)
        self.width = width

        if brushes is None:
            self.brushes = [pg.mkBrush(100, 100, 255, 150) for _ in height]
        else:
            self.brushes = brushes
            
        if pens is None:
            self.pens = [pg.mkPen(None) for _ in height]
        else:
            self.pens = pens
            
        self._picture = None
        self._boundingRect = None
        self.generatePicture()
    
    def generatePicture(self):
        """Pre-render the bars as a QPicture object"""
        self._picture = QPicture()
        painter = QPainter(self._picture)
        
        for i in range(len(self.x)):
            x, h = self.x[i], self.height[i]
            
            if h == 0:  # Skip zero-height bars
                continue
                
            rect = QRectF(x - self.width/2, 0, self.width, h)
            painter.setBrush(self.brushes[i])
            painter.setPen(self.pens[i])
            
            # Draw rectangle with rounded corners for positive values
            if h > 0:
                painter.drawRoundedRect(rect, 2, 2)
            else:
                painter.drawRect(rect)
                
        painter.end()
        
        # Calculate bounding rect
        xmin = min(self.x) - self.width/2
        xmax = max(self.x) + self.width/2
        ymin = min(0, min(self.height))
        ymax = max(0, max(self.height))
        
        self._boundingRect = QRectF(xmin, ymin, xmax-xmin, ymax-ymin)
    
    def paint(self, painter, option, widget):
        painter.drawPicture(0, 0, self._picture)
    
    def boundingRect(self):
        return self._boundingRect


class DecisionPoint(pg.ScatterPlotItem):
    """Enhanced scatter plot item that highlights decision points"""
    
    def __init__(self, x, y, decision_type, importance, actions=None, **kwargs):
        self.decision_type = decision_type  # e.g., 'positive', 'negative', 'neutral'
        self.importance = importance  # Numeric importance (determines size)
        self.actions = actions if actions else []
        
        # Determine symbol based on decision type
        symbol = 'o'  # Default
        if decision_type == 'positive':
            symbol = 't'  # Triangle
        elif decision_type == 'negative':
            symbol = 'd'  # Diamond
        
        # Determine size based on importance
        size = 10 + (importance * 5)
        
        # Determine color based on decision type
        brush = pg.mkBrush(50, 50, 200, 200)  # Default blue
        if decision_type == 'positive':
            brush = pg.mkBrush(50, 200, 50, 200)  # Green
        elif decision_type == 'negative':
            brush = pg.mkBrush(200, 50, 50, 200)  # Red
        
        # Call parent constructor with calculated properties
        super().__init__(x=x, y=y, size=size, symbol=symbol, brush=brush, **kwargs)


class VisualizationWidget(QWidget):
    """Widget for displaying interactive visualizations of agent data"""
    
    def __init__(self):
        super().__init__()

        # Setup the main window properties
        self.setWindowTitle("Crafter Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.frame_offset = 1 

        # Initialize the UI
        self.init_ui()
        
        # Initialize data containers
        self.time_steps = []
        self.reward_log = []
        self.action_log = []
        self.reward_components = {}
        self.cumulative_rewards = []  # Store pre-calculated cumulative rewards
        
        # Variable to track current position
        self.current_step = 0
        
        # Set initial view state
        self.view_state = {
            'cumulative': True,
            'components': True
        }
        
        # Initialize tracking variables for hover items
        self.component_curves = {}
        self.cumulative_data_points = []
        self.hover_text = None
        self.components_hover_text = None
        self.highlight_point = None
        self.hover_line = None
    
    def init_ui(self):
        """Initialize UI components for visualization"""
        
        # Set up the layout
        layout = QVBoxLayout(self)
        layout.setSpacing(4)  # Reduced spacing between plots
        
        # Create info panel at top with lower height
        self.info_panel = InfoPanel()
        layout.addWidget(self.info_panel, 2)  # Lower stretch factor
        
        # Create a plot widget for the cumulative reward with more space
        self.cumulative_plot = pg.PlotWidget(title="Agent Reward Timeline")
        self.cumulative_plot.setBackground('w')  # White background
        self.cumulative_plot.setLabel('left', 'Reward')
        self.cumulative_plot.setLabel('bottom', 'Time Step')
        self.cumulative_plot.showGrid(x=True, y=True, alpha=0.3)
        self.cumulative_plot.setMouseEnabled(x=True, y=True)
        
        # Change text color to dark for better visibility
        self.cumulative_plot.getAxis('left').setTextPen('k')  # Black text
        self.cumulative_plot.getAxis('bottom').setTextPen('k')  # Black text
        self.cumulative_plot.setTitle("Agent Reward Timeline", color="#333", size="12pt")
        
        # Install event filter to clear hover items when mouse leaves plot
        self.cumulative_plot.installEventFilter(self)
        
        # Style improvements for cumulative plot
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        axis_pen = pg.mkPen(color=(120, 120, 120), width=1)
        self.cumulative_plot.getAxis('bottom').setPen(axis_pen)
        self.cumulative_plot.getAxis('left').setPen(axis_pen)
        self.cumulative_plot.getAxis('bottom').setStyle(tickFont=font)
        self.cumulative_plot.getAxis('left').setStyle(tickFont=font)
        
        # Set higher stretch factor for plots
        layout.addWidget(self.cumulative_plot, 3)  # Higher stretch factor
        
        # Create a plot for reward components with more space
        self.components_plot = pg.PlotWidget(title="Reward Component Breakdown")
        self.components_plot.setBackground('w')  # White background
        self.components_plot.setLabel('left', 'Component Value')
        self.components_plot.setLabel('bottom', 'Time Step')
        self.components_plot.showGrid(x=True, y=True, alpha=0.3)
        self.components_plot.setMouseEnabled(x=True, y=True)
        
        # Change text color to dark for better visibility
        self.components_plot.getAxis('left').setTextPen('k')  # Black text
        self.components_plot.getAxis('bottom').setTextPen('k')  # Black text
        self.components_plot.setTitle("Reward Component Breakdown", color="#333", size="12pt")
        
        # Install event filter to clear hover items when mouse leaves plot
        self.components_plot.installEventFilter(self)
        
        # Style improvements for components plot
        self.components_plot.getAxis('bottom').setPen(axis_pen)
        self.components_plot.getAxis('left').setPen(axis_pen)
        self.components_plot.getAxis('bottom').setStyle(tickFont=font)
        self.components_plot.getAxis('left').setStyle(tickFont=font)
        
        # Set higher stretch factor for plots
        layout.addWidget(self.components_plot, 3)  # Higher stretch factor
        
        # Create position markers (vertical lines)
        self.cumulative_position_line = pg.InfiniteLine(
            angle=90, 
            movable=False, 
            pen=pg.mkPen('r', width=2, style=Qt.DashLine)
        )
        self.components_position_line = pg.InfiniteLine(
            angle=90, 
            movable=False, 
            pen=pg.mkPen('r', width=2, style=Qt.DashLine)
        )
        
        self.cumulative_plot.addItem(self.cumulative_position_line)
        self.components_plot.addItem(self.components_position_line)
        
        # Add legend to components plot
        self.components_legend = self.components_plot.addLegend(offset=(-10, 10))
        
        # Set up proxy for mouse hover events
        self.cumulative_proxy = pg.SignalProxy(
            self.cumulative_plot.scene().sigMouseMoved, 
            rateLimit=60, 
            slot=self.on_cumulative_hover
        )
        
        self.components_proxy = pg.SignalProxy(
            self.components_plot.scene().sigMouseMoved, 
            rateLimit=60, 
            slot=self.on_components_hover
        )
        
        # Add highlighted step region for context
        self.highlighted_region = pg.LinearRegionItem(
            values=[0, 0],
            brush=pg.mkBrush(100, 100, 255, 20),
            pen=pg.mkPen(None),
            movable=False
        )
        self.cumulative_plot.addItem(self.highlighted_region)
        self.highlighted_region.setVisible(False)
        
        # Add current step annotation
        self.current_step_text = pg.TextItem(
            text="",
            color=(200, 0, 0),
            anchor=(0.5, 0)
        )
        self.cumulative_plot.addItem(self.current_step_text)
        
        # Set plot size policies to allow expansion
        self.cumulative_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.components_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    
    def eventFilter(self, obj, event):
        """Event filter to clear hover items when mouse leaves plot area"""
        if event.type() == QEvent.Leave:
            if obj == self.cumulative_plot:
                self.clear_cumulative_hover_items()
            elif obj == self.components_plot:
                self.clear_components_hover_items()
        return super().eventFilter(obj, event)
    
    def clear_cumulative_hover_items(self):
        """Clear all hover items from cumulative plot"""
        if hasattr(self, 'hover_text') and self.hover_text is not None:
            try:
                self.cumulative_plot.removeItem(self.hover_text)
            except:
                pass
        
        if hasattr(self, 'highlight_point') and self.highlight_point is not None:
            try:
                self.cumulative_plot.removeItem(self.highlight_point)
            except:
                pass
    
    def clear_components_hover_items(self):
        """Clear all hover items from components plot"""
        if hasattr(self, 'components_hover_text') and self.components_hover_text is not None:
            try:
                self.components_plot.removeItem(self.components_hover_text)
            except:
                pass
        
        if hasattr(self, 'hover_line') and self.hover_line is not None:
            try:
                self.components_plot.removeItem(self.hover_line)
            except:
                pass
    
    def set_data(self, time_steps, reward_log, action_log, reward_components):
        """Set the data for visualization"""
        
        # Store the data
        self.time_steps = time_steps
        self.reward_log = reward_log
        self.action_log = action_log
        self.reward_components = reward_components
        
        # Pre-calculate cumulative rewards
        self.cumulative_rewards = np.cumsum(self.reward_log)
        
        # Clear existing plots
        self.cumulative_plot.clear()
        self.components_plot.clear()
        
        # Re-add position markers
        self.cumulative_plot.addItem(self.cumulative_position_line)
        self.components_plot.addItem(self.components_position_line)
        self.cumulative_plot.addItem(self.current_step_text)
        self.cumulative_plot.addItem(self.highlighted_region)
        
        # Reset the components legend
        if hasattr(self, 'components_legend') and self.components_legend:
            if hasattr(self.components_legend, 'scene') and callable(self.components_legend.scene) and self.components_legend.scene():
                self.components_legend.scene().removeItem(self.components_legend)
        self.components_legend = self.components_plot.addLegend(offset=(-10, 10))
        
        # Update the plots with new data
        self.update_cumulative_plot()
        self.update_components_plot()
        
        # Initialize position at step 0
        self.update_position(0)
        
        # Store data points for hover lookup
        self.cumulative_data_points = []
        for i, (t, r, a, c) in enumerate(zip(self.time_steps, self.reward_log, 
                                           self.action_log, self.cumulative_rewards)):
            self.cumulative_data_points.append({
                'x': t,
                'y': c,
                'time_step': t,
                'action': a,
                'reward': r,
                'cumulative': c,
                'index': i
            })
        
        # Calculate visualization ranges and adjust axes
        if time_steps:
            x_min, x_max = min(time_steps), max(time_steps)
            x_padding = (x_max - x_min) * 0.05  # 5% padding
            
            # Find y-ranges with padding
            if self.cumulative_rewards.size > 0:
                y_min = min(0, np.min(self.cumulative_rewards))
                y_max = max(0, np.max(self.cumulative_rewards))
                y_padding = max((y_max - y_min) * 0.1, 0.5)  # 10% padding or at least 0.5
                
                # Set cumulative plot range
                self.cumulative_plot.setXRange(x_min - x_padding, x_max + x_padding)
                self.cumulative_plot.setYRange(y_min - y_padding, y_max + y_padding)
            
            # Set component plot range
            self.components_plot.setXRange(x_min - x_padding, x_max + x_padding)
    
    def update_cumulative_plot(self):
        """Update the cumulative reward plot with interactive features"""
        
        if not self.time_steps or not self.reward_log:
            return
        
        # Add a legend for decision points with dark text
        legend = self.cumulative_plot.addLegend(offset=(10, 10), loc = 'top-left')
        legend.addItem(pg.ScatterPlotItem(symbol='t', brush=(50, 200, 50, 200)), "Positive Reward")
        legend.addItem(pg.ScatterPlotItem(symbol='d', brush=(200, 50, 50, 200)), "Negative Reward")
        legend.addItem(pg.PlotDataItem(pen=pg.mkPen(color=(0, 0, 255), width=2)), "Cumulative Reward")
        
        # Apply dark text to legend
        for item in legend.items:
            item[1].setText(item[1].text, color='#333')
        
        # Create main line plot
        pen = pg.mkPen(color=(0, 0, 255), width=2.5)
        self.cumulative_curve = self.cumulative_plot.plot(
            self.time_steps, 
            self.cumulative_rewards, 
            pen=pen, 
            name=None
        )
        
        # Identify reward change points (non-zero rewards)
        non_zero_indices = [i for i, r in enumerate(self.reward_log) if abs(r) > 0.001]
        
        if non_zero_indices:
            # Create improved scatter points for reward changes
            self.reward_points = []
            
            for i in non_zero_indices:
                 # Get action name instead of just the value
                action_value = self.action_log[i]
                action_name = "Unknown"
                if self.data_manager:
                    action_name = self.data_manager.get_action_name(action_value)
                
                # Create decision point with action name
                point = DecisionPoint(
                    x=[self.time_steps[i]], 
                    y=[self.cumulative_rewards[i]],
                    decision_type='positive' if self.reward_log[i] > 0 else 'negative',
                    importance=min(abs(self.reward_log[i]) / 0.5, 2.0),
                    data=[{
                        'step': self.time_steps[i],
                        'action_name': action_name,
                        'action_value': action_value,
                        'reward': self.reward_log[i]
                    }]
                )
                # Set hover template for the point
                point.setToolTip(f"Step: {self.time_steps[i]}\nAction: {action_name} ({action_value})\nReward: {self.reward_log[i]:.2f}")

                self.cumulative_plot.addItem(point)
                self.reward_points.append(point)
        
        # Add step rewards as a bar graph with custom styling
        if self.reward_log:
            # Create custom brushes based on reward values
            brushes = []
            pens = []
            
            for r in self.reward_log:
                if r > 0:
                    # Positive reward - green gradient
                    brushes.append(pg.mkBrush(100, min(100 + r*100, 255), 100, 150))
                    pens.append(pg.mkPen(0, 150, 0, 100, width=0.5))
                elif r < 0:
                    # Negative reward - red gradient
                    brushes.append(pg.mkBrush(min(100 + abs(r)*100, 255), 100, 100, 150))
                    pens.append(pg.mkPen(150, 0, 0, 100, width=0.5))
                else:
                    # Zero reward - grey
                    brushes.append(pg.mkBrush(150, 150, 150, 50))
                    pens.append(pg.mkPen(None))
            
            # Create and add custom bar graph
            try:
                reward_bars = CustomBarGraphItem(
                    x=self.time_steps,
                    height=self.reward_log,
                    width=0.6,
                    brushes=brushes,
                    pens=pens
                )
                self.cumulative_plot.addItem(reward_bars)
            except Exception as e:
                print(f"Error creating custom bar graph: {e}")
            
        # Add zero line
        zero_line = pg.InfiniteLine(
            pos=0, 
            angle=0, 
            pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.DashLine)
        )
        self.cumulative_plot.addItem(zero_line)

        # Create decision point with action name
        point = DecisionPoint(
            x=[self.time_steps[i]], 
            y=[self.cumulative_rewards[i]],
            decision_type='positive' if self.reward_log[i] > 0 else 'negative',
            importance=min(abs(self.reward_log[i]) / 0.5, 2.0),
            data=[{
                'step': self.time_steps[i],
                'action_name': action_name,
                'action_value': action_value,
                'reward': self.reward_log[i]
            }]
        )
        # Set hover template for the point
        point.setToolTip(f"Step: {self.time_steps[i]}\nAction: {action_name} ({action_value})\nReward: {self.reward_log[i]:.2f}")

        # Add this line to add the point to the plot
        self.cumulative_plot.addItem(point)
        self.reward_points.append(point)

    
    def update_components_plot(self):
        """Update the reward components plot with interactive features"""
        
        if not self.time_steps or not self.reward_components:
            return
        
        # Create a colormap with more vibrant, distinguishable colors
        colors = [
            (255, 50, 50),     # Bright Red
            (0, 180, 0),       # Bright Green
            (50, 100, 255),    # Bright Blue
            (255, 200, 0),     # Bright Yellow
            (200, 0, 200),     # Bright Magenta
            (0, 200, 200),     # Bright Cyan
            (255, 100, 0),     # Bright Orange
            (150, 50, 250),    # Bright Purple
            (0, 100, 100),     # Teal
            (180, 0, 100)      # Crimson
        ]
        
        # Store curves for hover functionality
        self.component_curves = {}
        
        # Limit to components with non-zero values
        active_components = {}
        for key, values in self.reward_components.items():
            if any(v != 0 for v in values):
                active_components[key] = values
        
        # Track area items for cleanup
        self.component_areas = []
        
        # Calculate baseline positions for stacked areas
        baseline = np.zeros(len(self.time_steps))
        positive_base = np.zeros(len(self.time_steps))
        negative_base = np.zeros(len(self.time_steps))
        
        # Get sorted components for stacking (always positive components on top)
        components_max = {name: max(abs(min(values)), abs(max(values))) 
                         for name, values in active_components.items()}
        sorted_components = sorted(active_components.items(), 
                                  key=lambda x: components_max[x[0]], 
                                  reverse=True)
        
        # For each component, create a filled area plot
        for i, (name, values) in enumerate(sorted_components):
            # Select color
            color = colors[i % len(colors)]
            
            # Create pens with gradient for fill
            pen = pg.mkPen(color=color, width=2)
            
            # Split into positive and negative
            values_array = np.array(values)
            pos_values = np.copy(values_array)
            pos_values[pos_values < 0] = 0
            
            neg_values = np.copy(values_array)
            neg_values[neg_values > 0] = 0
            
            # Create gradient fill for positive values
            if np.any(pos_values > 0):
                fill_brush = pg.mkBrush(color[0], color[1], color[2], 80)  # More opacity
                
                # Plot as stacked area
                fill_curve = pg.FillBetweenItem(
                    pg.PlotDataItem(self.time_steps, positive_base + pos_values), 
                    pg.PlotDataItem(self.time_steps, positive_base), 
                    brush=fill_brush
                )
                self.components_plot.addItem(fill_curve)
                self.component_areas.append(fill_curve)
                
                # Update the baseline for next component
                positive_base = positive_base + pos_values
            
            # Create gradient fill for negative values 
            if np.any(neg_values < 0):
                fill_brush = pg.mkBrush(color[0], color[1], color[2], 80)  # More opacity
                
                # Plot as stacked area
                fill_curve = pg.FillBetweenItem(
                    pg.PlotDataItem(self.time_steps, negative_base), 
                    pg.PlotDataItem(self.time_steps, negative_base + neg_values), 
                    brush=fill_brush
                )
                self.components_plot.addItem(fill_curve)
                self.component_areas.append(fill_curve)
                
                # Update the baseline for next component
                negative_base = negative_base + neg_values
            
            # Draw the line on top
            curve = self.components_plot.plot(
                self.time_steps, 
                values, 
                pen=pen, 
                name=name
            )
            
            self.component_curves[name] = {
                'curve': curve,
                'values': values,
                'color': color
            }
        
        # Add zero line
        zero_line = pg.InfiniteLine(
            pos=0, 
            angle=0, 
            pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.DashLine)
        )
        self.components_plot.addItem(zero_line)
    
    def update_position(self, step, from_video=False, from_timeline=False):
        """Update the position marker in visualizations"""
        
        if not self.time_steps or step >= len(self.time_steps):
            print(f"Invalid step: {step}, total steps: {len(self.time_steps)}")
            return
        
        # Update current step
        self.current_step = step
        
        # Get the x-position (time step)
        x_pos = self.time_steps[step]
        
        # Update position lines
        self.cumulative_position_line.setValue(x_pos)
        self.components_position_line.setValue(x_pos)
        
        # Update highlighted region for context (5 steps before and after)
        context_start = max(0, step - 5)
        context_end = min(len(self.time_steps) - 1, step + 5)
        if context_start < context_end:
            self.highlighted_region.setRegion([
                self.time_steps[context_start], 
                self.time_steps[context_end]
            ])
            self.highlighted_region.setVisible(True)
        else:
            self.highlighted_region.setVisible(False)
        
        # Update step annotation (without Step/Action/Reward labels)
        if self.action_log and step < len(self.action_log):
            action = self.action_log[step]
            
            # Create minimal marker for the current position
            self.current_step_text.setHtml(
                f"<div style='background-color: rgba(255, 255, 255, 0.8); padding: 2px 5px; "
                f"border: 1px solid #aaa; font-size: 9px;'>"
                f"<span style='color: #000;'>{x_pos}</span>"
                f"</div>"
            )
            self.current_step_text.setPos(x_pos, self.cumulative_rewards[step])
            
            # Update info panel with detailed state information
            if hasattr(self, 'info_panel'):
                # Gather all data for this step
                step_data = {
                    'time_step': x_pos,
                    'action': action,
                    'reward': self.reward_log[step],
                    'cumulative_reward': self.cumulative_rewards[step]
                }
                
                # Add component values
                for name, values in self.reward_components.items():
                    if step < len(values):
                        step_data[name] = values[step]
                
                self.info_panel.update_state(step_data)
    
    def on_cumulative_hover(self, event):
        """Handle mouse hover over cumulative plot"""
        # Clear previous hover items
        self.clear_cumulative_hover_items()
        
        if not event or not self.time_steps or not self.reward_log:
            return
        
        # Map mouse position to data coordinates
        pos = event[0]
        plot_item = self.cumulative_plot.getPlotItem()
        view_box = plot_item.getViewBox()
        mouse_point = view_box.mapSceneToView(pos)
        
        # Find closest point to mouse
        if len(self.time_steps) == 0:
            return
            
        # Find closest x coordinate
        x_coord = mouse_point.x()
        closest_idx = np.argmin(np.abs(np.array(self.time_steps) - x_coord))
        
        if closest_idx >= len(self.time_steps):
            return
            
        closest_time = self.time_steps[closest_idx]
        closest_reward = self.cumulative_rewards[closest_idx]
        
        # If mouse is too far from point, don't show tooltip
        if abs(closest_time - x_coord) > (max(self.time_steps) - min(self.time_steps)) * 0.02:
            return
        
        # Get action name
        action_value = self.action_log[closest_idx]
        action_name = "Unknown"
        if self.data_manager:
            action_name = self.data_manager.get_action_name(action_value)
        
        # Create hover text
        self.hover_text = pg.TextItem(
            html=f"<span style='background-color: rgba(255, 255, 255, 0.9); color: black;'>"
                f"Step: {closest_time}<br>"
                f"Action: {action_name} ({action_value})<br>"
                f"Reward: {self.reward_log[closest_idx]:.2f}<br>"
                f"Cumulative: {closest_reward:.2f}</span>",
            anchor=(0, 1), fill=(255, 255, 255, 200)
        )
        self.cumulative_plot.addItem(self.hover_text)
        self.hover_text.setPos(closest_time, closest_reward)
        
        # Create highlight point
        self.highlight_point = pg.ScatterPlotItem(
            [closest_time], [closest_reward], 
            size=12, pen=(200, 200, 200), brush=(255, 255, 0, 200),
            symbol='o'
        )
        self.cumulative_plot.addItem(self.highlight_point)

    
    def on_components_hover(self, event):
        """Handle hover events on the components plot with compact tooltips"""
        
        # Check if component_curves exists
        if not hasattr(self, 'component_curves') or not self.component_curves:
            return
            
        # Convert the event position to plot coordinates
        pos = event[0]
        if not self.components_plot.sceneBoundingRect().contains(pos):
            return
            
        mouse_point = self.components_plot.getViewBox().mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Find the closest time step
        if not self.time_steps:
            return
            
        closest_idx = min(range(len(self.time_steps)), 
                         key=lambda i: abs(self.time_steps[i] - x))
            
        # Only update if we're close enough to a time step
        if abs(self.time_steps[closest_idx] - x) > (max(self.time_steps) - min(self.time_steps)) / 20:
            return
        
        # Clear previous hover items
        self.clear_components_hover_items()
        
        # Gather component values at this time step
        component_values = {}
        for name, data in self.component_curves.items():
            values = data['values']
            if closest_idx < len(values):
                component_values[name] = values[closest_idx]
        
        # Create hover tooltip with compact sizing
        self.components_hover_text = pg.TextItem(
            anchor=(0, 0),
            border=pg.mkPen((50, 50, 50, 100), width=1),
            fill=pg.mkBrush(255, 255, 255, 230)
        )
        
        # Format tooltip HTML with smaller font and dark text color
        tooltip_html = (
            f"<span style='color: #333; font-size: 9px;'>"
            f"<b>Step:</b> {self.time_steps[closest_idx]}<br>"
            f"<b>Action:</b> {self.action_log[closest_idx] if closest_idx < len(self.action_log) else ''}<br>"
        )
        
        # Add component values (limited to top 5 by value)
        sorted_components = sorted(component_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for name, value in sorted_components:
            color = self.component_curves[name]['color']
            tooltip_html += (
                f"<span style='color: rgb({color[0]},{color[1]},{color[2]});'>"
                f"<b>{name}:</b> {value:.2f}</span><br>"
            )
        
        tooltip_html += "</span>"
        
        self.components_hover_text.setHtml(tooltip_html)
        
        # Position tooltip to remain visible within view
        view_rect = self.components_plot.viewRect()
        x_pos = self.time_steps[closest_idx] + 1
        
        # Adjust position if tooltip would go outside view
        if x_pos + 100 > view_rect.right():  # Assuming tooltip width ~100px
            x_pos = self.time_steps[closest_idx] - 1
            self.components_hover_text.setAnchor((1, 0))  # Right-aligned
        
        self.components_hover_text.setPos(x_pos, y)
        self.components_plot.addItem(self.components_hover_text)
        
        # Create a temporary vertical line at hover position
        self.hover_line = pg.InfiniteLine(
            pos=self.time_steps[closest_idx], 
            angle=90, 
            pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.DotLine)
        )
        self.components_plot.addItem(self.hover_line)
    
    def toggle_view(self, view_type, visible):
        """Toggle visibility of different visualization types"""
        
        self.view_state[view_type] = visible
        
        # Update visibility
        if view_type == 'cumulative':
            self.cumulative_plot.setVisible(visible)
        elif view_type == 'components':
            self.components_plot.setVisible(visible)
