import os
import slicer
import sys
import subprocess
from slicer.ScriptedLoadableModule import *

# Import các module cần thiết

# Trong main module định nghĩa class chính
class Tracker(ScriptedLoadableModule):
    """Module for tracking slice positions in 3D Slicer."""
    
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Slice Tracking With Dictation"
        self.parent.categories = ["VsData"]
        self.parent.dependencies = []
        self.parent.contributors = ["sowwn.dev"]
        self.parent.helpText = """
        Module for recording slice position in real-time and generating timeline plots from CSV data.
        """
        self.parent.acknowledgementText = """
        Developed by Sho from sowwn.dev.
        """
        
        # Icon path
        moduleDir = os.path.dirname(self.parent.path)
        iconPath = os.path.join(moduleDir, 'Resources/Icons', 'SliceTracker.png')
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)


# Tạo các class cần thiết cho ScriptedLoadableModule
from Libs.LogicLib import TrackerLogic
from Libs.WidgetLib import TrackerWidget
from Testing.Test import TrackerTest

# Quy định rõ các class cần thiết để Slicer biết
# Class nào sẽ được dùng làm Logic và Widget
TrackerLogic = TrackerLogic
TrackerWidget = TrackerWidget
TrackerTest  = TrackerTest