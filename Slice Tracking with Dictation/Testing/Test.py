import os
import slicer
from slicer.ScriptedLoadableModule import *
import SampleData


# Cập nhật đường dẫn import để phù hợp với cấu trúc mới
from Libs.LogicLib import TrackerLogic

class TrackerTest(ScriptedLoadableModuleTest):
    """Test class for SliceTracker."""

    def setUp(self):
        """Setup for tests."""
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run tests."""
        self.setUp()
        volumeNode = SampleData.SampleDataLogic().downloadCTChest()
        self.delayDisplay("Test volume loaded")
