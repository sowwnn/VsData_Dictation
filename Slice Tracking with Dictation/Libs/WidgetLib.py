import os
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
import logging
import json

class TrackerWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Set logic variable
        from Libs.LogicLib import TrackerLogic
        from Utils.AudioRecorder import AudioRecorder
        
        self.logic = TrackerLogic()
        self.recorder = AudioRecorder()  # Create recorder object

        # 1. TRACKING SECTION
        trackingCollapsibleButton = ctk.ctkCollapsibleButton()
        trackingCollapsibleButton.text = "Tracking"
        self.layout.addWidget(trackingCollapsibleButton)
        trackingFormLayout = qt.QFormLayout(trackingCollapsibleButton)
        
        # Output directory selector
        self.outputDirSelector = ctk.ctkPathLineEdit()
        self.outputDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        defaultDir = os.path.join(os.path.expanduser('~'), 'Documents', 'SliceTracker')
        if not os.path.exists(defaultDir):
            try:
                os.makedirs(defaultDir)
            except:
                pass
        self.outputDirSelector.currentPath = defaultDir
        self.outputDirSelector.toolTip = "Select directory where CSV files will be saved"
        trackingFormLayout.addRow("Output directory:", self.outputDirSelector)
        
        # Microphone selector
        self.microphoneSelector = qt.QComboBox()
        self.microphoneSelector.toolTip = "Select microphone for audio recording"
        self.populateMicrophoneSelector()  # Fill the combobox
        trackingFormLayout.addRow("Microphone:", self.microphoneSelector)
        
        # Language model selector
        self.languageModelSelector = qt.QComboBox()
        self.languageModelSelector.addItems(["English (Medical Dictation)", "Vietnamese (Default)"])
        self.languageModelSelector.setToolTip("Select the language and model for transcription")
        trackingFormLayout.addRow("Language Model:", self.languageModelSelector)
        
        # Transcription service selector (can be hidden or removed if only Google is supported)
        self.transcriptionServiceSelector = qt.QComboBox()
        self.transcriptionServiceSelector.addItems(['Google']) # Changed to Google
        self.transcriptionServiceSelector.setToolTip("Select transcription service")
        # self.transcriptionServiceSelector.currentTextChanged.connect(self.onTranscriptionServiceChanged) # Removed
        trackingFormLayout.addRow("Transcription Service:", self.transcriptionServiceSelector)
        self.transcriptionServiceSelector.setVisible(False) # Hide it as we only have one option

        # Auto-transcribe checkbox
        self.autoTranscribeCheckbox = qt.QCheckBox("Auto-transcribe recorded audio")
        self.autoTranscribeCheckbox.setChecked(True)
        trackingFormLayout.addRow("Auto-transcribe:", self.autoTranscribeCheckbox)
        
        # Refresh microphone list button
        refreshButton = qt.QPushButton("Refresh")
        refreshButton.toolTip = "Refresh microphone list"
        refreshButton.connect('clicked(bool)', self.populateMicrophoneSelector)
        trackingFormLayout.addRow("", refreshButton)
        
        # Tracking info label
        self.trackingInfoLabel = qt.QLabel("Will track Axial, Sagittal, and Coronal views simultaneously")
        trackingFormLayout.addRow(self.trackingInfoLabel)
        
        # Start and End buttons in the same row
        buttonsLayout = qt.QHBoxLayout()
        
        # Start button
        self.startButton = qt.QPushButton("Start")
        self.startButton.toolTip = "Start tracking slice position for all views"
        buttonsLayout.addWidget(self.startButton)
        
        # End button
        self.endButton = qt.QPushButton("End")
        self.endButton.toolTip = "End slice tracking and save data"
        self.endButton.enabled = False
        buttonsLayout.addWidget(self.endButton)
        
        # Add buttons layout to form
        trackingFormLayout.addRow("", buttonsLayout)
        
        # 2. VISUALIZATION SECTION (Hidden from UI but code kept for future use)
        # vizCollapsibleButton = ctk.ctkCollapsibleButton()
        # vizCollapsibleButton.text = "Visualization"
        # self.layout.addWidget(vizCollapsibleButton)
        # vizFormLayout = qt.QFormLayout(vizCollapsibleButton)
        
        # Input CSV file selector
        # self.inputFileSelector = ctk.ctkPathLineEdit()
        # self.inputFileSelector.filters = ctk.ctkPathLineEdit.Files
        # self.inputFileSelector.nameFilters = ["CSV Files (*.csv)"]
        # self.inputFileSelector.setCurrentPath("")
        # self.inputFileSelector.toolTip = "Select a CSV file containing slice tracking data"
        # vizFormLayout.addRow("Input CSV file:", self.inputFileSelector)
        
        # Generate Plot button
        # self.plotButton = qt.QPushButton("Generate Timeline Plot")
        # self.plotButton.toolTip = "Create a timeline plot from the selected CSV file"
        # vizFormLayout.addRow("", self.plotButton)
        
        # Auto-generate plot checkbox
        # self.showPlotCheckBox = qt.QCheckBox()
        # self.showPlotCheckBox.checked = True
        # self.showPlotCheckBox.toolTip = "Automatically generate timeline plot when tracking stops"
        # vizFormLayout.addRow("Auto-generate plot:", self.showPlotCheckBox)
        
        # 3. ANATOMY DETECTION SECTION
        anatomyCollapsibleButton = ctk.ctkCollapsibleButton()
        anatomyCollapsibleButton.text = "Anatomy Detection"
        self.layout.addWidget(anatomyCollapsibleButton)
        anatomyFormLayout = qt.QFormLayout(anatomyCollapsibleButton)
        
        # Enable anatomy detection checkbox
        self.enableAnatomyDetectionCheckbox = qt.QCheckBox("Enable anatomy detection")
        self.enableAnatomyDetectionCheckbox.setChecked(True)
        self.enableAnatomyDetectionCheckbox.toolTip = "Enable automatic organ detection and segmentation"
        anatomyFormLayout.addRow("Enable:", self.enableAnatomyDetectionCheckbox)
        
        # Server API URL input
        self.serverApiUrlInput = qt.QLineEdit()
        self.serverApiUrlInput.setPlaceholderText("http://localhost:8000")
        self.serverApiUrlInput.toolTip = "Server API base URL (e.g., http://localhost:8000)"
        anatomyFormLayout.addRow("Server API URL:", self.serverApiUrlInput)
        
        # Connect events
        self.startButton.connect('clicked(bool)', self.onStartButton)
        self.endButton.connect('clicked(bool)', self.onEndButton)
        # self.plotButton.connect('clicked(bool)', self.onPlotButton)  # Hidden - Visualization section disabled
        
        # Final spacer
        self.layout.addStretch(1)

    def onStartButton(self):
        # Check output directory
        outputDir = self.outputDirSelector.currentPath
        if not os.path.exists(outputDir):
            try:
                os.makedirs(outputDir)
            except Exception as e:
                slicer.util.errorDisplay(f"Cannot create output directory: {str(e)}")
                return
        
        # Update output directory for logic
        self.logic.outputDirectory = outputDir
        
        # Set language and model from selector
        selected_option = self.languageModelSelector.currentText
        if "English" in selected_option:
            self.logic.setLanguageAndModel(language_code="en-US", model="medical_dictation")
        else: # Vietnamese
            self.logic.setLanguageAndModel(language_code="vi-VN", model=None)

        # Set selected microphone if available
        if self.microphoneSelector.currentIndex > 0:  # Not "Default"
            deviceIndex = self.microphoneSelector.currentData
            if deviceIndex is not None:
                self.logic.audioRecorder.setInputDevice(deviceIndex)
        
        # Set transcription options
        self.logic.enableTranscription = self.autoTranscribeCheckbox.checked
        self.logic.transcriptionService = self.transcriptionServiceSelector.currentText
        
        # Set anatomy detection options
        self.logic.enableAnatomyDetection = self.enableAnatomyDetectionCheckbox.checked
        
        if self.logic.enableAnatomyDetection:
            # Configure server API
            server_url = self.serverApiUrlInput.text if self.serverApiUrlInput.text else None
            self.logic.setServerConfig(server_url)
        
        # Start tracking
        if self.logic.startTracking():
            # Update UI
            self.startButton.enabled = False
            self.endButton.enabled = True
            # Notification
            slicer.util.infoDisplay("Slice tracking started for all views")
        else:
            slicer.util.warningDisplay("Started tracking, but some views may not have volumes loaded.")

    def onEndButton(self):
        """Stop tracking and save data."""
        # Disable the End button immediately to prevent multiple clicks
        self.endButton.enabled = False
        
        # Stop tracking
        self.logic.stopTracking()
        
        # Export the data
        self.finishExportData()
        
        # Show message to user
        slicer.util.showStatusMessage("Tracking stopped and data saved.", 3000)

    def finishExportData(self):
        """Complete the export process after recording has stopped."""
        # Export data using the original method
        csvFilePaths = self.logic.exportData()
        
        if csvFilePaths and isinstance(csvFilePaths, dict) and len(csvFilePaths) > 0:
            # Generate a combined timeline plot with all views in one HTML file
            if hasattr(self.logic, 'createCombinedTimelinePlot'):
                htmlPath = self.logic.createCombinedTimelinePlot(csvFilePaths)
                if htmlPath:
                    slicer.util.showStatusMessage(f"Combined timeline plot generated: {os.path.basename(htmlPath)}", 3000)
                    
                    # Update file selector with one of the CSV files (if Visualization section is enabled)
                    # csvFilePath = csvFilePaths.get('Red')
                    # if csvFilePath and hasattr(self, 'inputFileSelector'):
                    #     self.inputFileSelector.setCurrentPath(csvFilePath)
        
        # Update UI
        self.startButton.enabled = True
        
        # Notification
        slicer.util.infoDisplay("Tracking data saved and visualization generated.", windowTitle="Tracking Complete")

    def onPlotButton(self):
        """Generate timeline plot from selected CSV file."""
        csvFilePath = self.inputFileSelector.currentPath
        if not csvFilePath or not os.path.exists(csvFilePath):
            slicer.util.errorDisplay("Please select a valid CSV file.")
            return
        
        # If this is a view-specific file, try to find all related files
        baseDir = os.path.dirname(csvFilePath)
        baseFileName = os.path.basename(csvFilePath)
        
        # Check if this is a view-specific file
        viewMatch = None
        for view in ['Red', 'Yellow', 'Green']:
            if view in baseFileName:
                viewMatch = view
                break
        
        if viewMatch:
            # Try to find other view files from the same recording
            baseTimestamp = baseFileName.replace(f"{viewMatch}_tracking_", "").split('.')[0]
            viewFiles = {}
            
            for view in ['Red', 'Yellow', 'Green']:
                potentialFile = os.path.join(baseDir, f"{view}_tracking_{baseTimestamp}.csv")
                if os.path.exists(potentialFile):
                    viewFiles[view] = potentialFile
            
            # If we found multiple views, ask user if they want combined plot
            if len(viewFiles) > 1:
                answer = slicer.util.confirmYesNoDisplay(
                    f"Found data for {len(viewFiles)} views. Generate combined plot?",
                    windowTitle="Combined Plot"
                )
                
                if answer:
                    # Generate combined plot
                    htmlPath = self.logic.createCombinedTimelinePlot(viewFiles)
                    if htmlPath:
                        slicer.util.showStatusMessage(f"Combined timeline plot generated: {os.path.basename(htmlPath)}", 3000)
                    else:
                        slicer.util.errorDisplay("Failed to generate combined timeline plot.")
                    return
        
        # Generate single view plot
        htmlPath = self.logic.createTimelinePlot(csvFilePath)
        if htmlPath:
            slicer.util.showStatusMessage(f"Timeline plot generated: {os.path.basename(htmlPath)}", 3000)
        else:
            slicer.util.errorDisplay("Failed to generate timeline plot.")

    def populateMicrophoneSelector(self):
        """Populate the microphone selector dropdown."""
        self.microphoneSelector.clear()
        
        # Check if we have sounddevice available
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            
            # Add a "Default" option first
            self.microphoneSelector.addItem("Default")
            
            # Add input devices
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    name = f"{device['name']} ({i})"
                    self.microphoneSelector.addItem(name, i)
                    
            logging.info(f"Found {self.microphoneSelector.count()-1} audio input devices")
            return True
                    
        except ImportError:
            # Try with PyAudio if sounddevice not available
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                
                # Add a "Default" option first
                self.microphoneSelector.addItem("Default")
                
                # Add input devices
                for i in range(p.get_device_count()):
                    device_info = p.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:
                        name = f"{device_info['name']} ({i})"
                        self.microphoneSelector.addItem(name, i)
                
                p.terminate()
                logging.info(f"Found {self.microphoneSelector.count()-1} audio input devices")
                return True
                
            except ImportError:
                self.microphoneSelector.addItem("No audio libraries available")
                logging.warning("Neither sounddevice nor PyAudio available. Audio recording disabled.")
                return False
                
        except Exception as e:
            logging.error(f"Error listing audio devices: {str(e)}")
            self.microphoneSelector.addItem("Error getting devices")
            return False

    # This method is no longer needed as we are not downloading any models.
    # def onTranscriptionServiceChanged(self, service):
    #     """Handle transcription service change."""
    #     if service.lower() == 'whisper':
    #         # Check if Whisper model exists
    #         model_path = os.path.join(os.path.expanduser('~'), '.cache', 'whisper', 'medium.pt')
    #         if not os.path.exists(model_path):
    #             # Show download progress dialog
    #             progressDialog = qt.QProgressDialog("Downloading Whisper model...", "Cancel", 0, 100, self)
    #             progressDialog.setWindowModality(qt.Qt.WindowModal)
    #             progressDialog.setAutoClose(True)
    #             progressDialog.setAutoReset(True)
    #             progressDialog.show()
                
    #             # Start download in background
    #             if self.logic.audioTranscriber.download_whisper_model():
    #                 slicer.util.showStatusMessage("Whisper model downloaded successfully", 3000)
    #             else:
    #                 slicer.util.errorDisplay("Failed to download Whisper model")
    #                 # Fallback to another service or disable transcription if no other options
    #                 self.transcriptionServiceSelector.setCurrentText('Whisper') # Keep it on whisper for now
    #                 return

