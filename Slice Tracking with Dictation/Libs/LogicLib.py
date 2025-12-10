import slicer
import os
import vtk
import qt
from datetime import datetime
import logging
import csv
import json
from slicer.ScriptedLoadableModule import *
import pandas as pd
import numpy as np
import traceback

# Import utility modules
from Utils.SliceProcess import getSliceIndexFromOffset
from Utils.PlotGenerator import generateTimelineHTML, generateCombinedTimelineHTML, get_transcription_segments
from Utils.AudioRecorder import AudioRecorder
from Utils.AudioTranscriber import AudioTranscriber
from Utils.CurveAnalyzer import CurveAnalyzer
from Utils.AnatomyDetectionAPI import AnatomyDetectionAPI


class TrackerLogic(ScriptedLoadableModuleLogic):
    """Logic class for SliceTracker module."""

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.timerObserverTag = None
        self.sliceNodes = {}  # Dictionary to track multiple views
        self.outputDirectory = "/home/sowwn/Documents/SliceTracker/"  # Set default output directory
        self.data = {
            'Red': [],     # Axial view data
            'Yellow': [],  # Sagittal view data 
            'Green': []    # Coronal view data
        }
        self.startTime = None
        self.startTimeStr = None
        self.timer = None
        self.samplingInterval = 100
        # Always track all views
        self.activeViews = ['Red', 'Yellow', 'Green']
        
        # Audio recording and transcription
        # Hardcode the path to the credentials file relative to this script's location
        base_dir = os.path.dirname(os.path.dirname(__file__)) # This should be "Slice Tracking with Dictation"
        self.credentialsPath = os.path.join(base_dir, "secret", "credentials.json")

        # --- DEBUGGING STEP ---
        print(f"[DEBUG] Attempting to use credentials file at: {self.credentialsPath}")
        logging.info(f"Attempting to use credentials file at: {self.credentialsPath}")
        # --- END DEBUGGING STEP ---

        self.languageCode = "en-US"
        self.transcriptionModel = "medical_dictation"
        self.audioRecorder = AudioRecorder()
        self.audioTranscriber = None # Will be initialized later
        self.enableAudioRecording = True
        self.enableTranscription = True
        
        # File paths
        self.sessionTimestamp = None
        self.audioFile = None
        self.transcriptionFile = None
        self.markersFile = None
        self.trackingFile = None
        self.isTracking = False
        self.currentTrackingData = []
        self.showPlot = True
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.outputDirectory):
            try:
                os.makedirs(self.outputDirectory)
                print(f"Created output directory: {self.outputDirectory}")
            except Exception as e:
                print(f"Error creating output directory: {e}")

        self.curveAnalyzer = CurveAnalyzer()
        self.curveFile = None
        
        # Anatomy detection API client
        self.enableAnatomyDetection = True
        self.anatomyDetectionAPI = None  # Will be initialized with server URL
        self.anatomyDetectionFile = None
        self.serverApiUrl = None

    def setLanguageAndModel(self, language_code, model=None):
        """Sets the language and model for transcription."""
        self.languageCode = language_code
        self.transcriptionModel = model
        logging.info(f"Transcription language set to '{language_code}' with model '{model or 'default'}'")

    def setOutputDirectory(self, path):
        """Set the output directory for all generated files.
        
        Args:
            path (str): Path to output directory
        """
        if not path:
            print("Output directory path cannot be empty")
            return
            
        # Create directory if it doesn't exist
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                print(f"Created output directory: {path}")
            except Exception as e:
                print(f"Error creating output directory: {e}")
                return
            
        self.outputDirectory = path
        print(f"Output directory set to: {path}")
        
        # Update file paths if session is active
        if self.sessionTimestamp:
            session_dir = os.path.join(self.outputDirectory, self.sessionTimestamp)
            self.trackingFile = os.path.join(session_dir, "tracking.csv")
            self.curveFile = os.path.join(session_dir, "curve.csv")
            self.audioFile = os.path.join(session_dir, "audio.wav")
            self.transcriptionFile = os.path.join(session_dir, "transcription.json")
            self.markersFile = os.path.join(session_dir, "markers.csv")
            self.combinedFile = os.path.join(session_dir, f"{self.sessionTimestamp}_combined.csv")
            self.htmlFile = os.path.join(session_dir, f"{self.sessionTimestamp}_combined.html")

    def isOutputDirectorySet(self):
        """Check if output directory is properly set and accessible.
        
        Returns:
            bool: True if output directory is set and accessible, False otherwise
        """
        if not self.outputDirectory:
            print("Output directory is not set. Please set it using setOutputDirectory()")
            return False
            
        try:
            if not os.path.exists(self.outputDirectory):
                os.makedirs(self.outputDirectory)
                print(f"Created output directory: {self.outputDirectory}")
            return os.access(self.outputDirectory, os.W_OK)
        except Exception as e:
            print(f"Error accessing output directory: {e}")
            return False

    def startTracking(self):
        """Start tracking slice position at regular intervals for all views."""
        # Check if output directory is set
        if not self.isOutputDirectorySet():
            slicer.util.warningDisplay("Please set output directory before starting tracking")
            return False
            
        # Initialize the audio transcriber with the right settings
        if self.enableTranscription:
            if not self.credentialsPath:
                slicer.util.errorDisplay("Credentials file not set. Cannot start transcription.")
                return False
            self.audioTranscriber = AudioTranscriber(
                credentials_path=self.credentialsPath,
                language_code=self.languageCode,
                model=self.transcriptionModel
            )

        # Clear old data
        for view in self.data:
            self.data[view] = []
        
        # Save start time and generate session timestamp
        self.startTime = datetime.now()
        self.startTimeStr = self.startTime.isoformat()
        self.sessionTimestamp = self.startTime.strftime("%Y%m%d_%H%M%S")
        
        # Set up file paths
        session_dir = os.path.join(self.outputDirectory, self.sessionTimestamp)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
            
        self.trackingFile = os.path.join(session_dir, "tracking.csv")
        self.curveFile = os.path.join(session_dir, "curve.csv")
        self.audioFile = os.path.join(session_dir, "audio.wav")
        self.transcriptionFile = os.path.join(session_dir, "transcription.json")
        self.markersFile = os.path.join(session_dir, "markers.csv")
        self.combinedFile = os.path.join(session_dir, f"{self.sessionTimestamp}_combined.csv")
        self.htmlFile = os.path.join(session_dir, f"{self.sessionTimestamp}_combined.html")
        self.anatomyDetectionFile = os.path.join(session_dir, f"{self.sessionTimestamp}_anatomy_detection.json")

        
        # Get slice nodes for all views
        allViewsValid = True
        for viewName in self.activeViews:
            self.sliceNodes[viewName] = slicer.app.layoutManager().sliceWidget(viewName).mrmlSliceNode()
            volumeNode = slicer.app.layoutManager().sliceWidget(viewName).sliceLogic().GetBackgroundLayer().GetVolumeNode()
            if not volumeNode:
                slicer.util.warningDisplay(f"No volume loaded in {viewName} view. Tracking will continue but this view may not be recorded properly.")
                allViewsValid = False
        
        # Record the initial position
        self.recordCurrentPosition()
        
        # Set up timer for regular sampling
        self.timer = qt.QTimer()
        self.timer.setInterval(self.samplingInterval)
        self.timer.connect('timeout()', self.onTimerTimeout)
        self.timer.start()
        
        # Start audio recording if enabled
        if self.enableAudioRecording:
            # Start the recording process
            if self.audioRecorder.startRecording(self.audioFile):
                logging.info(f"Recording session started to {self.audioFile}")
            else:
                logging.warning("Failed to start recording session")
        
        # Set tracking state
        self.isTracking = True
        
        # Log the start time
        logging.info(f"Slice tracking started at {self.startTimeStr} for all views")
        
        return allViewsValid


    def stopTracking(self):
        """Stop tracking and recording."""
        try:
            # Stop the timer
            if self.timer:
                self.timer.stop()
                self.timer = None

            # Stop audio recording
            if self.audioRecorder.isRecording:
                self.audioRecorder.stopRecording()
                # Đảm bảo thread ghi âm đã kết thúc ở đây

                # Save dictation markers
                if self.outputDirectory:
                    self.audioRecorder.saveDictationMarkers(self.markersFile)

                # Transcribe audio if enabled
                if self.enableTranscription:
                    if os.path.exists(self.audioFile) and os.path.getsize(self.audioFile) > 1024:
                        if self.audioTranscriber:
                            ok = self.audioTranscriber.transcribe_audio(self.audioFile)
                            if ok:
                                self.audioTranscriber.save_transcription(self.transcriptionFile)
                            else:
                                logging.error("Transcription failed.")
                        else:
                            logging.error("Transcriber not initialized.")
                    else:
                        logging.error("Audio file not found or too small.")
            logging.info("Slice tracking stopped")
        except Exception as e:
            logging.error(f"Lỗi khi stopTracking: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # Nếu có thể, hiển thị thông báo lỗi lên giao diện
            try:
                slicer.util.errorDisplay(f"Lỗi khi dừng tracking: {e}")
            except:
                pass
        # Gọi tiếp các bước sau nếu cần
        self.onStopTracking()
    
    def markDictation(self):
        """Mark current time as having dictation."""
        if self.audioRecorder.isRecording:
            timestamp = self.audioRecorder.markDictation()
            logging.info(f"Dictation marked at {timestamp} seconds")
            return True
        return False
    
    def onTimerTimeout(self):
        """Called on timer event to record current slice positions."""
        self.recordCurrentPosition()
    
    def recordCurrentPosition(self):
        """Record current slice position for all tracked views."""
        if not self.startTime:
            return
            
        currentTime = datetime.now()
        
        for viewName in self.activeViews:
            if viewName in self.sliceNodes:
                sliceNode = self.sliceNodes[viewName]
                # Get slice widget and logic
                sliceWidget = slicer.app.layoutManager().sliceWidget(viewName)
                sliceLogic = sliceWidget.sliceLogic()
                
                # Get volume node
                volumeNode = sliceLogic.GetBackgroundLayer().GetVolumeNode()
                if not volumeNode:
                    # Only log once per view to avoid spamming
                    if not hasattr(self, '_logged_missing_volume'):
                        self._logged_missing_volume = set()
                    if viewName not in self._logged_missing_volume:
                        logging.warning(f"No volume loaded in {viewName} view. Cannot track position.")
                        self._logged_missing_volume.add(viewName)
                    continue
                    
                # Get the slice offset (in mm)
                sliceOffset = sliceNode.GetSliceOffset()
                
                # Simpler approach to get the current slice index
                sliceIndex = None
                
                try:
                    # Get the slice position in RAS space at the center of the view
                    position = [0, 0, 0, 1]  # Center of slice in RAS space
                    sliceNode.GetSliceToRAS().MultiplyPoint([0, 0, 0, 1], position)
                    
                    # Convert RAS position to IJK coordinates
                    rasToIJK = vtk.vtkMatrix4x4()
                    volumeNode.GetRASToIJKMatrix(rasToIJK)
                    
                    # Transform the point from RAS to IJK
                    ijkFloat = [0, 0, 0, 1]
                    rasToIJK.MultiplyPoint(position, ijkFloat)
                    
                    # Round to get integer indices
                    ijk = [int(round(ijkFloat[0])), int(round(ijkFloat[1])), int(round(ijkFloat[2]))]
                    
                    # Select the appropriate index based on view orientation
                    if viewName == 'Red':  # Axial view - K changes with scrolling
                        sliceIndex = ijk[2]
                    elif viewName == 'Yellow':  # Sagittal view - I changes with scrolling
                        sliceIndex = ijk[0]
                    elif viewName == 'Green':  # Coronal view - J changes with scrolling
                        sliceIndex = ijk[1]
                    
                    # Print for debugging
                    # print(f"{viewName} slice: IJK={ijk}, Selected index={sliceIndex}")
                    
                except Exception as e:
                    # print(f"Error calculating slice index for {viewName}: {e}")
                    # Simple fallback - just convert offset to an index
                    # This won't be accurate but better than nothing
                    sliceIndex = int(abs(sliceOffset))
                    # print(f"Fallback to offset-based index: {sliceIndex}")
                
                # Ensure we have a valid index
                if sliceIndex is None:
                    sliceIndex = 0
                
                # Add to data
                self.data[viewName].append({
                    'time': currentTime,
                    'offset': sliceOffset,
                    'sliceIndex': sliceIndex
                })
    
    def exportData(self):
        """Export tracking data to CSV files."""
        # print("[DEBUG] Starting exportData")
        
        # Check if we have any actual data points across all views
        total_points = 0
        if self.data:
            for view in ['Red', 'Yellow', 'Green']:
                if view in self.data:
                    total_points += len(self.data[view])
        
        if total_points == 0:
            # If we have no data in memory, but the file exists and has content,
            # assume it was already written (e.g., double call from Widget) and return success.
            if self.trackingFile and os.path.exists(self.trackingFile) and os.path.getsize(self.trackingFile) > 0:
                print(f"[WARNING] No data in memory to export, but {self.trackingFile} exists. Skipping overwrite.")
                return {'combined': self.trackingFile, 'transcription': self.transcriptionFile}
            
            print("[ERROR] No tracking data to export")
            return None
            
        # print(f"[DEBUG] Data to export: {[f'{view}: {len(data)}' for view, data in self.data.items()]}")
        # print(f"[DEBUG] Tracking file path: {self.trackingFile}")
            
        try:
            with open(self.trackingFile, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['timestamp', 'view', 'slice_number', 'slice_position'])
                
                # Write data for all views
                rows_written = 0
                for view in ['Red', 'Yellow', 'Green']:
                    if view in self.data:
                        view_data = self.data[view]
                        # print(f"[DEBUG] Writing {len(view_data)} rows for {view} view")
                        for data in view_data:
                            writer.writerow([
                                data['time'].strftime("%Y-%m-%d %H:%M:%S.%f"),
                                view,
                                data['sliceIndex'],
                                data['offset']
                            ])
                            rows_written += 1
            
            print(f"Successfully wrote {rows_written} rows to {self.trackingFile}")
            return {'combined': self.trackingFile, 'transcription': self.transcriptionFile}
                
        except Exception as e:
            print(f"[ERROR] Error exporting data: {str(e)}")
            print(traceback.format_exc())
            return None

    def generateTimelinePlot(self, trackingFile):
        """Generate timeline plot from tracking data."""
        print(f"Generating timeline plot from {trackingFile}")
        
        if not os.path.exists(trackingFile):
            print(f"ERROR: Tracking file not found: {trackingFile}")
            return None
            
        try:
            df = pd.read_csv(trackingFile)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            start_time = df['timestamp'].min()
            df['elapsed_time'] = (df['timestamp'] - start_time).dt.total_seconds()
            
            views_data = {}
            for view_code, view_name in {'Red': 'Axial', 'Yellow': 'Sagittal', 'Green': 'Coronal'}.items():
                view_df = df[df['view'] == view_code]
                if not view_df.empty:
                    views_data[view_name] = {
                        'times': view_df['elapsed_time'].tolist(),
                        'indices': view_df['slice_number'].tolist(),
                        'uniqueSlices': sorted(view_df['slice_number'].unique().tolist()),
                        'csv_path': trackingFile
                    }
            
            transcription_data = None
            if self.transcriptionFile and os.path.exists(self.transcriptionFile):
                try:
                    with open(self.transcriptionFile, 'r') as f:
                        raw_transcription = json.load(f)
                    
                    # The get_transcription_segments function from PlotGenerator is robust
                    # and can handle the structure directly.
                    transcription_data = get_transcription_segments(raw_transcription)
                    print(f"Loaded {len(transcription_data)} transcription segments.")

                except Exception as e:
                    print(f"ERROR: Failed to load or parse transcription data: {e}")
            else:
                print("No transcription file found or specified.")
            
            classification_data = None
            output_dir = os.path.dirname(trackingFile)
            base_name = os.path.splitext(os.path.basename(trackingFile))[0]
            
            # First try to load report.csv (from anatomy detection pipeline)
            report_file = os.path.join(output_dir, "report.csv")
            classification_file = os.path.join(output_dir, f"{base_name}_classification.csv")
            
            # Prefer report.csv if it exists
            file_to_load = report_file if os.path.exists(report_file) else classification_file
            
            if os.path.exists(file_to_load):
                try:
                    class_df = pd.read_csv(file_to_load)
                    # Ensure numeric types are correct for JSON serialization
                    # report.csv may have different columns, adapt accordingly
                    for col in ['start_time', 'end_time', 'start', 'end']:
                        if col in class_df.columns:
                            class_df[col] = pd.to_numeric(class_df[col], errors='coerce')
                    
                    # Handle different column names
                    if 'class' in class_df.columns:
                        class_df['class'] = pd.to_numeric(class_df['class'], errors='coerce').astype(int)
                    
                    class_df.dropna(inplace=True) # Drop rows where conversion failed
                    classification_data = class_df.to_dict('records')
                    print(f"Loaded {len(classification_data)} classification segments from {os.path.basename(file_to_load)}.")
                except Exception as e:
                    print(f"ERROR: Failed to load classification data from {file_to_load}: {e}")

            html_content = generateCombinedTimelineHTML(views_data, transcription_data, classification_data)
            
            if not html_content:
                print("ERROR: Failed to generate HTML content from PlotGenerator")
                return None
            
            with open(self.htmlFile, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Saved HTML timeline to {self.htmlFile}")
            
            import webbrowser
            webbrowser.open('file://' + self.htmlFile)
            
            return self.htmlFile
            
        except Exception as e:
            print(f"ERROR: Failed to generate plot: {e}")
            print(traceback.format_exc())
            return None

    def onStopTracking(self):
        """Handle stop tracking event."""
        if self.isTracking:
            print("Stopping tracking...")
            self.isTracking = False
            
            # Stop audio recording and transcribe
            if self.audioRecorder and self.audioRecorder.isRecording:
                print("Stopping audio recording...")
                self.audioRecorder.stopRecording()
                
                if self.enableTranscription:
                    audio_file = self.audioRecorder.getLastRecordingPath()
                    if audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 1024:
                        print(f"Transcribing audio file: {audio_file}")
                        if self.audioTranscriber and self.audioTranscriber.transcribe_audio(self.audioFile):
                            if self.audioTranscriber.save_transcription(self.transcriptionFile):
                                print(f"Transcription saved to {self.transcriptionFile}")
        
            # Export data
            print("Exporting tracking data...")
            if not self.exportData():
                print("Failed to export tracking data")
                return

            # Anatomy detection pipeline (replaces classification)
            if self.enableAnatomyDetection:
                print("Running anatomy detection pipeline...")
                self.runAnatomyDetectionPipeline()
            else:
                # Old classification (disabled by default)
                print("Analyzing and classifying curve...")
                analyzer = CurveAnalyzer()
                analyzer.load_tracking_csv(self.trackingFile)
                results = analyzer.sliding_window_classify()
                
                if results:
                    df = pd.DataFrame(results)
                    output_dir = os.path.dirname(self.trackingFile)
                    base_name = os.path.splitext(os.path.basename(self.trackingFile))[0]
                    classification_file = os.path.join(output_dir, f"{base_name}_classification.csv")
                    df.to_csv(classification_file, index=False)
                    print(f"Saved classification results to: {classification_file}")

            # Generate plot
            print("Generating timeline plot...")
            if self.showPlot and self.trackingFile:
                htmlPath = self.generateTimelinePlot(self.trackingFile)
                if htmlPath:
                    print(f"Timeline plot generated at: {htmlPath}")
                else:
                    print("Failed to generate timeline plot")

            # Create combined CSV for archival/external analysis
            print("Creating combined CSV...")
            self.createCombinedCSV()
                
            # Reset state
            self.data = {'Red': [], 'Yellow': [], 'Green': []}
            self.startTime = None
            print("Tracking stopped successfully")

    def createCombinedCSV(self):
        """Create a combined CSV from tracking and transcription data for archival."""
        if not self.trackingFile or not os.path.exists(self.trackingFile):
            print("Tracking file not found, cannot create combined CSV.")
            return

        try:
            tracking_df = pd.read_csv(self.trackingFile)
            tracking_df['timestamp'] = pd.to_datetime(tracking_df['timestamp'])
            
            if not self.transcriptionFile or not os.path.exists(self.transcriptionFile):
                print("Transcription file not found, saving tracking data only to combined file.")
                tracking_df.to_csv(self.combinedFile, index=False)
                return

            with open(self.transcriptionFile, 'r') as f:
                transcription = json.load(f)
            
            segments = get_transcription_segments(transcription)
            if not segments:
                print("No segments in transcription, saving tracking data only.")
                tracking_df.to_csv(self.combinedFile, index=False)
                return

            # This part is complex and its primary value is for human-readable CSV.
            # We will simplify or ensure it's robust. For now, this logic can remain
            # as it doesn't feed into the plot generation anymore.
            # The logic to associate text with slice movements can be kept for the CSV.
            
            session_start_time = tracking_df['timestamp'].min()
            
            # Create a new dataframe from segments
            segments_df = pd.DataFrame(segments)
            segments_df.rename(columns={'start': 'start_time', 'end': 'end_time'}, inplace=True)
            
            # Find the slice info for each segment
            combined_records = []
            for _, segment in segments_df.iterrows():
                seg_start = session_start_time + pd.Timedelta(seconds=segment['start_time'])
                seg_end = session_start_time + pd.Timedelta(seconds=segment['end_time'])

                # Find tracking data within this segment's time range
                relevant_tracking = tracking_df[(tracking_df['timestamp'] >= seg_start) & (tracking_df['timestamp'] <= seg_end)]

                if not relevant_tracking.empty:
                    # Aggregate info from the relevant tracking data
                    agg_info = relevant_tracking.groupby('view')['slice_number'].agg(['min', 'max', 'nunique']).reset_index()
                    agg_info['activity_score'] = agg_info['nunique'] * (agg_info['max'] - agg_info['min'] + 1)
                    
                    most_active = agg_info.sort_values('activity_score', ascending=False).iloc[0]
                    
                    record = {
                        'start_time_sec': segment['start_time'],
                        'end_time_sec': segment['end_time'],
                        'text': segment['text'],
                        'view': most_active['view'],
                        'slice_min': most_active['min'],
                        'slice_max': most_active['max']
                    }
                    combined_records.append(record)
                else:
                    # If no tracking data, still log the text
                    combined_records.append({
                        'start_time_sec': segment['start_time'],
                        'end_time_sec': segment['end_time'],
                        'text': segment['text'],
                        'view': None,
                        'slice_min': None,
                        'slice_max': None
                    })
            
            if combined_records:
                combined_df = pd.DataFrame(combined_records)
                combined_df.to_csv(self.combinedFile, index=False)
                print(f"Combined data saved to {self.combinedFile}")
            else:
                print("No combined data generated.")

        except Exception as e:
            print(f"Error creating combined CSV: {str(e)}")
            print(traceback.format_exc())
            
    def _format_time_hhmmss(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        seconds = round(seconds or 0)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    def analyzeTracking(self, tracking_file=None):
        """Phân tích dữ liệu tracking và phân loại các đoạn."""
        try:
            if tracking_file is None:
                tracking_file = self.trackingFile
            if not tracking_file or not os.path.exists(tracking_file):
                logging.error(f"Tracking file not found: {tracking_file}")
                return False
                
            if not self.curveAnalyzer.load_tracking_csv(tracking_file):
                logging.error("Could not read tracking file for analysis.")
                return False
            
            result_file = self.curveAnalyzer.analyze_and_save(tracking_file)
            if result_file:
                self.curveFile = result_file
                logging.info(f"Analysis results saved to: {result_file}")
                return True
            else:
                logging.error("Analysis failed.")
                return False
                
        except Exception as e:
            logging.error(f"Error during tracking analysis: {str(e)}")
            logging.error(traceback.format_exc())
            return False
    
    def setServerConfig(self, api_url=None):
        """Set server API configuration.
        
        Args:
            api_url: Server API base URL (e.g., "http://localhost:8000")
        """
        self.serverApiUrl = api_url
        # Initialize API client
        if api_url:
            try:
                self.anatomyDetectionAPI = AnatomyDetectionAPI(
                    base_url=api_url
                )
                logging.info(f"AnatomyDetectionAPI initialized with URL: {api_url}")
            except Exception as e:
                logging.error(f"Error initializing AnatomyDetectionAPI: {e}")
                self.anatomyDetectionAPI = None
        else:
            self.anatomyDetectionAPI = None
    
    def extract_slice_images(self, output_dir: str) -> str:
        """Extract slice images from volume and save to directory.
        
        Args:
            output_dir: Directory to save slice images
            
        Returns:
            str: Path to directory containing slice images
        """
        try:
            # Get volume node from Slicer
            volume_node = None
            for view_name in self.activeViews:
                slice_widget = slicer.app.layoutManager().sliceWidget(view_name)
                if slice_widget:
                    volume_node = slice_widget.sliceLogic().GetBackgroundLayer().GetVolumeNode()
                    if volume_node:
                        break
            
            if not volume_node:
                logging.error("No volume node found")
                return None
            
            # Create slice images directory
            slice_images_dir = os.path.join(output_dir, "slice_images")
            if not os.path.exists(slice_images_dir):
                os.makedirs(slice_images_dir)
            
            # Save volume to file for server processing (Required for TotalSegmentator)
            try:
                volume_path = os.path.join(slice_images_dir, "volume.nrrd")
                slicer.util.saveNode(volume_node, volume_path)
                logging.info(f"Saved volume to {volume_path}")
            except Exception as e:
                logging.error(f"Failed to save volume node: {e}")

            # Get image data
            image_data = volume_node.GetImageData()
            if not image_data:
                logging.error("No image data in volume node")
                return None
            
            dims = image_data.GetDimensions()
            view_map = {'Red': ('Axial', 2), 'Yellow': ('Sagittal', 0), 'Green': ('Coronal', 1)}
            
            # Extract slices for each view
            import vtk.util.numpy_support as vtk_numpy
            
            for view_name in self.activeViews:
                view_label, dim_idx = view_map.get(view_name, ('Unknown', 2))
                max_slices = dims[dim_idx]
                
                # Get slice range from tracking data if available
                if os.path.exists(self.trackingFile):
                    tracking_df = pd.read_csv(self.trackingFile)
                    view_tracking = tracking_df[tracking_df['view'] == view_name]
                    if not view_tracking.empty:
                        slice_min = int(view_tracking['slice_number'].min())
                        slice_max = int(view_tracking['slice_number'].max())
                        # Add some padding
                        slice_min = max(0, slice_min - 5)
                        slice_max = min(max_slices - 1, slice_max + 5)
                    else:
                        slice_min, slice_max = 0, max_slices - 1
                else:
                    slice_min, slice_max = 0, max_slices - 1
                
                # Extract slices
                for slice_num in range(slice_min, slice_max + 1):
                    try:
                        # Extract slice based on view
                        if view_name == 'Red':  # Axial
                            slice_array = vtk_numpy.vtk_to_numpy(
                                image_data.GetPointData().GetScalars()
                            ).reshape(dims[2], dims[1], dims[0])[slice_num, :, :]
                        elif view_name == 'Yellow':  # Sagittal
                            slice_array = vtk_numpy.vtk_to_numpy(
                                image_data.GetPointData().GetScalars()
                            ).reshape(dims[2], dims[1], dims[0])[:, :, slice_num]
                        elif view_name == 'Green':  # Coronal
                            slice_array = vtk_numpy.vtk_to_numpy(
                                image_data.GetPointData().GetScalars()
                            ).reshape(dims[2], dims[1], dims[0])[:, slice_num, :]
                        else:
                            continue
                        
                        # Normalize to 0-255
                        if slice_array.max() > 255:
                            slice_array = ((slice_array - slice_array.min()) / 
                                         (slice_array.max() - slice_array.min()) * 255).astype(np.uint8)
                        
                        # Save as PNG
                        from PIL import Image
                        img = Image.fromarray(slice_array)
                        filename = f"{view_label}_slice{slice_num:04d}.png"
                        filepath = os.path.join(slice_images_dir, filename)
                        img.save(filepath)
                        
                    except Exception as e:
                        logging.warning(f"Error extracting slice {slice_num} for {view_name}: {e}")
                        continue
            
            logging.info(f"Extracted slice images to {slice_images_dir}")
            return slice_images_dir
            
        except Exception as e:
            logging.error(f"Error extracting slice images: {e}")
            logging.error(traceback.format_exc())
            return None
    
    def runAnatomyDetectionPipeline(self):
        """Run anatomy detection pipeline via server API.
        
        Steps:
        1. Extract slice images from volume and get volume file path
        2. Send data to server API
        3. Wait for processing results
        4. Save results locally
        """
        try:
            # Check prerequisites
            if not self.anatomyDetectionAPI:
                logging.error("Anatomy detection API not configured. Please set server URL.")
                return False
            
            if not self.trackingFile or not os.path.exists(self.trackingFile):
                logging.error("Tracking file not found")
                return False
            
            if not self.transcriptionFile or not os.path.exists(self.transcriptionFile):
                logging.warning("Transcription file not found, skipping anatomy detection")
                return False
            
            # Step 1: Extract slice images and get volume file path
            print("Step 1: Extracting slice images from volume...")
            output_dir = os.path.dirname(self.trackingFile)
            slice_images_dir = self.extract_slice_images(output_dir)
            
            # Find volume file path
            volume_file_path = None
            if slice_images_dir and os.path.exists(slice_images_dir):
                # Look for volume file in slice_images_dir
                for filename in os.listdir(slice_images_dir):
                    if filename.lower().endswith(('.nrrd', '.nii', '.nii.gz')) and "segmentation" not in filename.lower():
                        volume_file_path = os.path.join(slice_images_dir, filename)
                        break
            
            # If not found in slice_images_dir, try to get volume directly from Slicer
            if not volume_file_path:
                print("Volume file not found in slice_images_dir, trying to get from Slicer volume node...")
                volume_node = None
                for view_name in self.activeViews:
                    slice_widget = slicer.app.layoutManager().sliceWidget(view_name)
                    if slice_widget:
                        volume_node = slice_widget.sliceLogic().GetBackgroundLayer().GetVolumeNode()
                        if volume_node:
                            break
                
                if volume_node:
                    # Save volume to temporary location
                    temp_volume_path = os.path.join(output_dir, "volume.nrrd")
                    try:
                        slicer.util.saveNode(volume_node, temp_volume_path)
                        volume_file_path = temp_volume_path
                        logging.info(f"Saved volume node to {volume_file_path}")
                    except Exception as e:
                        logging.error(f"Failed to save volume node: {e}")
            
            if not volume_file_path or not os.path.exists(volume_file_path):
                logging.error("Volume file not found. Cannot proceed with anatomy detection.")
                return False
            
            # Step 2: Send data to server
            print("Step 2: Sending data to server for processing...")
            response = self.anatomyDetectionAPI.send_data_for_processing(
                tracking_file=self.trackingFile,
                transcription_file=self.transcriptionFile,
                volume_file=volume_file_path,
                session_id=self.sessionTimestamp
            )
            
            if not response:
                logging.error("Failed to send data to server")
                return False
            
            print(f"Data sent successfully. Session ID: {self.sessionTimestamp}")
            
            # Step 3: Wait for results
            print("Step 3: Waiting for server processing...")
            results = self.anatomyDetectionAPI.get_results(
                session_id=self.sessionTimestamp,
                wait_for_completion=True,
                poll_interval=5,
                max_wait_time=600  # 10 minutes max
            )
            
            if not results:
                logging.error("Failed to get results from server")
                return False
            
            # Step 4: Download and save all result files
            print("Step 4: Downloading result files from server...")
            
            # 4a. Download report.csv
            report_path = os.path.join(output_dir, "report.csv")
            print("  - Downloading report.csv...")
            if not self.anatomyDetectionAPI.download_report(self.sessionTimestamp, report_path):
                logging.warning("Failed to download report.csv, but continuing...")
            
            # 4b. Download filtered segmentation
            seg_path = os.path.join(output_dir, "filtered_segmentation.nrrd")
            print("  - Downloading filtered segmentation...")
            if not self.anatomyDetectionAPI.download_segmentation(self.sessionTimestamp, seg_path, seg_type="filtered"):
                logging.warning("Failed to download segmentation file, but continuing...")
            
            # 4c. Extract and save en_transcription.json from results if available
            en_transcription_path = os.path.join(output_dir, "en_transcription.json")
            en_transcription_saved = False
            
            # Check top-level english_translation
            if results.get('english_translation'):
                english_translation = results.get('english_translation')
                try:
                    if isinstance(english_translation, str):
                        # Try to parse as JSON string
                        try:
                            en_transcription_data = json.loads(english_translation)
                        except json.JSONDecodeError:
                            # If not JSON, treat as plain text and create a simple structure
                            en_transcription_data = {"translation": english_translation}
                    else:
                        en_transcription_data = english_translation
                    
                    with open(en_transcription_path, 'w', encoding='utf-8') as f:
                        json.dump(en_transcription_data, f, indent=2, ensure_ascii=False)
                    print(f"  - Saved en_transcription.json")
                    en_transcription_saved = True
                except Exception as e:
                    logging.warning(f"Could not save en_transcription.json: {e}")
            
            # If not found at top level, check in detections (might be in first detection)
            if not en_transcription_saved and 'detections' in results and results['detections']:
                for detection in results['detections']:
                    if 'english_translation' in detection:
                        try:
                            english_translation = detection.get('english_translation')
                            if isinstance(english_translation, str):
                                try:
                                    en_transcription_data = json.loads(english_translation)
                                except json.JSONDecodeError:
                                    en_transcription_data = {"translation": english_translation}
                            else:
                                en_transcription_data = english_translation
                            
                            with open(en_transcription_path, 'w', encoding='utf-8') as f:
                                json.dump(en_transcription_data, f, indent=2, ensure_ascii=False)
                            print(f"  - Saved en_transcription.json from detection")
                            en_transcription_saved = True
                            break
                        except Exception as e:
                            logging.warning(f"Could not save en_transcription.json from detection: {e}")
            
            if not en_transcription_saved:
                logging.info("No English translation found in results - en_transcription.json will not be created")
            
            # 4d. Save full results JSON
            self.anatomyDetectionFile = os.path.join(output_dir, f"{self.sessionTimestamp}_anatomy_detection.json")
            with open(self.anatomyDetectionFile, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  - Saved anatomy detection results JSON")
            
            # Step 5: Generate combined.html if report.csv exists
            if os.path.exists(report_path):
                print("Step 5: Generating combined HTML timeline...")
                # Use existing generateTimelinePlot method but ensure it uses report.csv
                html_path = self.generateTimelinePlot(self.trackingFile)
                if html_path:
                    print(f"  - Generated combined HTML: {html_path}")
                else:
                    logging.warning("Failed to generate combined HTML")
            
            print("All result files downloaded and saved successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Error in anatomy detection pipeline: {e}")
            logging.error(traceback.format_exc())
            return False