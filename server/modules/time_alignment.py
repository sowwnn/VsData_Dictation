# Port from client Utils/TimeAlignment.py
import numpy as np
import pandas as pd
import logging
import os
import tempfile
from modules.curve_analyzer import CurveAnalyzer

logger = logging.getLogger(__name__)

class TimeAlignment:
    """Align transcription segments with slice ranges - ported from client."""
    
    def __init__(self):
        self.behavior_classifier = CurveAnalyzer()
        self.class_offsets = {1: -2, 2: 0, 3: +2, 4: +1, 5: 0}
        self.window_extend = [0, -5, -10, +5, +10]
        self.confidence_threshold = 0.3
        
    def detect_behavior_class(self, tracking_df):
        try:
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            try:
                tracking_df.to_csv(temp_path, index=False)
                if not self.behavior_classifier.load_tracking_csv(temp_path):
                    logger.warning("Could not load tracking data for behavior classification")
                    return 3
                segments = self.behavior_classifier.sliding_window_classify()
                if not segments:
                    logger.warning("No behavior segments found, defaulting to class 3")
                    return 3
                class_counts = {}
                for seg in segments:
                    class_num = seg.get('class', 3)
                    class_counts[class_num] = class_counts.get(class_num, 0) + 1
                dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
                logger.info(f"Detected behavior class: {dominant_class}")
                return dominant_class
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        except Exception as e:
            logger.error(f"Error detecting behavior class: {e}")
            return 3
    
    def calculate_stability_score(self, slice_range, oscillation_count):
        range_size = slice_range[1] - slice_range[0] + 1
        stability = 1.0 / (range_size + oscillation_count + 1)
        return stability
    
    def oscillation_count(self, slices):
        if len(slices) < 2:
            return 0
        diff = np.diff(slices)
        sign = np.sign(diff)
        sign = sign[sign != 0]
        if len(sign) < 2:
            return 0
        return np.sum(np.diff(sign) != 0)
    
    def generate_candidates(self, transcription_segment, tracking_df, base_offset, session_start_time):
        candidates = []
        seg_start_sec = transcription_segment['start_time']
        seg_end_sec = transcription_segment['end_time']
        
        # Ensure elapsed_time column exists
        if 'elapsed_time' not in tracking_df.columns:
            if pd.api.types.is_numeric_dtype(tracking_df['timestamp']):
                tracking_df['elapsed_time'] = tracking_df['timestamp']
            else:
                tracking_df['elapsed_time'] = (tracking_df['timestamp'] - tracking_df['timestamp'].min()).dt.total_seconds()
        
        for window_offset in self.window_extend:
            total_offset = base_offset + window_offset
            window_start_sec = seg_start_sec + total_offset
            window_end_sec = seg_end_sec + total_offset
            
            # Use elapsed_time for comparison (more reliable than datetime)
            window_tracking = tracking_df[
                (tracking_df['elapsed_time'] >= window_start_sec) & 
                (tracking_df['elapsed_time'] <= window_end_sec)
            ]
            
            if window_tracking.empty:
                continue
            
            for view in window_tracking['view'].unique():
                view_tracking = window_tracking[window_tracking['view'] == view]
                if view_tracking.empty:
                    continue
                
                slices = view_tracking['slice_number'].values
                slice_min = int(slices.min())
                slice_max = int(slices.max())
                oscillation_count = self.oscillation_count(slices)
                stability_score = self.calculate_stability_score((slice_min, slice_max), oscillation_count)
                
                candidate = {
                    'transcription_segment': transcription_segment,
                    'window_start': window_start_sec,
                    'window_end': window_end_sec,
                    'offset_applied': total_offset,
                    'view': view,
                    'slice_range': (slice_min, slice_max),
                    'slice_min': slice_min,
                    'slice_max': slice_max,
                    'oscillation_count': oscillation_count,
                    'stability_score': stability_score,
                    'data_points': len(view_tracking)
                }
                candidates.append(candidate)
        
        return candidates
    
    def select_best_by_stability(self, candidates):
        if not candidates:
            return None
        candidates_sorted = sorted(candidates, key=lambda x: x['stability_score'], reverse=True)
        best = candidates_sorted[0].copy()
        best['confidence_score'] = best['stability_score']
        best['alternative_hypotheses'] = [
            {
                'view': c['view'],
                'slice_range': c['slice_range'],
                'stability_score': c['stability_score'],
                'offset_applied': c['offset_applied']
            }
            for c in candidates_sorted[1:4]
        ]
        return best
    
    def align(self, transcription_segments, tracking_df):
        if tracking_df.empty or not transcription_segments:
            logger.warning("Empty tracking data or transcription segments")
            return []

        # Ensure elapsed_time column exists and is calculated correctly
        if 'elapsed_time' not in tracking_df.columns:
            tracking_df = tracking_df.copy()
            if pd.api.types.is_numeric_dtype(tracking_df['timestamp']):
                tracking_df['elapsed_time'] = tracking_df['timestamp']
            else:
                tracking_df['timestamp'] = pd.to_datetime(tracking_df['timestamp'])
                session_start_time = tracking_df['timestamp'].min()
                tracking_df['elapsed_time'] = (tracking_df['timestamp'] - session_start_time).dt.total_seconds()

        alignments = []
        for seg in transcription_segments:
            seg_start_sec = seg['start_time']
            seg_end_sec = seg['end_time']

            # Filter tracking data for the duration of the transcription segment
            window_tracking = tracking_df[
                (tracking_df['elapsed_time'] >= seg_start_sec) & 
                (tracking_df['elapsed_time'] <= seg_end_sec)
            ]

            if window_tracking.empty:
                logger.warning(f"No tracking data found for segment: {seg.get('text', '')[:50]}")
                continue

            # Determine the dominant view (the one with the most tracking points)
            if not window_tracking['view'].empty:
                try:
                    dominant_view = window_tracking['view'].mode()[0]
                except IndexError:
                    logger.warning(f"Could not determine dominant view for segment: {seg.get('text', '')[:50]}")
                    continue
            else:
                logger.warning(f"No view data in tracking for segment: {seg.get('text', '')[:50]}")
                continue

            # Filter for the dominant view
            view_tracking = window_tracking[window_tracking['view'] == dominant_view]

            if view_tracking.empty:
                logger.warning(f"Dominant view tracking data is empty for segment: {seg.get('text', '')[:50]}")
                continue
            
            slices = view_tracking['slice_number'].values
            slice_min = int(slices.min())
            slice_max = int(slices.max())

            # Create a simple confidence score based on data points
            confidence = 1.0 - (1.0 / (len(slices) + 1))

            alignment_result = {
                'transcription_segment': seg,
                'transcription_text': seg.get('text', ''),
                'transcription_time': {
                    'start': seg['start_time'],
                    'end': seg['end_time']
                },
                'slice_range': {
                    'start': slice_min,
                    'end': slice_max,
                    'view': dominant_view
                },
                'behavior_class': 3, # Defaulting behavior class, can be refined if needed
                'alignment_offset_applied': 0, # No longer applying complex offsets
                'stability_score': confidence, # Using a simpler confidence metric
                'confidence_score': confidence,
                'alternative_hypotheses': [] # Simplified model does not generate alternatives
            }
            
            alignments.append(alignment_result)

        logger.info(f"Completed simple alignment: {len(alignments)} segments aligned")
        return alignments

