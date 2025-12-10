import numpy as np
import pandas as pd
import logging
import os
from Utils.CurveAnalyzer import CurveAnalyzer

class TimeAlignment:
    """Align transcription segments with slice ranges using behavior class-aware multi-window search."""
    
    def __init__(self):
        self.behavior_classifier = CurveAnalyzer()
        self.class_offsets = {1: -2, 2: 0, 3: +2, 4: +1, 5: 0}  # seconds
        self.window_extend = [0, -5, -10, +5, +10]  # seconds
        self.confidence_threshold = 0.3
        
    def detect_behavior_class(self, tracking_df):
        """Detect dominant behavior class for the session.
        
        Args:
            tracking_df: DataFrame with tracking data
            
        Returns:
            int: Dominant behavior class (1-5)
        """
        try:
            # Create temporary CSV file for CurveAnalyzer
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            try:
                # Save tracking_df to temp file
                tracking_df.to_csv(temp_path, index=False)
                
                # Load into classifier
                if not self.behavior_classifier.load_tracking_csv(temp_path):
                    logging.warning("Could not load tracking data for behavior classification")
                    return 3  # Default to class 3
                
                # Classify segments
                segments = self.behavior_classifier.sliding_window_classify()
                
                if not segments:
                    logging.warning("No behavior segments found, defaulting to class 3")
                    return 3
                
                # Find dominant class (most frequent)
                class_counts = {}
                for seg in segments:
                    class_num = seg.get('class', 3)
                    class_counts[class_num] = class_counts.get(class_num, 0) + 1
                
                dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
                logging.info(f"Detected behavior class: {dominant_class}")
                return dominant_class
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
        except Exception as e:
            logging.error(f"Error detecting behavior class: {e}")
            return 3  # Default to class 3
    
    def calculate_stability_score(self, slice_range, oscillation_count):
        """Calculate stability score for a slice range.
        
        Args:
            slice_range: Tuple (min_slice, max_slice)
            oscillation_count: Number of oscillations in the range
            
        Returns:
            float: Stability score (higher = more stable)
        """
        range_size = slice_range[1] - slice_range[0] + 1
        # Formula: stability = 1 / (range_size + oscillation_count + 1)
        # Smaller range and fewer oscillations = higher score
        stability = 1.0 / (range_size + oscillation_count + 1)
        return stability
    
    def oscillation_count(self, slices):
        """Count number of oscillations in slice sequence.
        
        Args:
            slices: Array of slice numbers
            
        Returns:
            int: Number of oscillations
        """
        if len(slices) < 2:
            return 0
        diff = np.diff(slices)
        sign = np.sign(diff)
        sign = sign[sign != 0]  # Remove zeros
        if len(sign) < 2:
            return 0
        return np.sum(np.diff(sign) != 0)
    
    def generate_candidates(self, transcription_segment, tracking_df, base_offset, session_start_time):
        """Generate candidate windows for alignment.
        
        Args:
            transcription_segment: Dict with 'start_time', 'end_time', 'text'
            tracking_df: DataFrame with tracking data
            base_offset: Base offset from behavior class (seconds)
            session_start_time: Session start timestamp
            
        Returns:
            list: List of candidate dicts with alignment info
        """
        candidates = []
        
        # Convert transcription times to datetime
        seg_start_sec = transcription_segment['start_time']
        seg_end_sec = transcription_segment['end_time']
        
        # Generate candidates for each window extension
        for window_offset in self.window_extend:
            total_offset = base_offset + window_offset
            
            # Calculate window times
            window_start_sec = seg_start_sec + total_offset
            window_end_sec = seg_end_sec + total_offset
            
            # Convert to datetime
            window_start = session_start_time + pd.Timedelta(seconds=window_start_sec)
            window_end = session_start_time + pd.Timedelta(seconds=window_end_sec)
            
            # Find tracking data in this window
            window_tracking = tracking_df[
                (tracking_df['timestamp'] >= window_start) & 
                (tracking_df['timestamp'] <= window_end)
            ]
            
            if window_tracking.empty:
                continue
            
            # Group by view and calculate stats
            for view in window_tracking['view'].unique():
                view_tracking = window_tracking[window_tracking['view'] == view]
                
                if view_tracking.empty:
                    continue
                
                slices = view_tracking['slice_number'].values
                slice_min = int(slices.min())
                slice_max = int(slices.max())
                oscillation_count = self.oscillation_count(slices)
                
                stability_score = self.calculate_stability_score(
                    (slice_min, slice_max), 
                    oscillation_count
                )
                
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
        """Select best candidate based on stability score.
        
        Args:
            candidates: List of candidate dicts
            
        Returns:
            dict: Best candidate with added confidence score
        """
        if not candidates:
            return None
        
        # Sort by stability score (descending)
        candidates_sorted = sorted(candidates, key=lambda x: x['stability_score'], reverse=True)
        best = candidates_sorted[0].copy()
        
        # Calculate confidence score
        # Confidence = stability_score * behavior_class_match (assumed 1.0 for now)
        best['confidence_score'] = best['stability_score']
        
        # Add alternative hypotheses (top 3)
        best['alternative_hypotheses'] = [
            {
                'view': c['view'],
                'slice_range': c['slice_range'],
                'stability_score': c['stability_score'],
                'offset_applied': c['offset_applied']
            }
            for c in candidates_sorted[1:4]  # Top 3 alternatives
        ]
        
        return best
    
    def align(self, transcription_segments, tracking_df):
        """Align transcription segments with slice ranges.
        
        Args:
            transcription_segments: List of dicts with 'start_time', 'end_time', 'text'
            tracking_df: DataFrame with tracking data (must have 'timestamp', 'view', 'slice_number')
            
        Returns:
            list: List of alignment results
        """
        if tracking_df.empty or not transcription_segments:
            logging.warning("Empty tracking data or transcription segments")
            return []
        
        # Ensure tracking_df has elapsed_time
        if 'elapsed_time' not in tracking_df.columns:
            tracking_df = tracking_df.copy()
            tracking_df['timestamp'] = pd.to_datetime(tracking_df['timestamp'])
            session_start_time = tracking_df['timestamp'].min()
            tracking_df['elapsed_time'] = (tracking_df['timestamp'] - session_start_time).dt.total_seconds()
        else:
            session_start_time = tracking_df['timestamp'].min()
        
        # Detect behavior class
        behavior_class = self.detect_behavior_class(tracking_df)
        base_offset = self.class_offsets.get(behavior_class, 0)
        
        logging.info(f"Starting alignment with behavior class {behavior_class}, base offset {base_offset}s")
        
        # Align each transcription segment
        alignments = []
        for seg in transcription_segments:
            candidates = self.generate_candidates(
                seg, 
                tracking_df, 
                base_offset,
                session_start_time
            )
            
            if not candidates:
                logging.warning(f"No candidates found for segment: {seg.get('text', '')[:50]}")
                continue
            
            best = self.select_best_by_stability(candidates)
            
            if best:
                alignment_result = {
                    'transcription_segment': seg,
                    'transcription_text': seg.get('text', ''),
                    'transcription_time': {
                        'start': seg['start_time'],
                        'end': seg['end_time']
                    },
                    'slice_range': {
                        'start': best['slice_min'],
                        'end': best['slice_max'],
                        'view': best['view']
                    },
                    'behavior_class': behavior_class,
                    'alignment_offset_applied': best['offset_applied'],
                    'stability_score': best['stability_score'],
                    'confidence_score': best['confidence_score'],
                    'alternative_hypotheses': best.get('alternative_hypotheses', [])
                }
                
                # Flag low confidence alignments
                if best['confidence_score'] < self.confidence_threshold:
                    alignment_result['needs_review'] = True
                    logging.warning(f"Low confidence alignment ({best['confidence_score']:.2f}) for: {seg.get('text', '')[:50]}")
                
                alignments.append(alignment_result)
        
        logging.info(f"Completed alignment: {len(alignments)} segments aligned")
        return alignments

