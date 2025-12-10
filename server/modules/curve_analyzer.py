# Port from client Utils/CurveAnalyzer.py
import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class CurveAnalyzer:
    """Behavior classification analyzer - ported from client."""
    
    def __init__(self):
        self.tracking_data = None
        self.classified_segments = []
        self.class_descriptions = {
            1: "Tần suất cao, range nhỏ (class 1)",
            2: "Class 2: Duyệt view khác ở giữa segment class 1",
            3: "Tần suất thấp, range lớn (class 3)",
            4: "Class 4: Gộp với class 3, có hành vi view khác trong đoạn",
            5: "Class 5: Có hành vi trên cả hai view còn lại"
        }
        self.axial_name = 'Red'
        self.slice_range_small = 15
        self.freq_high = 3
        self.window_size = 15
        self.window_step = 5

    def load_tracking_csv(self, csv_path):
        if not os.path.exists(csv_path):
            logger.error(f"File không tồn tại: {csv_path}")
            return False
        try:
            self.tracking_data = pd.read_csv(csv_path)
            if 'timestamp' not in self.tracking_data.columns:
                logger.error("Cột 'timestamp' không tồn tại trong file CSV.")
                return False
            
            # Handle timestamp: check if it's numeric (seconds) or datetime string
            if pd.api.types.is_numeric_dtype(self.tracking_data['timestamp']):
                # Timestamp is already in seconds (numeric)
                self.tracking_data['elapsed_time'] = self.tracking_data['timestamp']
            else:
                # Timestamp is datetime string, convert it
                self.tracking_data['timestamp'] = pd.to_datetime(self.tracking_data['timestamp'])
                first_time = self.tracking_data['timestamp'].min()
                self.tracking_data['elapsed_time'] = (self.tracking_data['timestamp'] - first_time).dt.total_seconds()
            return True
        except Exception as e:
            logger.error(f"Lỗi khi đọc file tracking: {e}")
            return False

    def oscillation_count(self, slices):
        diff = np.diff(slices)
        sign = np.sign(diff)
        sign = sign[sign != 0]
        return np.sum(np.diff(sign) != 0)
    
    def merge_adjacent_segments(self, segments):
        if not segments:
            return []
        merged = [segments[0].copy()]
        for seg in segments[1:]:
            if seg['class'] == 1 or seg['class'] == 3:
                last = merged[-1]
                if seg['class'] == last['class'] and seg['start_time'] <= last['end_time'] + self.window_step:
                    merged[-1]['end_time'] = max(last['end_time'], seg['end_time'])
                else:
                    merged.append(seg.copy())
        return merged

    def check_other_view_action(self, t0, t1):
        df = self.tracking_data
        count = 0
        for view in df['view'].unique():
            if view == self.axial_name:
                continue
            view_df = df[(df['view'] == view) & (df['elapsed_time'] >= t0) & (df['elapsed_time'] <= t1)]
            if len(view_df['slice_number'].unique()) > 1:
                count += 1
        return count

    def is_class_2(self, seg):
        t0, t1 = seg['start_time'], seg['end_time']
        t0_after = t0 + 0.2 * (t1 - t0)
        t1_before = t1 - 0.2 * (t1 - t0)
        return self.check_other_view_action(t0_after, t1_before) > 0

    def is_class_4(self, seg1, seg3):
        t0 = seg1['start_time']
        t1 = seg1['end_time']
        t1_next = seg3['end_time']
        action_count_full = self.check_other_view_action(t0 + 0.7 * (t1 - t0), t1_next)
        return action_count_full > 0

    def is_class_5(self, seg):
        t0, t1 = seg['start_time'], seg['end_time']
        return self.check_other_view_action(t0, t1) == 2

    def detect_class_2_4(self, segments):
        new_segments = []
        i = 0
        n = len(segments)
        while i < n:
            seg = segments[i]
            if seg['class'] == 1:
                if self.is_class_2(seg):
                    seg2 = seg.copy()
                    seg2['class'] = 2
                    seg2['description'] = self.class_descriptions[2]
                    new_segments.append(seg2)
                    i += 1
                    continue
                elif i + 1 < n and segments[i+1]['class'] == 3 and self.is_class_4(seg, segments[i+1]):
                    seg4 = seg.copy()
                    seg4['end_time'] = segments[i+1]['end_time']
                    seg4['class'] = 4
                    seg4['description'] = self.class_descriptions[4]
                    new_segments.append(seg4)
                    i += 2
                    continue
            new_segments.append(seg)
            i += 1
        final_segments = []
        skip_next = False
        for i, seg in enumerate(new_segments):
            if skip_next:
                skip_next = False
                continue
            if seg['class'] == 4 and i+1 < len(new_segments) and new_segments[i+1]['class'] == 3 and seg['end_time'] == new_segments[i+1]['end_time']:
                skip_next = True
            final_segments.append(seg)
        return final_segments

    def postprocess_class_5(self, segments):
        new_segments = []
        for seg in segments:
            if self.is_class_5(seg):
                seg5 = seg.copy()
                seg5['class'] = 5
                seg5['description'] = self.class_descriptions[5]
                new_segments.append(seg5)
            else:
                new_segments.append(seg)
        return new_segments

    def sliding_window_classify(self):
        df = self.tracking_data[self.tracking_data['view'] == self.axial_name].reset_index(drop=True)
        n = len(df)
        segments = []
        if n == 0:
            return segments
        t_min = df['elapsed_time'].iloc[0]
        t_max = df['elapsed_time'].iloc[-1]
        start_time = t_min
        while start_time < t_max:
            end_time = start_time + self.window_size
            window_df = df[(df['elapsed_time'] >= start_time) & (df['elapsed_time'] < end_time)]
            slices = window_df['slice_number'].values
            if len(slices) == 0:
                start_time += self.window_step
                continue
            rng = np.max(slices) - np.min(slices)
            t0 = window_df['elapsed_time'].iloc[0] if len(window_df) > 0 else start_time
            t1 = window_df['elapsed_time'].iloc[-1] if len(window_df) > 0 else end_time
            if rng > self.slice_range_small:
                segments.append({'start_time': t0, 'end_time': t1, 'class': 3, 'description': self.class_descriptions[3]})
            else:
                segments.append({'start_time': t0, 'end_time': t1, 'class': 1, 'description': self.class_descriptions[1]})
            start_time += self.window_step

        segments = self.merge_adjacent_segments(segments)
        segments = self.detect_class_2_4(segments)
        segments = self.postprocess_class_5(segments)
        self.classified_segments = segments
        return segments

