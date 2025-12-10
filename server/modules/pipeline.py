import os
import json
import logging
import csv
from typing import Optional, Dict, List
import pandas as pd
import shutil
import glob
import numpy as np
import nibabel as nib

from modules.time_alignment import TimeAlignment
from modules.medical_ner import MedicalNER
from modules.totalsegmentator_service import TotalSegmentatorRunner
from modules.mask_processor import MaskProcessor
from utils.organ_mapping import get_totalsegmentator_class

logger = logging.getLogger(__name__)

class AnatomyDetectionPipeline:
    """Orchestrator for anatomy detection pipeline."""
    
    def __init__(self):
        """Initialize pipeline components."""
        self.timeAlignment = TimeAlignment()
        self.medicalNER = None  # Will be initialized with config
        self.segmentator = None  # TotalSegmentator runner
        self.maskProcessor = None # Will be initialized with temp dir
        
        # Configuration
        self.llm_api_key = os.environ.get("GOOGLE_API_KEY")
        
        print("")
        print(f"LLM API Key: {self.llm_api_key}")
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize pipeline components."""
        try:
            # Initialize Medical NER
            if self.llm_api_key:
                self.medicalNER = MedicalNER(
                    api_key=self.llm_api_key
                )
                logger.info("MedicalNER initialized")
            else:
                logger.warning("LLM API key not found, Medical NER will be disabled")
            
            # Initialize TotalSegmentator
            self.segmentator = TotalSegmentatorRunner()
            self.segmentator.check_gpu()
            logger.info("TotalSegmentatorRunner initialized")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def _generate_csv_report(self, result: Dict, output_path: str):
        """Generate CSV report from pipeline results."""
        try:
            headers = ['Organ', 'Slice Position', 'Class Name', 'Label ID (Full)', 'Label ID (Filtered)', 'Text Transcript (EN)']
            
            # Create reverse mappings for both full and filtered segmentations
            full_class_to_label = {}
            filtered_class_to_label = {}
            
            full_label_mapping = result.get('full_label_mapping', {})
            filtered_label_mapping = result.get('filtered_label_mapping', {})
            
            # Build reverse mappings
            if full_label_mapping:
                for label_id, class_name in full_label_mapping.items():
                    full_class_to_label[class_name] = label_id
            
            if filtered_label_mapping:
                for label_id, class_name in filtered_label_mapping.items():
                    filtered_class_to_label[class_name] = label_id
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                # Get detections and filtered_label_mapping
                detections = result.get('detections', [])
                
                # Create mapping from detection index to label ID
                # Since merge_filtered_masks now processes masks in detection order,
                # label IDs (1,2,3...) correspond to detection indices (0,1,2...)
                detection_to_label = {}
                if filtered_label_mapping:
                    # Sort label IDs to get ordered list
                    sorted_label_ids = sorted(filtered_label_mapping.keys())
                    for idx, label_id in enumerate(sorted_label_ids):
                        if idx < len(detections):
                            detection_to_label[idx] = label_id
                
                for idx, detection in enumerate(detections):
                    # 1. Problematic Part
                    organ = detection.get('organ_name', '')
                    
                    # 2. Slice Position (show both original and padded range)
                    sr = detection.get('slice_range', {})
                    sr_padded = detection.get('slice_range_with_padding', {})
                    
                    if sr_padded:
                        # Show original range and padded range
                        slice_pos = f"{sr.get('view', 'Unknown')}: {sr.get('start', '?')}-{sr.get('end', '?')} (filtered: {sr_padded.get('start', '?')}-{sr_padded.get('end', '?')})"
                    else:
                        # Fallback to original range only
                        slice_pos = f"{sr.get('view', 'Unknown')}: {sr.get('start', '?')}-{sr.get('end', '?')}"
                    
                    # 3. Class name (from totalsegmentator_class)
                    ts_class = detection.get('totalsegmentator_class', '')
                    
                    # 4. Label ID in full segmentation
                    full_label_id = ''
                    if ts_class:
                        # Handle composite classes (comma-separated)
                        if ', ' in ts_class:
                            # For composite, show all label IDs
                            classes = [c.strip() for c in ts_class.split(',')]
                            full_labels = []
                            for cls in classes:
                                if cls in full_class_to_label:
                                    full_labels.append(str(full_class_to_label[cls]))
                            full_label_id = ', '.join(full_labels) if full_labels else ''
                        else:
                            if ts_class in full_class_to_label:
                                full_label_id = str(full_class_to_label[ts_class])
                    
                    # 5. Label ID in filtered segmentation
                    # Use detection index to get label ID (since merge order matches detection order)
                    filtered_label_id = ''
                    if idx in detection_to_label:
                        # Direct mapping by detection index (label IDs are 1,2,3... matching detection order)
                        filtered_label_id = str(detection_to_label[idx])
                    else:
                        # Fallback: Try to find by class name or segment file name
                        if ts_class:
                            # Handle composite classes (can be comma-separated string or list)
                            classes_to_check = []
                            if isinstance(ts_class, str):
                                if ', ' in ts_class:
                                    classes_to_check = [c.strip() for c in ts_class.split(',')]
                                else:
                                    classes_to_check = [ts_class]
                            elif isinstance(ts_class, list):
                                classes_to_check = ts_class
                            
                            # Try to find label ID for each class
                            filtered_labels = []
                            for cls in classes_to_check:
                                if cls in filtered_class_to_label:
                                    filtered_labels.append(str(filtered_class_to_label[cls]))
                            
                            # If no direct match, try to find by organ name (for composite organs)
                            if not filtered_labels:
                                organ_name = detection.get('organ_name', '').lower().replace(' ', '_')
                                for lid, cname in filtered_label_mapping.items():
                                    # Check if organ name matches the class name in mapping
                                    if organ_name in cname.lower() or cname.lower() in organ_name:
                                        filtered_labels.append(str(lid))
                                        break
                                    # Also check if any of the classes match
                                    for cls in classes_to_check:
                                        if cls.lower() in cname.lower() or cname.lower() in cls.lower():
                                            if str(lid) not in filtered_labels:
                                                filtered_labels.append(str(lid))
                            
                            filtered_label_id = ', '.join(filtered_labels) if filtered_labels else ''
                        
                        # Fallback: Try to find by segment file name
                        if not filtered_label_id:
                            segment_path = detection.get('segmentation_3d_path')
                            if segment_path and filtered_label_mapping:
                                segment_file = os.path.basename(segment_path)
                                base_name = segment_file.replace("_filtered.nii.gz", "").replace(".nii.gz", "")
                                for lid, cname in filtered_label_mapping.items():
                                    if base_name == cname or base_name in cname or cname in base_name:
                                        filtered_label_id = str(lid)
                                        break
                    
                    # 6. Text Transcript (EN)
                    transcript_en = detection.get('context', '')
                    
                    writer.writerow([organ, slice_pos, ts_class, full_label_id, filtered_label_id, transcript_en])
            
            # Also generate a summary at the end of CSV
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([])  # Empty row
                writer.writerow(['=== SEGMENTATION FILES ==='])
                
                full_seg_path = result.get('full_segmentation_path', '')
                filtered_seg_path = result.get('filtered_segmentation_path', '')
                
                if full_seg_path:
                    writer.writerow(['Full Segmentation (Unfiltered)', os.path.basename(full_seg_path)])
                if filtered_seg_path:
                    writer.writerow(['Filtered Segmentation (By Position)', os.path.basename(filtered_seg_path)])
            
        except Exception as e:
            logger.error(f"Error generating CSV report: {e}")

    def run(self, tracking_file: str, transcription_file: str, 
            slice_images_dir: Optional[str] = None, language: str = "en") -> Dict:
        """Run the complete anatomy detection pipeline.
        
        Args:
            tracking_file: Path to tracking CSV file
            transcription_file: Path to transcription JSON file
            slice_images_dir: Directory containing slice images AND the original volume file (NRRD/NIfTI)
            language: Language code ("en" or "vi")
            
        Returns:
            dict: Results with detections and segmentation paths
        """
        
        # Step 1: Load data
        tracking_df = pd.read_csv(tracking_file)
        
        # Normalize view names from Slicer colors to Anatomical names
        # Red -> Axial, Yellow -> Sagittal, Green -> Coronal
        tracking_df['view'] = tracking_df['view'].replace({
            'Red': 'Axial', 
            'Yellow': 'Sagittal', 
            'Green': 'Coronal'
        })
        
        # Handle timestamp
        if pd.api.types.is_numeric_dtype(tracking_df['timestamp']):
            if 'elapsed_time' not in tracking_df.columns:
                tracking_df['elapsed_time'] = tracking_df['timestamp']
            tracking_df['timestamp'] = pd.Timestamp('2000-01-01') + pd.to_timedelta(tracking_df['timestamp'], unit='s')
        else:
            tracking_df['timestamp'] = pd.to_datetime(tracking_df['timestamp'])
            # Calculate elapsed time relative to the start of the session
            # This aligns absolute timestamps with the dictation timeline (which starts at 0)
            session_start = tracking_df['timestamp'].min()
            tracking_df['elapsed_time'] = (tracking_df['timestamp'] - session_start).dt.total_seconds()
        
        with open(transcription_file, 'r', encoding='utf-8') as f:
            transcription = json.load(f)
        
        # Extract transcription segments
        from utils.file_handler import get_transcription_segments
        segments = get_transcription_segments(transcription)
        
        if not segments:
            logger.warning("No transcription segments found")
            return {"session_id": None, "detections": []}
        
        # Step 2: Time alignment
        alignments = self.timeAlignment.align(segments, tracking_df)
        
        if not alignments:
            logger.warning("No alignments found")
            return {"session_id": None, "detections": []}
        
        # Step 3: Medical NER (Abnormality Detection Only)
        if not self.medicalNER:
            logger.error("MedicalNER not initialized")
            return {"session_id": None, "detections": []}
        
        # Process full transcript in a single LLM call for both Vietnamese and English
        # This allows LLM to correct STT errors and extract abnormal organs efficiently
        organ_detections = self.medicalNER.extract_from_transcript(alignments, language)
        
        # Print detected organs
        if organ_detections:
            print(f"\n[ORGAN DETECTIONS] Found {len(organ_detections)} abnormal organ(s):")
            for det in organ_detections:
                organ = det.get('organ_name', 'Unknown')
                context = det.get('context', '')
                print(f"  - {organ}: {context}")
        
        # Step 4: TotalSegmentator (Full Volume Segmentation)
        logger.info("Step 4: Running TotalSegmentator on full volume...")
        
        segmentation_output_path = None # Path to the single .nii.gz file
        segmentation_output_dir = None
        input_volume_path = None  # Store reference volume path for merge
        
        # Base output dir
        output_dir_base = os.path.dirname(transcription_file)
        
        if slice_images_dir and os.path.exists(slice_images_dir):
            # Find the volume file (NRRD or NIfTI)
            volume_files = glob.glob(os.path.join(slice_images_dir, "*.nrrd")) + \
                           glob.glob(os.path.join(slice_images_dir, "*.nii.gz")) + \
                           glob.glob(os.path.join(slice_images_dir, "*.nii"))
            
            # Filter out "segmentation" files
            volume_files = [f for f in volume_files if "segmentation" not in os.path.basename(f).lower()]
            
            if volume_files:
                input_volume_path = volume_files[0]
                logger.info(f"Found input volume: {input_volume_path}")
                
                # Define explicit output directory for segmentation
                segmentation_output_dir = os.path.join(output_dir_base, "masks")
                os.makedirs(segmentation_output_dir, exist_ok=True)
                
                try:
                    # Set output_dir to directory (not file path) for TotalSegmentator
                    self.segmentator.output_dir = segmentation_output_dir
                    # Use 'fast' mode, 'total' task (without --ml flag to create individual files)
                    self.segmentator.run(input_volume_path, task="total", fast=True)
                    
                    # TotalSegmentator without --ml creates individual files for each organ
                    # Files are named like: organ_name.nii.gz (e.g., liver.nii.gz, spleen.nii.gz)
                    logger.info(f"Segmentation completed. Individual organ files created in {segmentation_output_dir}")
                        
                except Exception as e:
                    logger.error(f"Segmentation failed: {e}")
                    segmentation_output_dir = None
            else:
                logger.warning("No volume file found. Skipping TotalSegmentator.")
        else:
            logger.warning("slice_images_dir invalid. Skipping TotalSegmentator.")

        # Initialize MaskProcessor
        filtered_masks_dir = os.path.join(output_dir_base, "filtered_masks")
        self.maskProcessor = MaskProcessor(filtered_masks_dir)

        # Step 5: Map Detections AND Filter by Position
        # Get available classes from individual segmentation files
        available_classes = {}
        if segmentation_output_dir and os.path.exists(segmentation_output_dir):
            # List all .nii.gz files in masks directory
            individual_files = glob.glob(os.path.join(segmentation_output_dir, "*.nii.gz"))
            
            # Map file names to class names and label IDs
            from utils.organ_mapping import TOTAL_SEGMENTATOR_LABELS
            for file_path in individual_files:
                file_name = os.path.basename(file_path).replace(".nii.gz", "")
                # Find matching class name and label ID
                for class_name, label_id in TOTAL_SEGMENTATOR_LABELS.items():
                    if class_name == file_name:
                        available_classes[label_id] = class_name
                        break
        
        # Step 5.1: Batch LLM matching for all organs at once
        llm_matches = {}
        if available_classes and self.medicalNER and organ_detections:
            organ_names = [detection.get('organ_name', '') for detection in organ_detections if detection.get('organ_name')]
            if organ_names:
                llm_matches = self.medicalNER.match_organs_to_classes_batch(organ_names, available_classes)
        
        final_detections = []
        
        for detection in organ_detections:
            organ_name = detection.get('organ_name', '')
            
            # Default values
            detection['segmentation_3d_path'] = None
            detection['totalsegmentator_class'] = None
            detection['label_id'] = None
            
            # Use LLM matching result from batch call
            ts_class, label_id = None, None
            is_composite = False
            composite_files = []
            
            if organ_name in llm_matches:
                llm_class, llm_label_id = llm_matches[organ_name]
                if llm_class and llm_label_id:
                    # Check if composite (lists) or single (strings/ints)
                    if isinstance(llm_class, list) and isinstance(llm_label_id, list):
                        is_composite = True
                        ts_class = llm_class  # List of classes
                        label_id = llm_label_id  # List of IDs
                        logger.info(f"Using LLM composite match for '{organ_name}' -> {ts_class} (IDs: {label_id})")
                    else:
                        ts_class = llm_class
                        label_id = llm_label_id
                        logger.info(f"Using LLM match for '{organ_name}' -> '{ts_class}' (ID: {label_id})")
            
            # Fallback to direct mapping if LLM matching failed or unavailable
            if (not ts_class or not label_id):
                ts_class, label_id = get_totalsegmentator_class(organ_name)
                if ts_class and label_id:
                    logger.info(f"Direct mapping matched '{organ_name}' -> '{ts_class}' (ID: {label_id})")
            
            # Find individual segmentation file(s) for this organ
            individual_seg_file = None
            
            if ts_class and segmentation_output_dir and os.path.exists(segmentation_output_dir):
                if is_composite:
                    # Composite organ - collect all component files
                    for class_name in ts_class:
                        possible_file = os.path.join(segmentation_output_dir, f"{class_name}.nii.gz")
                        if os.path.exists(possible_file):
                            composite_files.append(possible_file)
                        else:
                            logger.warning(f"Component file not found: {possible_file}")
                    
                    if composite_files:
                        # Merge composite files into single mask
                        merged_name = f"{organ_name.replace(' ', '_').lower()}_composite"
                        individual_seg_file = self.maskProcessor.merge_composite_masks(composite_files, merged_name)
                else:
                    # Single organ - find file
                    possible_file = os.path.join(segmentation_output_dir, f"{ts_class}.nii.gz")
                    if os.path.exists(possible_file):
                        individual_seg_file = possible_file
            
            if ts_class and individual_seg_file:
                print(f"\n[ORGAN] {organ_name}")
                
                # Step 1: Always calculate mask bbox first (actual organ position)
                mask_bbox_raw = {}
                tracking_bbox_raw = {}
                final_bbox = {}
                bbox_source = None
                
                if individual_seg_file and os.path.exists(individual_seg_file):
                    mask_bbox_raw = self.maskProcessor.calculate_mask_bbox(individual_seg_file)
                    if mask_bbox_raw and 'z' in mask_bbox_raw:
                        mask_z_range = mask_bbox_raw['z']
                        print(f"  [MASK POSITION] Slice Range: {mask_z_range[0]} - {mask_z_range[1]}")
                
                # Step 2: Calculate tracking bbox from transcription time range
                start_time = detection.get('transcription_time', {}).get('start')
                end_time = detection.get('transcription_time', {}).get('end')
                
                if start_time is not None and end_time is not None:
                    if 'elapsed_time' in tracking_df.columns:
                        subset = tracking_df[(tracking_df['elapsed_time'] >= start_time) & 
                                            (tracking_df['elapsed_time'] <= end_time)]
                        
                        if not subset.empty:
                            ax = subset[subset['view'] == 'Axial']
                            if not ax.empty:
                                ax_min = int(ax['slice_number'].min())
                                ax_max = int(ax['slice_number'].max())
                                tracking_bbox_raw['z'] = (ax_min, ax_max)
                                print(f"  [TRACKING POSITION] Slice Range: {ax_min} - {ax_max}")
                
                # Step 3: Decide which bbox to use based on mask and tracking relationship
                if mask_bbox_raw and 'z' in mask_bbox_raw:
                    mask_z = mask_bbox_raw['z']
                    mask_min, mask_max = mask_z[0], mask_z[1]
                    
                    if tracking_bbox_raw and 'z' in tracking_bbox_raw:
                        track_z = tracking_bbox_raw['z']
                        track_min, track_max = track_z[0], track_z[1]
                        
                        # Check if tracking overlaps with mask
                        overlaps = not (track_max < mask_min or track_min > mask_max)
                        
                        if overlaps:
                            # Case 1: Tracking nằm trong mask → dùng tracking (vị trí bác sĩ đang nói)
                            final_bbox = tracking_bbox_raw
                            bbox_source = "tracking_in_mask"
                        else:
                            # Case 2: Tracking nằm ngoài mask → align số slice từ tracking vào mask
                            # Tính số slice của tracking
                            tracking_slice_count = track_max - track_min + 1
                            
                            # Tìm điểm trong mask gần với tracking nhất
                            if track_max < mask_min:
                                # Tracking ở trước mask → điểm gần nhất là mask_min
                                nearest_point = mask_min
                                # Từ mask_min, đếm xuôi số slice của tracking
                                aligned_end = min(mask_min + tracking_slice_count - 1, mask_max)
                                final_bbox = {'z': (mask_min, aligned_end)}
                                bbox_source = "tracking_before_mask_aligned"
                            elif track_min > mask_max:
                                # Tracking ở sau mask → điểm gần nhất là mask_max
                                nearest_point = mask_max
                                # Từ mask_max, đếm ngược số slice của tracking
                                aligned_start = max(mask_min, mask_max - tracking_slice_count + 1)
                                final_bbox = {'z': (aligned_start, mask_max)}
                                bbox_source = "tracking_after_mask_aligned"
                            else:
                                # Should not happen, but fallback
                                final_bbox = mask_bbox_raw
                                bbox_source = "mask_fallback"
                    else:
                        # Case 3: Không có tracking → dùng mask bbox
                        final_bbox = mask_bbox_raw
                        bbox_source = "mask_only"
                else:
                    # Case 4: Không có mask bbox → báo lỗi (không dùng full mask)
                    logger.error(f"  Cannot calculate mask bbox for '{organ_name}'. Cannot proceed without mask position.")
                    final_bbox = None
                
                # Use final_bbox for filtering
                mask_bbox = final_bbox if final_bbox else {}
                
                # Step 4: Filter Mask by zeroing out slices outside range (Axial only)
                filtered_mask_path = None
                if mask_bbox and 'z' in mask_bbox:
                    # Store original slice range (without padding) for report
                    original_z_range = mask_bbox['z']
                    
                    # Update slice_range in detection for the report/API (Axial only)
                    detection['slice_range'] = {
                        'view': 'Axial',
                        'start': original_z_range[0],
                        'end': original_z_range[1],
                        'source': bbox_source
                    }
                    
                    # Filter mask by zeroing out slices outside range (Axial only)
                    padding = 10
                    filtered_mask_path = self.maskProcessor.crop_mask_to_box(
                        mask_path=individual_seg_file,
                        bbox=mask_bbox,
                        padding=padding
                    )
                    
                    if filtered_mask_path:
                        logger.info(f"Created filtered mask for '{organ_name}': {os.path.basename(filtered_mask_path)}")
                    else:
                        logger.warning(f"Failed to create filtered mask for '{organ_name}'")
                    
                    # Calculate and store slice range with padding for reference
                    if filtered_mask_path:
                        try:
                            img = nib.load(individual_seg_file)
                            dims = img.get_fdata().shape
                            z_min_padded = max(0, original_z_range[0] - padding)
                            z_max_padded = min(dims[2], original_z_range[1] + padding + 1) - 1
                            
                            detection['slice_range_with_padding'] = {
                                'view': 'Axial',
                                'start': z_min_padded,
                                'end': z_max_padded,
                                'padding': padding,
                                'source': bbox_source
                            }
                        except Exception as e:
                            logger.warning(f"Could not calculate padded slice range: {e}")
                    
                    # Print final slice range (selected for cutting)
                    sr = detection.get('slice_range', {})
                    print(f"  [SELECTED FOR CUTTING] Slice Range: {sr.get('start')} - {sr.get('end')} (Axial)")
                    
                    if not filtered_mask_path:
                        logger.warning(f"  Failed to filter mask for '{organ_name}'")
                else:
                    # No mask bbox available - cannot proceed
                    logger.error(f"  Cannot create filtered mask for '{organ_name}' - no valid bbox available.")
                    filtered_mask_path = None

            else:
                if not ts_class:
                    logger.warning(f"Could not map organ '{organ_name}' to any TotalSegmentator class.")
                elif not individual_seg_file:
                    logger.warning(f"Individual segmentation file not found for '{organ_name}' (class: {ts_class}).")
                filtered_mask_path = None
                
            # Update detection
            detection['segmentation_3d_path'] = filtered_mask_path
            # Store composite info if applicable
            if is_composite:
                detection['totalsegmentator_class'] = ', '.join(ts_class) if isinstance(ts_class, list) else ts_class
                detection['label_id'] = ', '.join(map(str, label_id)) if isinstance(label_id, list) else label_id
                detection['is_composite'] = True
            else:
                detection['totalsegmentator_class'] = ts_class
                detection['label_id'] = label_id
                detection['is_composite'] = False
            
            # Always add to final result
            final_detections.append(detection)
            
        # Prepare results
        result = {
            'session_id': None,
            'detections': final_detections,
            'segments': alignments,
            'segmentation_dir': segmentation_output_dir
        }
        
        # Step 6a: Merge FULL masks (NOT filtered) for detected organs
        logger.info("Step 6a: Merging full (unfiltered) masks into single segmentation file...")
        try:
            # Collect all unique class names from detections and auto-expand related organs
            full_class_names = []
            
            # Define related organ groups that should be merged together
            organ_expansion_map = {
                # If we detect any lung lobe, include ALL lobes of that side
                'lung_upper_lobe_left': ['lung_upper_lobe_left', 'lung_lower_lobe_left'],
                'lung_lower_lobe_left': ['lung_upper_lobe_left', 'lung_lower_lobe_left'],
                'lung_upper_lobe_right': ['lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right'],
                'lung_middle_lobe_right': ['lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right'],
                'lung_lower_lobe_right': ['lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right'],
                
                # If we detect any kidney part, include the whole kidney
                'kidney_left': ['kidney_left', 'kidney_cyst_left'],
                'kidney_right': ['kidney_right', 'kidney_cyst_right'],
            }
            
            for detection in final_detections:
                ts_class = detection.get('totalsegmentator_class')
                if ts_class:
                    # Handle composite classes (comma-separated string or list)
                    classes_to_process = []
                    if isinstance(ts_class, str):
                        if ', ' in ts_class:
                            classes_to_process = [c.strip() for c in ts_class.split(',')]
                        else:
                            classes_to_process = [ts_class]
                    elif isinstance(ts_class, list):
                        classes_to_process = ts_class
                    
                    # For each class, check if we should expand it
                    for cls in classes_to_process:
                        if cls in organ_expansion_map:
                            # Add all related organs
                            full_class_names.extend(organ_expansion_map[cls])
                        else:
                            full_class_names.append(cls)
            
            # Remove duplicates while preserving order
            unique_full_classes = []
            seen = set()
            for cls in full_class_names:
                if cls and cls not in seen:
                    unique_full_classes.append(cls)
                    seen.add(cls)
            
            if unique_full_classes and segmentation_output_dir:
                # Temporarily change output_dir to save full segmentation to filtered_masks_dir
                original_output_dir = self.maskProcessor.output_dir
                self.maskProcessor.output_dir = filtered_masks_dir
                
                # Merge full masks from original masks directory, but save to filtered_masks_dir for consistency
                full_seg_path, full_label_mapping = self.maskProcessor.merge_full_masks(
                    masks_dir=segmentation_output_dir,
                    class_names=unique_full_classes,
                    output_filename="full_segmentation.nii.gz",
                    reference_volume_path=input_volume_path
                )
                
                # Restore original output_dir
                self.maskProcessor.output_dir = original_output_dir
                
                if full_seg_path:
                    result['full_segmentation_path'] = full_seg_path
                    result['full_label_mapping'] = full_label_mapping
                    logger.info(f"Full segmentation saved to: {full_seg_path}")
                    logger.info(f"Total labels in full segmentation: {len(full_label_mapping)}")
                else:
                    logger.warning("Failed to merge full masks")
            else:
                logger.warning("No classes to merge for full segmentation")
        except Exception as e:
            logger.error(f"Error merging full masks: {e}")
        
        # Step 6b: Merge FILTERED masks into a single segmentation file
        try:
            # Check if filtered_masks_dir exists and has files
            if os.path.exists(filtered_masks_dir):
                all_files = glob.glob(os.path.join(filtered_masks_dir, "*.nii.gz"))
                filtered_files = [f for f in all_files if "_filtered" in os.path.basename(f)]
                logger.info(f"Checking filtered_masks_dir: {filtered_masks_dir}")
                logger.info(f"Found {len(filtered_files)} filtered mask files: {[os.path.basename(f) for f in filtered_files]}")
            
            # Create ordered list of filtered mask files based on detection order
            # This ensures label IDs (1,2,3...) match the order in report.csv
            ordered_filtered_files = []
            for detection in final_detections:
                seg_path = detection.get('segmentation_3d_path')
                if seg_path and os.path.exists(seg_path):
                    ordered_filtered_files.append(seg_path)
                    logger.debug(f"Adding to merge order: {os.path.basename(seg_path)}")
            
            # Use reference volume path to ensure same orientation/affine
            # Save as NRRD format with 3D Slicer-compatible metadata
            # Pass ordered list to ensure label IDs match report.csv order
            filtered_seg_path, filtered_label_mapping = self.maskProcessor.merge_filtered_masks(
                masks_dir=filtered_masks_dir,
                output_filename="filtered_segmentation.nii.gz",
                reference_volume_path=input_volume_path,
                ordered_mask_files=ordered_filtered_files  # Pass ordered list
            )
            
            if filtered_seg_path:
                result['filtered_segmentation_path'] = filtered_seg_path
                result['filtered_label_mapping'] = filtered_label_mapping
                # Keep backwards compatibility
                result['merged_segmentation_path'] = filtered_seg_path
                result['label_mapping'] = filtered_label_mapping
                logger.info(f"Filtered segmentation saved to: {filtered_seg_path}")
                logger.info(f"Total labels in filtered segmentation: {len(filtered_label_mapping)}")
            else:
                logger.warning("Failed to merge filtered masks - no filtered mask files found or merge failed")
        except Exception as e:
            logger.error(f"Error merging filtered masks: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Generate CSV Report
        try:
            csv_output_path = os.path.join(output_dir_base, "report.csv")
            self._generate_csv_report(result, csv_output_path)
        except Exception as e:
            logger.error(f"Failed to generate CSV report: {e}")
        
        return result
