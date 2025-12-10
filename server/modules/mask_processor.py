import os
import logging
import numpy as np
import nibabel as nib
import glob
import json
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

class MaskProcessor:
    """Process and filter 3D segmentation masks."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def merge_composite_masks(self, mask_paths: List[str], output_name: str) -> Optional[str]:
        """
        Merge multiple individual masks into a single composite mask.
        Used for composite organs like "Right Lung" (upper + middle + lower lobes).
        
        Args:
            mask_paths: List of paths to individual mask files to merge
            output_name: Name for the merged output file
            
        Returns:
            Path to merged mask file or None if failed
        """
        if not mask_paths:
            logger.warning("No mask paths provided for merging")
            return None
        
        # Filter out non-existent files
        valid_paths = [p for p in mask_paths if os.path.exists(p)]
        if not valid_paths:
            logger.warning(f"No valid mask files found from {len(mask_paths)} paths")
            return None
        
        try:
            # Load first mask to get shape and affine
            first_img = nib.load(valid_paths[0])
            merged_data = first_img.get_fdata().astype(np.uint8)
            affine = first_img.affine
            
            logger.info(f"Merging {len(valid_paths)} masks into composite: {output_name}")
            
            # Merge remaining masks (binary OR operation)
            for mask_path in valid_paths[1:]:
                try:
                    img = nib.load(mask_path)
                    data = img.get_fdata().astype(np.uint8)
                    
                    # Check shape compatibility
                    if data.shape != merged_data.shape:
                        logger.warning(f"Shape mismatch for {mask_path}: {data.shape} vs {merged_data.shape}, skipping")
                        continue
                    
                    # Binary OR: any voxel that is non-zero in either mask becomes non-zero
                    merged_data = np.maximum(merged_data, data)
                    
                except Exception as e:
                    logger.warning(f"Error loading mask {mask_path}: {e}, skipping")
                    continue
            
            # Save merged mask
            output_path = os.path.join(self.output_dir, f"{output_name}.nii.gz")
            merged_img = nib.Nifti1Image(merged_data, affine)
            nib.save(merged_img, output_path)
            
            voxel_count = np.sum(merged_data > 0)
            logger.info(f"Saved composite mask to {output_path} ({voxel_count} voxels)")
            return output_path
            
        except Exception as e:
            logger.error(f"Error merging composite masks: {e}")
            return None
    
    def check_label_exists(self, mask_path: str, label_id: int) -> bool:
        """Check if a label exists in the segmentation mask."""
        if not os.path.exists(mask_path):
            return False
        try:
            img = nib.load(mask_path)
            data = img.get_fdata().astype(np.uint8)
            return np.any(data == label_id)
        except Exception as e:
            logger.error(f"Error checking label existence: {e}")
            return False
    
    def get_available_classes(self, mask_path: str) -> Dict[int, str]:
        """
        Extract list of available classes from segmentation mask.
        Returns dict mapping label_id -> class_name.
        """
        if not os.path.exists(mask_path):
            return {}
        
        try:
            img = nib.load(mask_path)
            data = img.get_fdata().astype(np.uint8)
            unique_labels = np.unique(data)
            
            # Remove background (0)
            unique_labels = unique_labels[unique_labels > 0]
            
            # Map label IDs to class names
            from utils.organ_mapping import TOTAL_SEGMENTATOR_LABELS
            
            available_classes = {}
            for label_id in unique_labels:
                # Find class name for this label ID
                for class_name, mapped_id in TOTAL_SEGMENTATOR_LABELS.items():
                    if mapped_id == int(label_id):
                        available_classes[int(label_id)] = class_name
                        break
            
            logger.info(f"Found {len(available_classes)} available classes in segmentation mask")
            return available_classes
        except Exception as e:
            logger.error(f"Error extracting available classes: {e}")
            return {}
    
    def find_similar_vertebrae_label(self, mask_path: str, target_label_id: int) -> Optional[int]:
        """
        Find a similar vertebrae label if target doesn't exist.
        For vertebrae, try nearby labels (L1->L2->L3->L4->L5, or T12->T11->...)
        """
        if not os.path.exists(mask_path):
            return None
        
        try:
            img = nib.load(mask_path)
            data = img.get_fdata().astype(np.uint8)
            unique_labels = np.unique(data)
            
            # Define vertebrae label groups (Lumbar, Thoracic, Cervical)
            lumbar_labels = [18, 19, 20, 21, 22]  # L5, L4, L3, L2, L1
            thoracic_labels = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]  # T12 to T1
            cervical_labels = [35, 36, 37, 38, 39, 40, 41]  # C7 to C1
            
            # Find which group target belongs to
            candidate_labels = []
            if target_label_id in lumbar_labels:
                candidate_labels = lumbar_labels
            elif target_label_id in thoracic_labels:
                candidate_labels = thoracic_labels
            elif target_label_id in cervical_labels:
                candidate_labels = cervical_labels
            
            # Try to find the closest available label
            for candidate_id in candidate_labels:
                if candidate_id in unique_labels:
                    logger.info(f"Found similar vertebrae label {candidate_id} instead of {target_label_id}")
                    return candidate_id
            
            return None
        except Exception as e:
            logger.error(f"Error finding similar vertebrae label: {e}")
            return None
        
    def filter_mask_by_slices(self, mask_path: str, slice_ranges: dict, label_id: int = None, padding: int = 5) -> Optional[str]:
        """
        Filter a 3D mask to keep only the regions within the specified slice ranges.
        
        Args:
            mask_path: Path to the input 3D mask file (NIfTI).
            slice_ranges: Dictionary with keys 'Axial', 'Coronal', 'Sagittal' and values (min, max) tuple.
                          Example: {'Axial': (100, 150)}
            label_id: If provided, only process this specific label ID within the mask (for multilabel files).
                      Other labels will be ignored/removed in the output.
            padding: Number of slices to expand the range by.
            
        Returns:
            Path to the filtered mask file, or None if failed.
        """
        if not os.path.exists(mask_path):
            logger.error(f"Mask file not found: {mask_path}")
            return None
            
        try:
            # Load the mask
            img = nib.load(mask_path)
            data = img.get_fdata().astype(np.uint8)
            affine = img.affine
            
            # If label_id is provided, we are isolating a specific organ from a multilabel file
            # We create a binary mask for this label (or keep it as is but zero out others)
            # But here we want to return a mask that might be displayed.
            # If we want to keep the original label ID:
            if label_id is not None:
                # Check if label exists in the data before filtering
                label_exists = np.any(data == label_id)
                if not label_exists:
                    logger.warning(f"Label {label_id} not found in segmentation mask {mask_path}. Creating empty mask file.")
                    # Create empty mask file instead of returning None
                    filtered_data = np.zeros_like(data)
                else:
                    # Create a new mask array for this specific result
                    filtered_data = np.zeros_like(data)
                    mask_indices = (data == label_id)
                    filtered_data[mask_indices] = label_id # Keep original ID
                
                # Log initial label presence
                initial_voxels = np.sum(filtered_data > 0)
                logger.debug(f"Label {label_id} has {initial_voxels} voxels before slice filtering")
            else:
                filtered_data = data.copy()
            
            # Apply slice filtering
            # IMPORTANT: CT scans typically only scroll through Axial view (I-S axis).
            # We should ONLY filter by Axial slices, not by Coronal or Sagittal.
            # Filtering by multiple axes would result in a too-small region.
            # 
            # Axis mapping for RAS orientation:
            # 0: Sagittal (R-L) - Left to Right
            # 1: Coronal (A-P) - Anterior to Posterior  
            # 2: Axial (I-S) - Inferior to Superior (this is the CT scan direction)
            
            dims = filtered_data.shape
            
            # Start with fully preserving all data
            spatial_mask = np.ones_like(filtered_data, dtype=bool)
            
            # ONLY apply Axial filter (Z-axis, dim 2) - the primary CT scan direction
            # This is the direction the doctor scrolls through during review
            if 'Axial' in slice_ranges:
                min_s, max_s = slice_ranges['Axial']
                # Apply padding
                min_s = max(0, min_s - padding)
                max_s = min(dims[2], max_s + padding)
                
                # Create mask: 0 everywhere, 1 in range
                z_mask = np.zeros(dims[2], dtype=bool)
                z_mask[min_s:max_s] = True
                
                # Broadcast to 3D and apply (this is the ONLY filter we apply)
                spatial_mask = z_mask[np.newaxis, np.newaxis, :]
                
                logger.debug(f"Applied Axial filter: slices {min_s}-{max_s} (dim 2)")
            
            # DO NOT filter by Coronal or Sagittal - these are just auxiliary views
            # The doctor primarily reviews CT scans in Axial view
            
            # Apply the spatial filter
            filtered_data[~spatial_mask] = 0
            
            # Check if we have anything left
            # For multilabel files, check if any voxels with the label_id remain
            # Note: We still create the file even if empty, to ensure every organ has a segment file
            if label_id is not None:
                remaining_voxels = np.sum(filtered_data == label_id)
                if remaining_voxels == 0:
                    logger.warning(f"Filtering resulted in empty mask for label {label_id}. Ranges: {slice_ranges}, Shape: {dims}. Creating empty file.")
                else:
                    logger.debug(f"After filtering, label {label_id} has {remaining_voxels} voxels remaining")
            else:
                if np.sum(filtered_data) == 0:
                    logger.warning(f"Filtering resulted in empty mask. Ranges: {slice_ranges}, Shape: {dims}")
                    return None
                
            # Save result
            # Create a unique filename
            base_name = os.path.basename(mask_path).replace(".nii.gz", "")
            if label_id:
                output_filename = f"{base_name}_label{label_id}_filtered.nii.gz"
            else:
                output_filename = f"{base_name}_filtered.nii.gz"
                
            output_path = os.path.join(self.output_dir, output_filename)
            
            new_img = nib.Nifti1Image(filtered_data, affine)
            nib.save(new_img, output_path)
            
            logger.info(f"Created filtered mask: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing mask: {e}")
            return None
    
    def calculate_mask_bbox(self, mask_path: str) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Calculate bounding box from segmentation mask by finding slices with non-zero voxels.
        This gives the actual spatial extent of the organ in the volume.
        
        Args:
            mask_path: Path to segmentation mask file
            
        Returns:
            Dictionary with 'z' key mapping to (min_slice, max_slice) tuple for Axial view,
            or None if mask is empty or error occurs
        """
        if not os.path.exists(mask_path):
            logger.error(f"Mask file not found: {mask_path}")
            return None
            
        try:
            img = nib.load(mask_path)
            data = img.get_fdata()
            affine = img.affine
            
            # Check if mask has any non-zero voxels
            if np.sum(data > 0) == 0:
                logger.warning(f"Mask {mask_path} is empty (no non-zero voxels)")
                return None
            
            # Determine which dimension corresponds to Axial (z-axis) using affine
            try:
                import nibabel.orientations as nibo
                axcodes = nibo.aff2axcodes(affine)
                logger.info(f"Volume axis codes from affine: {axcodes}")
                # In RAS: Axial slices are perpendicular to Superior (S) axis, which is dimension 2
                axial_dim = 2
                logger.info(f"Using dimension {axial_dim} for Axial bounding box calculation")
            except Exception as e:
                logger.warning(f"Could not determine axis codes from affine: {e}. Using default dimension 2.")
                axial_dim = 2
            
            # Find slices with non-zero voxels along the axial dimension
            # Sum along other dimensions to get a 1D array indicating which slices have data
            if axial_dim == 2:
                # Sum along x and y dimensions (dim 0 and 1)
                slice_sums = np.sum(data, axis=(0, 1))
            elif axial_dim == 1:
                slice_sums = np.sum(data, axis=(0, 2))
            else:  # axial_dim == 0
                slice_sums = np.sum(data, axis=(1, 2))
            
            # Find first and last slice with non-zero values
            non_zero_slices = np.where(slice_sums > 0)[0]
            
            if len(non_zero_slices) == 0:
                logger.warning(f"No non-zero slices found in mask {mask_path}")
                return None
            
            z_min = int(non_zero_slices.min())
            z_max = int(non_zero_slices.max())
            
            bbox = {'z': (z_min, z_max)}
            logger.info(f"Calculated bounding box from mask: z range {z_min}-{z_max} (total slices with data: {len(non_zero_slices)})")
            return bbox
            
        except Exception as e:
            logger.error(f"Error calculating bounding box from mask {mask_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def crop_mask_to_box(self, mask_path: str, bbox: Dict[str, Tuple[int, int]], padding: int = 5) -> Optional[str]:
        """
        Filter 3D mask by zeroing out slices outside the specified bounding box.
        This preserves the original volume shape and affine, making it easier to merge.
        
        Args:
            mask_path: Path to input mask
            bbox: Dictionary with keys 'x', 'y', 'z' mapping to (min, max) tuples.
                  Example: {'z': (76, 92)} for Axial view filtering only
            padding: Padding to add around the box
            
        Returns:
            Path to filtered mask file (same shape as original, but slices outside range are zeroed)
        """
        if not os.path.exists(mask_path):
            logger.error(f"Mask file not found: {mask_path}")
            return None
            
        try:
            img = nib.load(mask_path)
            data = img.get_fdata()
            affine = img.affine
            
            # Get volume dimensions
            # Note: NIfTI shape from get_fdata() is typically (x, y, z) = (dim0, dim1, dim2)
            # For Axial filtering, we filter along dimension 2 (z-axis)
            dims = data.shape
            logger.info(f"Mask shape: {dims} (expected: x, y, z = dim0, dim1, dim2)")
            
            # Determine which dimension corresponds to Axial (z-axis) using affine
            # In RAS orientation: x=Sagittal (dim 0), y=Coronal (dim 1), z=Axial (dim 2)
            # But we should verify this from the affine matrix to be safe
            try:
                import nibabel.orientations as nibo
                axcodes = nibo.aff2axcodes(affine)
                logger.info(f"Volume axis codes from affine: {axcodes}")
                # axcodes is typically ('R', 'A', 'S') for RAS orientation
                # In RAS: R=Right (x, dim 0), A=Anterior (y, dim 1), S=Superior (z, dim 2)
                # Axial slices are perpendicular to Superior (S) axis, which is dimension 2
                axial_dim = 2  # Default to dimension 2 for RAS orientation
                logger.info(f"Using dimension {axial_dim} for Axial filtering (S axis in RAS)")
            except Exception as e:
                logger.warning(f"Could not determine axis codes from affine: {e}. Using default dimension 2 for Axial.")
                axial_dim = 2
            
            # Create a copy to filter (preserve original shape)
            filtered_data = data.copy()
            
            # Apply filtering for each axis if specified
            # For Axial-only filtering, only 'z' will be in bbox
            # Axial corresponds to dimension 2 (z-axis) in RAS orientation
            
            if 'z' in bbox:
                z_range = bbox['z']
                z_min = max(0, z_range[0] - padding)
                z_max = min(dims[axial_dim], z_range[1] + padding + 1)  # +1 because slicing is exclusive on end
                
                # Log before filtering
                original_nonzero = np.sum(filtered_data > 0)
                logger.info(f"Before filtering: {original_nonzero} non-zero voxels")
                logger.info(f"Filtering Axial slices: keeping slices {z_min} to {z_max-1} (dimension {axial_dim})")
                print(f"[FILTER MASK] Shape: {dims} | Filtering dimension {axial_dim} (Axial) | Keeping slices {z_min} to {z_max-1}")
                print(f"[FILTER MASK] Original slice range: {z_range[0]}-{z_range[1]} | With padding: {z_min}-{z_max-1}")
                
                # Zero out slices outside the range (keep original shape)
                # Use the determined axial dimension
                if axial_dim == 0:
                    filtered_data[:z_min, :, :] = 0
                    filtered_data[z_max:, :, :] = 0
                elif axial_dim == 1:
                    filtered_data[:, :z_min, :] = 0
                    filtered_data[:, z_max:, :] = 0
                else:  # axial_dim == 2 (default)
                    filtered_data[:, :, :z_min] = 0
                    filtered_data[:, :, z_max:] = 0
                
                # Log after filtering
                remaining_nonzero = np.sum(filtered_data > 0)
                logger.info(f"After filtering: {remaining_nonzero} non-zero voxels remaining")
                logger.info(f"Zeroed out slices outside Z range {z_min}-{z_max-1} (original range: {z_range[0]}-{z_range[1]})")
                print(f"[FILTER MASK] Zeroed slices outside Z={z_min} to {z_max-1} | Remaining voxels: {remaining_nonzero}/{original_nonzero}")
            
            if 'y' in bbox:
                y_range = bbox['y']
                y_min = max(0, y_range[0] - padding)
                y_max = min(dims[1], y_range[1] + padding + 1)
                filtered_data[:, :y_min, :] = 0
                filtered_data[:, y_max:, :] = 0
                logger.info(f"Zeroed out slices outside Y range {y_min}-{y_max-1}")
            
            if 'x' in bbox:
                x_range = bbox['x']
                x_min = max(0, x_range[0] - padding)
                x_max = min(dims[0], x_range[1] + padding + 1)
                filtered_data[:x_min, :, :] = 0
                filtered_data[x_max:, :, :] = 0
                logger.info(f"Zeroed out slices outside X range {x_min}-{x_max-1}")
            
            # Check if result is empty (all zeros)
            if np.sum(filtered_data) == 0:
                logger.warning(f"Filtered mask is empty for {mask_path}. All slices were zeroed out.")
            
            # Count remaining voxels for logging
            remaining_voxels = np.sum(filtered_data > 0)
            original_voxels = np.sum(data > 0)
            logger.info(f"Filtered mask: {remaining_voxels} voxels remaining (from {original_voxels} original)")
            
            # Save filtered file (same shape and affine as original)
            base_name = os.path.basename(mask_path).replace(".nii.gz", "")
            output_filename = f"{base_name}_filtered.nii.gz"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Use same dtype and affine as original (no affine update needed since shape is preserved)
            new_img = nib.Nifti1Image(filtered_data.astype(data.dtype), affine)
            nib.save(new_img, output_path)
            
            logger.info(f"Created filtered mask: {output_path} (shape preserved: {dims})")
            return output_path
            
        except Exception as e:
            logger.error(f"Error filtering mask: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def merge_filtered_masks(self, masks_dir: Optional[str] = None, output_filename: str = "merged_segmentation.nii.gz", 
                            reference_volume_path: Optional[str] = None,
                            ordered_mask_files: Optional[List[str]] = None) -> Tuple[Optional[str], Dict[int, str]]:
        """
        Merge all filtered mask files in filtered_masks directory into a single segmentation file.
        Each organ will have a unique sequential label ID (1, 2, 3, ...).
        Saves as NIfTI format (.nii.gz) - 3D Slicer will auto-detect labels from data.
        
        Args:
            masks_dir: Directory containing filtered mask files. If None, uses self.output_dir.
            output_filename: Name of the output merged segmentation file (should be .nrrd for 3D Slicer).
            reference_volume_path: Path to reference volume file (NIfTI/NRRD) to use its affine/orientation.
                                  If provided, merged segmentation will use the same affine as reference volume.
            ordered_mask_files: Optional list of mask file paths in the desired order.
                               If provided, only these files will be merged in this order.
                               This ensures label IDs (1,2,3...) match the order in report.csv.
            
        Returns:
            Tuple of (output_path, label_mapping) where:
            - output_path: Path to the merged segmentation file, or None if failed
            - label_mapping: Dictionary mapping label_id -> class_name
        """
        if masks_dir is None:
            masks_dir = self.output_dir
        
        if not os.path.exists(masks_dir):
            logger.error(f"Masks directory not found: {masks_dir}")
            return None, {}
        
        # Use ordered list if provided, otherwise get all filtered files from directory
        if ordered_mask_files:
            # Filter to only include files that exist and are in the ordered list
            mask_files = [f for f in ordered_mask_files if os.path.exists(f) and "_filtered" in os.path.basename(f)]
            logger.info(f"Using ordered list: {len(mask_files)} filtered mask files to merge in specified order")
        else:
            # Get only filtered .nii.gz files in the directory (must have "_filtered" suffix)
            # This ensures we don't merge unfiltered composite masks
            all_mask_files = glob.glob(os.path.join(masks_dir, "*.nii.gz"))
            mask_files = [f for f in all_mask_files if "_filtered" in os.path.basename(f)]
            
            if not mask_files:
                logger.warning(f"No filtered mask files found in {masks_dir}")
                # Log which files were skipped
                skipped_files = [os.path.basename(f) for f in all_mask_files if f not in mask_files]
                if skipped_files:
                    logger.info(f"Skipped unfiltered files: {skipped_files}")
                return None, {}
            
            logger.info(f"Found {len(mask_files)} filtered mask files to merge (skipped {len(all_mask_files) - len(mask_files)} unfiltered files)")
        
        # Import organ mapping to resolve class names
        from utils.organ_mapping import TOTAL_SEGMENTATOR_LABELS
        
        # Dictionary to store class name for each file
        file_class_mapping = {}
        
        # Process each file to determine its class name
        for mask_file in mask_files:
            filename = os.path.basename(mask_file).replace(".nii.gz", "")
            
            # Remove "_filtered" suffix if present
            base_name = filename.replace("_filtered", "")
            
            # Try to extract class name from filename
            class_name = None
            
            # Case 1: Direct class name match (e.g., "aorta_filtered" -> "aorta")
            if base_name in TOTAL_SEGMENTATOR_LABELS:
                class_name = base_name
            # Case 2: Check if filename contains a known class name
            else:
                for ts_class in TOTAL_SEGMENTATOR_LABELS.keys():
                    if ts_class in base_name:
                        class_name = ts_class
                        break
            
            # Case 3: Check for label pattern (e.g., "segmentation_label15_filtered" -> label 15)
            if class_name is None:
                label_match = None
                if "label" in base_name:
                    # Extract label number
                    import re
                    match = re.search(r'label(\d+)', base_name)
                    if match:
                        label_id = int(match.group(1))
                        # Find class name for this label ID
                        for ts_class, ts_label_id in TOTAL_SEGMENTATOR_LABELS.items():
                            if ts_label_id == label_id:
                                class_name = ts_class
                                break
            
            # Case 4: Use filename as class name if no match found
            if class_name is None:
                class_name = base_name
                logger.warning(f"Could not determine class name for {filename}, using '{class_name}'")
            
            file_class_mapping[mask_file] = class_name
            logger.debug(f"File {filename} -> class: {class_name}")
        
        # Load reference volume to get affine/orientation if provided
        reference_affine = None
        reference_shape = None
        reference_img = None
        reference_nrrd_header = None  # Store NRRD header if reference is NRRD
        if reference_volume_path and os.path.exists(reference_volume_path):
            try:
                logger.info(f"Loading reference volume for affine: {reference_volume_path}")
                
                # Handle NRRD files
                if reference_volume_path.lower().endswith('.nrrd'):
                    try:
                        import nrrd
                        data, header = nrrd.read(reference_volume_path)
                        reference_shape = data.shape
                        reference_nrrd_header = header.copy()  # Store header for later use
                        
                        # Construct affine from NRRD header (same logic as convert_nrrd_to_nifti)
                        affine = np.eye(4)
                        if 'space directions' in header:
                            directions = header['space directions']
                            # Fix: Check if directions is string 'none', not compare array with string
                            if not (isinstance(directions, str) and directions == 'none') and directions is not None:
                                if isinstance(directions, (list, tuple)):
                                    valid_dirs = []
                                    for d in directions:
                                        # Fix: Check if d is string 'none'
                                        if not (isinstance(d, str) and d == 'none') and d is not None:
                                            if isinstance(d, (list, tuple, np.ndarray)):
                                                valid_dirs.append(np.array(d))
                                            else:
                                                valid_dirs.append(np.array([d, 0, 0]) if len(valid_dirs) == 0 else 
                                                                 np.array([0, d, 0]) if len(valid_dirs) == 1 else 
                                                                 np.array([0, 0, d]))
                                    if len(valid_dirs) == 3:
                                        affine[:3, :3] = np.column_stack(valid_dirs)
                                elif isinstance(directions, np.ndarray):
                                    if directions.shape == (3, 3):
                                        affine[:3, :3] = directions
                                    elif directions.shape == (3,):
                                        affine[:3, :3] = np.diag(directions)
                        
                        if 'space origin' in header:
                            origin = header['space origin']
                            # Fix: Check if origin is string 'none', not compare array with string
                            if not (isinstance(origin, str) and origin == 'none') and origin is not None:
                                if isinstance(origin, (list, tuple, np.ndarray)):
                                    affine[:3, 3] = np.array(origin)
                        
                        reference_affine = affine
                        logger.info(f"Loaded NRRD reference volume shape: {reference_shape}")
                        logger.info(f"NRRD reference affine:\n{reference_affine}")
                        logger.info(f"Stored NRRD header for later use")
                    except ImportError:
                        logger.warning("pynrrd not installed. Cannot load NRRD reference. Will use affine from first mask.")
                        reference_nrrd_header = None  # Reset to None on error
                    except Exception as e:
                        logger.warning(f"Could not load NRRD reference volume: {e}. Will use affine from first mask.")
                        reference_nrrd_header = None  # Reset to None on error
                else:
                    # Handle NIfTI files
                    reference_img = nib.load(reference_volume_path)
                    reference_affine = reference_img.affine
                    reference_shape = reference_img.shape
                
                if reference_affine is not None:
                    # Get orientation codes for reference
                    ref_ornt = nib.orientations.io_orientation(reference_affine)
                    ref_axcodes = nib.orientations.aff2axcodes(reference_affine)
                    logger.info(f"Reference volume shape: {reference_shape}")
                    logger.info(f"Reference orientation codes: {ref_ornt}")
                    logger.info(f"Reference axis codes: {ref_axcodes}")
                    logger.info(f"Reference affine:\n{reference_affine}")
            except Exception as e:
                logger.warning(f"Could not load reference volume: {e}. Will use affine from first mask.")
                reference_nrrd_header = None  # Reset to None on error
                reference_affine = None  # Reset to None on error
        
        # Load all masks and merge them
        merged_data = None
        merged_affine = None
        label_mapping = {}  # Maps new label_id -> class_name
        new_label_id = 1
        
        for mask_file in mask_files:
            try:
                img = nib.load(mask_file)
                data = img.get_fdata()
                affine = img.affine
                
                # Initialize merged_data with first mask
                if merged_data is None:
                    # Use reference affine if available, otherwise use first mask's affine
                    if reference_affine is not None:
                        merged_affine = reference_affine.copy()
                        # Use reference shape if available, otherwise use first mask's shape
                        if reference_shape is not None:
                            merged_data = np.zeros(reference_shape, dtype=np.uint16)
                            logger.info(f"Initialized merged segmentation with reference shape {reference_shape} and affine")
                        else:
                            merged_data = np.zeros_like(data, dtype=np.uint16)
                            logger.info(f"Initialized merged segmentation with mask shape {merged_data.shape} and reference affine")
                    else:
                        merged_data = np.zeros_like(data, dtype=np.uint16)
                        merged_affine = affine.copy()
                        logger.info(f"Initialized merged segmentation with shape {merged_data.shape} and mask affine")
                
                # Check if shapes match
                if data.shape != merged_data.shape:
                    logger.warning(f"Shape mismatch for {mask_file}: {data.shape} vs {merged_data.shape}.")
                    # If using reference shape, we might need to resample or skip
                    # For now, skip mismatched shapes
                    logger.warning(f"Skipping {mask_file} due to shape mismatch.")
                    continue
                
                # Check if affine matches (only warn, we'll use reference affine for output)
                if reference_affine is not None:
                    if not np.allclose(affine, reference_affine, atol=1e-3):
                        logger.debug(f"Affine mismatch for {mask_file}. Will use reference affine for output.")
                elif not np.allclose(affine, merged_affine, atol=1e-3):
                    logger.debug(f"Affine mismatch for {mask_file}. Will use first mask affine for output.")
                
                # Get non-zero voxels (mask) - convert to binary mask
                # Filtered masks should be binary (0 or 1), but we ensure binary here
                mask = (data > 0).astype(bool)
                
                # Get class name for this mask
                class_name = file_class_mapping[mask_file]
                
                # Always assign label ID to maintain order (matching report.csv)
                # Assign label to ALL voxels in the mask (not just background)
                # This ensures each mask gets its own label, even if there's overlap
                if np.any(mask):
                    total_mask_voxels = np.sum(mask)
                    
                    # Check for overlap with existing labels (for logging)
                    overlap_mask = mask & (merged_data > 0)
                    overlap_count = np.sum(overlap_mask)
                    background_mask = mask & (merged_data == 0)
                    background_count = np.sum(background_mask)
                    
                    # IMPORTANT: Assign label to ALL voxels in mask, not just background
                    # This ensures each organ gets its own label ID
                    # If there's overlap, the later mask will overwrite the earlier one at overlap locations
                    merged_data[mask] = new_label_id
                    label_mapping[new_label_id] = class_name
                    
                    if overlap_count > 0:
                        logger.warning(f"Overlap detected for {class_name} (label {new_label_id}): "
                                     f"{overlap_count} voxels overlap with existing labels. "
                                     f"Overlapping voxels will use label {new_label_id} (later mask takes precedence).")
                    else:
                        logger.info(f"Added {class_name} (label {new_label_id}) with {total_mask_voxels} voxels")
                    
                    logger.debug(f"Processing {class_name}: total mask voxels={total_mask_voxels}, "
                               f"background voxels={background_count}, overlap voxels={overlap_count}")
                else:
                    # Empty mask: still add to label_mapping to maintain order with report.csv
                    # But don't add to merged_data (no voxels to assign)
                    label_mapping[new_label_id] = class_name
                    logger.warning(f"Empty mask for {class_name} ({os.path.basename(mask_file)}), "
                                 f"assigned label {new_label_id} but no voxels in merged segmentation")
                
                new_label_id += 1
                    
            except Exception as e:
                logger.error(f"Error processing {mask_file}: {e}")
                continue
        
        if merged_data is None:
            logger.error("No valid masks to merge")
            return None, {}
        
        # Save merged segmentation as NIfTI format (.nii.gz)
        # Use self.output_dir (class output directory) to save the merged segmentation
        save_dir = self.output_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, output_filename)
        
        # Ensure output path has .nii.gz extension
        if not output_path.lower().endswith(('.nii.gz', '.nii')):
            output_path = output_path + '.nii.gz'
        
        # Convert affine from LPS (NRRD) to RAS+ (NIfTI) if reference was NRRD
        # NRRD uses LPS convention, NIfTI uses RAS+ convention
        # Need to flip X and Y axes: LPS -> RAS+ conversion
        nifti_affine = merged_affine.copy() if merged_affine is not None else np.eye(4)
        
        # If reference was NRRD, convert from LPS to RAS+
        if reference_volume_path and reference_volume_path.lower().endswith('.nrrd'):
            logger.info("Reference volume is NRRD (LPS), converting affine to RAS+ for NIfTI")
            # LPS to RAS+ conversion: flip X and Y axes
            # Create flip matrix: flip X (L->R) and Y (P->A)
            flip_matrix = np.array([
                [-1,  0,  0,  0],  # Flip X: L->R
                [ 0, -1,  0,  0],  # Flip Y: P->A
                [ 0,  0,  1,  0],  # Keep Z: S->S
                [ 0,  0,  0,  1]
            ])
            # Apply flip: new_affine = flip_matrix @ old_affine
            nifti_affine = flip_matrix @ nifti_affine
            logger.info(f"Converted affine from LPS to RAS+:\n{nifti_affine}")
        
        # Log affine information before saving
        if nifti_affine is not None:
            try:
                import nibabel.orientations as nibo
                ornt = nibo.io_orientation(nifti_affine)
                axcodes = nibo.aff2axcodes(nifti_affine)
                logger.info(f"Saving NIfTI with affine (RAS+):\n{nifti_affine}")
                logger.info(f"Orientation codes: {ornt}")
                logger.info(f"Axis codes: {axcodes}")
            except Exception as e:
                logger.warning(f"Could not determine orientation: {e}")
        
        # Save as NIfTI format (3D Slicer will auto-detect labels from data)
        # Use RAS+ affine for NIfTI format
        merged_img = nib.Nifti1Image(merged_data.astype(np.uint16), nifti_affine)
        nib.save(merged_img, output_path)
        
        # Verify saved file has correct affine
        try:
            verify_img = nib.load(output_path)
            verify_affine = verify_img.affine
            if not np.allclose(nifti_affine, verify_affine, atol=1e-6):
                logger.warning(f"Affine mismatch! Saved affine differs from intended affine")
                logger.warning(f"Intended (RAS+):\n{nifti_affine}")
                logger.warning(f"Saved:\n{verify_affine}")
            else:
                logger.info(f"Verified: Saved NIfTI affine matches intended RAS+ affine")
                try:
                    import nibabel.orientations as nibo
                    verify_axcodes = nibo.aff2axcodes(verify_affine)
                    logger.info(f"Saved NIfTI axis codes (should be RAS+): {verify_axcodes}")
                except:
                    pass
        except Exception as e:
            logger.warning(f"Could not verify saved file: {e}")
        
        logger.info(f"Saved merged segmentation to {output_path} (NIfTI format - 3D Slicer will auto-detect {len(label_mapping)} labels)")
        logger.info(f"Total labels: {len(label_mapping)}")
        
        # Note: Label mapping is now included in CSV report instead of separate JSON file
        # But we can still save JSON for reference if needed
        mapping_file = output_path.replace(".nii.gz", "_labels.json").replace(".nii", "_labels.json")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved label mapping to {mapping_file} (for reference)")
        
        # Count voxels for each label in merged segmentation
        logger.info("=" * 60)
        logger.info("Voxel count for each label in merged segmentation:")
        total_voxels = 0
        labels_with_voxels = []
        for label_id in sorted(label_mapping.keys()):
            class_name = label_mapping[label_id]
            voxel_count = np.sum(merged_data == label_id)
            total_voxels += voxel_count
            if voxel_count > 0:
                labels_with_voxels.append(label_id)
            logger.info(f"  Label {label_id} ({class_name}): {voxel_count:,} voxels")
        logger.info(f"Total voxels (all labels): {total_voxels:,}")
        logger.info(f"Background voxels (label 0): {np.sum(merged_data == 0):,}")
        
        # Check unique labels in merged_data
        unique_labels = np.unique(merged_data)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        logger.info(f"Unique labels in merged_data (excluding background): {sorted(unique_labels.tolist())}")
        logger.info(f"Labels with voxels: {labels_with_voxels}")
        logger.info(f"Total unique labels: {len(unique_labels)}")
        
        if len(unique_labels) != len(labels_with_voxels):
            logger.warning(f"Mismatch! Expected {len(labels_with_voxels)} labels with voxels, but found {len(unique_labels)} unique labels in data")
        
        logger.info("=" * 60)
        
        # Verify by reloading the file to ensure it was saved correctly
        try:
            if output_path.endswith('.nrrd'):
                import nrrd
                verify_data, _ = nrrd.read(output_path)
            else:
                verify_img = nib.load(output_path)
                verify_data = verify_img.get_fdata()
            
            verify_unique_labels = np.unique(verify_data)
            verify_unique_labels = verify_unique_labels[verify_unique_labels > 0]
            logger.info(f"Verification: Unique labels in saved file: {sorted(verify_unique_labels.tolist())}")
            logger.info(f"Verification: Total unique labels in saved file: {len(verify_unique_labels)}")
            
            if len(verify_unique_labels) != len(unique_labels):
                logger.error(f"ERROR: Mismatch between merged_data and saved file! "
                           f"Merged data has {len(unique_labels)} labels, saved file has {len(verify_unique_labels)} labels")
        except Exception as e:
            logger.warning(f"Could not verify saved file: {e}")
        
        return output_path, label_mapping

    def merge_full_masks(self, masks_dir: str, class_names: List[str], 
                         output_filename: str = "full_segmentation.nii.gz",
                         reference_volume_path: Optional[str] = None) -> Tuple[Optional[str], Dict[int, str]]:
        """
        Merge individual full organ masks (NOT filtered) into a single segmentation file.
        This is used to create the "full" version before slice filtering.
        Saves as NIfTI format (.nii.gz) - 3D Slicer will auto-detect labels from data.
        
        Args:
            masks_dir: Directory containing full organ mask files from TotalSegmentator
            class_names: List of class names to include (e.g., ['aorta', 'liver', 'lung_lower_lobe_right'])
            output_filename: Name of the output merged segmentation file (should be .nrrd)
            reference_volume_path: Path to reference volume file to use its affine/orientation
            
        Returns:
            Tuple of (output_path, label_mapping) where:
            - output_path: Path to the merged segmentation file, or None if failed
            - label_mapping: Dictionary mapping label_id -> class_name
        """
        if not os.path.exists(masks_dir):
            logger.error(f"Masks directory not found: {masks_dir}")
            return None, {}
        
        # Collect mask files for specified classes
        mask_files = []
        for class_name in class_names:
            mask_file = os.path.join(masks_dir, f"{class_name}.nii.gz")
            if os.path.exists(mask_file):
                mask_files.append(mask_file)
            else:
                logger.warning(f"Full mask file not found for class '{class_name}': {mask_file}")
        
        if not mask_files:
            logger.warning(f"No valid mask files found for specified classes")
            return None, {}
        
        logger.info(f"Found {len(mask_files)} full mask files to merge")
        
        # Load reference volume to get affine/orientation if provided
        reference_affine = None
        reference_shape = None
        reference_nrrd_header = None
        if reference_volume_path and os.path.exists(reference_volume_path):
            try:
                logger.info(f"Loading reference volume for affine: {reference_volume_path}")
                
                # Handle NRRD files
                if reference_volume_path.lower().endswith('.nrrd'):
                    try:
                        import nrrd
                        data, header = nrrd.read(reference_volume_path)
                        reference_shape = data.shape
                        reference_nrrd_header = header.copy()
                        
                        # Construct affine from NRRD header
                        affine = np.eye(4)
                        if 'space directions' in header:
                            directions = header['space directions']
                            if not (isinstance(directions, str) and directions == 'none') and directions is not None:
                                if isinstance(directions, (list, tuple)):
                                    valid_dirs = []
                                    for d in directions:
                                        if not (isinstance(d, str) and d == 'none') and d is not None:
                                            if isinstance(d, (list, tuple, np.ndarray)):
                                                valid_dirs.append(np.array(d))
                                            else:
                                                valid_dirs.append(np.array([d, 0, 0]) if len(valid_dirs) == 0 else 
                                                                 np.array([0, d, 0]) if len(valid_dirs) == 1 else 
                                                                 np.array([0, 0, d]))
                                    if len(valid_dirs) == 3:
                                        affine[:3, :3] = np.column_stack(valid_dirs)
                                elif isinstance(directions, np.ndarray):
                                    if directions.shape == (3, 3):
                                        affine[:3, :3] = directions
                                    elif directions.shape == (3,):
                                        affine[:3, :3] = np.diag(directions)
                        
                        if 'space origin' in header:
                            origin = header['space origin']
                            if not (isinstance(origin, str) and origin == 'none') and origin is not None:
                                if isinstance(origin, (list, tuple, np.ndarray)):
                                    affine[:3, 3] = np.array(origin)
                        
                        reference_affine = affine
                        logger.info(f"Loaded NRRD reference volume shape: {reference_shape}")
                    except ImportError:
                        logger.warning("pynrrd not installed. Cannot load NRRD reference.")
                    except Exception as e:
                        logger.warning(f"Could not load NRRD reference volume: {e}")
                else:
                    # Handle NIfTI files
                    reference_img = nib.load(reference_volume_path)
                    reference_affine = reference_img.affine
                    reference_shape = reference_img.shape
            except Exception as e:
                logger.warning(f"Could not load reference volume: {e}")
        
        # Load all masks and merge them
        merged_data = None
        merged_affine = None
        label_mapping = {}
        new_label_id = 1
        
        for mask_file in mask_files:
            try:
                img = nib.load(mask_file)
                data = img.get_fdata()
                affine = img.affine
                
                # Extract class name from filename
                filename = os.path.basename(mask_file).replace(".nii.gz", "")
                class_name = filename
                
                # Initialize merged_data with first mask
                if merged_data is None:
                    if reference_affine is not None:
                        merged_affine = reference_affine.copy()
                        if reference_shape is not None:
                            merged_data = np.zeros(reference_shape, dtype=np.uint16)
                            logger.info(f"Initialized full merged segmentation with reference shape {reference_shape}")
                        else:
                            merged_data = np.zeros_like(data, dtype=np.uint16)
                            logger.info(f"Initialized full merged segmentation with mask shape {merged_data.shape}")
                    else:
                        merged_data = np.zeros_like(data, dtype=np.uint16)
                        merged_affine = affine.copy()
                        logger.info(f"Initialized full merged segmentation with shape {merged_data.shape}")
                
                # Check if shapes match
                if data.shape != merged_data.shape:
                    logger.warning(f"Shape mismatch for {mask_file}: {data.shape} vs {merged_data.shape}. Skipping.")
                    continue
                
                # Get non-zero voxels (mask) - convert to binary mask
                mask = (data > 0).astype(bool)
                
                if np.any(mask):
                    total_mask_voxels = np.sum(mask)
                    
                    # Check for overlap with existing labels (for logging)
                    overlap_mask = mask & (merged_data > 0)
                    overlap_count = np.sum(overlap_mask)
                    
                    # Assign label to ALL voxels in mask
                    # If there's overlap, the later mask will overwrite the earlier one
                    merged_data[mask] = new_label_id
                    label_mapping[new_label_id] = class_name
                    
                    if overlap_count > 0:
                        logger.warning(f"Overlap detected for {class_name} (label {new_label_id}): "
                                     f"{overlap_count} voxels overlap with existing labels. "
                                     f"Overlapping voxels will use label {new_label_id} (later mask takes precedence).")
                    
                    logger.info(f"Added full {class_name} (label {new_label_id}) with {total_mask_voxels} voxels")
                    new_label_id += 1
                else:
                    logger.warning(f"Empty mask for {mask_file}, skipping")
                    
            except Exception as e:
                logger.error(f"Error processing {mask_file}: {e}")
                continue
        
        if merged_data is None:
            logger.error("No valid masks to merge")
            return None, {}
        
        # Save as NIfTI format (.nii.gz)
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Ensure output path has .nii.gz extension
        if not output_path.lower().endswith(('.nii.gz', '.nii')):
            output_path = output_path + '.nii.gz'
        
        # Convert affine from LPS (NRRD) to RAS+ (NIfTI) if reference was NRRD
        # NRRD uses LPS convention, NIfTI uses RAS+ convention
        # Need to flip X and Y axes: LPS -> RAS+ conversion
        nifti_affine = merged_affine.copy() if merged_affine is not None else np.eye(4)
        
        # If reference was NRRD, convert from LPS to RAS+
        if reference_volume_path and reference_volume_path.lower().endswith('.nrrd'):
            logger.info("Reference volume is NRRD (LPS), converting affine to RAS+ for NIfTI")
            # LPS to RAS+ conversion: flip X and Y axes
            # Create flip matrix: flip X (L->R) and Y (P->A)
            flip_matrix = np.array([
                [-1,  0,  0,  0],  # Flip X: L->R
                [ 0, -1,  0,  0],  # Flip Y: P->A
                [ 0,  0,  1,  0],  # Keep Z: S->S
                [ 0,  0,  0,  1]
            ])
            # Apply flip: new_affine = flip_matrix @ old_affine
            nifti_affine = flip_matrix @ nifti_affine
            logger.info(f"Converted affine from LPS to RAS+:\n{nifti_affine}")
        
        # Log affine information before saving
        if nifti_affine is not None:
            try:
                import nibabel.orientations as nibo
                ornt = nibo.io_orientation(nifti_affine)
                axcodes = nibo.aff2axcodes(nifti_affine)
                logger.info(f"Saving full NIfTI with affine (RAS+):\n{nifti_affine}")
                logger.info(f"Orientation codes: {ornt}")
                logger.info(f"Axis codes: {axcodes}")
            except Exception as e:
                logger.warning(f"Could not determine orientation: {e}")
        
        # Save as NIfTI format (3D Slicer will auto-detect labels from data)
        # Use RAS+ affine for NIfTI format
        merged_img = nib.Nifti1Image(merged_data.astype(np.uint16), nifti_affine)
        nib.save(merged_img, output_path)
        
        # Verify saved file has correct affine
        try:
            verify_img = nib.load(output_path)
            verify_affine = verify_img.affine
            if not np.allclose(nifti_affine, verify_affine, atol=1e-6):
                logger.warning(f"Affine mismatch! Saved affine differs from intended affine")
                logger.warning(f"Intended (RAS+):\n{nifti_affine}")
                logger.warning(f"Saved:\n{verify_affine}")
            else:
                logger.info(f"Verified: Saved full NIfTI affine matches intended RAS+ affine")
                try:
                    import nibabel.orientations as nibo
                    verify_axcodes = nibo.aff2axcodes(verify_affine)
                    logger.info(f"Saved full NIfTI axis codes (should be RAS+): {verify_axcodes}")
                except:
                    pass
        except Exception as e:
            logger.warning(f"Could not verify saved file: {e}")
        
        logger.info(f"Saved full merged segmentation to {output_path} (NIfTI format - 3D Slicer will auto-detect {len(label_mapping)} labels)")
        logger.info(f"Total labels in full segmentation: {len(label_mapping)}")
        
        # Count voxels for each label in full segmentation
        logger.info("=" * 60)
        logger.info("Voxel count for each label in FULL segmentation:")
        total_voxels = 0
        for label_id in sorted(label_mapping.keys()):
            class_name = label_mapping[label_id]
            voxel_count = np.sum(merged_data == label_id)
            total_voxels += voxel_count
            logger.info(f"  Label {label_id} ({class_name}): {voxel_count:,} voxels")
        logger.info(f"Total voxels (all labels): {total_voxels:,}")
        logger.info(f"Background voxels (label 0): {np.sum(merged_data == 0):,}")
        logger.info("=" * 60)
        
        return output_path, label_mapping


