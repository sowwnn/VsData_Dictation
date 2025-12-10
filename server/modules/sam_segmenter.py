import os
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
import nrrd # For 3D annotation creation

class SAMSegmenter:
    """Segment anatomical organs in slice images using SAM Medical."""
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "medsam"):
        """Initialize SAM Medical segmenter.
        
        Args:
            model_path: Path to SAM model checkpoint (optional, will download if not provided)
            model_type: "medsam" or "sam2" (default: "medsam")
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.model = None
        self.predictor = None
        self.device = "cuda" if self._check_cuda() else "cpu"
        self.florence2_detector = None  # Will be initialized if needed
        
        self._initialize_model()
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _initialize_model(self):
        """Initialize SAM Medical model."""
        try:
            if self.model_type == "medsam":
                self._initialize_medsam()
            elif self.model_type == "sam2":
                self._initialize_sam2()
            else:
                logging.error(f"Unsupported model type: {self.model_type}")
                self.model = None
        except Exception as e:
            logging.error(f"Error initializing SAM model: {e}")
            self.model = None
    
    def _initialize_medsam(self):
        """Initialize MedSAM model."""
        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor
            
            # Default MedSAM checkpoint path
            if not self.model_path:
                # Try to find default path or download
                default_path = "server/model/medsam_vit_b.pth"
                if os.path.exists(default_path):
                    self.model_path = default_path
                else:
                    logging.warning(f"MedSAM model not found at {default_path}. Please download it.")
                    self.model = None
                    return
            
            if not os.path.exists(self.model_path):
                logging.error(f"MedSAM model file not found: {self.model_path}")
                self.model = None
                return
            
            # Load model state_dict first with weights_only=False
            model_state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # Initialize model and load state_dict
            sam = sam_model_registry["vit_b"]()
            sam.load_state_dict(model_state_dict)

            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            self.model = sam
            
            logging.info(f"MedSAM model loaded from {self.model_path}")
            
        except ImportError:
            logging.error("segment_anything package not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
            self.model = None
        except Exception as e:
            logging.error(f"Error loading MedSAM: {e}")
            self.model = None
    
    def _initialize_sam2(self):
        """Initialize SAM 2 model."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import torch
            
            # SAM2 requires config file
            if not self.model_path:
                logging.error("SAM2 requires model_path to be specified")
                self.model = None
                return
            
            # Load SAM2
            sam2_cfg = self.model_path  # This should be config file path
            sam2 = build_sam2(sam2_cfg, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2)
            self.model = sam2
            
            logging.info(f"SAM2 model loaded")
            
        except ImportError:
            logging.error("sam2 package not installed. Please install SAM2.")
            self.model = None
        except Exception as e:
            logging.error(f"Error loading SAM2: {e}")
            self.model = None
    
    def get_slice_image(self, volume_node, slice_number: int, view: str = "Axial") -> Optional[np.ndarray]:
        """NOT USED ON SERVER - Slicer-specific method."""
        # This method is for client-side use with Slicer
        # On server, use segment_organ_in_slices_from_dir instead
        return None
        """Extract slice image from Slicer volume.
        
        Args:
            volume_node: Slicer volume node
            slice_number: Slice index
            view: View name ("Axial", "Sagittal", "Coronal")
            
        Returns:
            numpy array: Slice image as numpy array
        """
        try:
            import vtk
            
            # Get image data
            image_data = volume_node.GetImageData()
            if not image_data:
                logging.error("No image data in volume node")
                return None
            
            # Get dimensions
            dims = image_data.GetDimensions()
            
            # Import vtk.util.numpy_support
            try:
                import vtk.util.numpy_support as vtk_numpy
            except ImportError:
                logging.error("vtk.util.numpy_support not available")
                return None
            
            # Extract slice based on view
            if view == "Axial":
                if slice_number < 0 or slice_number >= dims[2]:
                    logging.warning(f"Slice number {slice_number} out of range [0, {dims[2]-1}]")
                    return None
                # Extract axial slice (z = slice_number)
                slice_array = vtk_numpy.vtk_to_numpy(
                    image_data.GetPointData().GetScalars()
                ).reshape(dims[2], dims[1], dims[0])[slice_number, :, :]
                
            elif view == "Sagittal":
                if slice_number < 0 or slice_number >= dims[0]:
                    logging.warning(f"Slice number {slice_number} out of range [0, {dims[0]-1}]")
                    return None
                # Extract sagittal slice (x = slice_number)
                slice_array = vtk_numpy.vtk_to_numpy(
                    image_data.GetPointData().GetScalars()
                ).reshape(dims[2], dims[1], dims[0])[:, :, slice_number]
                
            elif view == "Coronal":
                if slice_number < 0 or slice_number >= dims[1]:
                    logging.warning(f"Slice number {slice_number} out of range [0, {dims[1]-1}]")
                    return None
                # Extract coronal slice (y = slice_number)
                slice_array = vtk_numpy.vtk_to_numpy(
                    image_data.GetPointData().GetScalars()
                ).reshape(dims[2], dims[1], dims[0])[:, slice_number, :]
            else:
                logging.error(f"Unsupported view: {view}")
                return None
            
            # Normalize to 0-255 if needed
            if slice_array.max() > 255:
                slice_array = ((slice_array - slice_array.min()) / (slice_array.max() - slice_array.min()) * 255).astype(np.uint8)
            
            # Convert to RGB if needed (SAM expects RGB)
            if len(slice_array.shape) == 2:
                slice_array = np.stack([slice_array] * 3, axis=-1)
            
            return slice_array
            
        except Exception as e:
            logging.error(f"Error extracting slice image: {e}")
            return None
    
    def segment_with_text_prompt(self, image: np.ndarray, organ_name: str, 
                                 use_detector: bool = True) -> Optional[Dict]:
        """Segment organ using text prompt.
        
        Uses a detector (Florence-2) to convert text to bounding box, then SAM for segmentation.
        
        Args:
            image: Image array (H, W, 3) or PIL Image
            organ_name: Name of organ to segment (e.g., "liver", "heart")
            use_detector: Whether to use the detector for text-to-bbox conversion
            
        Returns:
            dict: Segmentation result with mask and bbox
        """
        # Convert numpy array to PIL Image if needed, as Florence-2 expects PIL Image
        from PIL import Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = image # Assume it's already a PIL Image

        if use_detector:
            # Initialize Florence-2 if not already done
            if self.florence2_detector is None:
                try:
                    from modules.florence2_detector import Florence2Detector
                    self.florence2_detector = Florence2Detector()
                except Exception as e:
                    logging.warning(f"Could not initialize Florence2Detector: {e}. Falling back to center point.")
                    use_detector = False
            
            # Use Florence-2 to get bounding box
            if self.florence2_detector and self.florence2_detector.model:
                bbox = self.florence2_detector.get_best_bbox(pil_image, organ_name)
                if bbox is not None:
                    logging.info(f"Found bounding box for '{organ_name}': {bbox}")
                    # SAM segmenter works with numpy array
                    return self.segment_with_bbox(np.array(pil_image), np.array(bbox))
                else:
                    logging.warning(f"No bounding box found for '{organ_name}'. Falling back to center point.")
        
        # Fallback: Use center point as prompt
        logging.warning(f"Text prompt segmentation using center point as fallback for '{organ_name}'.")
        np_image = np.array(pil_image)
        h, w = np_image.shape[:2]
        center_point = np.array([[w // 2, h // 2]])
        
        return self.segment_with_points(np_image, center_point)
    
    def segment_with_bbox(self, image: np.ndarray, bbox: np.ndarray) -> Optional[Dict]:
        """Segment using bounding box prompt.
        
        Args:
            image: Image array (H, W, 3)
            bbox: Bounding box [x1, y1, x2, y2] in image coordinates
            
        Returns:
            dict: Segmentation result with 'mask' and 'bbox'
        """
        if not self.predictor:
            logging.error("SAM predictor not initialized")
            return None
        
        try:
            # Set image
            self.predictor.set_image(image)
            
            # Convert bbox to SAM format (x1, y1, x2, y2)
            bbox_sam = np.array([bbox])
            
            # Predict mask using bounding box
            masks, scores, logits = self.predictor.predict(
                box=bbox_sam,
                multimask_output=True
            )
            
            # Select best mask (highest score)
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]
            
            # Calculate bounding box from mask
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                logging.warning("Empty mask")
                return None
            
            bbox_result = [
                int(x_indices.min()),
                int(y_indices.min()),
                int(x_indices.max()),
                int(y_indices.max())
            ]
            
            return {
                'mask': mask,
                'bbox': bbox_result,
                'score': float(score),
                'organ_name': None  # Will be set by caller
            }
            
        except Exception as e:
            logging.error(f"Error in SAM segmentation with bbox: {e}")
            return None
    
    def segment_with_points(self, image: np.ndarray, points: np.ndarray, labels: Optional[np.ndarray] = None) -> Optional[Dict]:
        """Segment using point prompts.
        
        Args:
            image: Image array (H, W, 3)
            points: Array of points [[x, y], ...] in image coordinates
            labels: Optional array of labels (1 = foreground, 0 = background)
            
        Returns:
            dict: Segmentation result with 'mask' and 'bbox'
        """
        if not self.predictor:
            logging.error("SAM predictor not initialized")
            return None
        
        try:
            # Set image
            self.predictor.set_image(image)
            
            # Default labels to foreground if not provided
            if labels is None:
                labels = np.ones(len(points), dtype=int)
            
            # Predict mask
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
            
            # Select best mask (highest score)
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]
            
            # Calculate bounding box
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                logging.warning("Empty mask")
                return None
            
            bbox = [
                int(x_indices.min()),
                int(y_indices.min()),
                int(x_indices.max()),
                int(y_indices.max())
            ]
            
            return {
                'mask': mask,
                'bbox': bbox,
                'score': float(score),
                'organ_name': None  # Will be set by caller
            }
            
        except Exception as e:
            logging.error(f"Error in SAM segmentation: {e}")
            return None
    
    def segment_organ_in_slices(self, organ_name: str, slice_range: Dict, volume_node, view: str = "Axial") -> List[Dict]:
        """NOT USED ON SERVER - Slicer-specific method.
        
        Use segment_organ_in_slices_from_dir instead.
        """
        # This method is for client-side use with Slicer
        # On server, use segment_organ_in_slices_from_dir instead
        return []
        """Segment organ across multiple slices.
        
        Args:
            organ_name: Name of organ
            slice_range: Dict with 'start', 'end', 'view'
            volume_node: Slicer volume node
            view: View name
            
        Returns:
            list: List of segmentation results for each slice
        """
        if not self.model:
            logging.error("SAM model not initialized")
            return []
        
        results = []
        start_slice = slice_range.get('start', 0)
        end_slice = slice_range.get('end', 0)
        
        # Get first slice to establish reference
        first_image = self.get_slice_image(volume_node, start_slice, view)
        if first_image is None:
            logging.error(f"Could not extract slice {start_slice}")
            return []
        
        # Use center point as initial prompt (in real implementation, this would come from text prompt or user input)
        h, w = first_image.shape[:2]
        center_point = np.array([[w // 2, h // 2]])
        
        # Segment first slice
        first_result = self.segment_with_points(first_image, center_point)
        if first_result:
            first_result['organ_name'] = organ_name
            first_result['slice_number'] = start_slice
            results.append(first_result)
        
        # For other slices, could use previous mask as prompt or continue with center point
        # This is a simplified implementation
        for slice_num in range(start_slice + 1, end_slice + 1):
            image = self.get_slice_image(volume_node, slice_num, view)
            if image is None:
                continue
            
            # Use center point (in production, would propagate mask from previous slice)
            result = self.segment_with_points(image, center_point)
            if result:
                result['organ_name'] = organ_name
                result['slice_number'] = slice_num
                results.append(result)
        
        logging.info(f"Segmented {organ_name} in {len(results)} slices")
        return results
    
    def segment_organ_in_slices_from_dir(self, organ_name: str, slice_range: Dict, 
                                         slice_images_dir: str) -> List[Dict]:
        """Segment organ across multiple slices from image directory.
        
        Args:
            organ_name: Name of organ
            slice_range: Dict with 'start', 'end', 'view'
            slice_images_dir: Directory containing slice images
            
        Returns:
            list: List of segmentation results for each slice
        """
        if not self.model:
            logging.error("SAM model not initialized")
            return []
        
        import os
        import re
        from PIL import Image
        
        results = []
        start_slice = slice_range.get('start', 0)
        end_slice = slice_range.get('end', 0)
        view = slice_range.get('view', 'Red') # Slicer view names: Red, Yellow, Green
        
        # Map Slicer view names to orientation
        view_map = {'Axial': 'Axial', 'Sagittal': 'Sagittal', 'Coronal': 'Coronal',
                    'Red': 'Axial', 'Yellow': 'Sagittal', 'Green': 'Coronal'}
        view_label = view_map.get(view, 'Axial')
        
        # Find slice images in directory
        if not os.path.exists(slice_images_dir):
            logging.error(f"Slice images directory not found: {slice_images_dir}")
            return []
        
        # Get all image files matching the view and slice range
        image_files = []
        for filename in os.listdir(slice_images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Check if matches view and slice number
                if view_label in filename:
                    # Extract slice number from filename (e.g., "Axial_slice0010.png")
                    match = re.search(r'slice(\d+)', filename)
                    if match:
                        slice_num = int(match.group(1))
                        if start_slice <= slice_num <= end_slice:
                            image_files.append((slice_num, os.path.join(slice_images_dir, filename)))
        
        # Sort by slice number
        image_files.sort(key=lambda x: x[0])
        
        if not image_files:
            logging.warning(f"No slice images found for {view_label} view in range {start_slice}-{end_slice}")
            return []
        
        # Create output directory for masks
        masks_dir = os.path.join(slice_images_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        
        # Process each slice
        for slice_num, image_path in image_files:
            try:
                # Use PIL to open image, Florence-2 works well with it
                pil_image = Image.open(image_path).convert("RGB")
                
                # Use text prompt for more accurate segmentation
                result = self.segment_with_text_prompt(pil_image, organ_name)
                
                if result:
                    # Save mask
                    mask_filename = f"{organ_name}_{view_label}_slice{slice_num:04d}_mask.png"
                    mask_path = os.path.join(masks_dir, mask_filename)
                    self.save_segmentation_mask(result['mask'], mask_path)
                    
                    # Update result with metadata
                    result['organ_name'] = organ_name
                    result['slice_number'] = slice_num
                    result['mask_path'] = mask_path
                    results.append(result)
            except Exception as e:
                logging.error(f"Error processing slice {slice_num}: {e}")
                continue
        
        logging.info(f"Segmented {organ_name} in {len(results)} slices")
        return results

    def create_3d_annotation(self, mask_paths: List[str], slice_numbers: List[int], 
                             output_dir: str, organ_name: str, view: str) -> Optional[str]:
        """Create a 3D NRRD annotation file from a series of 2D masks.
        
        Args:
            mask_paths: List of paths to the 2D mask images.
            slice_numbers: List of corresponding slice numbers.
            output_dir: Directory to save the NRRD file.
            organ_name: Name of the organ for filename.
            view: View name for filename.
        
        Returns:
            Path to the created NRRD file or None if failed.
        """
        if not mask_paths:
            return None
            
        from PIL import Image

        try:
            # Sort masks by slice number to ensure correct order
            sorted_masks = sorted(zip(slice_numbers, mask_paths))
            
            # Load all masks and store them in a list
            mask_arrays = [np.array(Image.open(path)) for _, path in sorted_masks]

            if not mask_arrays:
                return None

            # Stack masks into a 3D volume
            # Stack along the last axis (z-axis) to match typical medical volume format (x, y, z)
            # Assuming mask_arrays are (height, width) or (width, height) - usually row-major
            # If images are loaded as (H, W), and we want (W, H, D) or (H, W, D), we need to check.
            # Typically Slicer expects (x, y, z).
            volume = np.stack(mask_arrays, axis=-1)

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Define output path
            output_filename = f"{organ_name}_{view}_segmentation.nrrd"
            output_path = os.path.join(output_dir, output_filename)
            
            # Add StartSlice to metadata
            start_slice = min(slice_numbers)
            header = {'StartSlice': str(start_slice)}

            # Write to NRRD file
            nrrd.write(output_path, volume.astype(np.uint8), header=header)
            
            logging.info(f"Created 3D annotation at: {output_path}")
            return output_path

        except Exception as e:
            logging.error(f"Failed to create 3D annotation: {e}")
            return None

    def save_segmentation_mask(self, mask: np.ndarray, output_path: str):
        """Save segmentation mask to file.
        
        Args:
            mask: Binary mask array
            output_path: Output file path
        """
        try:
            from PIL import Image
            
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Save as image
            Image.fromarray(mask_uint8).save(output_path)
            logging.info(f"Saved mask to {output_path}")
            
        except ImportError:
            logging.error("PIL (Pillow) not installed. Cannot save mask.")
        except Exception as e:
            logging.error(f"Error saving mask: {e}")

def get_nrrd_header(path: str):
    """Safely get the header of an NRRD file."""
    try:
        header = nrrd.read_header(path)
        return header
    except Exception as e:
        logging.error(f"Could not read NRRD header for {path}: {e}")
        return None

def merge_3d_annotations(segmentation_data: Dict[str, List[Tuple[int, str]]], output_path: str, reference_volume_path: Optional[str] = None):
    """
    Merge multiple 2D mask slices into a single 3D NRRD file with labels.
    
    Args:
        segmentation_data: Dictionary mapping organ names to a list of (slice_number, mask_path) tuples.
        output_path: Path to save the merged NRRD file.
        reference_volume_path: Path to the original volume (to copy geometry).
    """
    if not segmentation_data:
        logging.warning("No segmentation data provided for merging.")
        return

    # Get the header from reference volume
    base_header = None
    if reference_volume_path and os.path.exists(reference_volume_path):
        base_header = get_nrrd_header(reference_volume_path)
    
    if not base_header:
        logging.error("Reference volume not found or invalid. Cannot determine volume geometry.")
        return
        
    # Initialize merged volume with zeros (background)
    # NRRD reads as (x, y, z) typically, so sizes matches that.
    merged_volume = np.zeros(base_header['sizes'], dtype=np.uint8)
    logging.info(f"Initialized merged volume with shape {merged_volume.shape}")

    # --- Create a new header compatible with 3D Slicer ---
    header = base_header.copy()
    
    # Define some colors for different segments
    colors = [
        (0.9, 0.2, 0.2), (0.2, 0.9, 0.2), (0.2, 0.2, 0.9),
        (0.9, 0.9, 0.2), (0.2, 0.9, 0.9), (0.9, 0.2, 0.9)
    ]
    
    # Add Slicer-specific segmentation fields
    header['Segmentation_ContainedRepresentationNames'] = 'LabelMap'
    header['Segmentation_MasterRepresentation'] = 'LabelMap'
    
    # Ensure type is appropriate for labels
    header['type'] = 'unsigned char'
    header['encoding'] = 'gzip' # Ensure compressed output

    from PIL import Image

    label_value = 1
    for organ_name, slices in segmentation_data.items():
        try:
            # Add segment metadata to the header for Slicer
            color = colors[(label_value - 1) % len(colors)]
            header[f'Segment{label_value-1}_ID'] = f'Segment_{label_value}'
            header[f'Segment{label_value-1}_Name'] = organ_name
            header[f'Segment{label_value-1}_Color'] = f'{color[0]} {color[1]} {color[2]}'
            header[f'Segment{label_value-1}_LabelValue'] = str(label_value)
            header[f'Segment{label_value-1}_Layer'] = '0'
            
            # Collect extents for metadata
            min_slice = float('inf')
            max_slice = float('-inf')

            for slice_num, mask_path in slices:
                if not os.path.exists(mask_path):
                    logging.warning(f"Mask file not found: {mask_path}")
                    continue

                # Check bounds
                if slice_num < 0 or slice_num >= merged_volume.shape[2]:
                    logging.warning(f"Slice number {slice_num} out of bounds [0, {merged_volume.shape[2]-1}]")
                    continue
                
                # Load mask
                # Mask is 2D (H, W) or (W, H) depending on PIL/Numpy
                # We assume it matches the first two dimensions of merged_volume
                mask_img = Image.open(mask_path)
                mask = np.array(mask_img) > 0 # Convert to binary boolean
                
                # Check mask dimensions
                if mask.shape != merged_volume.shape[:2]:
                     # Try transposing if dimensions are swapped
                    if mask.T.shape == merged_volume.shape[:2]:
                        mask = mask.T
                    else:
                        logging.warning(f"Mask shape {mask.shape} does not match volume slice shape {merged_volume.shape[:2]}")
                        continue

                # Assign label to volume at this slice
                # We use bitwise OR logic or just overwrite? Overwrite is fine for now, or check for overlap.
                # merged_volume[:, :, slice_num][mask] = label_value
                
                # Safe assignment
                current_slice = merged_volume[:, :, slice_num]
                current_slice[mask] = label_value
                merged_volume[:, :, slice_num] = current_slice
                
                min_slice = min(min_slice, slice_num)
                max_slice = max(max_slice, slice_num)
            
            # Add Extent metadata if we processed any slices
            if min_slice != float('inf'):
                header[f'Segment{label_value-1}_Extent'] = f'0 {merged_volume.shape[0]-1} 0 {merged_volume.shape[1]-1} {min_slice} {max_slice}'
            
            label_value += 1
            
        except Exception as e:
            logging.error(f"Error processing organ {organ_name}: {e}")
            continue
    
    # Save the merged volume with the Slicer-compatible header
    if np.any(merged_volume):
        try:
            nrrd.write(output_path, merged_volume, header=header)
            logging.info(f"Successfully merged segmentations into {output_path}")
        except Exception as e:
            logging.error(f"Error writing merged NRRD file: {e}")

def delete_files(file_paths: List[str]):
    """Delete a list of files."""
    for path in file_paths:
        try:
            os.remove(path)
            logging.info(f"Deleted temporary file: {path}")
        except OSError as e:
            logging.error(f"Error deleting file {path}: {e}")

