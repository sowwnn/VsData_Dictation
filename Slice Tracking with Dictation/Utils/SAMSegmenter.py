import os
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
import slicer

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
                default_path = os.path.expanduser("~/.cache/medsam/medsam_vit_b.pth")
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
            
            # Load model
            sam = sam_model_registry["vit_b"](checkpoint=self.model_path)
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
    
    def segment_with_text_prompt(self, image: np.ndarray, organ_name: str) -> Optional[Dict]:
        """Segment organ using text prompt (if supported).
        
        Note: Standard SAM doesn't support text prompts directly.
        This is a placeholder for future SAM variants with text support.
        
        Args:
            image: Image array
            organ_name: Name of organ to segment
            
        Returns:
            dict: Segmentation result with mask and bbox
        """
        # For now, SAM doesn't support text prompts directly
        # This would require a text-to-prompt model or manual point selection
        # Placeholder implementation
        
        logging.warning("Text prompt segmentation not yet implemented. Using center point as default.")
        
        # Use center point as prompt (fallback)
        h, w = image.shape[:2]
        center_point = np.array([[w // 2, h // 2]])
        
        return self.segment_with_points(image, center_point)
    
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

