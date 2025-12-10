"""
Utility to extract slice images from NRRD volume files.
"""
import os
import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import nrrd
except ImportError:
    logger.warning("pynrrd not installed. NRRD support will be disabled.")
    nrrd = None


def extract_slices_from_nrrd(
    nrrd_path: str,
    tracking_file: str,
    output_dir: str,
    views: Optional[List[str]] = None
) -> bool:
    """Extract slice images from NRRD file based on tracking data.
    
    Args:
        nrrd_path: Path to NRRD file
        tracking_file: Path to tracking CSV file
        output_dir: Directory to save extracted slice images
        views: List of views to extract (default: ['Axial', 'Coronal', 'Sagittal'])
        
    Returns:
        bool: True if successful, False otherwise
    """
    if nrrd is None:
        logger.error("pynrrd not installed. Cannot extract slices from NRRD.")
        return False
    
    if views is None:
        views = ['Axial', 'Coronal', 'Sagittal']
    
    try:
        # Load NRRD file
        logger.info(f"Loading NRRD file: {nrrd_path}")
        data, header = nrrd.read(nrrd_path)
        logger.info(f"NRRD shape: {data.shape}, dtype: {data.dtype}")
        
        # Load tracking data
        tracking_df = pd.read_csv(tracking_file)
        
        # Get unique slice numbers for each view
        slice_info = {}
        for view in views:
            view_df = tracking_df[tracking_df['view'] == view]
            if not view_df.empty:
                unique_slices = sorted(view_df['slice_number'].unique())
                slice_info[view] = unique_slices
                logger.info(f"View {view}: {len(unique_slices)} unique slices")
        
        # Extract slices for each view
        os.makedirs(output_dir, exist_ok=True)
        
        for view in views:
            if view not in slice_info:
                continue
                
            slices = slice_info[view]
            dims = data.shape
            
            # Determine axis and extract slices
            if view == 'Axial':
                # Axial: extract along z-axis (axis 2)
                axis = 2
                max_slice = dims[2]
            elif view == 'Coronal':
                # Coronal: extract along y-axis (axis 1)
                axis = 1
                max_slice = dims[1]
            elif view == 'Sagittal':
                # Sagittal: extract along x-axis (axis 0)
                axis = 0
                max_slice = dims[0]
            else:
                logger.warning(f"Unknown view: {view}")
                continue
            
            # Extract each slice
            for slice_num in slices:
                if slice_num < 0 or slice_num >= max_slice:
                    logger.warning(f"Slice {slice_num} out of range [0, {max_slice-1}] for {view}")
                    continue
                
                # Extract slice
                if axis == 0:
                    slice_data = data[slice_num, :, :]
                elif axis == 1:
                    slice_data = data[:, slice_num, :]
                else:  # axis == 2
                    slice_data = data[:, :, slice_num]
                
                # Normalize to 0-255
                if slice_data.max() > 255 or slice_data.min() < 0:
                    slice_min = slice_data.min()
                    slice_max = slice_data.max()
                    if slice_max > slice_min:
                        slice_data = ((slice_data - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
                    else:
                        slice_data = np.zeros_like(slice_data, dtype=np.uint8)
                else:
                    slice_data = slice_data.astype(np.uint8)
                
                # Convert to RGB if grayscale
                if len(slice_data.shape) == 2:
                    slice_data = np.stack([slice_data] * 3, axis=-1)
                
                # Save as PNG
                filename = f"{view}_slice{slice_num:04d}.png"
                filepath = os.path.join(output_dir, filename)
                
                image = Image.fromarray(slice_data)
                image.save(filepath)
                logger.debug(f"Saved {filename}")
        
        logger.info(f"Extracted slices to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting slices from NRRD: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def get_nrrd_info(nrrd_path: str) -> Optional[dict]:
    """Get information about NRRD file.
    
    Args:
        nrrd_path: Path to NRRD file
        
    Returns:
        dict: Information about the NRRD file or None if error
    """
    if nrrd is None:
        return None
    
    try:
        data, header = nrrd.read(nrrd_path)
        return {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'header': header,
            'min': float(data.min()),
            'max': float(data.max()),
            'mean': float(data.mean())
        }
    except Exception as e:
        logger.error(f"Error reading NRRD info: {e}")
        return None




