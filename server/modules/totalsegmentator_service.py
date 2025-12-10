import os
import logging
import subprocess
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
import time

logger = logging.getLogger(__name__)

class TotalSegmentatorRunner:
    """Wrapper for running TotalSegmentator."""

    def __init__(self, output_dir: str = None):
        """
        Initialize the runner.
        
        Args:
            output_dir: Directory to save segmentation results. 
                        If None, a temporary directory will be created per run.
        """
        self.output_dir = output_dir

    def convert_nrrd_to_nifti(self, nrrd_path: str) -> str:
        """
        Convert NRRD file to NIfTI format, preserving affine/orientation.
        
        Args:
            nrrd_path: Path to NRRD file.
            
        Returns:
            Path to converted NIfTI file.
        """
        try:
            import nrrd
        except ImportError:
            logger.error("pynrrd not installed. Cannot convert NRRD to NIfTI.")
            raise ImportError("pynrrd is required for NRRD conversion")

        logger.info(f"Converting NRRD to NIfTI: {nrrd_path}")
        
        # Read NRRD
        data, header = nrrd.read(nrrd_path)
        
        # Construct affine matrix from NRRD header
        # NRRD uses space directions and space origin to define the affine transformation
        affine = np.eye(4)
        
        try:
            # Extract space directions (vectors defining the orientation of each axis)
            if 'space directions' in header:
                directions = header['space directions']
                
                # Handle different formats: can be 'none', list of lists, or numpy array
                # Fix: Check if directions is string 'none', not compare array with string
                if not (isinstance(directions, str) and directions == 'none') and directions is not None:
                    # Convert to numpy array if needed
                    if isinstance(directions, (list, tuple)):
                        # Filter out 'none' entries
                        valid_dirs = []
                        for d in directions:
                            # Fix: Check if d is string 'none'
                            if not (isinstance(d, str) and d == 'none') and d is not None:
                                if isinstance(d, (list, tuple, np.ndarray)):
                                    valid_dirs.append(np.array(d))
                                else:
                                    # Single value (spacing)
                                    valid_dirs.append(np.array([d, 0, 0]) if len(valid_dirs) == 0 else 
                                                     np.array([0, d, 0]) if len(valid_dirs) == 1 else 
                                                     np.array([0, 0, d]))
                        
                        if len(valid_dirs) == 3:
                            # Stack directions as columns (each direction is a column vector)
                            affine[:3, :3] = np.column_stack(valid_dirs)
                    elif isinstance(directions, np.ndarray):
                        if directions.shape == (3, 3):
                            affine[:3, :3] = directions
                        elif directions.shape == (3,):
                            # Diagonal spacing
                            affine[:3, :3] = np.diag(directions)
            
            # Extract space origin (translation)
            if 'space origin' in header:
                origin = header['space origin']
                # Fix: Check if origin is string 'none', not compare array with string
                if not (isinstance(origin, str) and origin == 'none') and origin is not None:
                    if isinstance(origin, (list, tuple, np.ndarray)):
                        affine[:3, 3] = np.array(origin)
                    else:
                        logger.warning(f"Unexpected origin format: {type(origin)}")
            
            # Log orientation info for debugging
            logger.info(f"NRRD space directions: {header.get('space directions', 'not found')}")
            logger.info(f"NRRD space origin: {header.get('space origin', 'not found')}")
            logger.info(f"Constructed affine:\n{affine}")
            
            # Check orientation codes
            ornt = nib.orientations.io_orientation(affine)
            axcodes = nib.orientations.aff2axcodes(affine)
            logger.info(f"NRRD orientation codes: {ornt}")
            logger.info(f"NRRD axis codes: {axcodes}")
            
        except Exception as e:
            logger.warning(f"Could not extract full affine from NRRD header: {e}")
            logger.warning("Using identity affine - orientation may not match original NRRD")
            affine = np.eye(4)

        # Create NIfTI image with the affine
        img = nib.Nifti1Image(data, affine)
        
        # Save as .nii.gz in the same directory
        output_path = nrrd_path.replace(".nrrd", ".nii.gz")
        nib.save(img, output_path)
        
        # Verify the saved file has correct affine
        verify_img = nib.load(output_path)
        logger.info(f"Saved NIfTI affine:\n{verify_img.affine}")
        verify_axcodes = nib.orientations.aff2axcodes(verify_img.affine)
        logger.info(f"Saved NIfTI axis codes: {verify_axcodes}")
        
        logger.info(f"Converted to: {output_path}")
        return output_path

    def run(self, input_path: str, task: str = "total", fast: bool = True) -> str:
        """
        Run TotalSegmentator on the input volume.

        Args:
            input_path: Path to the input NRRD or NIfTI file.
            task: Task to run (default: "total").
            fast: Whether to use --fast mode (lower resolution, faster).

        Returns:
            Path to the directory containing the segmentation masks.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Handle NRRD input by converting to NIfTI
        original_input_path = input_path
        temp_nifti_path = None
        
        if input_path.lower().endswith('.nrrd'):
            try:
                input_path = self.convert_nrrd_to_nifti(input_path)
                temp_nifti_path = input_path
            except Exception as e:
                logger.error(f"Failed to convert NRRD to NIfTI: {e}")
                raise

        # Determine output directory
        if self.output_dir:
            output_path = self.output_dir
        else:
            # Default to a folder named after the input file in the same directory
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            # remove .nii if it was .nii.gz
            if base_name.endswith('.nii'):
                base_name = base_name[:-4]
            output_path = os.path.join(os.path.dirname(input_path), f"{base_name}_segmentation")

        # Create directory if output_path is a directory (doesn't end in .nii/.nii.gz)
        # If it is a file path, create the parent directory.
        if output_path.endswith('.nii') or output_path.endswith('.nii.gz'):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            os.makedirs(output_path, exist_ok=True)
        
        logger.info(f"Running TotalSegmentator on {input_path}...")
        logger.info(f"Output directory: {output_path}")

        start_time = time.time()

        # Construct command
        # totalsegmentator -i input.nii.gz -o output_dir --task total --fast
        # Without --ml flag, TotalSegmentator creates individual files for each organ
        cmd = [
            "TotalSegmentator",
            "-i", input_path,
            "-o", output_path,
            "--task", task
            # No --ml flag: creates individual organ files
        ]

        if fast:
            cmd.append("--fast")
            
        # Ensure we are using GPU if available
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            # Run the command
            process = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True
            )
            logger.info("TotalSegmentator completed successfully.")
            # logger.debug(f"Stdout: {process.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"TotalSegmentator failed with error: {e.stderr}")
            # Clean up temp file if failed
            if temp_nifti_path and os.path.exists(temp_nifti_path):
                 try:
                    os.remove(temp_nifti_path)
                 except:
                    pass
            raise RuntimeError(f"TotalSegmentator failed: {e.stderr}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            if temp_nifti_path and os.path.exists(temp_nifti_path):
                 try:
                    os.remove(temp_nifti_path)
                 except:
                    pass
            raise

        # Cleanup temp NIfTI file
        if temp_nifti_path and os.path.exists(temp_nifti_path):
            try:
                # We might want to keep it for debugging, but usually delete
                # os.remove(temp_nifti_path) 
                pass 
            except Exception as e:
                logger.warning(f"Could not delete temp NIfTI file: {e}")

        elapsed_time = time.time() - start_time
        logger.info(f"Segmentation took {elapsed_time:.2f} seconds.")

        return output_path

    def check_gpu(self):
        """Check if GPU is available for TotalSegmentator."""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"GPU is available: {torch.cuda.get_device_name(0)}")
                return True
            else:
                logger.warning("GPU is NOT available. TotalSegmentator will be slow.")
                return False
        except ImportError:
            logger.warning("Torch not installed, cannot check GPU.")
            return False
