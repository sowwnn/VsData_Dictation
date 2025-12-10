#!/usr/bin/env python3
"""
Script to merge all filtered mask files into a single segmentation file.
"""

import os
import sys
import logging
from modules.mask_processor import MaskProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Default paths
    filtered_masks_dir = os.path.join(os.path.dirname(__file__), "temp_data", "filtered_masks")
    output_filename = "merged_segmentation.nii.gz"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        filtered_masks_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_filename = sys.argv[2]
    
    logger.info(f"Merging segmentation files from: {filtered_masks_dir}")
    logger.info(f"Output file: {output_filename}")
    
    # Initialize MaskProcessor
    processor = MaskProcessor(filtered_masks_dir)
    
    # Merge masks
    output_path, label_mapping = processor.merge_filtered_masks(
        masks_dir=filtered_masks_dir,
        output_filename=output_filename
    )
    
    if output_path:
        logger.info("=" * 60)
        logger.info("Merge completed successfully!")
        logger.info(f"Output file: {output_path}")
        logger.info(f"Label mapping file: {output_path.replace('.nii.gz', '_labels.json')}")
        logger.info("=" * 60)
        logger.info("Label mapping:")
        for label_id, class_name in sorted(label_mapping.items()):
            logger.info(f"  Label {label_id}: {class_name}")
        return 0
    else:
        logger.error("Failed to merge segmentation files")
        return 1

if __name__ == "__main__":
    sys.exit(main())





