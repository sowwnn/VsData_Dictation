import os
import logging
import json
import sys

# Add current directory to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__)))

from modules.pipeline import AnatomyDetectionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_pipeline_run():
    """
    Test AnatomyDetectionPipeline with TotalSegmentator logic.
    """
    logging.info("Initializing AnatomyDetectionPipeline...")
    
    # Initialize the main pipeline
    # Ensure GOOGLE_API_KEY is set in your environment
    pipeline = AnatomyDetectionPipeline()
    
    # Check if TotalSegmentator component loaded
    if not hasattr(pipeline, 'segmentator') or not pipeline.segmentator:
        logging.error("TotalSegmentator failed to initialize.")
        return

    logging.info("Pipeline initialized successfully.")

    # --- Setup Test Data Paths ---
    # Assuming running from server/ directory or root
    base_data_path = os.path.abspath("server/temp_data")
    if not os.path.exists(base_data_path):
        # Try relative path if running from server dir
        base_data_path = os.path.abspath("temp_data")
    
    if not os.path.exists(base_data_path):
        logging.error(f"Test data directory not found at {base_data_path}")
        return

    tracking_file = os.path.join(base_data_path, "tracking.csv")
    transcription_file = os.path.join(base_data_path, "transcription.json")
    
    # IMPORTANT: slice_images_dir must contain the NRRD/NIfTI file for TotalSegmentator
    slice_images_dir = base_data_path 

    logging.info(f"Running pipeline with:")
    logging.info(f"  - Tracking: {tracking_file}")
    logging.info(f"  - Transcription: {transcription_file}")
    logging.info(f"  - Input Dir (Volume): {slice_images_dir}")

    # Check if NRRD file exists in input dir
    has_volume = False
    for f in os.listdir(slice_images_dir):
        if f.endswith('.nrrd') or f.endswith('.nii.gz'):
            has_volume = True
            logging.info(f"  - Found volume file: {f}")
            break
    
    if not has_volume:
        logging.warning("No .nrrd or .nii.gz file found in input dir! TotalSegmentator step will likely be skipped.")

    # Execute the pipeline
    try:
        result_data = pipeline.run(
            tracking_file=tracking_file,
            transcription_file=transcription_file,
            slice_images_dir=slice_images_dir,
            language="vi" # Test with Vietnamese assuming transcript is VI based on your context
        )

        # Save the results
        output_file_path = os.path.join(base_data_path, "test_output_totalsegmentator.json")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        
        logging.info("="*50)
        logging.info(f"SUCCESS! Results saved to {output_file_path}")
        logging.info("Check 'detections' in JSON for 'segmentation_3d_path'")
        logging.info("Check 'segmentation_dir' in JSON for folder containing all generated masks")
        logging.info("="*50)

    except Exception as e:
        logging.error(f"Error during execution: {e}", exc_info=True)

if __name__ == "__main__":
    test_pipeline_run()
