from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import os
import json
import logging
import zipfile
import tempfile
import shutil
from typing import Optional
from datetime import datetime

from modules.pipeline import AnatomyDetectionPipeline
from utils.nrrd_extractor import extract_slices_from_nrrd

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage for processing status (in production, use Redis or database)
processing_status = {}
processing_results = {}

@router.post("/anatomy-detection")
async def anatomy_detection(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    tracking_file: UploadFile = File(...),
    transcription_file: UploadFile = File(...),
    volume_file: UploadFile = File(...)
):
    """Process anatomy detection pipeline.
    
    Args:
        session_id: Session ID
        tracking_file: Tracking CSV file
        transcription_file: Transcription JSON file
        volume_file: Volume file (NIfTI .nii, .nii.gz, or NRRD .nrrd)
        
    Returns:
        dict: Session ID and status
    """
    try:
        # Validate inputs
        if not tracking_file or not transcription_file or not volume_file:
            raise HTTPException(
                status_code=400, 
                detail="tracking_file, transcription_file, and volume_file are required"
            )
        
        # Validate volume file format
        volume_filename = volume_file.filename.lower() if volume_file.filename else ""
        valid_extensions = ['.nii', '.nii.gz', '.nrrd']
        if not any(volume_filename.endswith(ext) for ext in valid_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported volume file format. Expected: {', '.join(valid_extensions)}"
            )
        
        # Create temp directory for this session
        temp_dir = tempfile.mkdtemp(prefix=f"anatomy_detection_{session_id}_")
        
        try:
            # Save uploaded files
            tracking_path = os.path.join(temp_dir, "tracking.csv")
            transcription_path = os.path.join(temp_dir, "transcription.json")
            
            with open(tracking_path, "wb") as f:
                shutil.copyfileobj(tracking_file.file, f)
            
            with open(transcription_path, "wb") as f:
                shutil.copyfileobj(transcription_file.file, f)
            
            # Save volume file to slice_images_dir (pipeline expects volume file here)
            slice_images_dir = os.path.join(temp_dir, "slice_images")
            os.makedirs(slice_images_dir, exist_ok=True)
            
            # Save volume file with original filename
            original_filename = volume_file.filename or "volume"
            volume_path = os.path.join(slice_images_dir, original_filename)
            with open(volume_path, "wb") as f:
                shutil.copyfileobj(volume_file.file, f)
            
            logger.info(f"Saved volume file: {volume_path} ({original_filename})")
            
            # Initialize processing status
            processing_status[session_id] = {
                "status": "processing",
                "started_at": datetime.now().isoformat(),
                "error": None,
                "temp_dir": temp_dir
            }
            
            # Run pipeline in background
            background_tasks.add_task(
                run_pipeline_background,
                session_id,
                tracking_path,
                transcription_path,
                slice_images_dir,
                temp_dir
            )
            
            return {
                "session_id": session_id,
                "status": "processing",
                "message": "Pipeline started"
            }
            
        except Exception as e:
            # Cleanup on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in anatomy-detection endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{session_id}")
async def get_status(session_id: str):
    """Get processing status.
    
    Args:
        session_id: Session ID
        
    Returns:
        dict: Status information
    """
    if session_id not in processing_status:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return processing_status[session_id]

@router.get("/results/{session_id}")
async def get_results(session_id: str):
    """Get processing results.
    
    Args:
        session_id: Session ID
        
    Returns:
        dict: Results data
    """
    if session_id not in processing_status:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if processing_status[session_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed yet")
    
    if session_id not in processing_results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return processing_results[session_id]

@router.get("/download-report/{session_id}")
async def download_report(session_id: str):
    """Download report.csv file.
    
    Args:
        session_id: Session ID
        
    Returns:
        FileResponse: CSV report file
    """
    if session_id not in processing_status:
        raise HTTPException(status_code=404, detail="Session not found")
    
    status = processing_status[session_id]
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed yet")
    
    temp_dir = status.get("temp_dir")
    if not temp_dir or not os.path.exists(temp_dir):
        raise HTTPException(status_code=404, detail="Result files not found (expired or deleted)")
    
    # Report CSV is generated in temp_dir (output_dir_base)
    report_path = os.path.join(temp_dir, "report.csv")
    
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        path=report_path,
        filename=f"report_{session_id}.csv",
        media_type='text/csv'
    )

@router.get("/download-segmentation/{session_id}")
async def download_segmentation(session_id: str, seg_type: str = "filtered"):
    """Download a specific segmentation file.
    
    Args:
        session_id: Session ID
        seg_type: Type of segmentation to download:
                 - "filtered" (default): Position-filtered segmentation
                 - "full": Full (unfiltered) segmentation
                 - "merged": Legacy merged segmentation (backward compatibility)
        
    Returns:
        FileResponse: Segmentation NIfTI file (.nii.gz)
    """
    if session_id not in processing_status:
        raise HTTPException(status_code=404, detail="Session not found")
    
    status = processing_status[session_id]
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed yet")
    
    temp_dir = status.get("temp_dir")
    if not temp_dir or not os.path.exists(temp_dir):
        raise HTTPException(status_code=404, detail="Result files not found (expired or deleted)")
    
    # Segmentation files are in filtered_masks subdirectory
    filtered_masks_dir = os.path.join(temp_dir, "filtered_masks")
    
    # Determine which file to return based on seg_type
    seg_path = None
    filename = None
    
    if seg_type == "full":
        # Full (unfiltered) segmentation
        full_seg_nifti = os.path.join(filtered_masks_dir, "full_segmentation.nii.gz")
        
        if os.path.exists(full_seg_nifti):
            seg_path = full_seg_nifti
            filename = f"full_segmentation_{session_id}.nii.gz"
    
    elif seg_type == "merged":
        # Legacy merged segmentation (backward compatibility)
        merged_seg_nifti = os.path.join(filtered_masks_dir, "merged_segmentation.nii.gz")
        
        if os.path.exists(merged_seg_nifti):
            seg_path = merged_seg_nifti
            filename = f"merged_segmentation_{session_id}.nii.gz"
    
    else:  # Default: "filtered"
        # Position-filtered segmentation
        filtered_seg_nifti = os.path.join(filtered_masks_dir, "filtered_segmentation.nii.gz")
        
        if os.path.exists(filtered_seg_nifti):
            seg_path = filtered_seg_nifti
            filename = f"filtered_segmentation_{session_id}.nii.gz"
        
        # Fallback to legacy merged_segmentation for backward compatibility
        if not seg_path:
            merged_seg_nifti = os.path.join(filtered_masks_dir, "merged_segmentation.nii.gz")
            
            if os.path.exists(merged_seg_nifti):
                seg_path = merged_seg_nifti
                filename = f"segmentation_{session_id}.nii.gz"
    
    if not seg_path:
        raise HTTPException(status_code=404, detail=f"Segmentation file not found (type: {seg_type})")
    
    # Media type for NIfTI files
    media_type = 'application/gzip'
    
    return FileResponse(
        path=seg_path,
        filename=filename,
        media_type=media_type
    )

@router.get("/download-results/{session_id}")
async def download_results(session_id: str):
    """Download all results as ZIP file (report.csv and segmentation files).
    
    Includes:
    - report.csv: Detection report with label mappings
    - full_segmentation.nii.gz: Full (unfiltered) segmentation of detected organs
    - filtered_segmentation.nii.gz: Position-filtered segmentation of detected organs
    - (Legacy) merged_segmentation.nii.gz: Backward compatibility
    
    Args:
        session_id: Session ID
        
    Returns:
        FileResponse: ZIP file containing results
    """
    if session_id not in processing_status:
        raise HTTPException(status_code=404, detail="Session not found")
    
    status = processing_status[session_id]
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed yet")
    
    temp_dir = status.get("temp_dir")
    if not temp_dir or not os.path.exists(temp_dir):
        raise HTTPException(status_code=404, detail="Result files not found (expired or deleted)")
        
    # Create a zip file with the results
    zip_filename = f"segmentation_results_{session_id}.zip"
    zip_path = os.path.join(temp_dir, zip_filename)
    
    # Files to include
    report_path = os.path.join(temp_dir, "report.csv")
    filtered_masks_dir = os.path.join(temp_dir, "filtered_masks")
    
    # New segmentation files (NII.GZ only)
    full_seg_nifti = os.path.join(filtered_masks_dir, "full_segmentation.nii.gz")
    filtered_seg_nifti = os.path.join(filtered_masks_dir, "filtered_segmentation.nii.gz")
    
    # Legacy names for backward compatibility
    merged_seg_nifti = os.path.join(filtered_masks_dir, "merged_segmentation.nii.gz")
    
    # Log file existence for debugging
    logger.info(f"Checking files for download-results:")
    logger.info(f"  report_path: {report_path} - exists: {os.path.exists(report_path)}")
    logger.info(f"  full_seg_nifti: {full_seg_nifti} - exists: {os.path.exists(full_seg_nifti)}")
    logger.info(f"  filtered_seg_nifti: {filtered_seg_nifti} - exists: {os.path.exists(filtered_seg_nifti)}")
    logger.info(f"  filtered_masks_dir: {filtered_masks_dir} - exists: {os.path.exists(filtered_masks_dir)}")
    
    if os.path.exists(filtered_masks_dir):
        import glob
        all_files = glob.glob(os.path.join(filtered_masks_dir, "*"))
        logger.info(f"  Files in filtered_masks_dir: {[os.path.basename(f) for f in all_files]}")
    
    files_to_include = []
    
    # Always include report
    if os.path.exists(report_path):
        files_to_include.append(("report.csv", report_path))
    
    # Include new segmentation files (NII.GZ only)
    if os.path.exists(full_seg_nifti):
        files_to_include.append(("full_segmentation.nii.gz", full_seg_nifti))
    
    if os.path.exists(filtered_seg_nifti):
        files_to_include.append(("filtered_segmentation.nii.gz", filtered_seg_nifti))
    
    # Backward compatibility: include legacy merged_segmentation if new files don't exist
    if not os.path.exists(filtered_seg_nifti):
        if os.path.exists(merged_seg_nifti):
            files_to_include.append(("merged_segmentation.nrrd", merged_seg_nrrd))
        elif os.path.exists(merged_seg_nifti):
            files_to_include.append(("merged_segmentation.nii.gz", merged_seg_nifti))
    
    logger.info(f"Files to include in ZIP: {[f[0] for f in files_to_include]}")
    
    if not files_to_include:
        raise HTTPException(status_code=404, detail="No results generated to download")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for arcname, file_path in files_to_include:
            zipf.write(file_path, arcname=arcname)
                
    return FileResponse(
        path=zip_path, 
        filename=zip_filename, 
        media_type='application/zip'
    )

def run_pipeline_background(session_id: str, tracking_path: str, transcription_path: str, 
                           slice_images_dir: Optional[str], temp_dir: str):
    """Run pipeline in background task.
    
    Args:
        session_id: Session ID
        tracking_path: Path to tracking CSV
        transcription_path: Path to transcription JSON
        slice_images_dir: Directory containing slice images
        temp_dir: Temporary directory for this session
    """
    try:
        logger.info(f"Starting pipeline for session {session_id}")
        
        # Initialize pipeline
        pipeline = AnatomyDetectionPipeline()
        
        # Determine language from transcription (default to en)
        language = "en"
        try:
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)
                if isinstance(transcription_data, dict) and 'language' in transcription_data:
                    lang_code = transcription_data['language']
                    language = "vi" if "vi" in lang_code.lower() else "en"
        except:
            pass
        
        # Run pipeline
        results = pipeline.run(
            tracking_file=tracking_path,
            transcription_file=transcription_path,
            slice_images_dir=slice_images_dir,
            language=language
        )
        
        # Set session_id in results
        results['session_id'] = session_id
        
        # Store results
        processing_results[session_id] = results
        
        # Update status but preserve temp_dir
        current_status = processing_status.get(session_id, {})
        processing_status[session_id] = {
            "status": "completed",
            "started_at": current_status.get("started_at"),
            "completed_at": datetime.now().isoformat(),
            "error": None,
            "temp_dir": current_status.get("temp_dir")
        }
        
        logger.info(f"Pipeline completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error in pipeline for session {session_id}: {e}")
        current_status = processing_status.get(session_id, {})
        processing_status[session_id] = {
            "status": "failed",
            "started_at": current_status.get("started_at"),
            "completed_at": datetime.now().isoformat(),
            "error": str(e),
            "temp_dir": current_status.get("temp_dir")
        }
    finally:
        # Cleanup temp directory after some time (or implement cleanup job)
        # For now, keep it for debugging
        pass

