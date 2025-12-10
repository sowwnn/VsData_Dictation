import os
import json
import logging
import requests
import time
from typing import Optional, Dict, List

class AnatomyDetectionAPI:
    """API client for anatomy detection server."""
    
    def __init__(self, base_url: str):
        """Initialize API client.
        
        Args:
            base_url: Base URL of the server API (e.g., "http://localhost:8000")
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = 600  # Increase to 10 minutes for large volume uploads
        
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {'Content-Type': 'application/json'}
        return headers
    
    def send_data_for_processing(self, tracking_file: str, transcription_file: str, 
                                 volume_file: str, session_id: str) -> Optional[Dict]:
        """Send data to server for processing.
        
        Args:
            tracking_file: Path to tracking CSV file
            transcription_file: Path to transcription JSON file
            volume_file: Path to volume file (.nrrd, .nii, or .nii.gz)
            session_id: Session ID
            
        Returns:
            dict: Response from server with session_id and status, or None if error
        """
        try:
            url = f"{self.base_url}/api/anatomy-detection"
            
            # Validate required files
            if not os.path.exists(tracking_file):
                logging.error(f"Tracking file not found: {tracking_file}")
                return None
            
            if not os.path.exists(transcription_file):
                logging.error(f"Transcription file not found: {transcription_file}")
                return None
            
            if not os.path.exists(volume_file):
                logging.error(f"Volume file not found: {volume_file}")
                return None
            
            # Validate volume file format
            volume_filename = os.path.basename(volume_file).lower()
            valid_extensions = ['.nrrd', '.nii', '.nii.gz']
            if not any(volume_filename.endswith(ext) for ext in valid_extensions):
                logging.error(f"Unsupported volume file format: {volume_file}. Expected: {', '.join(valid_extensions)}")
                return None
            
            # Prepare files for upload
            files = {}
            
            # Add tracking CSV
            files['tracking_file'] = ('tracking.csv', open(tracking_file, 'rb'), 'text/csv')
            
            # Add transcription JSON
            files['transcription_file'] = ('transcription.json', open(transcription_file, 'rb'), 'application/json')
            
            # Add volume file (required by server API)
            volume_basename = os.path.basename(volume_file)
            # Determine content type based on extension
            if volume_file.lower().endswith('.nrrd'):
                content_type = 'application/octet-stream'
            elif volume_file.lower().endswith(('.nii', '.nii.gz')):
                content_type = 'application/gzip'
            else:
                content_type = 'application/octet-stream'
            
            files['volume_file'] = (volume_basename, open(volume_file, 'rb'), content_type)
            logging.info(f"Prepared volume file for upload: {volume_file}")
            
            # Prepare data
            data = {
                'session_id': session_id
            }
            
            # Headers for multipart/form-data (requests will set Content-Type automatically)
            headers = {}
            
            # Send request
            response = requests.post(
                url,
                files=files,
                data=data,
                headers=headers,
                timeout=self.timeout
            )
            
            # Close file handles
            for file_tuple in files.values():
                if hasattr(file_tuple[1], 'close'):
                    file_tuple[1].close()
            
            if response.status_code == 200:
                result = response.json()
                logging.info(f"Successfully sent data for processing. Session ID: {session_id}")
                return result
            else:
                logging.error(f"Error sending data: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Error in send_data_for_processing: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    def check_status(self, session_id: str) -> Optional[Dict]:
        """Check processing status.
        
        Args:
            session_id: Session ID
            
        Returns:
            dict: Status information, or None if error
        """
        try:
            url = f"{self.base_url}/api/status/{session_id}"
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logging.warning(f"Status check failed: {response.status_code}")
                return None
                
        except Exception as e:
            logging.error(f"Error checking status: {e}")
            return None
    
    def get_results(self, session_id: str, wait_for_completion: bool = True, 
                   poll_interval: int = 5, max_wait_time: int = 600) -> Optional[Dict]:
        """Get results from server.
        
        Args:
            session_id: Session ID
            wait_for_completion: If True, poll until processing is complete
            poll_interval: Seconds between polls
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            dict: Results data, or None if error or timeout
        """
        try:
            if wait_for_completion:
                # Poll until completion
                start_time = time.time()
                while time.time() - start_time < max_wait_time:
                    status = self.check_status(session_id)
                    if status:
                        if status.get('status') == 'completed':
                            # Get results
                            url = f"{self.base_url}/api/results/{session_id}"
                            response = requests.get(
                                url,
                                headers=self._get_headers(),
                                timeout=30
                            )
                            if response.status_code == 200:
                                return response.json()
                        elif status.get('status') == 'failed':
                            logging.error(f"Processing failed: {status.get('error', 'Unknown error')}")
                            return None
                        elif status.get('status') == 'processing':
                            logging.info(f"Processing... ({int(time.time() - start_time)}s elapsed)")
                            time.sleep(poll_interval)
                            continue
                
                logging.error(f"Timeout waiting for results (>{max_wait_time}s)")
                return None
            else:
                # Just get results directly
                url = f"{self.base_url}/api/results/{session_id}"
                response = requests.get(
                    url,
                    headers=self._get_headers(),
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logging.warning(f"Results not ready: {response.status_code}")
                    return None
                    
        except Exception as e:
            logging.error(f"Error getting results: {e}")
            return None
    
    def download_report(self, session_id: str, output_path: str) -> bool:
        """Download report.csv file from server.
        
        Args:
            session_id: Session ID
            output_path: Local path to save the report file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            url = f"{self.base_url}/api/download-report/{session_id}"
            response = requests.get(
                url,
                timeout=60
            )
            
            if response.status_code == 200:
                # Save file
                output_dir = os.path.dirname(output_path)
                if output_dir:  # Only create directory if path has a directory component
                    os.makedirs(output_dir, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Report downloaded to: {output_path}")
                return True
            else:
                logging.error(f"Failed to download report: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logging.error(f"Error downloading report: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def download_segmentation(self, session_id: str, output_path: str, seg_type: str = "filtered") -> bool:
        """Download segmentation file from server.
        
        Args:
            session_id: Session ID
            output_path: Local path to save the segmentation file
            seg_type: Type of segmentation ("filtered", "full", or "merged")
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            url = f"{self.base_url}/api/download-segmentation/{session_id}"
            params = {"seg_type": seg_type}
            response = requests.get(
                url,
                params=params,
                timeout=300  # Longer timeout for large files
            )
            
            if response.status_code == 200:
                # Save file
                output_dir = os.path.dirname(output_path)
                if output_dir:  # Only create directory if path has a directory component
                    os.makedirs(output_dir, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Segmentation downloaded to: {output_path}")
                return True
            else:
                logging.error(f"Failed to download segmentation: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logging.error(f"Error downloading segmentation: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False




