"""
Test script for Anatomy Detection API
Usage: python test_api.py
"""
import requests
import json
import time
import os
from pathlib import Path

# API base URL
BASE_URL = os.environ.get("API_URL", "http://localhost:8000")
API_BASE = f"{BASE_URL}/api"

def test_health_check():
    """Test health check endpoint"""
    print("=" * 50)
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200, "Health check failed"
        print("✓ Health check passed\n")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}\n")
        return False

def test_anatomy_detection(tracking_file=None, transcription_file=None, slice_images=None):
    """Test anatomy detection endpoint"""
    print("=" * 50)
    print("Testing anatomy detection endpoint...")
    
    # Generate session ID
    import uuid
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")
    
    # Prepare form data
    files = {}
    data = {"session_id": session_id}
    
    # Add files if provided
    if tracking_file and os.path.exists(tracking_file):
        files["tracking_file"] = open(tracking_file, "rb")
        print(f"Using tracking file: {tracking_file}")
    
    if transcription_file and os.path.exists(transcription_file):
        files["transcription_file"] = open(transcription_file, "rb")
        print(f"Using transcription file: {transcription_file}")
    
    if slice_images and os.path.exists(slice_images):
        files["slice_images"] = open(slice_images, "rb")
        print(f"Using slice images: {slice_images}")
    
    try:
        # Send request
        response = requests.post(
            f"{API_BASE}/anatomy-detection",
            data=data,
            files=files
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Close file handles
        for f in files.values():
            f.close()
        
        if response.status_code == 200:
            print("✓ Anatomy detection request submitted successfully\n")
            return session_id
        else:
            print(f"✗ Request failed: {response.text}\n")
            return None
            
    except Exception as e:
        print(f"✗ Request failed: {e}\n")
        # Close file handles on error
        for f in files.values():
            f.close()
        return None

def test_status(session_id):
    """Test status endpoint"""
    print("=" * 50)
    print(f"Testing status endpoint for session: {session_id}...")
    
    try:
        response = requests.get(f"{API_BASE}/status/{session_id}")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            status_data = response.json()
            print(f"✓ Status retrieved: {status_data.get('status')}\n")
            return status_data.get('status')
        else:
            print(f"✗ Status check failed: {response.text}\n")
            return None
            
    except Exception as e:
        print(f"✗ Status check failed: {e}\n")
        return None

def test_results(session_id):
    """Test results endpoint"""
    print("=" * 50)
    print(f"Testing results endpoint for session: {session_id}...")
    
    try:
        response = requests.get(f"{API_BASE}/results/{session_id}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"Response: {json.dumps(results, indent=2)}")
            print("✓ Results retrieved successfully\n")
            return results
        else:
            print(f"Response: {response.text}")
            print(f"✗ Results retrieval failed\n")
            return None
            
    except Exception as e:
        print(f"✗ Results retrieval failed: {e}\n")
        return None

def wait_for_completion(session_id, timeout=300, check_interval=5):
    """Wait for processing to complete"""
    print("=" * 50)
    print(f"Waiting for processing to complete (timeout: {timeout}s)...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = test_status(session_id)
        
        if status == "completed":
            print("✓ Processing completed!\n")
            return True
        elif status == "failed":
            print("✗ Processing failed!\n")
            return False
        elif status == "processing":
            print(f"Still processing... (elapsed: {int(time.time() - start_time)}s)")
            time.sleep(check_interval)
        else:
            print(f"Unknown status: {status}")
            time.sleep(check_interval)
    
    print(f"✗ Timeout after {timeout}s\n")
    return False

def create_sample_files():
    """Create sample test files"""
    print("=" * 50)
    print("Creating sample test files...")
    
    # Create sample tracking CSV
    tracking_content = """timestamp,x,y,z,organ
0.0,100,200,50,liver
1.0,105,205,52,liver
2.0,110,210,55,kidney
"""
    with open("sample_tracking.csv", "w") as f:
        f.write(tracking_content)
    print("Created: sample_tracking.csv")
    
    # Create sample transcription JSON
    transcription_content = {
        "language": "en",
        "cues": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "This is the liver"
            },
            {
                "start": 1.0,
                "end": 2.0,
                "text": "Now we see the kidney"
            }
        ]
    }
    with open("sample_transcription.json", "w") as f:
        json.dump(transcription_content, f, indent=2)
    print("Created: sample_transcription.json")
    
    print("✓ Sample files created\n")

def main():
    """Main test function"""
    print("\n" + "=" * 50)
    print("Anatomy Detection API Test Suite")
    print("=" * 50 + "\n")
    
    # Test 1: Health check
    if not test_health_check():
        print("Server is not running. Please start the server first.")
        print(f"Run: python main.py or uvicorn main:app --host 0.0.0.0 --port 8000")
        return
    
    # Test 2: Create sample files if they don't exist
    if not os.path.exists("sample_tracking.csv") or not os.path.exists("sample_transcription.json"):
        create_sample_files()
    
    # Test 3: Submit anatomy detection request
    session_id = test_anatomy_detection(
        tracking_file="sample_tracking.csv",
        transcription_file="sample_transcription.json",
        slice_images=None  # Optional
    )
    
    if not session_id:
        print("Failed to submit request. Exiting.")
        return
    
    # Test 4: Monitor status
    completed = wait_for_completion(session_id, timeout=300)
    
    # Test 5: Get results if completed
    if completed:
        test_results(session_id)
    
    print("=" * 50)
    print("Test suite completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()




