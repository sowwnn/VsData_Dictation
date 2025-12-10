# Anatomy Detection API Server

API server for anatomy detection pipeline processing.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install SAM (Segment Anything Model):
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

3. Download MedSAM model (optional):
- Download from: https://github.com/bowang-lab/MedSAM
- Set `SAM_MODEL_PATH` environment variable

## Configuration

Set environment variables:

- `LLM_PROVIDER`: "openai" or "anthropic" (default: "openai")
- `LLM_API_KEY` or `OPENAI_API_KEY`: API key for LLM
- `SAM_MODEL_PATH`: Path to SAM model checkpoint (optional)
- `SAM_MODEL_TYPE`: "medsam" or "sam2" (default: "medsam")
- `PORT`: Server port (default: 8000)

## Run Server

```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /api/anatomy-detection
Process anatomy detection pipeline.

**Request:**
- `session_id` (form): Session ID
- `tracking_file` (file): Tracking CSV file
- `transcription_file` (file): Transcription JSON file
- `slice_images` (file, optional): Zip file containing slice images

**Response:**
```json
{
  "session_id": "...",
  "status": "processing",
  "message": "Pipeline started"
}
```

### GET /api/status/{session_id}
Get processing status.

**Response:**
```json
{
  "status": "processing|completed|failed",
  "started_at": "...",
  "completed_at": "...",
  "error": null
}
```

### GET /api/results/{session_id}
Get processing results.

**Response:**
```json
{
  "session_id": "...",
  "detections": [
    {
      "organ_name": "...",
      "transcription_text": "...",
      "slice_range": {...},
      "segmentation": {...}
    }
  ]
}
```

## Notes

- Processing runs in background tasks
- Results are stored in memory (use Redis/database for production)
- Temp files are cleaned up automatically




