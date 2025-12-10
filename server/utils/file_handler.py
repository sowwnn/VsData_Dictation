import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def get_transcription_segments(transcriptionData):
    """Extract transcription segments from transcription data.
    
    Ported from client Utils/PlotGenerator.py
    
    Args:
        transcriptionData: Transcription data (dict or list)
        
    Returns:
        list: List of segments with 'start_time', 'end_time', 'text'
    """
    if isinstance(transcriptionData, dict) and 'segments' in transcriptionData:
        return transcriptionData['segments']
    elif isinstance(transcriptionData, list):
        return transcriptionData
    else:
        return []




