import os
import logging
import json
import time
import re
from typing import List, Dict, Optional

def _format_transcript_for_prompt(segments: List[Dict]) -> str:
    """Formats the transcript segments into a JSON string for the LLM prompt."""
    transcript_parts = []
    for segment in segments:
        start_time = segment.get('transcription_time', {}).get('start')
        end_time = segment.get('transcription_time', {}).get('end')
        text = segment.get('transcription_text', '')
        
        if start_time is not None and text:
            transcript_parts.append({
                "start_time": start_time,
                "end_time": end_time,
                "text": text
            })
    return json.dumps(transcript_parts, indent=2, ensure_ascii=False)

class MedicalNER:
    """Extract anatomical organ names from transcription text using LLM."""
    
    def __init__(self, api_key=None, model="gemini-2.5-flash"):
        """Initialize Medical NER with Google Gemini.
        
        Args:
            api_key: API key for the provider (or use environment variable GOOGLE_API_KEY)
            model: Model name (defaults to gemini-2.5-flash)
        """
        self.api_provider = "google"
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model = model
        self.client = None
        
        # Rate limiting: track last API call time
        self.last_api_call_time = 0
        self.min_delay_between_calls = 1.0  # Minimum 1 second between API calls
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            logging.info(f"Google Gemini client initialized with model {self.model}")
        except ImportError:
            logging.error("google-generativeai package not installed. Install with: pip install google-generativeai")
            self.client = None
        except Exception as e:
            logging.error(f"Error initializing Google Gemini client: {e}")
            self.client = None
    
    def _create_translation_extraction_prompt(self, full_transcript_json: str) -> str:
        """
        Creates a prompt to translate a Vietnamese medical transcript to English,
        correct STT errors, and extract anatomical organs.
        """
        prompt = f"""You are an expert medical assistant AI. Your task is to process a Vietnamese medical dictation transcript.
The user has provided a transcript segmented by timestamps. Due to potential Speech-to-Text (STT) errors, some Vietnamese medical terms might be slightly incorrect.
Your tasks are:
1.  Read the entire Vietnamese transcript below.
2.  Translate the full text into fluent, professional English.
3.  While translating, use your medical expertise to correct any likely STT errors in the Vietnamese source to ensure the most accurate meaning.
4.  **CRITICAL**: Extract ONLY anatomical organs or body parts that are described as **ABNORMAL**, **PATHOLOGICAL**, or having some **MEDICAL ISSUE** (e.g., cysts, tumors, inflammation, stones, dilation, lesions, etc.).
    - DO NOT extract organs that are described as "normal", "unremarkable", "clear", or mentioned just for context without any findings.
    - If an organ is mentioned but the doctor says it is normal, IGNORE it.
5.  For each extracted abnormal organ, you MUST provide the `start_time` from the original transcript segment where the abnormality was described.

Input Transcript (JSON format):
{full_transcript_json}

Provide your response as a single JSON object with the following structure. Do NOT include any other text or explanations outside of this JSON object.
{{
    "english_translation": "The complete, corrected English translation of the entire transcript.",
    "extracted_organs": [
        {{
            "organ_name": "name of the abnormal organ in English",
            "start_time": "the start_time of the transcript segment where the abnormality was mentioned",
            "context": "the English sentence describing the abnormality"
        }}
    ]
}}
"""
        return prompt
    
    def _create_correction_extraction_prompt(self, full_transcript_json: str) -> str:
        """
        Creates a prompt to correct STT errors in English medical transcript
        and extract anatomical organs (no translation needed).
        """
        prompt = f"""You are an expert medical assistant AI. Your task is to process an English medical dictation transcript.
The user has provided a transcript segmented by timestamps. Due to potential Speech-to-Text (STT) errors, some English medical terms might be slightly incorrect or misspelled.
Your tasks are:
1.  Read the entire English transcript below.
2.  Use your medical expertise to correct any likely STT errors, misspellings, or transcription mistakes to ensure the most accurate medical meaning.
3.  **CRITICAL**: Extract ONLY anatomical organs or body parts that are described as **ABNORMAL**, **PATHOLOGICAL**, or having some **MEDICAL ISSUE** (e.g., cysts, tumors, inflammation, stones, dilation, lesions, etc.).
    - DO NOT extract organs that are described as "normal", "unremarkable", "clear", or mentioned just for context without any findings.
    - If an organ is mentioned but the doctor says it is normal, IGNORE it.
4.  For each extracted abnormal organ, you MUST provide the `start_time` from the original transcript segment where the abnormality was described.

Input Transcript (JSON format):
{full_transcript_json}

Provide your response as a single JSON object with the following structure. Do NOT include any other text or explanations outside of this JSON object.
{{
    "corrected_transcript": "The complete, corrected English transcript with STT errors fixed.",
    "extracted_organs": [
        {{
            "organ_name": "name of the abnormal organ in English",
            "start_time": "the start_time of the transcript segment where the abnormality was mentioned",
            "context": "the corrected English sentence describing the abnormality"
        }}
    ]
}}
"""
        return prompt

    def _create_prompt(self, text: str, language: str = "en") -> str:
        """Create prompt for LLM to extract anatomical terms.
        
        Args:
            text: Transcription text
            language: "en" or "vi"
            
        Returns:
            str: Formatted prompt
        """
        if language == "vi":
            prompt = f"""Bạn là một chuyên gia y tế. Hãy trích xuất tên các bộ phận/cơ quan trong cơ thể được đề cập trong đoạn văn sau.

Đoạn văn:
"{text}"

Hãy trả về kết quả dưới dạng JSON với format:
{{
    "organs": [
        {{
            "name": "tên bộ phận",
            "confidence": 0.0-1.0,
            "context": "ngữ cảnh trong câu"
        }}
    ]
}}

Chỉ trả về JSON, không có text thêm."""
        else:
            prompt = f"""You are a medical expert. Extract anatomical organ/body part names mentioned in the following text.

Text:
"{text}"

Return the result as JSON with format:
{{
    "organs": [
        {{
            "name": "organ name",
            "confidence": 0.0-1.0,
            "context": "context in sentence"
        }}
    ]
}}

Return only JSON, no additional text."""
        
        return prompt
    
    def _parse_retry_delay(self, error_message: str) -> float:
        """Parse retry_delay from API error message.
        
        Args:
            error_message: Error message from API
            
        Returns:
            float: Retry delay in seconds, or None if not found
        """
        # Try to extract retry_delay from error message
        # Format: "Please retry in X.XXXXXXs" or "retry_delay { seconds: X }"
        patterns = [
            r"retry in ([\d.]+)s",
            r"retry_delay.*?seconds[:\s]+(\d+)",
            r"seconds[:\s]+(\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                try:
                    delay = float(match.group(1))
                    return delay
                except ValueError:
                    continue
        
        return None
    
    def _call_gemini(self, prompt: str, max_retries: int = 3) -> Optional[Dict]:
        """Call Google Gemini API with retry logic and rate limiting.
        
        Args:
            prompt: Prompt text
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            dict: Response from API
        """
        # Rate limiting: ensure minimum delay between API calls
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        if time_since_last_call < self.min_delay_between_calls:
            sleep_time = self.min_delay_between_calls - time_since_last_call
            logging.debug(f"Rate limiting: waiting {sleep_time:.2f}s before API call")
            time.sleep(sleep_time)
        
        for attempt in range(max_retries + 1):
            try:
                # Update last API call time
                self.last_api_call_time = time.time()
                
                # Generate content with JSON response format
                # Note: response_mime_type may not be supported in all models
                try:
                    response = self.client.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0.3,
                            "response_mime_type": "application/json",
                        }
                    )
                except Exception as e:
                    error_str = str(e)
                    
                    # Check if it's a quota/rate limit error (429)
                    if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                        # Parse retry_delay from error message
                        retry_delay = self._parse_retry_delay(error_str)
                        
                        if retry_delay is None:
                            # Use exponential backoff: 2^attempt seconds
                            retry_delay = min(2 ** attempt, 60)  # Cap at 60 seconds
                        else:
                            # Add small buffer to retry_delay from API
                            retry_delay = retry_delay + 1.0
                        
                        if attempt < max_retries:
                            logging.warning(f"API quota/rate limit exceeded (attempt {attempt + 1}/{max_retries + 1}). "
                                          f"Retrying in {retry_delay:.2f}s...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            logging.error(f"API quota/rate limit exceeded after {max_retries + 1} attempts. Giving up.")
                            return None
                    
                    # Fallback if response_mime_type is not supported (non-quota error)
                    logging.warning(f"JSON mode not supported, using standard generation: {e}")
                    response = self.client.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0.3,
                        }
                    )
                
                content = response.text
                
                # Parse JSON response
                try:
                    result = json.loads(content)
                    return result
                except json.JSONDecodeError as e:
                    logging.warning(f"Error parsing Gemini JSON response: {e}, trying to extract JSON...")
                    # Try to extract JSON from response if it's wrapped in text
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = content[start:end]
                        try:
                            result = json.loads(json_str)
                            return result
                        except json.JSONDecodeError:
                            logging.error(f"Could not parse extracted JSON: {json_str[:200]}")
                            return None
                    else:
                        logging.warning(f"No JSON found in Gemini response. Content: {content[:200]}")
                        return None
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a quota/rate limit error (429)
                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                    # Parse retry_delay from error message
                    retry_delay = self._parse_retry_delay(error_str)
                    
                    if retry_delay is None:
                        # Use exponential backoff: 2^attempt seconds
                        retry_delay = min(2 ** attempt, 60)  # Cap at 60 seconds
                    else:
                        # Add small buffer to retry_delay from API
                        retry_delay = retry_delay + 1.0
                    
                    if attempt < max_retries:
                        logging.warning(f"API quota/rate limit exceeded (attempt {attempt + 1}/{max_retries + 1}). "
                                      f"Retrying in {retry_delay:.2f}s...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logging.error(f"API quota/rate limit exceeded after {max_retries + 1} attempts. Giving up.")
                        return None
                else:
                    # Non-quota error: log and return None
                    logging.error(f"Error calling Google Gemini API (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    if attempt < max_retries:
                        # Exponential backoff for other errors too
                        retry_delay = min(2 ** attempt, 10)  # Cap at 10 seconds for non-quota errors
                        logging.info(f"Retrying in {retry_delay:.2f}s...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return None
        
        return None
    
    def match_organs_to_classes_batch(self, organ_names: List[str], available_classes: Dict[int, str]) -> Dict[str, tuple]:
        """
        Use LLM to match multiple organ names to TotalSegmentator classes in a single call.
        Supports composite organs (e.g., "Right Lung" -> multiple lobe classes).
        
        Args:
            organ_names: List of organ names to match (e.g., ["Lung (right base)", "Liver"])
            available_classes: Dict mapping label_id -> class_name (e.g., {15: "lung_upper_lobe_right", ...})
            
        Returns:
            dict: Mapping from organ_name -> (class_name, label_id) OR (list[class_names], list[label_ids]) for composites
                  Returns (None, None) if no match found
        """
        if not self.client:
            logging.error("LLM client not initialized. Cannot match organs to classes.")
            return {name: (None, None) for name in organ_names}
        
        if not available_classes:
            logging.warning("No available classes provided for matching")
            return {name: (None, None) for name in organ_names}
        
        if not organ_names:
            return {}
        
        # Remove duplicates while preserving order
        unique_organs = []
        seen = set()
        for org in organ_names:
            if org and org not in seen:
                unique_organs.append(org)
                seen.add(org)
        
        # Format organ names for prompt
        organs_list = []
        for i, organ_name in enumerate(unique_organs, 1):
            organs_list.append(f"  {i}. {organ_name}")
        
        organs_text = "\n".join(organs_list)
        
        # Format available classes for prompt
        classes_list = []
        for label_id, class_name in sorted(available_classes.items()):
            classes_list.append(f"  - {class_name} (label_id: {label_id})")
        
        classes_text = "\n".join(classes_list)
        
        prompt = f"""You are a medical imaging expert. Match each given organ name to the most appropriate TotalSegmentator class(es) from the available classes list.

Organ names to match:
{organs_text}

Available TotalSegmentator classes in this segmentation:
{classes_text}

Instructions:
1. For each organ name, find the best matching TotalSegmentator class(es)
2. Consider synonyms and variations (e.g., "Lung (right base)" matches "lung_lower_lobe_right")
3. **IMPORTANT**: If an organ is a COMPOSITE structure (multiple parts), return ALL matching classes as an array
   - Example: "Right Lung" should match ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
   - Example: "Spine" should match all vertebrae classes if mentioned generally
4. For single-part organs, still use an array with one element
5. Return ONLY a JSON object with this exact format:

{{
    "matches": [
        {{
            "organ_name": "exact organ name from the list above",
            "matched_classes": ["class_name1", "class_name2"],
            "label_ids": [label_id1, label_id2],
            "is_composite": true or false,
            "confidence": 0.0-1.0,
            "reason": "brief explanation"
        }}
    ]
}}

For organs with no good match, include them with:
{{
    "organ_name": "exact organ name",
    "matched_classes": [],
    "label_ids": [],
    "is_composite": false,
    "confidence": 0.0,
    "reason": "explanation"
}}

Return only JSON, no additional text."""

        try:
            result = self._call_gemini(prompt)
            matches_dict = {}
            
            if result and 'matches' in result:
                # Build mapping from LLM response
                for match in result['matches']:
                    organ_name = match.get('organ_name', '')
                    matched_classes = match.get('matched_classes', [])
                    label_ids = match.get('label_ids', [])
                    is_composite = match.get('is_composite', False)
                    confidence = match.get('confidence', 0.5)
                    reason = match.get('reason', '')
                    
                    if matched_classes and label_ids:
                        # Convert to list if composite, or tuple if single
                        if is_composite and len(matched_classes) > 1:
                            matches_dict[organ_name] = (matched_classes, [int(lid) for lid in label_ids])
                            logging.info(f"LLM matched composite '{organ_name}' -> {matched_classes} (IDs: {label_ids}, confidence: {confidence:.2f}, reason: {reason})")
                        else:
                            # Single class
                            matches_dict[organ_name] = (matched_classes[0], int(label_ids[0]))
                            logging.info(f"LLM matched '{organ_name}' -> '{matched_classes[0]}' (ID: {label_ids[0]}, confidence: {confidence:.2f}, reason: {reason})")
                    else:
                        matches_dict[organ_name] = (None, None)
                        logging.warning(f"LLM could not match '{organ_name}' to any available class. Reason: {reason}")
                
                # Fill in missing organs (if LLM didn't return all)
                for organ_name in unique_organs:
                    if organ_name not in matches_dict:
                        matches_dict[organ_name] = (None, None)
                        logging.warning(f"LLM did not return a match for '{organ_name}'")
            else:
                logging.warning(f"LLM response format invalid. Expected 'matches' array. Response: {result}")
                matches_dict = {name: (None, None) for name in unique_organs}
            
            # Map back to original organ_names list (including duplicates)
            result_dict = {}
            for org in organ_names:
                if org in matches_dict:
                    result_dict[org] = matches_dict[org]
                else:
                    result_dict[org] = (None, None)
            
            return result_dict
            
        except Exception as e:
            logging.error(f"Error matching organs to classes with LLM: {e}")
            return {name: (None, None) for name in organ_names}
    
    def match_organ_to_class(self, organ_name: str, available_classes: Dict[int, str]) -> Optional[tuple]:
        """
        Use LLM to match an organ name to the best TotalSegmentator class from available classes.
        (Legacy method - use match_organs_to_classes_batch for multiple organs)
        
        Args:
            organ_name: Organ name to match (e.g., "Lung (right base)")
            available_classes: Dict mapping label_id -> class_name (e.g., {15: "lung_upper_lobe_right", ...})
            
        Returns:
            tuple: (class_name, label_id) or (None, None) if no match found
        """
        results = self.match_organs_to_classes_batch([organ_name], available_classes)
        return results.get(organ_name, (None, None))
    
    def extract_organs(self, text: str, language: str = "en") -> List[Dict]:
        """Extract organ names from text.
        
        Args:
            text: Transcription text
            language: "en" or "vi"
            
        Returns:
            list: List of dicts with 'organ_name', 'original_text', 'confidence'
        """
        if not self.client:
            logging.error("LLM client not initialized. Cannot extract organs.")
            return []
        
        if not text or not text.strip():
            return []
        
        # Create prompt
        prompt = self._create_prompt(text, language)
        
        # Call API
        result = self._call_gemini(prompt)
        
        if not result:
            return []
        
        # Parse results
        organs = []
        if isinstance(result, dict) and 'organs' in result:
            for org in result['organs']:
                organs.append({
                    'organ_name': org.get('name', ''),
                    'original_text': text,
                    'confidence': float(org.get('confidence', 0.5)),
                    'context': org.get('context', '')
                })
        else:
            logging.warning(f"Unexpected API response format: {result}")
        
        logging.info(f"Extracted {len(organs)} organs from text: {organs}")
        return organs
    
    def extract_from_transcript(self, alignment_results: List[Dict], language: str = "vi") -> List[Dict]:
        """
        Processes a full transcript to correct STT errors, translate (if Vietnamese), and extract organs
        using a single LLM call. Works for both Vietnamese and English.

        Args:
            alignment_results: List of alignment results from TimeAlignment.
            language: The source language of the transcript ("vi" or "en").

        Returns:
            A list of dictionaries, each containing detailed information about a detected organ.
        """
        if not self.client:
            logging.error("LLM client not initialized. Cannot extract organs.")
            return []
            
        if not alignment_results:
            logging.warning("No alignment results provided to process.")
            return []

        # Prepare the full transcript for the prompt
        full_transcript_json = _format_transcript_for_prompt(alignment_results)
        
        # Choose appropriate prompt based on language
        if language == "vi":
            prompt = self._create_translation_extraction_prompt(full_transcript_json)
            logging.info("Processing full Vietnamese transcript: translate, correct STT errors, and extract abnormal organs")
        else:
            prompt = self._create_correction_extraction_prompt(full_transcript_json)
            logging.info("Processing full English transcript: correct STT errors and extract abnormal organs")

        # Call the LLM
        llm_response = self._call_gemini(prompt)

        if not llm_response or "extracted_organs" not in llm_response:
            logging.error("Failed to get a valid response from the LLM or response is missing 'extracted_organs'.")
            return []

        # Log the corrected/translated transcript
        if language == "vi":
            logging.info(f"LLM Response received. Full translation: {llm_response.get('english_translation', 'N/A')}")
        else:
            logging.info(f"LLM Response received. Corrected transcript: {llm_response.get('corrected_transcript', 'N/A')[:200]}...")
        
        # Create a lookup map for faster access to alignment data by start_time
        alignment_map = {}
        for align in alignment_results:
            start_time = align.get('transcription_time', {}).get('start')
            if start_time is not None:
                alignment_map[start_time] = align

        # Process the extracted organs and map them back to the original segments
        final_results = []
        for organ in llm_response["extracted_organs"]:
            start_time = organ.get("start_time")
            
            # Find nearest matching start_time in alignment_map
            # LLM might round timestamps or return slightly different values
            matched_start_time = None
            min_diff = float('inf')
            alignment = None
            
            try:
                organ_start = float(start_time)
                
                for t in alignment_map.keys():
                    diff = abs(float(t) - organ_start)
                    if diff < min_diff:
                        min_diff = diff
                        matched_start_time = t
                
                # Accept match if difference is within 1.0 second
                if matched_start_time is not None and min_diff <= 1.0:
                    alignment = alignment_map[matched_start_time]
                    logging.info(f"Matched LLM time {organ_start} to alignment time {matched_start_time} (diff: {min_diff:.3f}s)")
                else:
                    logging.warning(f"No close match for time {organ_start}. Nearest: {matched_start_time} (diff: {min_diff:.3f}s)")
                    
            except (ValueError, TypeError) as e:
                logging.warning(f"Error processing start_time '{start_time}': {e}")
                continue

            if not alignment:
                logging.warning(f"Could not find matching alignment for organ '{organ.get('organ_name')}' with start_time {start_time}")
                continue
            
            # Use a default confidence for now, as the new prompt doesn't ask for it.
            # This could be added back to the prompt if needed.
            ner_confidence = 0.9 

            result = {
                'organ_name': organ['organ_name'],
                'transcription_text': alignment.get('transcription_text', ''),
                'transcription_time': alignment.get('transcription_time', {}),
                'slice_range': alignment.get('slice_range', {}),
                'behavior_class': alignment.get('behavior_class'),
                'alignment_confidence': alignment.get('confidence_score', 0.0),
                'ner_confidence': ner_confidence,
                'overall_confidence': alignment.get('confidence_score', 0.0) * ner_confidence,
                'context': organ.get('context', '')
            }
            final_results.append(result)

        logging.info(f"Processed full transcript and found {len(final_results)} organ detections.")
        return final_results

    def extract_from_segments(self, alignment_results: List[Dict], language: str = "en") -> List[Dict]:
        """Extract organs from aligned transcription segments.
        
        Args:
            alignment_results: List of alignment results from TimeAlignment
            language: "en" or "vi"
            
        Returns:
            list: List of dicts with organ info + alignment info
        """
        results = []
        
        for alignment in alignment_results:
            text = alignment.get('transcription_text', '')
            if not text:
                continue
            
            # Extract organs from text
            organs = self.extract_organs(text, language)
            
            # Combine organ info with alignment info
            for organ in organs:
                result = {
                    'organ_name': organ['organ_name'],
                    'transcription_text': text,
                    'transcription_time': alignment.get('transcription_time', {}),
                    'slice_range': alignment.get('slice_range', {}),
                    'behavior_class': alignment.get('behavior_class'),
                    'alignment_confidence': alignment.get('confidence_score', 0.0),
                    'ner_confidence': organ['confidence'],
                    'overall_confidence': alignment.get('confidence_score', 0.0) * organ['confidence'],
                    'context': organ.get('context', '')
                }
                results.append(result)
        
        logging.info(f"Extracted organs from {len(alignment_results)} segments: {len(results)} organ detections")
        return results

