import os
import logging
import json
from typing import List, Dict, Optional

class MedicalNER:
    """Extract anatomical organ names from transcription text using LLM."""
    
    def __init__(self, api_provider="openai", api_key=None, model=None):
        """Initialize Medical NER with LLM provider.
        
        Args:
            api_provider: "openai" or "anthropic"
            api_key: API key for the provider (or use environment variable)
            model: Model name (defaults to gpt-4 for OpenAI, claude-3-opus for Anthropic)
        """
        self.api_provider = api_provider.lower()
        self.api_key = api_key or os.environ.get(f"{api_provider.upper()}_API_KEY")
        self.model = model or self._get_default_model()
        self.client = None
        
        self._initialize_client()
    
    def _get_default_model(self):
        """Get default model for provider."""
        if self.api_provider == "openai":
            return "gpt-4"
        elif self.api_provider == "anthropic":
            return "claude-3-opus-20240229"
        else:
            return "gpt-4"
    
    def _initialize_client(self):
        """Initialize LLM client."""
        try:
            if self.api_provider == "openai":
                try:
                    import openai
                    self.client = openai.OpenAI(api_key=self.api_key)
                    logging.info("OpenAI client initialized")
                except ImportError:
                    logging.error("openai package not installed. Install with: pip install openai")
                    self.client = None
                    
            elif self.api_provider == "anthropic":
                try:
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=self.api_key)
                    logging.info("Anthropic client initialized")
                except ImportError:
                    logging.error("anthropic package not installed. Install with: pip install anthropic")
                    self.client = None
            else:
                logging.error(f"Unsupported API provider: {self.api_provider}")
                self.client = None
                
        except Exception as e:
            logging.error(f"Error initializing LLM client: {e}")
            self.client = None
    
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
    
    def _call_openai(self, prompt: str) -> Optional[Dict]:
        """Call OpenAI API.
        
        Args:
            prompt: Prompt text
            
        Returns:
            dict: Response from API
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical expert assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            return result
            
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            return None
    
    def _call_anthropic(self, prompt: str) -> Optional[Dict]:
        """Call Anthropic API.
        
        Args:
            prompt: Prompt text
            
        Returns:
            dict: Response from API
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = message.content[0].text
            # Try to extract JSON from response
            try:
                # Find JSON in response
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    result = json.loads(json_str)
                    return result
                else:
                    logging.warning("No JSON found in Anthropic response")
                    return None
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing Anthropic JSON response: {e}")
                return None
                
        except Exception as e:
            logging.error(f"Error calling Anthropic API: {e}")
            return None
    
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
        if self.api_provider == "openai":
            result = self._call_openai(prompt)
        elif self.api_provider == "anthropic":
            result = self._call_anthropic(prompt)
        else:
            logging.error(f"Unsupported API provider: {self.api_provider}")
            return []
        
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




