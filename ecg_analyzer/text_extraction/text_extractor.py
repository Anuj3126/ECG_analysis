"""
Text extraction from ECG images using OpenAI Vision API.
Clean, focused approach for extracting all text information.
"""

import os
import base64
import json
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

from ..core.models import ECGTextData, PatientInfo, ECGParameters, DeviceInfo

# Load environment variables
load_dotenv()

class TextExtractor:
    """Extract text information from ECG images using OpenAI Vision API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the text extractor."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def correct_ocr_errors(self, text: str) -> str:
        """Correct common OCR errors in extracted text."""
        if not text:
            return text
        
        # Common OCR corrections
        corrections = {
            # Numbers to letters (common in names)
            '0': 'O',  # Zero to letter O
            '1': 'I',  # One to letter I
            '5': 'S',  # Five to letter S
            '8': 'B',  # Eight to letter B
            '3': 'E',  # Three to letter E
            '6': 'G',  # Six to letter G
            
            # Letter to number corrections (less common but possible)
            'O': '0',  # Letter O to zero (in IDs/numbers)
            'I': '1',  # Letter I to one (in IDs/numbers)
            'S': '5',  # Letter S to five (in IDs/numbers)
            'B': '8',  # Letter B to eight (in IDs/numbers)
            'E': '3',  # Letter E to three (in IDs/numbers)
            'G': '6',  # Letter G to six (in IDs/numbers)
        }
        
        # Apply corrections
        corrected_text = text
        for wrong, correct in corrections.items():
            corrected_text = corrected_text.replace(wrong, correct)
        
        return corrected_text
    
    
    def extract_text_from_ecg(self, image_path: str) -> ECGTextData:
        """Extract all text from ECG image using OpenAI Vision API."""
        
        # Encode the image
        base64_image = self.encode_image(image_path)
        
        # Create the prompt for text extraction
        prompt = """
        You are an expert at reading ECG (electrocardiogram) images. 
        
        Please extract ALL text information from this ECG image and return it in a structured JSON format.
        
        CRITICAL FOR PATIENT NAMES: 
        - Patient names are ALWAYS composed of LETTERS ONLY - never numbers
        - Do not extract name as numbers from the image, it should be a string of characters only.
        - Make sure each and every letter from the name is extracted properly as it is.
        - Do not mix or interpret the letters as numbers or a different character or letter.
        
        Look for and extract:
        1. Patient information (name, age, gender, date, time) - CONVERT NUMBERS TO LETTERS IN NAMES
        2. ECG parameters (heart rate, PR interval, QRS duration, QT interval, QTc interval, P/QRS/T axes)
        3. Device information (speed, gain, filters, device model, serial number)
        4. Any comments or interpretations
        5. All other visible text
        
        Return the data in this exact JSON format:
        {
            "patient": {
                "name": "extracted name (Has only characters, no numbers) or null",
                "age": extracted_age_or_null,
                "gender": "extracted gender or null", 
                "date": "extracted date or null",
                "time": "extracted time or null"
            },
            "parameters": {
                "heart_rate_bpm": extracted_hr_or_null,
                "pr_interval_ms": extracted_pr_or_null,
                "qrs_duration_ms": extracted_qrs_or_null,
                "qt_interval_ms": extracted_qt_or_null,
                "qtc_interval_ms": extracted_qtc_or_null,
                "p_axis_degrees": extracted_p_axis_or_null,
                "qrs_axis_degrees": extracted_qrs_axis_or_null,
                "t_axis_degrees": extracted_t_axis_or_null
            },
            "device_info": {
                "speed": "extracted speed or null",
                "gain": "extracted gain or null",
                "filters": "extracted filters or null",
                "device_model": "extracted device model or null",
                "serial_number": "extracted serial number or null"
            },
            "comments": ["any extracted comments or interpretations"],
            "raw_text": "all visible text as a single string"
        }
        
        Be very thorough and extract every piece of text you can see. Remember: patient names contain ONLY letters, never numbers. If you can't find a specific value, use null.
        """
        
        try:
            # Call OpenAI Vision API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Extract the response
            response_text = response.choices[0].message.content
            
            # Try to parse as JSON
            try:
                # Clean the response to extract JSON
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                else:
                    json_text = response_text.strip()
                
                # Parse JSON
                data = json.loads(json_text)
                
                # Create ECGTextData object (no post-processing needed - prompt handles corrections)
                return ECGTextData(
                    patient=PatientInfo(**data.get("patient", {})),
                    parameters=ECGParameters(**data.get("parameters", {})),
                    device_info=DeviceInfo(**data.get("device_info", {})),
                    comments=data.get("comments", []),
                    raw_text=data.get("raw_text", response_text)
                )
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Response text: {response_text}")
                
                # Fallback: return raw text
                return ECGTextData(
                    patient=PatientInfo(),
                    parameters=ECGParameters(),
                    device_info=DeviceInfo(),
                    comments=[],
                    raw_text=response_text
                )
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ECGTextData(
                patient=PatientInfo(),
                parameters=ECGParameters(),
                device_info=DeviceInfo(),
                comments=[],
                raw_text=f"Error: {str(e)}"
            )
