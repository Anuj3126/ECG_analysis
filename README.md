# ECG Analysis System

A professional, modular ECG analysis system that extracts both text information and digital signals from ECG images using advanced AI and computer vision techniques.

## Features

- **Text Extraction**: Uses OpenAI Vision API to extract patient information, ECG parameters, and clinical findings
- **Signal Processing**: Converts ECG waveforms to digital signals with proper calibration
- **Modular Design**: Clean, professional code structure suitable for presentation
- **Comprehensive Output**: Generates structured JSON data, signal plots, and analysis reports

## Project Structure

```
ecg_analyzer/
├── core/                    # Core data models
│   ├── __init__.py
│   └── models.py           # Pydantic models for data structures
├── text_extraction/        # Text extraction module
│   ├── __init__.py
│   └── text_extractor.py   # OpenAI Vision API integration
├── signal_processing/      # Signal processing module
│   ├── __init__.py
│   └── signal_processor.py # Waveform digitization
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── file_utils.py       # File I/O operations
├── __init__.py
└── ecg_analyzer.py         # Main analyzer class

data/                       # ECG image files
output/                     # Analysis results
tests/                      # Unit tests
docs/                       # Documentation
main.py                     # Main execution script
requirements.txt            # Dependencies
README.md                   # This file
```

## Installation

1. **Clone or download the project**
2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key**:
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Basic Usage

```python
from ecg_analyzer import ECGAnalyzer

# Initialize analyzer
analyzer = ECGAnalyzer()

# Analyze ECG image
analysis = analyzer.analyze_ecg("data/ecg1.jpeg")

# Access results
print(f"Patient: {analysis.text_data.patient.name}")
print(f"Heart Rate: {analysis.text_data.parameters.heart_rate_bpm} bpm")
print(f"Signals: {len(analysis.signal_data.leads)} leads extracted")
```

### Command Line Usage

```bash
python main.py
```

This will process all ECG images in the `data/` directory and save results to `output/`.

### Advanced Usage

```python
# Text extraction only
text_data = analyzer.analyze_text_only("data/ecg1.jpeg")

# Signal processing only
signal_data = analyzer.analyze_signals_only("data/ecg1.jpeg")

# Custom sampling rate
analyzer = ECGAnalyzer(sampling_rate_hz=1000.0)
```

## Output Files

For each ECG image, the system generates:

- `*_complete.json`: Complete analysis with both text and signal data
- `*_text.json`: Text extraction results only
- `*_signals.json`: Signal data only
- `*_signals_plot.png`: Visualization of extracted signals

## Data Models

### Patient Information
- Name, age, gender
- Date and time of ECG

### ECG Parameters
- Heart rate, PR interval, QRS duration
- QT/QTc intervals, P/QRS/T axes

### Signal Data
- 12-lead ECG signals as time-voltage arrays
- Sampling rate and calibration information
- Grid detection and measurement conversion

## API Reference

### ECGAnalyzer

Main class for ECG analysis.

```python
analyzer = ECGAnalyzer(openai_api_key=None, sampling_rate_hz=500.0)
analysis = analyzer.analyze_ecg(image_path, extract_signals=True)
```

### TextExtractor

Extracts text information from ECG images.

```python
extractor = TextExtractor(api_key="your_key")
text_data = extractor.extract_text_from_ecg("image.jpg")
```

### SignalProcessor

Processes ECG waveforms to digital signals.

```python
processor = SignalProcessor(sampling_rate_hz=500.0)
signal_data = processor.process_ecg_image("image.jpg")
```

## Requirements

- Python 3.8+
- OpenAI API key
- OpenCV for image processing
- NumPy/SciPy for signal processing
- Matplotlib for visualization

## License

This project is for educational and research purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For questions or issues, please create an issue in the repository.