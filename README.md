# News Classifier

## Content

- **Terminal UI** — amber-on-black, scanline overlay, monospace throughout
- **Confidence threshold slider** — filter results dynamically (sidebar)
- **Export results** — JSON and CSV download per scan
- **Session stats** — scan count and total inference time tracked live

## Manual:
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- First run downloads the model (~80 MB) once; cached after that
- GPU auto-detected if available (CUDA)
