# NEURAL-X v2.0 — AI Text Classifier

Zero-shot semantic text classification.

## What's New in v2.0

- **Terminal UI** — amber-on-black, scanline overlay, monospace throughout
- **Confidence threshold slider** — filter results dynamically (sidebar)
- **Max categories slider** — control how many results to display
- **Multi-label toggle** — switch between multi-label and single-label mode
- **Customizable category set** — add/remove from 32 built-in categories
- **Confidence tiers** — HIGH / MEDIUM / LOW with color-coded indicators
- **Export results** — JSON and CSV download per scan
- **Bulk analysis mode** — classify multiple texts at once, export as CSV
- **Session stats** — scan count and total inference time tracked live
- **Glitch flicker** title animation + scanline overlay

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