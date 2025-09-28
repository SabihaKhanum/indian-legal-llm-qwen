# indian-legal-llm-qwen


Fine-tuned Qwen 2.5-1.5B model for answering Indian legal questions, specifically trained on IPC sections and legal Q&A data.


### Option 1: Google Colab (Recommended)
https://colab.research.google.com/drive/12CvvlXuLs9ITIDjMSTrWTltLqQwEjvhR?authuser=1#scrollTo=2LmVNId12abb

### Option 2: Local Setup
```bash
git clone https://github.com/SabihaKhanum/indian-legal-llm-qwen.git
cd indian-legal-llm-qwen
pip install -r requirements.txt
python scripts/run_inference.py --question "What is Section 302 of IPC?"
