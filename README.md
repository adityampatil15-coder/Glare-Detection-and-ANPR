# Glare-Detection-and-ANPR
This is a project I worked on that captures the numberplates of vehicles that has they high beams on and blinded other road users. 

Glare Detection & Automatic Number Plate Recognition (ANPR)
A real-time vehicle number plate detection and recognition system built to operate reliably under adverse lighting conditions including nighttime glare and high-beam headlights. Built on a Raspberry Pi 4B with a custom-trained YOLOv8 model and EasyOCR.

Overview
Standard ANPR systems fail under glare conditions common in Indian traffic — oncoming high beams, streetlight reflections, and overexposed plate surfaces. This project tackles that problem with a two-stage pipeline:

Glare Detection — A multi-zone OpenCV pipeline detects high-beam glare and gates OCR processing to avoid noisy reads
Number Plate Detection — A YOLOv8 model trained on a custom dataset of 800+ annotated Indian vehicle images detects plate regions
OCR — EasyOCR extracts the plate text from the detected region

Project Structure
anpr-glare-detection/
├── glare_test.py               # Glare detection pipeline (multi-zone OpenCV)
├── OCRwithGlareandANPR.py      # Main pipeline — YOLO + Glare gating + EasyOCR
├── trainplate.py               # YOLOv8 training and validation script
├── data.yaml                   # Dataset config (update paths for your system)
├── requirements.txt            # Python dependencies

Install dependencies
bash 
pip install -r requirements.txt

Donwload model weights 
Use the .pt and .onnx file in train4 folder

Usage
Run the full ANPR pipeline
bashpython OCRwithGlareandANPR.py
Test glare detection only
bashpython glare_test.py
Train the model on your own dataset

Update paths in data.yaml
Run:

bashpython trainplate.py

Dataset
The model was trained on a proprietary dataset of 800+ custom-annotated Indian vehicle images, annotated using Label Studio. The dataset is not included in this repository.

Technologies Used

YOLOv8 (Ultralytics)
EasyOCR
OpenCV
Python, NumPy
Raspberry Pi 4B + Camera Module v2

Author
Aditya Patil
B.Tech ECE (AI-ML) — MIT World Peace University, Pune
GitHub: @adityampatil15-coder
