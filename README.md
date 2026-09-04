# ocr-openvino-int8-cpu
CPU-only VLM/OCR application using OpenVINO INT8, FastAPI and React for local document and image text extraction.
A local AI-powered OCR application for extracting text from images and documents using CPU-based inference.

The application provides a simple web interface where users can upload an image, process it locally, view the extracted text, and save the result as a `.txt` file.

## Project Overview

This project demonstrates how an OCR application can be built and deployed locally using:

- Python
- FastAPI
- OpenVINO INT8
- React
- Vite
- CPU-based inference

The system is designed for local document and image text extraction without requiring cloud-based OCR services.

## Features

- Upload images for OCR processing
- Extract visible text from uploaded images
- CPU-only inference
- INT8 optimized inference
- FastAPI backend
- React + Vite frontend
- Real-time backend health status
- Preview uploaded images
- Display extracted OCR text
- Download extracted text as a `.txt` file
- Local processing

## Technologies Used

### Backend

- Python
- FastAPI
- Uvicorn
- Pillow
- NumPy
- Transformers
- OpenVINO
- Optimum Intel
- PyTorch

### Frontend

- React
- Vite
- JavaScript
- CSS
- Lucide React

## Project Structure

```text
ocr-openvino-int8-cpu/
│
├── backend/
│   ├── glm_cpu.py
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── App.css
│       ├── components/
│       │   ├── Header.jsx
│       │   ├── HealthStatus.jsx
│       │   ├── UploadArea.jsx
│       │   ├── ImagePreview.jsx
│       │   ├── OCRResult.jsx
│       │   └── ActionButtons.jsx
│       └── services/
│           └── ocrApi.js
│
├── screenshots/
│
├── README.md
└── .gitignore

Application Architecture
                ┌─────────────────────┐
                │    React Frontend   │
                │      Vite UI        │
                └──────────┬──────────┘
                           │
                           │ HTTP API
                           ▼
                ┌─────────────────────┐
                │    FastAPI Backend  │
                │      Python         │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  OCR Inference      │
                │  OpenVINO INT8      │
                │  CPU Processing     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Extracted Text     │
                │      Result         │
                └─────────────────────┘
How It Works
User opens the React web application.
The user uploads an image.
The frontend sends the image to the FastAPI backend.
The backend processes the image.
OCR inference runs locally using CPU-based optimized inference.
Extracted text is returned to the frontend.
The text is displayed in the application.
The user can download the extracted text as a .txt file.

