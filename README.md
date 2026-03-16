# 🔱 TRIYAMBAKA
### The Three-Eyed AI Surveillance System

> *Named after the three-eyed form of Lord Shiva — the all-seeing divine eye that misses nothing.*

TRIYAMBAKA is a free, AI-powered real-time violence detection system that uses Google Gemini Vision AI to analyze live camera feeds, uploaded videos, and images for signs of violence or threatening behavior.

## Features
- 📷 Live webcam / RTSP CCTV camera monitoring
- 📁 Upload & analyze recorded videos and images
- 🤖 Google Gemini 1.5 Flash Vision AI detection
- 🚨 Siren alarm on violence detection
- 📧 Email alerts with snapshot via SMTP
- ⟳ Auto-analysis mode
- 100% Free to use

## Technologies
- Python 3 + Flask
- OpenCV (video streaming)
- Google Gemini 1.5 Flash (AI vision)
- SMTP (email alerts)
- Web Audio API (siren)

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000

## Deploy on Render.com
1. Fork this repository
2. Go to render.com → New Web Service
3. Connect this repo
4. Add environment variable: `GEMINI_API_KEY=your_key`
5. Build: `pip install -r requirements.txt`
6. Start: `gunicorn app:app --workers 2 --threads 4 --timeout 60`

## About the Name
TRIYAMBAKA (त्र्यम्बक) is a Sanskrit name for Lord Shiva meaning "The Three-Eyed One". Just as Shiva's third eye sees all threats and destroys evil, TRIYAMBAKA uses AI vision to detect violence and protect people.
- OpenCV — camera streaming & video processing
- Google Gemini 1.5 Flash — AI vision analysis
- PIL/Pillow — image processing
- smtplib — email alerts
- Web Audio API — siren sound
- Render.com — free cloud hosting
