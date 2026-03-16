# VisionGuard — Deployment Guide

## Run Locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Deploy FREE on Render.com (Public Website)

### Step 1 — Get Free Gemini API Key
1. Go to https://aistudio.google.com
2. Sign in with Google
3. Click "Get API Key" → "Create API key"
4. Copy the key (starts with AIza...)

### Step 2 — Upload to GitHub
1. Go to github.com → New Repository → name it "visionguard"
2. Upload all project files
3. Click "Commit changes"

### Step 3 — Deploy on Render
1. Go to https://render.com → Sign up free
2. Click "New +" → "Web Service"
3. Connect your GitHub → select "visionguard"
4. Fill in:
   - Name: visionguard
   - Runtime: Python 3
   - Build Command: pip install -r requirements.txt
   - Start Command: gunicorn app:app --workers 2 --threads 4 --timeout 60
5. Click "Advanced" → "Add Environment Variable"
   - Key: GEMINI_API_KEY
   - Value: your AIza... key
6. Click "Create Web Service"
7. Wait 2-3 minutes → your site is live!

### Your public URL will be:
https://visionguard.onrender.com

### Get on Google Search
1. Go to https://search.google.com/search-console
2. Add your Render URL
3. Submit sitemap
4. Google will index it within a few days

## Project Structure
```
visionguard/
├── app.py              # Flask backend (Python)
├── requirements.txt    # Python dependencies  
├── Procfile            # Render deployment config
├── .env.example        # Environment variables template
└── templates/
    └── index.html      # Frontend (HTML/CSS/JS)
```

## Technologies
- Python 3 + Flask — web server
- OpenCV — camera streaming & video processing
- Google Gemini 1.5 Flash — AI vision analysis
- PIL/Pillow — image processing
- smtplib — email alerts
- Web Audio API — siren sound
- Render.com — free cloud hosting
