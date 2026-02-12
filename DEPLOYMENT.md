# Deployment Guide

## Architecture

```
Frontend (React)  ──→  Vercel / Netlify  (Static Site)
Backend  (FastAPI) ──→  Render           (Python Web Service)
```

The frontend calls the backend API via the `VITE_API_URL` environment variable.

---

## Step 1: Deploy Backend on Render (Free)

1. Go to [render.com](https://render.com) → Sign in with GitHub
2. Click **"New" → "Web Service"**
3. Connect your repo: `NarendraThanda/Context-Aware-Emoji-Prediction-from-Text-Using-NLP-Techniques`
4. Configure:
   | Setting         | Value                                      |
   |-----------------|--------------------------------------------|
   | **Name**        | `emoji-predictor-api`                      |
   | **Root Directory** | `backend`                               |
   | **Runtime**     | `Python`                                   |
   | **Build Command** | `pip install -r requirements.txt`         |
   | **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
5. Click **"Create Web Service"**
6. Wait for deploy → Copy your URL (e.g., `https://emoji-predictor-api.onrender.com`)

---

## Step 2A: Deploy Frontend on Vercel

1. Go to [vercel.com](https://vercel.com) → Sign in with GitHub
2. Click **"Import Project"** → Select your repo
3. Configure:
   | Setting              | Value       |
   |----------------------|-------------|
   | **Root Directory**   | `frontend`  |
   | **Framework Preset** | `Vite`      |
   | **Build Command**    | `npm run build` |
   | **Output Directory** | `dist`      |
4. Add **Environment Variable**:
   | Key            | Value                                           |
   |----------------|-------------------------------------------------|
   | `VITE_API_URL` | Your Render URL (e.g., `https://emoji-predictor-api.onrender.com`) |
5. Click **"Deploy"**

---

## Step 2B: Deploy Frontend on Netlify

1. Go to [netlify.com](https://netlify.com) → Sign in with GitHub
2. Click **"Add new site" → "Import an existing project"**
3. Connect your GitHub repo
4. Configure:
   | Setting              | Value       |
   |----------------------|-------------|
   | **Base directory**   | `frontend`  |
   | **Build command**    | `npm run build` |
   | **Publish directory** | `frontend/dist` |
5. Add **Environment Variable** (Site settings → Environment variables):
   | Key            | Value                                           |
   |----------------|-------------------------------------------------|
   | `VITE_API_URL` | Your Render URL (e.g., `https://emoji-predictor-api.onrender.com`) |
6. Click **"Deploy site"**

---

## Run Locally

```bash
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
python main.py
```

Open http://localhost:8000

---

## Environment Variables

| Variable       | Where     | Description                          |
|----------------|-----------|--------------------------------------|
| `VITE_API_URL` | Frontend  | Backend API URL (empty for local)    |

---

## Project Structure

```
├── main.py                 # Full-stack launcher (python main.py)
├── requirements.txt        # Python dependencies
├── netlify.toml            # Netlify deployment config
│
├── frontend/               # React + Vite frontend
│   ├── vercel.json         # Vercel deployment config
│   ├── public/_redirects   # Netlify SPA routing
│   ├── src/
│   │   ├── main.jsx        # React entry point
│   │   ├── App.jsx         # Router setup
│   │   ├── index.css       # Glassmorphism design system
│   │   ├── components/
│   │   │   └── Navbar.jsx
│   │   └── pages/
│   │       ├── HomePage.jsx
│   │       └── PredictorPage.jsx
│   └── dist/               # Built output (auto-generated)
│
└── backend/                # FastAPI + NLP backend
    ├── main.py             # API server
    ├── nlp_pipeline.py     # Text preprocessing & sentiment
    ├── model.py            # LSTM/GRU/CNN models
    ├── full_emoji.csv      # Emoji dataset (1816 emojis)
    └── requirements.txt    # Python dependencies
```
