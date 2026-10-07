# SocialPulse — AI-Powered Social Media Analytics Platform

A modern, high-performance web application designed to analyze cross-platform social media performance, audience sentiment, emotional resonance, and draft post engagement.

---

## 🚀 Key Features

1. **Multi-Platform Analytics Dashboard**
   - Track key metrics: Impressions, Total Engagements (Likes, Comments, Shares), Average & Peak Engagement Rates.
   - Interactive growth trajectories and timeline graphs (Chart.js).
   - Channel comparison across **Instagram, X / Twitter, YouTube, LinkedIn, and TikTok**.
   - Content Format ROI (Carousels vs. Reels vs. Videos vs. Images vs. Text).
   - Trending hashtags ranking and high-impact posts table.
   - Algorithmic detection of **Prime Posting Days and Peak Hours**.

2. **Sentiment & Emotion Intelligence Lab**
   - Natural language polarity classification (Positive, Neutral, Negative).
   - Multi-spectrum emotional analysis (**Joy, Anticipation, Surprise, Frustration, Sadness**).
   - Subjectivity meter (Objective vs. Opinionated content).
   - Automatic keyword & topic extraction.
   - Pre-loaded interactive testing prompts.

3. **Draft Post & Caption Optimizer**
   - Live character, word, and hashtag counter.
   - Platform-aware engagement prediction score (0–100) with letter grading (A+, A, B, C).
   - Rule-based algorithmic feedback for length, hashtag count, and Call-to-Action (CTA) presence.
   - Suggested high-performing trending hashtags that can be appended directly into drafts with one click.

4. **Dataset Upload & Exportable Reports**
   - Drag & drop CSV/JSON file uploader with flexible column mapping.
   - Pre-loaded demo datasets (Global Brand Campaign & B2B Tech SaaS Launch).
   - Downloadable CSV template for formatting custom data exports.
   - Executive summary report with printable PDF styling, CSV export, and structured JSON export.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.12, Flask 3.1
- **Data & Analytics Engine**: Pandas, NumPy, Scikit-Learn
- **Frontend**: HTML5, Tailwind CSS (modern SaaS dark glassmorphism theme), Lucide Icons
- **Data Visualization**: Chart.js (interactive line, bar, doughnut, and polar area charts)

---

## 📦 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📊 CSV Dataset Format

To upload your own data, use the sample template provided at `/api/download-template` or ensure your CSV has any variation of these columns:

| Column Header | Example Values | Description |
|---|---|---|
| `platform` | Instagram, X, LinkedIn, YouTube, TikTok | Social media channel |
| `content` | "Big launch today! Link in bio..." | Post caption or text body |
| `likes` | 4200 | Number of likes / reactions |
| `comments` | 310 | Number of comments / replies |
| `shares` | 540 | Number of retweets / reposts |
| `impressions` | 85000 | Views or total reach |
| `date` | 2024-10-01 10:00:00 | Date and time posted |
| `type` | Carousel, Reel, Video, Image, Text | Content format |
| `author` | @BrandHandle | Creator or page account |

---

## 🔗 Key Endpoints

- `GET /` - Redirects to Dashboard
- `GET /dashboard` - Main analytics dashboard
- `GET /sentiment` - Real-time Sentiment & Emotion Lab
- `GET /optimizer` - Post Engagement Optimizer
- `GET /upload` - CSV/JSON dataset manager
- `GET /reports` - Executive audit report view
- `POST /api/analyze-sentiment` - Live text sentiment endpoint
- `POST /api/optimize-post` - Draft engagement scoring endpoint
- `POST /api/load-preset` - Preset dataset loader
- `GET /api/export-csv` - Export cleaned dataset as CSV
- `GET /api/export-report-json` - Export structured JSON report
- `GET /api/download-template` - Download template CSV
