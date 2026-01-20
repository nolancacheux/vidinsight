# VidInsight

AI-powered YouTube comment analysis tool with sentiment detection, topic modeling, and aspect-based analysis.

## Features

- **Comment Extraction**: Fetch comments from any YouTube video using yt-dlp
- **Sentiment Analysis**: BERT-powered multilingual sentiment classification (positive/negative/neutral/suggestion)
- **Topic Modeling**: BERTopic clustering to identify key discussion themes
- **Aspect-Based Sentiment Analysis (ABSA)**: Zero-shot analysis across 5 video dimensions:
  - Content (information quality, explanations)
  - Audio (sound quality, voice clarity)
  - Production (editing, visual quality)
  - Pacing (video length, rhythm)
  - Presenter (personality, delivery)
- **Health Scoring**: 0-100 channel health score with trend tracking
- **Smart Recommendations**: Prioritized action items based on viewer feedback
- **Interactive Dashboard**: Real-time analysis progress with Recharts visualizations

## Quick Start

### Prerequisites

- Node.js 20+
- pnpm
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Frontend Setup

```bash
# Install dependencies
pnpm install

# Run development server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Backend Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create venv
uv sync

# Run API server
uv run uvicorn api.main:app --reload --port 8000
```

API available at [http://localhost:8000](http://localhost:8000)

## Tech Stack

- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS v4, shadcn/ui, Recharts
- **Backend**: FastAPI, Python 3.11+, yt-dlp
- **Database**: SQLite with SQLAlchemy
- **AI/ML**:
  - `nlptown/bert-base-multilingual-uncased-sentiment` (sentiment analysis)
  - `facebook/bart-large-mnli` (zero-shot aspect detection)
  - BERTopic (topic modeling)

## Project Structure

```
vidinsight/
├── src/                    # Next.js frontend
│   ├── components/         # React components
│   │   ├── charts/         # Recharts visualizations
│   │   ├── results/        # Topic cards, ABSA section
│   │   └── ui/             # shadcn components
│   ├── hooks/              # useAnalysis hook
│   └── types/              # TypeScript interfaces
├── api/                    # FastAPI backend
│   ├── services/           # ML services
│   │   ├── absa.py         # Aspect-based sentiment
│   │   ├── insights.py     # Recommendations engine
│   │   ├── sentiment.py    # BERT sentiment
│   │   └── topics.py       # Topic modeling
│   ├── routers/            # API endpoints
│   └── db/                 # SQLAlchemy models
└── tests/                  # pytest tests
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analysis/analyze` | Start analysis (SSE stream) |
| GET | `/api/analysis/result/{id}` | Get analysis with ABSA data |
| GET | `/api/analysis/history` | List past analyses |
| DELETE | `/api/analysis/history/{id}` | Delete an analysis |

## ABSA Response Example

```json
{
  "absa": {
    "health": {
      "overall_score": 72,
      "strengths": ["content", "presenter"],
      "weaknesses": ["audio"]
    },
    "recommendations": [
      {
        "aspect": "audio",
        "priority": "high",
        "title": "Improve Audio Quality",
        "action_items": ["Check microphone setup"]
      }
    ]
  }
}
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## License

MIT
