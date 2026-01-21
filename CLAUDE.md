
# Instructions

These instructions must be followed for all work in this repository.

## Core Rules
- UPDATE this CLAUDE.md file with any new features, endpoints, or changes! MUST BE DONE
- Small, clean commits with best practices
- NEVER mention Claude in commits, branches, or PRs
- Always keep code ultra clean

## General Preferences

- Never use emojis or smileys in any output
- Always communicate in English


### Commits
- Commit small, incremental changes frequently
- All CI checks must pass before merging
- PRs must be clean
- **Use Conventional Commits format:**
  ```
  <type>: <description>
  ```
  **Types:**
  - `feat:` - New feature
  - `fix:` - Bug fix
  - `docs:` - Documentation only
  - `style:` - Formatting, no code change
  - `refactor:` - Code restructuring, no behavior change
  - `test:` - Adding/updating tests
  - `chore:` - Maintenance, dependencies, CI
  - `perf:` - Performance improvement

  **Examples:**
  - `feat: add user authentication`
  - `fix: resolve login redirect issue`
  - `chore: update dependencies`
  - `refactor: simplify error handling`



# AI-Video-Comment-Analyzer

Portfolio-quality YouTube comment analysis tool with ML-powered sentiment detection and topic modeling.

## Tech Stack

- **Frontend**: Next.js 15 + React 19 + Tailwind v4 + shadcn/ui + Recharts
- **Backend**: FastAPI + yt-dlp + Transformers (BERT sentiment, BART zero-shot)
- **Database**: SQLite
- **Package Manager**: pnpm (frontend), uv (backend)
- **GPU**: NVIDIA GPU with CUDA recommended for fast ML inference (works on CPU but slower)

## Running the Application

```bash
# Terminal 1 - Frontend
pnpm dev

# Terminal 2 - Backend
uv run uvicorn api.main:app --reload --port 8000
```

## Project Structure

```
src/
├── app/
│   ├── page.tsx              # Main dashboard with 3 states (input/analyzing/results)
│   ├── layout.tsx            # Root layout with Geist fonts
│   └── globals.css           # Tailwind v4 + custom CSS variables
├── components/
│   ├── layout/
│   │   ├── sidebar.tsx       # Resizable sidebar with history + delete confirmation
│   │   ├── video-header.tsx  # Video info bar (thumbnail, title, stats)
│   │   └── dashboard-grid.tsx # Grid layout components + stat cards
│   ├── charts/
│   │   ├── sentiment-pie.tsx      # Donut chart (Love/Dislike/Suggestions/Neutral)
│   │   ├── engagement-bar.tsx     # Horizontal bar chart (likes by sentiment)
│   │   ├── topic-bubble.tsx       # Scatter plot (size=mentions, color=sentiment)
│   │   └── confidence-histogram.tsx # ML confidence distribution
│   ├── analysis/
│   │   ├── ml-info-panel.tsx      # ML Pipeline panel (model, speed, tokens)
│   │   └── progress-terminal.tsx  # Clean stepper progress with 7 stages + cancel button
│   ├── results/
│   │   ├── topic-card.tsx         # Topic cards with priority badges
│   │   ├── comment-card.tsx       # Comments with word highlighting + like emphasis
│   │   └── absa-section.tsx       # ABSA visualization (health score, aspects, recommendations)
│   ├── search-results.tsx    # Search results dropdown for video search
│   └── ui/                        # shadcn components (includes AlertDialog)
├── hooks/
│   └── useAnalysis.ts        # Analysis state + real-time ML metrics + cancel support
├── lib/
│   ├── api.ts                # API client with SSE streaming + AbortController
│   ├── utils.ts              # cn() utility
│   └── highlight-words.ts    # Sentiment word detection
└── types/
    └── index.ts              # TypeScript interfaces (includes ML metrics)

api/
├── main.py                   # FastAPI app with CORS (ports 3000, 3001)
├── models/
│   ├── schemas.py            # Pydantic models (includes MLMetadata)
│   └── __init__.py
├── routers/
│   └── analysis.py           # SSE streaming with real ML metrics + delete endpoint
├── services/
│   ├── youtube.py            # yt-dlp extraction (max 100 top comments)
│   ├── sentiment.py          # BERT sentiment with streaming progress
│   ├── topics.py             # Topic modeling
│   ├── absa.py               # Aspect-Based Sentiment Analysis (BART zero-shot)
│   └── insights.py           # Recommendations and health scoring
└── db/
    ├── models.py             # SQLAlchemy models
    └── __init__.py
```

## Dashboard States

### 1. Input State
- Centered URL input with validation and **video search**
- Type search queries (3+ chars) to search YouTube directly via yt-dlp
- Auto-search with 500ms debounce, results dropdown below input
- Inline feature indicators (Sentiment Analysis, Topic Detection, Actionable Insights)
- History in sidebar (resizable)

### 2. Analyzing State
- Clean stepper progress UI with 7 stages:
  1. Validating - Checking URL format and availability
  2. Fetching Video - Getting video information
  3. Extracting Comments - Downloading top 100 comments
  4. Sentiment Analysis - BERT classifies each comment (with progress bar)
  5. Aspect Analysis - ABSA across 5 aspects (content, audio, production, pacing, presenter)
  6. Topic Detection - Grouping comments by theme
  7. Generating Insights - Creating recommendations
- **Cancel button** - Stop analysis at any time
- Real-time ML metrics from backend:
  - Processing speed (comments/sec)
  - Tokens processed
  - Batch progress
  - Elapsed time
- Live metrics (comments found, analyzed)

### 3. Results State
- Video header with thumbnail, title, channel, date, comment count
- 4 stat cards: Love (emerald), Dislike (rose), Suggestions (blue), Neutral (slate)
- 4 charts in a row:
  - Sentiment Distribution (pie)
  - Engagement by Sentiment (bar)
  - Topic Analysis (bubble)
  - ML Confidence (histogram)
- Topics section with clickable cards (filters comments)
- Sample Comments sorted by likes (most engaged first):
  - Tabs (All/Love/Dislike/Suggestions)
  - Word highlighting (positive=green, negative=red, suggestions=blue)
  - Confidence badges
  - Topic labels
  - Like emphasis (gold for 100+, darker for 10+)

## Sidebar Features

- **Resizable sidebar** - Drag right edge to resize (56px to 400px)
- **Auto-collapse** - Automatically collapses when dragged small enough
- **Collapsed mode** - Shows only thumbnails, click to select analysis
- **Expanded mode** - Shows thumbnail, title, time ago, and delete button
- **Delete button (trash icon)** - Left side of each history item
- **Delete confirmation** - AlertDialog modal with Cancel/Delete buttons

## Color Palette

| Element | Color | Hex |
|---------|-------|-----|
| Background | Light gray | #FAFAFA |
| Cards | White | #FFFFFF |
| Love/Positive | Emerald | #10B981 |
| Dislike/Negative | Rose | #F43F5E |
| Suggestions | Blue | #3B82F6 |
| Neutral | Slate | #64748B |
| Accent | Indigo | #6366F1 |
| High Likes | Amber | #D97706 |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analysis/analyze` | SSE stream with real ML metrics |
| GET | `/api/analysis/result/{id}` | Get complete analysis with ML metadata |
| GET | `/api/analysis/history` | List recent analyses |
| DELETE | `/api/analysis/history/{id}` | Delete an analysis |
| GET | `/api/analysis/video/{id}/latest` | Get latest analysis for video |
| GET | `/api/analysis/search` | Search YouTube videos by query (params: q, limit)

## ML Pipeline

### Sentiment Analysis
- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Batch size**: 32 comments
- **Max comments**: 100 (top by engagement)
- **Streaming progress**: Real-time updates every 10 comments

### ABSA (Aspect-Based Sentiment Analysis)
- **Model**: `facebook/bart-large-mnli` (zero-shot classification)
- **Aspects**: content, audio, production, pacing, presenter
- **Health Score**: 0-100 overall channel health
- **Recommendations**: Prioritized action items
- **Performance**: GPU (CUDA) recommended - CPU is slow (~2-3 min for 100 comments)

### SSE Metrics
- `ml_batch` / `ml_total_batches` - Current batch progress
- `ml_processed` / `ml_total` - Comments processed
- `ml_speed` - Comments per second
- `ml_tokens` - Tokens processed
- `ml_batch_time_ms` - Time per batch
- `ml_elapsed_seconds` - Total elapsed time

## Key Features

- **Single-page dashboard**: Full 1920x1080 layout, no scrolling
- **YouTube video search**: Type queries to search YouTube directly via yt-dlp
- **Real ML Pipeline**: BERT sentiment + BART zero-shot aspect detection
- **Cancel analysis**: Stop pipeline at any time via UI button
- **Reload protection**: beforeunload warning prevents accidental page leave during analysis
- **Resizable sidebar**: Drag to resize, auto-collapse on small width
- **Delete with confirmation**: AlertDialog modal for safe deletion
- **ABSA Analysis**: Zero-shot aspect detection across 5 video dimensions
- **Health Scoring**: 0-100 channel health with strengths/weaknesses
- **Smart Recommendations**: Prioritized action items with evidence
- **Interactive charts**: 4 Recharts visualizations
- **Topic filtering**: Click topics to filter comments
- **Word highlighting**: Sentiment words colored in comments
- **Like-sorted comments**: Most engaged comments shown first
- **Backend logging**: INFO-level logging for all services (YouTube, sentiment, ABSA, topics)

## Development Commands

### Running the App
```bash
# Frontend (Terminal 1)
pnpm dev

# Backend (Terminal 2)
uv run uvicorn api.main:app --reload --port 8000
```

### Tests
```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=api --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_youtube.py -v
uv run pytest tests/test_sentiment.py -v
uv run pytest tests/test_absa.py -v
```

### Code Quality (Ruff)
```bash
# Lint check
uv run ruff check api/ tests/

# Auto-fix lint issues
uv run ruff check api/ tests/ --fix

# Format code
uv run ruff format api/ tests/

# Check formatting only (for CI)
uv run ruff format --check api/ tests/
```

### CI Pipeline
GitHub Actions (`.github/workflows/ci.yml`) runs:
1. **Lint job**: `ruff check` + `ruff format --check`
2. **Test job**: `pytest` with 75% coverage threshold

## Quality Checklist

- [x] No scrolling on 1920x1080
- [x] All charts render correctly
- [x] ML pipeline with real metrics during analysis
- [x] Clean stepper progress animation (7 stages)
- [x] Cancel button stops analysis
- [x] Reload warning during analysis (beforeunload)
- [x] Comments show highlighted sentiment words
- [x] Sidebar resizable via drag
- [x] Delete with AlertDialog confirmation
- [x] History loads previous analyses
- [x] Topic click filters comments
- [x] Confidence scores displayed
- [x] Comments sorted by likes
- [x] ABSA health score displays correctly
- [x] Aspect cards show sentiment breakdown
- [x] Recommendations sorted by priority
- [x] YouTube video search with debounce
- [x] Search results dropdown scrollable (max-h-80)
- [x] Topic display shows helpful message for few comments
- [x] Backend logging enabled (INFO level)
- [x] 215 unit tests passing
- [x] 75% code coverage

