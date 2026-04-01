# dashboard/src — React + TypeScript Frontend

Vite + React + TypeScript SPA. Visualizes predictions, bankroll, performance, and model analytics.

## STRUCTURE

```
dashboard/src/
├── main.tsx           # App entry, renders <App />
├── App.tsx            # Root component, routing
├── Layout.tsx         # Page layout shell
├── api/               # API client layer (calls FastAPI backend)
├── pages/             # Route-level page components
├── components/        # Shared UI components
│   └── charts/        # Chart components (Plotly)
├── charts/            # Top-level chart configurations
├── hooks/             # Custom React hooks
├── assets/            # Static assets
├── App.css            # App styles
├── Layout.css         # Layout styles
└── index.css          # Global styles
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add a new page | `pages/` — add component + route in App.tsx |
| Add a new chart | `components/charts/` or `charts/` |
| Change API endpoint | `api/` — all backend calls centralized here |
| Change layout | `Layout.tsx`, `Layout.css` |
| Add shared component | `components/` |

## BACKEND API

- FastAPI backend at `src/dashboard/` (Python)
- Routes defined in `src/dashboard/routes/`
- Schemas in `src/dashboard/schemas.py`
- Run backend: `python -m uvicorn src.dashboard.main:app --host 127.0.0.1 --port 8000`

## COMMANDS

```bash
cd dashboard
npm run dev      # Dev server on :5173
npm run build    # tsc -b && vite build
npm run lint     # eslint .
```

## NOTES

- API calls go to `http://localhost:8000/api/...` (FastAPI backend)
- E2E tests in `tests/integration/dashboard_e2e/` use Playwright
- E2E conftest starts both backend (uvicorn) and frontend (npm run dev) automatically
