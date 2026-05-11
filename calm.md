# Claude Code — Build Prompt: CALM Architecture Platform

You are building the **CALM Architecture Platform** — a full enterprise multi-page web application for creating, governing, discovering, and consuming reusable software architectural patterns using the FINOS CALM standard.

Read every section before writing a single line of code. Understand the full system, then build it in the order defined in Section 14.

---

## 1. Non-negotiable constraints

1. `@finos/calm` npm package is the **only** CALM schema authority. No custom CALM schemas anywhere in the codebase.
2. **Multi-page application.** Every route in Section 6 is a distinct TanStack Router page with its own typed loader function that fetches data before rendering.
3. **No Nginx.** FastAPI gateway serves the React build via Starlette `StaticFiles`.
4. **No PgBouncer.** FastAPI services connect directly to PostgreSQL via asyncpg.
5. **Docker is for stores only.** FalkorDB, Elasticsearch, and PostgreSQL run in Docker. All Python and Node.js processes run on the host system.
6. **No Helm.** Kubernetes manifests are raw YAML with Kustomize overlays.
7. **No observability stack.** Health exposed via `/healthz` endpoints; aggregated by gateway; displayed in React UI.
8. **No collaboration or social features.** No comments, reactions, mentions, email, or activity feeds.
9. **No subscription tiers, no Vercel, no Cloudflare.**
10. **LDAP is the sole identity mechanism.** No OAuth, no Keycloak, no OIDC.
11. **Three stores, three data domains.** No CALM JSON ever in PostgreSQL. No governance state in FalkorDB.
12. **UI/UX design system in Section 4 is mandatory.** Every component must follow these rules.

---

## 2. Repository structure

```
calm-platform/
├── Makefile                           # start/stop targets for local dev
├── docker-compose.yml                 # stores only: falkordb, elasticsearch, postgres
├── .env.example
├── README.md
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── router.tsx
│       ├── lib/
│       │   ├── calm-schema.ts         # parses @finos/calm JSON Schema → palette types
│       │   ├── api.ts                 # typed fetch wrapper
│       │   ├── auth.ts                # JWT cookie helpers
│       │   ├── query-client.ts
│       │   └── design-tokens.ts       # re-exports CSS custom property names as TS constants
│       ├── stores/
│       │   └── canvas.ts              # Zustand canvas state
│       ├── components/
│       │   ├── ui/                    # Radix UI + Tailwind wrappers
│       │   │   ├── button.tsx
│       │   │   ├── badge.tsx
│       │   │   ├── card.tsx
│       │   │   ├── dialog.tsx
│       │   │   ├── dropdown-menu.tsx
│       │   │   ├── input.tsx
│       │   │   ├── label.tsx
│       │   │   ├── select.tsx
│       │   │   ├── tabs.tsx
│       │   │   ├── toast.tsx
│       │   │   ├── tooltip.tsx
│       │   │   ├── separator.tsx
│       │   │   ├── skeleton.tsx
│       │   │   ├── scroll-area.tsx
│       │   │   └── alert-dialog.tsx
│       │   ├── canvas/
│       │   │   ├── CalmCanvas.tsx     # React Flow canvas wrapper
│       │   │   ├── NodePalette.tsx    # draggable node type palette
│       │   │   ├── CalmNode.tsx       # custom React Flow node
│       │   │   └── CalmEdge.tsx       # custom React Flow edge
│       │   ├── layout/
│       │   │   ├── AppShell.tsx       # sidebar + main area wrapper
│       │   │   ├── Sidebar.tsx        # persistent nav sidebar
│       │   │   ├── Header.tsx         # page header with breadcrumb
│       │   │   └── Breadcrumb.tsx
│       │   └── shared/
│       │       ├── PatternCard.tsx
│       │       ├── StatusBadge.tsx
│       │       ├── QualityBadge.tsx
│       │       ├── DriftBadge.tsx
│       │       ├── EmptyState.tsx
│       │       ├── ErrorState.tsx
│       │       └── LoadingSkeleton.tsx
│       └── pages/                     # one file per route
│           ├── index.tsx              # /
│           ├── login.tsx
│           ├── dashboard.tsx
│           ├── search.tsx
│           ├── profile.tsx
│           ├── docs/
│           ├── patterns/
│           ├── marketplace/
│           ├── workspace/
│           ├── governance/
│           ├── developer/
│           ├── analytics/
│           └── admin/
│
├── services/
│   ├── gateway/
│   │   ├── Dockerfile                 # multi-stage: React build → slim Python
│   │   ├── requirements.txt
│   │   └── main.py
│   ├── pattern-svc/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── main.py
│   │   ├── routes/
│   │   ├── db/falkordb.py
│   │   ├── db/queries.py
│   │   ├── es/sync.py
│   │   └── quality.py
│   ├── calm-validator/
│   │   ├── Dockerfile
│   │   ├── package.json               # @finos/calm + express
│   │   └── server.js
│   ├── discovery-svc/
│   ├── governance-svc/
│   ├── lineage-svc/
│   ├── auth-svc/
│   └── export-svc/
│
├── k8s/
│   ├── namespace.yaml
│   ├── kustomization.yaml
│   ├── stores/
│   │   ├── falkordb-statefulset.yaml
│   │   ├── elasticsearch-statefulset.yaml
│   │   └── postgres-statefulset.yaml
│   ├── services/
│   │   ├── gateway-deployment.yaml
│   │   ├── pattern-svc-deployment.yaml    # calm-validator as sidecar
│   │   ├── discovery-svc-deployment.yaml
│   │   ├── governance-svc-deployment.yaml
│   │   ├── lineage-svc-deployment.yaml
│   │   ├── auth-svc-deployment.yaml
│   │   └── export-svc-deployment.yaml
│   ├── config/
│   │   ├── configmap.yaml
│   │   ├── secret.yaml
│   │   └── hpa.yaml
│   └── overlays/
│       ├── dev/
│       └── prod/
│
└── scripts/
    ├── seed-falkordb.cypher
    ├── seed-postgres.sql
    └── create-es-indices.sh
```

---

## 3. Package dependencies

### Frontend (`frontend/package.json`)
```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "@tanstack/react-router": "^1.48.0",
    "@tanstack/react-query": "^5.56.0",
    "@tanstack/react-table": "^8.20.0",
    "@xyflow/react": "^12.3.0",
    "@radix-ui/react-dialog": "^1.1.0",
    "@radix-ui/react-dropdown-menu": "^2.1.0",
    "@radix-ui/react-tabs": "^1.1.0",
    "@radix-ui/react-select": "^2.1.0",
    "@radix-ui/react-tooltip": "^1.1.0",
    "@radix-ui/react-popover": "^1.1.0",
    "@radix-ui/react-toast": "^1.2.0",
    "@radix-ui/react-separator": "^1.1.0",
    "@radix-ui/react-alert-dialog": "^1.1.0",
    "@radix-ui/react-scroll-area": "^1.2.0",
    "@radix-ui/react-label": "^2.1.0",
    "@radix-ui/react-checkbox": "^1.1.0",
    "@radix-ui/react-switch": "^1.1.0",
    "@radix-ui/react-progress": "^1.1.0",
    "@monaco-editor/react": "^4.6.0",
    "zustand": "^5.0.0",
    "@finos/calm": "latest",
    "recharts": "^2.12.0",
    "lucide-react": "^0.439.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.5.0",
    "date-fns": "^4.1.0",
    "zod": "^3.23.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "tailwindcss": "^3.4.0",
    "autoprefixer": "^10.4.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0"
  }
}
```

### Python services (shared base `requirements.txt`)
```
fastapi==0.115.0
uvicorn[standard]==0.31.0
pydantic==2.9.0
python-jose[cryptography]==3.3.0
httpx==0.27.0
```

### Service-specific additions
```
pattern-svc:   falkordb==1.0.0  elasticsearch[async]==8.15.0  pyyaml==6.0.2  slowapi==0.1.9
governance-svc: asyncpg==0.29.0  sqlalchemy[asyncio]==2.0.35
auth-svc:      asyncpg==0.29.0  sqlalchemy[asyncio]==2.0.35  ldap3==2.9.1  bcrypt==4.2.0
discovery-svc: elasticsearch[async]==8.15.0  sentence-transformers==3.1.0
lineage-svc:   falkordb==1.0.0
export-svc:    falkordb==1.0.0  reportlab==4.2.0  pyyaml==6.0.2  jinja2==3.1.4
```

### calm-validator (`package.json`)
```json
{ "dependencies": { "@finos/calm": "latest", "express": "^4.21.0" } }
```

---

## 4. UI/UX design system — mandatory rules

Every component, every page must follow these rules. No exceptions.

### 4.1 Colour — semantic tokens only

Define all colours as CSS custom properties in `index.css` and map to Tailwind tokens in `tailwind.config.ts`. **No raw hex in component files.**

```css
/* index.css — light theme */
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;    /* slate-50 */
  --bg-tertiary: #f1f5f9;     /* slate-100 */
  --border: #e2e8f0;          /* slate-200 */
  --border-strong: #94a3b8;   /* slate-400 */
  --text-primary: #0f172a;    /* slate-900 */
  --text-secondary: #475569;  /* slate-600 */
  --text-tertiary: #94a3b8;   /* slate-400 */
  --brand: #4f46e5;           /* indigo-600 */
  --brand-hover: #4338ca;     /* indigo-700 */
  --brand-muted: #eef2ff;     /* indigo-50 */
  --status-draft: #f59e0b;    /* amber-500 */
  --status-review: #3b82f6;   /* blue-500 */
  --status-published: #10b981;/* emerald-500 */
  --status-deprecated: #ef4444;/* red-500 */
  --error: #ef4444;
  --warning: #f59e0b;
  --success: #10b981;
  --info: #3b82f6;
}

/* dark theme */
.dark {
  --bg-primary: #09090b;      /* zinc-950 */
  --bg-secondary: #18181b;    /* zinc-900 */
  --bg-tertiary: #27272a;     /* zinc-800 */
  --border: #27272a;          /* zinc-800 */
  --border-strong: #52525b;   /* zinc-600 */
  --text-primary: #f4f4f5;    /* zinc-100 */
  --text-secondary: #a1a1aa;  /* zinc-400 */
  --text-tertiary: #52525b;   /* zinc-600 */
  --brand: #818cf8;           /* indigo-400 */
  --brand-hover: #a5b4fc;     /* indigo-300 */
  --brand-muted: #1e1b4b;     /* indigo-950 */
}
```

Verify all foreground/background pairs: body text ≥ 4.5:1, large text and UI components ≥ 3:1. Both themes independently.

### 4.2 Typography

```css
/* Base */
body { font-family: 'Inter', sans-serif; font-size: 14px; line-height: 1.6; }
code, pre, .monaco-editor { font-family: 'JetBrains Mono', monospace; font-size: 13px; }

/* Scale: 11 / 12 / 13 / 14 / 16 / 18 / 20 / 24 / 32 */
/* Never below 11px for any visible text */
```

Load Inter and JetBrains Mono via `@fontsource` packages or Google Fonts preload in `index.html`.

### 4.3 Spacing — 4pt grid

Use Tailwind's spacing scale (multiples of 4px). Reference values:
- `gap-1` = 4px (icon-to-text), `gap-2` = 8px (related elements), `p-4` = 16px (card padding), `gap-6` = 24px (section gaps), `gap-8` = 32px (major sections).

### 4.4 App shell

```tsx
// AppShell.tsx — persistent sidebar + main content area
<div className="flex h-screen overflow-hidden bg-[var(--bg-primary)]">
  <Sidebar />                          {/* 220px, fixed, collapsible at <1024px */}
  <main className="flex-1 overflow-y-auto">
    <Header />                         {/* breadcrumb + page title + action buttons */}
    <div className="px-8 py-6 max-w-[1280px]">
      <Outlet />                       {/* TanStack Router page content */}
    </div>
  </main>
</div>
```

### 4.5 Sidebar

```tsx
// Sidebar nav item — active state pattern
<NavItem
  to="/patterns"
  icon={<FileCode2 size={20} />}
  label="Patterns"
  // active: border-l-2 border-[var(--brand)] bg-[var(--bg-tertiary)] text-[var(--text-primary)] font-medium
  // inactive: text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)]
/>

// Group header — 10px uppercase letter-spacing-wide text-[var(--text-tertiary)]
```

Sidebar collapses to 56px icon bar below 1024px. Labels hidden; Tooltip shows label on hover.

### 4.6 Buttons

```tsx
// Button variants using class-variance-authority
const buttonVariants = cva(
  "inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        primary:     "bg-[var(--brand)] text-white hover:bg-[var(--brand-hover)]",
        secondary:   "border border-[var(--border)] bg-[var(--bg-secondary)] text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]",
        destructive: "bg-red-600 text-white hover:bg-red-700",
        ghost:       "text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] hover:text-[var(--text-primary)]",
      },
      size: {
        sm: "h-8 px-3 text-xs",
        md: "h-9 px-4 text-sm",    // default — compact tables
        lg: "h-10 px-6 text-sm",   // forms, primary CTAs
      },
    },
    defaultVariants: { variant: "secondary", size: "md" },
  }
)
// Loading state: show <Loader2 className="animate-spin" size={16} /> instead of label; set disabled
```

One primary button per page. Destructive actions spatially separated and always have a confirm AlertDialog before proceeding.

### 4.7 Status and quality badges

```tsx
// StatusBadge.tsx
const statusConfig = {
  draft:       { label: "Draft",       className: "bg-amber-100  text-amber-800  dark:bg-amber-900/30  dark:text-amber-400"  },
  in_review:   { label: "In Review",   className: "bg-blue-100   text-blue-800   dark:bg-blue-900/30   dark:text-blue-400"   },
  approved:    { label: "Approved",    className: "bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-400" },
  published:   { label: "Published",   className: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400" },
  deprecated:  { label: "Deprecated",  className: "bg-red-100    text-red-800    dark:bg-red-900/30    dark:text-red-400"    },
}
// Always text + colour. Never colour alone.

// QualityBadge.tsx — colour transitions at 40 and 70
// 0–40: red, 41–70: amber, 71–100: green
// DriftBadge.tsx — 0: green, 1–5: amber, >5: red
```

### 4.8 Cards

```tsx
<div className="rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] p-4
                hover:border-[var(--border-strong)] hover:shadow-md transition-all duration-150">
```

Shadow scale: `hover:shadow-sm` → `hover:shadow-md`. Do not shift layout on hover — use `transition-shadow` and `transition-border-color` only.

### 4.9 Tables (TanStack Table)

```tsx
// Header row
<th className="h-10 px-3 text-left text-[11px] font-medium uppercase tracking-wider
               text-[var(--text-tertiary)] bg-[var(--bg-secondary)] sticky top-0">

// Data row (standard 48px, compact 40px)
<tr className="border-b border-[var(--border)] hover:bg-[var(--bg-tertiary)] transition-colors">

// Sortable column — show sort icon on hover, directional arrow when active
// aria-sort="ascending" | "descending" | "none" on <th>
```

### 4.10 Forms

```tsx
// Label always above input
<div className="flex flex-col gap-1.5">
  <Label htmlFor="name" className="text-sm font-medium text-[var(--text-primary)]">
    Name <span aria-hidden="true" className="text-[var(--error)]">*</span>
  </Label>
  <Input id="name" className="h-10 ..." />
  {error && (
    <p role="alert" className="text-xs text-[var(--error)]">{error}</p>   // error below field
  )}
  {helperText && (
    <p className="text-xs text-[var(--text-tertiary)]">{helperText}</p>
  )}
</div>
```

Validate on blur only. Auto-focus first invalid field after form submit. Disabled inputs: `opacity-50 cursor-not-allowed`. Never placeholder-as-label.

### 4.11 CALM Canvas (React Flow)

```tsx
// Canvas configuration
<ReactFlow
  nodeTypes={calmNodeTypes}           // derived from @finos/calm schema
  defaultEdgeOptions={{ type: 'smoothstep', animated: false }}
  style={{ background: 'var(--bg-primary)' }}
>
  <Background variant={BackgroundVariant.Dots} color="var(--border)" gap={16} size={1} />
  <Controls position="top-right" />
  <MiniMap position="bottom-right" style={{ width: 160, height: 100 }} />
  <Panel position="top-left"><NodePalette /></Panel>
</ReactFlow>

// CalmNode styling
// fill: var(--bg-secondary)  stroke: var(--border) 1.5px  radius: 8px
// Selected: stroke var(--brand) 2px + box-shadow 0 0 0 3px var(--brand-muted)
// Error: stroke var(--error) 2px + box-shadow 0 0 0 3px rgba(239,68,68,0.15)

// Node type accent (left border, not fill):
// service=indigo  database=emerald  system=amber  network=cyan  human-actor=violet  generic=zinc
```

Canvas and Monaco JSON editor share state via Zustand. Bidirectional sync debounced at 500ms. CALM `unique-id` is the React Flow node `id`.

### 4.12 Empty and loading states

```tsx
// EmptyState.tsx — shown when list/table has no results
<div className="flex flex-col items-center gap-4 py-16 text-center">
  <Icon size={40} className="text-[var(--text-tertiary)]" />
  <div>
    <h3 className="text-sm font-medium text-[var(--text-primary)]">{title}</h3>
    <p className="mt-1 text-sm text-[var(--text-secondary)]">{description}</p>
  </div>
  {action && <Button variant="primary" size="lg">{action}</Button>}
</div>

// LoadingSkeleton.tsx — shimmer placeholder for any async content
// Animate background-position shift for shimmer effect
// Never show empty axis frames for charts, never blank pages
```

### 4.13 Analytics charts (Recharts)

```tsx
// Colour order for multi-series: indigo → emerald → amber → rose → cyan
const CHART_COLORS = ['#6366f1', '#10b981', '#f59e0b', '#f43f5e', '#06b6d4']

// Every chart must have:
// - visible Legend component
// - XAxis and YAxis with labels and units
// - Tooltip with exact values
// - date range picker (7d / 30d / 90d / custom) above the chart

// Empty state: <EmptyState /> when data array is empty
// Loading state: <LoadingSkeleton /> — never a blank chart
// All interactive elements (bars, points, slices): aria-label with data value
```

### 4.14 Accessibility — non-negotiable

- Every icon-only button: `aria-label="Description"`.
- Every form input: associated `<label htmlFor>`.
- Status changes: `aria-live="polite"`. Errors: `role="alert"`.
- Focus ring: `focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-2` on every interactive element. Never `outline-none` without the custom ring class.
- Headings: sequential h1 → h6, no skipped levels per page.
- Tables: `aria-sort` on sortable `<th>`, `scope="col"` on all header cells.
- Charts: `aria-label` on chart container describing key insight.
- `prefers-reduced-motion`: wrap all `transition-*` and `animate-*` in `@media (prefers-reduced-motion: reduce)` override that sets `transition: none`.

### 4.15 Animation

```css
/* Only animate transform and opacity */
.animate-enter { animation: enter 200ms ease-out; }
@keyframes enter { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

/* Toast: slide from bottom-right + fade, auto-dismiss 4s */
/* Canvas node add: scale(0.85) + opacity(0) → scale(1) + opacity(1) in 150ms */
/* Page transition: fade 300ms */

@media (prefers-reduced-motion: reduce) {
  *, ::before, ::after { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; }
}
```

Duration: 150–250ms micro, 300ms page transitions. Easing: ease-out enter, ease-in exit. Animate only `transform` and `opacity`. Never `width`, `height`, `top`, `left`.

---

## 5. @finos/calm integration — critical implementation

### 5.1 Frontend canvas palette (`src/lib/calm-schema.ts`)

```typescript
import { getCalmSchema } from '@finos/calm'

export interface CalmNodeType {
  id: string
  label: string
  description: string
  requiredProperties: string[]
  optionalProperties: string[]
  accentColor: string    // from NODE_ACCENT_COLORS map
}

export interface CalmRelType {
  id: string
  label: string
}

const NODE_ACCENT_COLORS: Record<string, string> = {
  service:      '#6366f1',   // indigo
  database:     '#10b981',   // emerald
  system:       '#f59e0b',   // amber
  network:      '#06b6d4',   // cyan
  'human-actor':'#8b5cf6',   // violet
  generic:      '#71717a',   // zinc
}

export function parseCalmSchema(): { nodeTypes: CalmNodeType[], relTypes: CalmRelType[] } {
  const schema = getCalmSchema()
  // Parse schema.definitions to extract node-type enum values + relationship types
  // Map to CalmNodeType[] with accentColor from NODE_ACCENT_COLORS
}
```

Gateway exposes `GET /api/v1/calm/schema` returning parsed palette types. Frontend fetches once at boot via TanStack Query with `staleTime: Infinity`.

### 5.2 Validator sidecar (`services/calm-validator/server.js`)

```javascript
const { validate, getCalmSchema } = require('@finos/calm')
const express = require('express')
const app = express()
app.use(express.json({ limit: '5mb' }))

app.get('/healthz', (_, res) => res.json({ status: 'ok' }))

app.post('/validate', (req, res) => {
  const { calmDocument, orgCustomTypes = [] } = req.body
  try {
    const result = validate(calmDocument, { additionalDefinitions: orgCustomTypes })
    // result: { valid: boolean, errors: Array<{ nodeId, path, message, keyword }> }
    res.json(result)
  } catch (err) {
    res.status(400).json({ valid: false, errors: [{ message: err.message }] })
  }
})

app.listen(3001, '127.0.0.1')
```

### 5.3 Pattern Service — save pipeline (every CALM write)

```python
async def save_pattern(pattern_id: str, calm_json: str, org_id: str):
    # 1. Fetch org custom types from FalkorDB
    custom_types = await falkordb.get_custom_node_types(org_id)

    # 2. Validate via calm-validator sidecar
    resp = await httpx_client.post("http://localhost:3001/validate",
        json={"calmDocument": json.loads(calm_json), "orgCustomTypes": custom_types})
    result = resp.json()
    if not result["valid"]:
        raise HTTPException(422, detail={"code": "CALM_INVALID", "errors": result["errors"]})

    # 3. Write to FalkorDB
    calm_yaml = yaml.dump(json.loads(calm_json))
    await falkordb.update_pattern(pattern_id, calm_json=calm_json, calm_yaml=calm_yaml)

    # 4. Extract CALM nodes for ES
    doc = json.loads(calm_json)
    node_labels = [n.get("node-type") for n in doc.get("nodes", [])]
    node_names  = [n.get("name", "") for n in doc.get("nodes", [])]

    # 5. Generate embedding
    embedding = await generate_embedding(f"{pattern_name} {description} {' '.join(node_names)}")

    # 6. Sync to Elasticsearch (synchronous; flag stale on failure)
    try:
        await es.index(index=f"calm_patterns_{org_id}", id=pattern_id,
            document={ ..., "calm_node_labels": node_labels, "calm_node_names": node_names,
                       "embedding": embedding })
    except Exception:
        await pg.execute("UPDATE es_sync_status SET stale=true WHERE pattern_id=$1", pattern_id)
```

---

## 6. All routes — TanStack Router

Every route has a `loader` function that prefetches data via TanStack Query. URL search params for filters use `validateSearch` with Zod schemas.

```
PUBLIC:
  /                 /login            /docs
  /docs/calm-reference  /docs/authoring  /docs/api  /docs/cli  /docs/sdk

AUTHENTICATED:
  /dashboard        /search           /profile

PATTERN STUDIO:
  /patterns                           /patterns/new
  /patterns/import                    /patterns/:id
  /patterns/:id/canvas                /patterns/:id/json
  /patterns/:id/versions              /patterns/:id/variations
  /patterns/:id/instances             /patterns/:id/lineage
  /patterns/:id/quality               /patterns/:id/validate

MARKETPLACE:
  /marketplace                        /marketplace/search
  /marketplace/collections            /marketplace/collections/:id
  /marketplace/domains                /marketplace/domains/:slug
  /marketplace/:id                    /marketplace/compare
  /marketplace/new

WORKSPACE:
  /workspace                          /workspace/:id
  /workspace/:id/canvas               /workspace/:id/validate
  /workspace/:id/diff                 /workspace/:id/export

GOVERNANCE:
  /governance                         /governance/queue
  /governance/queue/:id               /governance/drift
  /governance/coverage                /governance/audit
  /governance/compliance              /governance/deprecations

DEVELOPER PORTAL:
  /developer                          /developer/scaffold
  /developer/api-keys                 /developer/webhooks
  /developer/api-docs                 /developer/cli
  /developer/sdk

ANALYTICS:
  /analytics                          /analytics/adoption
  /analytics/quality                  /analytics/governance
  /analytics/teams

ADMIN:
  /admin                              /admin/users
  /admin/teams                        /admin/roles
  /admin/ldap                         /admin/taxonomy
  /admin/node-types                   /admin/templates
  /admin/settings                     /admin/health
```

---

## 7. Docker Compose — stores only

```yaml
version: '3.9'

services:
  falkordb:
    image: falkordb/falkordb:latest
    ports: ["6379:6379"]
    volumes: [falkordb-data:/data]
    networks: [calm-net]

  elasticsearch:
    image: elasticsearch:8.13.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    ports: ["9200:9200"]
    volumes: [es-data:/usr/share/elasticsearch/data]
    networks: [calm-net]

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: calmplatform
      POSTGRES_USER: calm
      POSTGRES_PASSWORD: ${PG_PASSWORD}
    ports: ["5432:5432"]
    volumes: [pg-data:/var/lib/postgresql/data]
    networks: [calm-net]

networks:
  calm-net:

volumes:
  falkordb-data:
  es-data:
  pg-data:
```

---

## 8. Makefile — local development

```makefile
.PHONY: stores start-all stop-stores migrate

# Start data stores
stores:
	docker compose up -d

# Stop stores
stop-stores:
	docker compose down

# Run database migrations
migrate:
	psql "postgresql://calm:$(PG_PASSWORD)@localhost:5432/calmplatform" -f scripts/seed-postgres.sql
	bash scripts/create-es-indices.sh

# Start all services (each in background; use 'make logs-{svc}' to tail)
start-all: start-validator start-auth start-pattern start-discovery start-governance start-lineage start-export start-gateway

start-validator:
	cd services/calm-validator && node server.js &

start-auth:
	cd services/auth-svc && uvicorn main:app --host 0.0.0.0 --port 8005 --reload &

start-pattern:
	cd services/pattern-svc && uvicorn main:app --host 0.0.0.0 --port 8001 --reload &

start-discovery:
	cd services/discovery-svc && uvicorn main:app --host 0.0.0.0 --port 8002 --reload &

start-governance:
	cd services/governance-svc && uvicorn main:app --host 0.0.0.0 --port 8003 --reload &

start-lineage:
	cd services/lineage-svc && uvicorn main:app --host 0.0.0.0 --port 8004 --reload &

start-export:
	cd services/export-svc && uvicorn main:app --host 0.0.0.0 --port 8006 --reload &

start-gateway:
	cd services/gateway && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend dev server (hot reload)
frontend:
	cd frontend && npm run dev

# Build frontend into gateway's dist/
build-frontend:
	cd frontend && npm run build && cp -r dist ../services/gateway/dist
```

---

## 9. FastAPI gateway

```python
# services/gateway/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import httpx, os
from jose import jwt, JWTError

app = FastAPI(title="CALM Platform Gateway")

JWT_SECRET = os.getenv("JWT_SECRET_KEY")

UPSTREAM = {
    "auth":       os.getenv("AUTH_SVC_URL",       "http://localhost:8005"),
    "patterns":   os.getenv("PATTERN_SVC_URL",     "http://localhost:8001"),
    "search":     os.getenv("DISCOVERY_SVC_URL",   "http://localhost:8002"),
    "governance": os.getenv("GOVERNANCE_SVC_URL",  "http://localhost:8003"),
    "lineage":    os.getenv("LINEAGE_SVC_URL",     "http://localhost:8004"),
    "export":     os.getenv("EXPORT_SVC_URL",      "http://localhost:8006"),
    "calm":       os.getenv("PATTERN_SVC_URL",     "http://localhost:8001"),
    "admin":      os.getenv("PATTERN_SVC_URL",     "http://localhost:8001"),
}

PUBLIC_PREFIXES = ["/api/v1/auth/"]

clients: dict[str, httpx.AsyncClient] = {}

@app.on_event("startup")
async def startup():
    for name, url in UPSTREAM.items():
        clients[name] = httpx.AsyncClient(base_url=url, timeout=30.0)

@app.on_event("shutdown")
async def shutdown():
    for c in clients.values():
        await c.aclose()

@app.api_route("/api/v1/{service}/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(service: str, path: str, request: Request):
    full_path = f"/api/v1/{service}/{path}"
    is_public = any(full_path.startswith(p) for p in PUBLIC_PREFIXES)
    if not is_public:
        token = request.cookies.get("access_token")
        if not token:
            raise HTTPException(401, "Not authenticated")
        try:
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except JWTError:
            raise HTTPException(401, "Invalid token")
    if service not in clients:
        raise HTTPException(404)
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    resp = await clients[service].request(
        method=request.method, url=f"/api/v1/{service}/{path}",
        content=body, headers=headers, params=dict(request.query_params)
    )
    return StreamingResponse(
        resp.aiter_bytes(), status_code=resp.status_code, headers=dict(resp.headers)
    )

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "gateway"}

# MUST be last — serves React multi-page build
app.mount("/", StaticFiles(directory="dist", html=True), name="spa")
```

---

## 10. Auth Service — LDAP

```python
# POST /api/v1/auth/login
async def login(body: LoginRequest, response: Response):
    user_dn = f"uid={body.username},{LDAP_BASE_DN}"
    try:
        server = Server(LDAP_URL, get_info=ALL)
        conn   = Connection(server, user=user_dn, password=body.password,
                            client_strategy=SAFE_SYNC, auto_bind=True)
    except (LDAPBindError, LDAPSocketOpenError):
        raise HTTPException(401, "Invalid credentials")

    conn.search(LDAP_BASE_DN, f"(member={user_dn})", attributes=["dn"])
    group_dns = [e.entry_dn for e in conn.entries]

    async with pg_pool.acquire() as db:
        rows = await db.fetch(
            "SELECT DISTINCT platform_role FROM role_mappings "
            "WHERE ldap_group_dn = ANY($1::text[]) AND org_id = $2",
            group_dns, org_id
        )
        roles = [r["platform_role"] for r in rows]

    token = jwt.encode(
        {"sub": user_dn, "roles": roles, "org_id": str(org_id),
         "exp": datetime.utcnow() + timedelta(hours=8), "jti": str(uuid4())},
        JWT_SECRET, algorithm="HS256"
    )
    response.set_cookie("access_token", token,
                        httponly=True, secure=True, samesite="strict", max_age=28800)
    return {"user": body.username, "roles": roles}
```

---

## 11. FalkorDB — key Cypher patterns

```cypher
-- Always include org_id filter on every query
MATCH (p:Pattern { org_id: $org_id }) ...

-- Create pattern (stores full CALM JSON as string property)
CREATE (p:Pattern {
  id: $id, name: $name, version: '1.0.0', domain: $domain,
  pattern_type: $pattern_type, status: 'draft',
  org_id: $org_id, team_id: $team_id,
  calm_json: $calm_json, calm_yaml: $calm_yaml,
  calm_schema_uri: $calm_schema_uri,
  quality_score: 0, created_at: timestamp(), updated_at: timestamp()
}) RETURN p

-- Fork to instance
MATCH (p:Pattern { id: $pattern_id, org_id: $org_id })
CREATE (i:Instance {
  id: $instance_id, name: $name, org_id: $org_id, team_id: $team_id,
  calm_json: p.calm_json, validated: false, drift_score: 0.0,
  created_at: timestamp()
})-[:DERIVED_FROM { forked_at: timestamp(), drift_score: 0.0, last_checked: timestamp() }]->(p)
RETURN i

-- Update drift score
MATCH (i:Instance { id: $iid })-[r:DERIVED_FROM]->(p:Pattern)
SET r.drift_score = $score, r.last_checked = timestamp(), i.drift_score = $score

-- Coverage heat map
MATCH (p:Pattern { org_id: $org_id, status: 'published' })
RETURN p.domain, p.pattern_type, count(p) AS n ORDER BY p.domain, p.pattern_type

-- Drift report (paginated)
MATCH (i:Instance)-[r:DERIVED_FROM]->(p:Pattern { org_id: $org_id })
WHERE r.drift_score > 0
RETURN i.id, i.name, p.id, p.name, r.drift_score
ORDER BY r.drift_score DESC SKIP $skip LIMIT $limit
```

---

## 12. Quality score calculation

```python
# services/pattern-svc/quality.py
import math
from datetime import datetime

def calculate_quality_score(pattern: dict, instance_count: int) -> int:
    score = 0
    calm  = json.loads(pattern.get("calm_json", "{}"))

    # Completeness — 25 pts
    c = 0
    if pattern.get("name"):                           c += 5
    if pattern.get("domain"):                         c += 5
    if len(pattern.get("description","").split()) >= 20: c += 5
    if pattern.get("compliance_tags"):                c += 5
    if pattern.get("has_variation"):                  c += 5
    score += c

    # Documentation — 25 pts
    d = 0
    words = len(pattern.get("description","").split())
    if   words >= 100: d += 12
    elif words >= 50:  d += 7
    elif words >= 20:  d += 3
    if pattern.get("compliance_tags"):  d += 8
    if pattern.get("usage_notes"):      d += 5
    score += d

    # Freshness — 25 pts, linear decay 90d → 365d
    updated = pattern.get("updated_at")
    if updated:
        days = (datetime.utcnow() - updated).days
        if   days <= 90:  score += 25
        elif days >= 365: score += 0
        else:             score += int(25 * (365 - days) / (365 - 90))

    # Usage — 25 pts, logarithmic
    if instance_count > 0:
        score += min(25, int(25 * math.log(instance_count + 1) / math.log(11)))

    return min(100, score)
```

---

## 13. Drift score calculation

```python
# services/lineage-svc — called on POST /api/v1/lineage/drift/recalculate/:instance_id
def calculate_drift_score(instance_json: str, parent_json: str) -> float:
    inst   = json.loads(instance_json)
    parent = json.loads(parent_json)

    p_nodes = {n["unique-id"] for n in parent.get("nodes", [])}
    i_nodes = {n["unique-id"] for n in inst.get("nodes", [])}

    p_rels  = {r["unique-id"]: r for r in parent.get("relationships", [])}
    i_rels  = {r["unique-id"]: r for r in inst.get("relationships",   [])}

    added_nodes   = len(i_nodes - p_nodes)
    removed_nodes = len(p_nodes - i_nodes)
    added_rels    = len(set(i_rels) - set(p_rels))
    removed_rels  = len(set(p_rels) - set(i_rels))
    modified_rels = sum(
        1 for uid in (set(p_rels) & set(i_rels))
        if p_rels[uid].get("relationship-type") != i_rels[uid].get("relationship-type")
    )
    return float(added_nodes + removed_nodes + added_rels + removed_rels + modified_rels)
```

---

## 14. Build order — follow exactly

Build in phases. Each phase must be functional before starting the next.

### Phase A: Infrastructure + stores

1. Create full directory structure (all folders and placeholder files)
2. Write `docker-compose.yml` (stores only — no app services)
3. Write `scripts/seed-postgres.sql` — all tables including `audit_log` (append-only constraint)
4. Write `scripts/seed-falkordb.cypher` — node indices, constraints, 12 archetype Pattern templates
5. Write `scripts/create-es-indices.sh` — `calm_patterns_{template}` + `calm_docs` index with full mapping
6. Write `Makefile` with all targets from Section 8
7. Write `.env.example`

### Phase B: CALM validator sidecar

8. Write `services/calm-validator/package.json`
9. Write `services/calm-validator/server.js` (Section 5.2)
10. Write `services/calm-validator/Dockerfile`
11. **Test:** `node services/calm-validator/server.js` → POST to `:3001/validate` with valid + invalid CALM JSON

### Phase C: Auth Service

12. Write `services/auth-svc/main.py` — LDAP bind + JWT issuance (Section 10)
13. Write `services/auth-svc/dependencies.py` — shared JWT validation function (to be imported by all services)
14. Write `GET /healthz`
15. **Test:** login returns httpOnly cookie containing JWT

### Phase D: Pattern Service

16. Write `services/pattern-svc/db/falkordb.py` — async FalkorDB client wrapper
17. Write `services/pattern-svc/db/queries.py` — all Cypher queries from Section 11
18. Write `services/pattern-svc/es/sync.py` — ES document sync
19. Write `services/pattern-svc/quality.py` (Section 12)
20. Implement all pattern routes: GET list, POST create, GET/PUT/DELETE detail, versions, variations, validate, import
21. Implement WebSocket canvas sync endpoint

### Phase E: Discovery, Lineage, Governance, Export

22. Write `services/discovery-svc/` — ES hybrid search, semantic similarity, suggest
23. Write `services/lineage-svc/` — lineage graph, drift calc (Section 13), coverage heat map
24. Write `services/governance-svc/` — state machine, 4 pre-checks, comments, deprecation, audit log writes
25. Write `services/export-svc/` — 7 output formats + Jinja2 templates for OpenAPI, Terraform, K8s

### Phase F: Gateway

26. Write `services/gateway/main.py` (Section 9) — proxy routes, JWT middleware, StaticFiles
27. Write `services/gateway/Dockerfile` (multi-stage: React build → Python runtime)

### Phase G: Frontend foundation

28. Bootstrap React app: Vite + TypeScript + TailwindCSS
29. Write `tailwind.config.ts` — map all CSS custom property tokens to Tailwind utilities
30. Write `index.css` with full light/dark token definitions (Section 4.1)
31. Configure TanStack Router with all 55+ routes from Section 6
32. Write `src/lib/calm-schema.ts` — parse `@finos/calm` into palette types (Section 5.1)
33. Write `src/lib/api.ts` — typed fetch against gateway
34. Write `src/stores/canvas.ts` — Zustand canvas state
35. Write all `src/components/ui/` — Radix UI + Tailwind wrappers (following Section 4.6–4.8 exactly)
36. Write `AppShell`, `Sidebar`, `Header`, `Breadcrumb` (Section 4.4–4.5)
37. Write `EmptyState`, `LoadingSkeleton`, `StatusBadge`, `QualityBadge`, `DriftBadge` (Section 4.12–4.7)

### Phase H: Canvas components

38. Write `CalmCanvas.tsx` — React Flow wrapper with dot-grid background (Section 4.11)
39. Write `CalmNode.tsx` — custom node with type accent border colours
40. Write `NodePalette.tsx` — draggable node palette from `@finos/calm` parsed types
41. Implement bidirectional canvas ↔ Monaco JSON sync via Zustand (500ms debounce)

### Phase I: All pages (in this order)

42. `/login` → LDAP form → POST /auth/login → httpOnly cookie → redirect /dashboard
43. `/dashboard` → recent patterns, assigned reviews, org stats
44. `/patterns` list + `/patterns/new` wizard + `/patterns/:id` overview
45. `/patterns/:id/canvas` — full canvas + Monaco split pane
46. `/patterns/:id/versions` + `/variations` + `/instances` + `/lineage` + `/quality` + `/validate`
47. `/marketplace` + `/marketplace/search` + `/marketplace/:id` + `/marketplace/compare`
48. `/marketplace/collections` + `/marketplace/domains`
49. `/workspace` + `/workspace/:id/canvas` + `/workspace/:id/diff` + `/workspace/:id/export` + `/workspace/:id/validate`
50. `/governance` + `/governance/queue` + `/governance/queue/:id` (split layout: 60/40)
51. `/governance/drift` + `/governance/coverage` + `/governance/audit` + `/governance/compliance`
52. `/developer` + `/developer/scaffold` + `/developer/api-keys` + `/developer/webhooks` + `/developer/api-docs`
53. `/analytics/*` — all 4 analytics pages with Recharts + date range picker
54. `/admin/*` — all 10 admin pages; `/admin/health` polls `/api/v1/admin/health` every 30s
55. `/` landing + `/docs/*` documentation hub (CALM reference auto-generated from `@finos/calm` schema)

### Phase J: Kubernetes manifests

56. Write all K8s YAML: namespace, StatefulSets (3 stores), Deployments (7 services), Services, HPA, ConfigMap, Secret
57. `pattern-svc-deployment.yaml` includes `calm-validator` as sidecar container (`localhost:3001`)
58. Write `k8s/overlays/dev/` (replicas=1, NodePort) and `k8s/overlays/prod/` (HPA, LoadBalancer, resource limits)

---

## 15. Environment variables (`.env.example`)

```bash
# Stores
FALKORDB_HOST=localhost
FALKORDB_PORT=6379

ES_HOST=http://localhost:9200

PG_HOST=localhost
PG_PORT=5432
PG_USER=calm
PG_PASSWORD=changeme
PG_DATABASE=calmplatform
DATABASE_URL=postgresql+asyncpg://calm:changeme@localhost:5432/calmplatform

# Auth
JWT_SECRET_KEY=change-this-to-a-long-random-string-in-production
LDAP_URL=ldap://ldap.example.com:389
LDAP_BASE_DN=dc=example,dc=com
LDAP_BIND_DN=cn=admin,dc=example,dc=com
LDAP_BIND_PASSWORD=changeme

# Inter-service (local dev — all on localhost)
CALM_VALIDATOR_URL=http://localhost:3001
PATTERN_SVC_URL=http://localhost:8001
DISCOVERY_SVC_URL=http://localhost:8002
GOVERNANCE_SVC_URL=http://localhost:8003
LINEAGE_SVC_URL=http://localhost:8004
AUTH_SVC_URL=http://localhost:8005
EXPORT_SVC_URL=http://localhost:8006
```

---

## 16. Acceptance criteria — Phase A complete

- [ ] `docker compose up -d` starts FalkorDB, Elasticsearch, PostgreSQL with no errors
- [ ] `make migrate` runs SQL seed and creates ES indices without errors
- [ ] `node services/calm-validator/server.js` starts on port 3001
- [ ] `POST localhost:3001/validate` with valid `@finos/calm` JSON returns `{"valid":true,"errors":[]}`
- [ ] `POST localhost:3001/validate` with invalid JSON returns `{"valid":false,"errors":[...]}` with node-level details
- [ ] `uvicorn main:app --port 8005` (auth-svc) starts; `POST /api/v1/auth/login` with mock LDAP creds returns httpOnly cookie
- [ ] `POST /api/v1/patterns` creates Pattern node in FalkorDB with `calm_json` property
- [ ] Pattern with invalid CALM document rejected HTTP 422 with `errors[].nodeId` in response
- [ ] `GET /api/v1/search/patterns?q=kafka` returns results from Elasticsearch
- [ ] `uvicorn main:app --port 8000` (gateway) starts; `GET localhost:8000/` serves React SPA
- [ ] All 55+ React routes navigable in browser with correct page title and data loaded
- [ ] React Flow canvas palette contains standard `@finos/calm` node types (no hardcoded types)
- [ ] Draft → Submitted → In Review → Approved → Published governance flow completes end-to-end
- [ ] `/admin/health` page shows correct status for all services
- [ ] Light and dark theme toggle works on all pages with correct colour tokens
- [ ] All interactive elements have visible focus rings on keyboard navigation
