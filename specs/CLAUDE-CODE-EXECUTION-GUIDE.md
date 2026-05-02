# EA Knowledge Hub — Claude Code Execution Guide v2
**Updated: infra removed, React Flow throughout, GitHub Enterprise, LDAP auth, in-app-only notifications**

---

## Prerequisites

- Docker Desktop (v4.28+)
- Node.js 20 LTS
- Python 3.12
- Access to enterprise LDAP server (ldaps://) and service account credentials
- GitHub Enterprise: PAT with `repo` scope, or GitHub App with Contents + Pull Requests permissions
- Access to internal LLM/embedding gateway (OpenAI-compatible API)

---

## Docker Compose (dev environment)

Create `docker-compose.yml`:

```yaml
version: '3.9'
services:

  falkordb:
    image: falkordb/falkordb:latest
    ports: ["6379:6379"]
    volumes: [falkordb_data:/data]
    command: --requirepass ${FALKORDB_PASSWORD} --save 900 1 --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${FALKORDB_PASSWORD}", "ping"]
      interval: 10s
      retries: 5

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ea_hub
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ea_hub
    ports: ["5432:5432"]
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_postgres.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ea_hub"]
      interval: 10s
      retries: 5

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.13.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms1g -Xmx1g
    ports: ["9200:9200"]
    volumes: [elasticsearch_data:/usr/share/elasticsearch/data]
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_health || exit 1"]
      interval: 15s
      retries: 5

  redis:
    image: redis:7-alpine
    ports: ["6380:6379"]
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes: [redis_data:/data]

  backend:
    build: { context: ./ea-hub-backend }
    environment:
      FALKORDB_HOST: falkordb
      FALKORDB_PORT: 6379
      FALKORDB_PASSWORD: ${FALKORDB_PASSWORD}
      POSTGRES_URL: postgresql+asyncpg://ea_hub:${POSTGRES_PASSWORD}@postgres:5432/ea_hub
      ELASTICSEARCH_URL: http://elasticsearch:9200
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379
      LDAP_SERVER: ${LDAP_SERVER}
      LDAP_BASE_DN: ${LDAP_BASE_DN}
      LDAP_BIND_DN: ${LDAP_BIND_DN}
      LDAP_BIND_PASSWORD: ${LDAP_BIND_PASSWORD}
      LDAP_USER_SEARCH_BASE: ${LDAP_USER_SEARCH_BASE}
      LDAP_CA_CERT_PATH: /certs/corp-ca.crt
      GITHUB_BASE_URL: ${GITHUB_BASE_URL}
      GITHUB_TOKEN: ${GITHUB_TOKEN}
      GITHUB_REPO_OWNER: ${GITHUB_REPO_OWNER}
      GITHUB_REPO_NAME: ${GITHUB_REPO_NAME}
      OPENAI_API_BASE: ${OPENAI_API_BASE}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      SECRET_KEY: ${SECRET_KEY}
      DEBUG: "true"
    ports: ["8000:8000"]
    volumes:
      - ./ea-hub-backend:/app
      - ./certs:/certs:ro       # mount enterprise CA cert bundle
    depends_on:
      falkordb: { condition: service_healthy }
      postgres: { condition: service_healthy }
      elasticsearch: { condition: service_healthy }
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  worker:
    build: { context: ./ea-hub-backend }
    environment:
      FALKORDB_HOST: falkordb
      POSTGRES_URL: postgresql+asyncpg://ea_hub:${POSTGRES_PASSWORD}@postgres:5432/ea_hub
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379
      OPENAI_API_BASE: ${OPENAI_API_BASE}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on: [backend, redis]
    command: celery -A app.tasks.celery_app worker --loglevel=info

  beat:
    build: { context: ./ea-hub-backend }
    depends_on: [redis]
    command: celery -A app.tasks.celery_app beat --loglevel=info

  frontend:
    build:
      context: ./ea-hub-frontend
      dockerfile: Dockerfile.dev
    ports: ["5173:5173"]
    volumes:
      - ./ea-hub-frontend:/app
      - /app/node_modules
    environment:
      VITE_API_URL: http://localhost:8000/api/v1
    command: npm run dev -- --host

volumes:
  falkordb_data:
  postgres_data:
  elasticsearch_data:
  redis_data:
```

---

## `.env.example`

```bash
SECRET_KEY=change-me-32-chars-min

FALKORDB_PASSWORD=change-me
POSTGRES_PASSWORD=change-me
REDIS_PASSWORD=change-me

# Enterprise LDAP
LDAP_SERVER=ldaps://ldap.corp.example.com:636
LDAP_BASE_DN=DC=corp,DC=example,DC=com
LDAP_BIND_DN=CN=svc-ea-hub,OU=ServiceAccounts,DC=corp,DC=example,DC=com
LDAP_BIND_PASSWORD=change-me
LDAP_USER_SEARCH_BASE=OU=Users,DC=corp,DC=example,DC=com
LDAP_USER_FILTER=(uid={username})
LDAP_CA_CERT_PATH=/certs/corp-ca.crt

# GitHub Enterprise
GITHUB_BASE_URL=https://github.corp.example.com/api/v3
GITHUB_TOKEN=ghp_...
GITHUB_REPO_OWNER=ea-architecture
GITHUB_REPO_NAME=ea-hub-content
GITHUB_DEFAULT_BRANCH=main

# Internal LLM/embedding gateway (OpenAI-compatible)
OPENAI_API_BASE=https://ai-gateway.corp.example.com/v1
OPENAI_API_KEY=internal-key
```

---

## Makefile

```makefile
.PHONY: up down reset seed indexes migrate test-backend test-frontend lint

up:
	docker-compose up -d

down:
	docker-compose down

reset:
	docker-compose down -v && docker-compose up -d

migrate:
	docker-compose exec backend alembic upgrade head

indexes:
	docker-compose exec backend python scripts/create_indexes.py

seed:
	docker-compose exec backend python scripts/seed_data.py

test-backend:
	docker-compose exec backend pytest tests/ -v

test-frontend:
	cd ea-hub-frontend && npm test

lint:
	cd ea-hub-backend && ruff check app/ && mypy app/
	cd ea-hub-frontend && npm run lint

logs:
	docker-compose logs -f backend worker
```

---

## Step-by-step Claude Code prompts

### Step 1 — Scaffold

```
Read ea-hub-PRD.md, SPEC-01-data-model.md, SPEC-02-backend.md,
SPEC-03-frontend.md, and SPEC-04-search.md.

Scaffold the full directory structure:
- ea-hub-backend/ with all folders from SPEC-02 Section 1,
  pyproject.toml with exact dependencies from SPEC-02 Section 10,
  and a Dockerfile (python:3.12-slim, pip install, uvicorn entrypoint)
- ea-hub-frontend/ with all folders from SPEC-03 Section 1,
  package.json with exact dependencies from SPEC-03 Section 12,
  vite.config.ts, tsconfig.json, tailwind.config.js

Also create docker-compose.yml, .env.example, and Makefile from the
Claude Code Execution Guide.

Do not write application logic yet — only directory structure, empty
__init__.py files, and config files.
```

### Step 2 — Data layer (SPEC-01)

```
Using SPEC-01-data-model.md, implement:

1. app/db/falkordb.py:
   - Async FalkorDB connection (falkordb Python client; wrap sync calls
     in asyncio executor if client is sync-only)
   - Connection pool, health-check ping
   - query(cypher, params) helper
   - Node CRUD helpers for every node type in Section 1.1
   - Relationship creation helpers for all types in Section 1.2
   - Named methods for the 5 key queries in Section 1.3

2. app/db/postgres.py:
   - Async SQLAlchemy engine + session factory
   - All table definitions from Section 2 as SQLAlchemy Core Table objects
     (not ORM classes — keep it simple)
   - alembic/versions/001_initial_schema.py migration

3. app/db/elasticsearch.py:
   - Async ES client
   - create_index() with exact mapping from Section 3
   - search(), index_artifact(), update_artifact(), update_embedding()

4. app/db/redis.py:
   - Redis client wrapper, get/set/setex/delete/xadd/xread helpers

5. scripts/create_indexes.py:
   - Creates ES index with mapping (idempotent)
   - Creates FalkorDB vector index on embedding_vector field

6. scripts/seed_data.py:
   - Seeds all data from Section 6 of SPEC-01
   - Creates GitHub repo branches and placeholder files for each seed artifact
     (use GitHub API from SPEC-02 Section 5)
   - Queues embedding generation for all seeded artifacts
```

### Step 3 — Backend (SPEC-02)

```
Using SPEC-02-backend.md, implement the full FastAPI backend:

1. app/config.py — exact Settings class from Section 2

2. app/services/ldap_service.py — exact LDAPService from Section 3:
   - authenticate(username, password) with ldap3 library
   - refresh_attributes(ldap_uid) with Redis caching
   - evaluate_auto_assign_rules(ldap_attrs, rules)
   - _apply_operator(field_val, operator, cond_val)

3. app/core/auth.py — LDAP-backed JWT from Section 4:
   - issue_jwt(user_id, role) → (token, jti)
   - get_current_user dependency
   - require_role(*roles) factory
   - require_active() dependency (blocks Pending users)

4. app/services/github_service.py — exact GitHubService from Section 5:
   - create_branch, upsert_file, create_pr, add_pr_comment,
     merge_pr, get_pr_comments (all via httpx)

5. app/services/notification_service.py — in-app only from Section 7.1:
   - create(), notify_all_admins(), notify_domain_stewards()
   - SSE stream endpoint at GET /users/me/notifications/stream

6. app/services/contribution_service.py — GitHub-backed from Section 7.2:
   - submit(), steward_triage(), certify() (all 5 gates)

7. All remaining services:
   - artifact_service.py, fitness_service.py (exact weights),
     validation_service.py (all rules), embedding_service.py

8. All routers from Section 6 with exact paths, query params,
   request/response shapes. Implement login flow logic exactly as
   described in Section 6.1 (LDAP bind → auto-assign rules → Pending).

9. Admin endpoints from Section 6.2 including:
   - PUT /admin/users/{id}/role (with JWT revocation)
   - POST /admin/auto-assign-rules/test

10. Celery tasks from Section 9.

11. tests/test_artifacts.py, tests/test_contributions.py,
    tests/test_ldap_auth.py (mock ldap3 library for tests)
```

### Step 4 — Search pipeline (SPEC-04)

```
Using SPEC-04-search.md, implement:

1. QueryUnderstanding class (exact hint maps from Section 2)
2. bm25_search() with exact ES query body (Section 3)
3. semantic_search() with KNN top-level query (Section 4)
4. graph_search() using FalkorDB traversal (Section 5)
5. reciprocal_rank_fusion() with exact RRF formula (Section 6)
6. apply_reranking_multipliers() with all multiplier rules (Section 6)
7. compute_dynamic_facets() ES aggregation (Section 7)
8. generate_synthesis() RAG prompt (Section 8)
9. SSE streaming synthesis endpoint (Section 9)
10. record_search_event() and get_gap_report() (Section 10)
11. Typeahead suggest endpoint with Redis caching (Section 11)
12. EmbeddingService with build_artifact_text() (Section 12)

Wire all into POST /search in app/routers/search.py.

Write tests/test_search.py covering intent classification,
filter clause generation, RRF ordering, and facet aggregation.
```

### Step 5 — Frontend (SPEC-03)

```
Using SPEC-03-frontend.md, build the React + Vite frontend:

1. Base:
   - App.tsx with React Router routes and guards from Section 2
   - authStore.ts, notificationStore.ts, studioStore.ts, searchStore.ts
   - api/client.ts with auth interceptor (redirects Pending → /pending)
   - styles/tokens.css with all CSS variables from Section (design tokens)
   - AppShell.tsx, Sidebar.tsx, Topbar.tsx, NotificationBell.tsx

2. Auth:
   - Login page — LDAP credential form as specified in Section 3
   - Pending landing page as specified in Section 4

3. Admin pages (Section 5):
   - PendingUsersPanel, UserTable, RoleAssignModal
   - AutoAssignRuleList (drag-to-reorder with @dnd-kit/core),
     AutoAssignRuleForm, RuleTester
   - /admin/users and /admin/rules page layouts

4. React Flow — two contexts (Section 6):
   A. StudioCanvas (editable):
      - PatternNode custom node with tech annotation input
      - Drag-and-drop from PatternPalette via dataTransfer API
      - Edge type selector dialog (Radix UI Popover) on connect
      - Background, Controls, MiniMap
      - Staleness indicator on nodes when source pattern is Under_Watch/Sunset
   B. RelationshipGraph (read-only):
      - ArtifactNode + CentralArtifactNode custom node types
      - Click-to-navigate to artifact detail
      - nodesDraggable=false, nodesConnectable=false
      - Dagre layout (install: npm install dagre @types/dagre)
      - applyDagreLayout() helper function

5. Notification SSE (Section 7):
   - useNotifications hook using @microsoft/fetch-event-source
   - NotificationBell popover with unread count badge

6. All pages as specified in Section 8:
   - Home, Search, Explorer (3-panel), ArtifactDetail (7 tabs),
     BlueprintStudio (3-panel), Contribute (5-step wizard),
     Lifecycle dashboard, Governance, Profile

7. Blueprint Studio specific:
   - PatternPalette: grouped by TOGAF domain, draggable items
   - StudioCanvas: React Flow editable with PatternNode
   - BlueprintManifest: all fields, TechChoiceRow with GitHub link
   - ValidationResults: hard errors red, warnings amber
   - Export buttons: ArchiMate/C4/Terraform trigger API calls → file download

8. ArtifactDetail Tab 1 (Blueprint):
   - Read-only React Flow canvas from architecture_diagram_json
   - nodesDraggable=false, nodesConnectable=false, fitView

9. Pattern Explorer detail pane:
   - Embed RelationshipGraph (React Flow read-only) below related artifacts list

10. RFCCommentThread:
    - Displays GitHub PR comments fetched from GET /contributions/{id}
      which calls GitHubService.get_pr_comments internally
    - Text input + "Add comment" posts to GitHub PR via API
```

### Step 6 — Integration smoke tests

```
1. Start all services: make up
2. make migrate && make indexes && make seed
3. Verify: curl http://localhost:8000/ready → all checks ok

4. Auth flow:
   - POST /api/v1/auth/login with seed user credentials
   - With pending_user: verify redirect to /pending works in browser
   - Admin assigns role → verify JWT revocation → re-login gets new role

5. Auto-assign rules:
   - POST /admin/auto-assign-rules/test with architect's LDAP attrs
   - Verify correct role would be assigned

6. Search:
   - POST /api/v1/search with query "distributed transaction consistency"
   - Expect: AI synthesis present, Saga pattern in top 3 results

7. Contribution:
   - Submit a new pattern as architect user
   - Verify GitHub PR created in configured repo
   - Steward triage → RFC opens → GitHub PR comment added

8. Notifications:
   - Open /users/me/notifications/stream SSE endpoint
   - Trigger any notification → verify it arrives in stream

9. Browser:
   - Login page: enter LDAP credentials
   - Home, Search, Explorer all render correctly
   - Blueprint Studio: drag pattern from palette onto canvas
   - Relationship graph renders with React Flow nodes/edges
   - Admin: Pending Users panel shows pending_user seed
   - Admin: Auto-assign rules list shows seeded rule
```

---

## Technical notes for Claude Code

1. **FalkorDB Python client async**: check if `falkordb` pip package exposes async API. If sync-only, wrap all `.execute()` calls with `asyncio.get_event_loop().run_in_executor(None, ...)` or use `anyio.to_thread.run_sync()`.

2. **LDAP TLS with enterprise CA**: `ldap3.Tls(ca_certs_file=path, validate=ssl.CERT_REQUIRED)`. The CA cert path is mounted as a Docker volume at `/certs/corp-ca.crt`. In local dev without a real LDAP, use a mock/stub in `tests/` that skips actual LDAP bind.

3. **GitHub Enterprise API vs GitHub.com API**: always use `settings.GITHUB_BASE_URL` (e.g. `https://github.corp.example.com/api/v3`) — never hardcode `api.github.com`. The `X-GitHub-Api-Version` header should still be included.

4. **React Flow v11 vs v12**: use `reactflow` v11 (the `npm install reactflow` package). The import is `from 'reactflow'` not `@xyflow/react`. If v12 is the latest, verify API compatibility with these specs before upgrading.

5. **Dagre layout for RelationshipGraph**: dagre is a peer of `reactflow`. After calling `dagre.layout(graph)`, extract `{x, y}` from `dagreGraph.node(id)` and set React Flow node `position: { x: x - nodeWidth/2, y: y - nodeHeight/2 }`.

6. **EventSource with Auth header**: native browser `EventSource` doesn't support custom headers. Use `@microsoft/fetch-event-source` which wraps `fetch()` and supports `Authorization`. Add to frontend deps.

7. **JWT revocation on role change**: `PUT /admin/users/{id}/role` must `UPDATE ldap_sessions SET revoked = TRUE WHERE user_id = $1` as part of the same transaction that updates the user's role. The user's next API call hits `get_current_user`, which checks the session table and gets 401, forcing re-login with a fresh JWT carrying the new role.

8. **RSA-4096 certification signatures**: generate the key pair once as part of `make seed` or a first-run admin setup script. Store the private key in a Docker secret (dev) or environment variable. Compute signature with `cryptography` library: `private_key.sign(content_hash.encode(), padding.PSS(...), hashes.SHA256())`.

9. **Elasticsearch dense_vector + KNN**: the `knn` clause must be at the **top level** of the request body, not nested inside `query`. ES 8.x KNN syntax: `{"knn": {"field": "embedding_vector", "query_vector": [...], "k": 50, "num_candidates": 500, "filter": {...}}}`.

10. **LDAP mock for local dev/testing**: add a `LDAP_MOCK=true` env variable. When set, `LDAPService.authenticate` accepts any username/password combination that exists in the `users` Postgres table (password ignored), returning fabricated LDAP attrs from the user record. This allows frontend development without a real LDAP connection.
```

---

*End of Execution Guide. Package: 5 files — PRD + 4 SPEC files.*
