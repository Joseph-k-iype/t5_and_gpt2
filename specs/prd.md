# EA Knowledge Hub — Master Product Requirements Document
**Version:** 2.0  
**Status:** Ready for implementation  
**Intended consumer:** Claude Code (agentic build)  
**Last updated:** May 2026

---

## 0. Document package

| File | Covers |
|------|--------|
| `ea-hub-PRD.md` | This file — vision, personas, content model, features, NFRs, tech stack |
| `SPEC-01-data-model.md` | FalkorDB graph schema, PostgreSQL schema, Elasticsearch mappings, Redis keys |
| `SPEC-02-backend.md` | FastAPI services, all endpoints, LDAP auth, business logic, GitHub integration |
| `SPEC-03-frontend.md` | React + Vite UI, all pages and components, React Flow usage for both canvas and graph |
| `SPEC-04-search.md` | Search pipeline, hybrid fusion, AI synthesis |

**Build order:** SPEC-01 → SPEC-02 → SPEC-04 → SPEC-03

---

## 1. Product vision

The EA Knowledge Hub is a **living semantic repository of enterprise architecture intelligence**. It is not a document store or wiki. It is a graph-backed, AI-assisted knowledge platform where architecture patterns, blueprints, decision records, and governance artifacts are first-class typed entities with explicit relationships, certification lifecycles, and multi-jurisdiction applicability metadata.

---

## 2. Users, personas, and onboarding

### 2.1 Personas

| Persona | Zachman row | Primary hub need |
|---------|-------------|-----------------|
| Executive / Strategist | Planner | Principles, capability maps, regulatory posture |
| Business Domain Owner | Owner | Domain blueprints, operating models, data ownership |
| Solution / Domain Architect | Designer | Full pattern catalog, blueprints, ADRs |
| Platform / Delivery Engineer | Builder | Infrastructure blueprints, tech standards |
| Specialist Implementer | Subcontractor | Config patterns, hardening guides |
| Domain Steward | All rows | Content governance, lifecycle |
| Reviewer | All rows | Structured review and certification |
| Board Member | All rows | Accept/reject, sign certifications |

### 2.2 LDAP-based onboarding and role assignment

All users authenticate via Enterprise LDAP. The hub does **not** use OIDC, SAML, or any other SSO mechanism.

**First-login flow:**
1. User hits `/login`, enters their LDAP credentials (username + password)
2. Backend performs LDAP bind against the enterprise LDAP server (ldaps://)
3. On successful bind, hub fetches the user's LDAP attributes: `cn`, `mail`, `department`, `title`, `manager`, `memberOf` (group DNs)
4. Hub checks if a hub user record exists for this LDAP `uid`
   - If not: create user record with role = `Pending`
   - If yes and role ≠ `Pending`: issue hub JWT, proceed to app
5. Pending users can view Certified/Mandated content (read-only) but cannot contribute, review, or govern
6. On next login while still Pending, user sees an in-app banner: "Your account is awaiting role assignment by an administrator"

**Admin role assignment workflow:**
1. Admin dashboard shows a **Pending Users** panel: list of users awaiting role, with their LDAP attributes displayed (department, title, manager)
2. Admin selects a user → picks a role from the role dropdown → clicks Assign
3. Hub writes the role to Postgres; user's next request or page refresh reflects the new role
4. An in-app notification is created: "Your account has been activated with role: {role}" — visible to the user on their next login

**Auto-assign rules (optional, configured by Admin):**
- Admin can define rules in the Admin UI: `IF ldap.department CONTAINS "Architecture" THEN role = Contributor`
- Rules are evaluated at first login before creating the Pending record
- If a rule matches, the user is provisioned directly with the matched role (skipping Pending)
- Rules are ordered; first match wins
- Rules can be enabled/disabled without deletion

**Role changes:**
- Admin can change any user's role at any time from the User Management screen
- Revoking a role sets it back to `Pending` (user loses contribute/review access immediately — enforced on next API call via JWT re-validation)
- Suspending a user sets `active = false`; all their API calls return 403 regardless of role

---

## 3. Content model

### 3.1 Content type hierarchy

```
Architecture Principles
  └── Patterns
        └── Reference Blueprints
              └── Implementation Guides
Architecture Decision Records (ADRs)
Anti-Pattern Library
Curated Sets
```

### 3.2 Certification tiers

| Tier | Badge | Signed by | Search weight |
|------|-------|-----------|---------------|
| Community | ○ | Contributor | 0.4× |
| Reviewed | ◑ | Steward + Reviewer | 0.7× |
| Certified | ● | Board Chair | 1.0× |
| Mandated | ★ | Group CTO / Chief Architect | 1.3× |

### 3.3 Lifecycle states

`Draft → RFC_Open → Reviewed → Accepted → Certified → Under_Watch → Sunset → Deprecated → Archived`

### 3.4 TOGAF coordinates

Domains: `Business | Data | Application | Technology | Intelligence`  
ADM phases: `Preliminary | A | B | C | D | E | F | G | H | Requirements_Management`

### 3.5 Zachman coordinates

Rows: `Planner | Owner | Designer | Builder | Subcontractor | Enterprise`  
Columns: `What | How | Where | Who | When | Why`  
Multi-cell tagging permitted; primary cell required.

### 3.6 Fitness score

```
fitness = recency×0.25 + adoption×0.25 + rating×0.20 + issues×0.15 + supersession×0.15
```
Under_Watch threshold: < 60. Auto-escalate to steward: < 40.

---

## 4. Core features

### 4.1 Discovery and search
NL search bar, three parallel retrieval engines (BM25, semantic KNN, graph traversal), hybrid fusion with re-ranking, dynamic facets, AI synthesis, saved searches (in-app), REST API.

### 4.2 Pattern Explorer
3-panel browse: navigation rail (TOGAF domain, ADM phase, Zachman lens, curated sets), card grid (with comparison mode), detail pane (problem, consequences, typed related artifacts). Relationship graph rendered as **read-only interactive React Flow canvas** showing artifact nodes and typed edges up to 2 hops.

### 4.3 Blueprint Studio
3-panel: pattern palette → **React Flow** composition canvas → manifest panel. Drag-and-drop pattern blocks, technology annotations, typed edges, scope boundary, staleness indicators. Blueprint manifest with ADR-linked technology choices. Validation engine. Export: ArchiMate, C4, Terraform scaffold, Markdown.

### 4.4 Contribution workflow (GitHub Enterprise)
All content lives in a GitHub Enterprise repository. Submissions create GitHub PRs via GitHub REST API. PR comments drive the RFC. PR merge = publication. 5-gate pipeline: Draft → Triage → RFC → Review → Board → Certify. Amendment and deprecation pathways.

### 4.5 Review and certification
Scorecard: 6 criteria × 20 pts. Board voting with quorum. RSA-4096 certification signatures stored in Postgres. Waiver system for Mandated artifacts (granted waivers visible to all users on the artifact page).

### 4.6 Lifecycle tracking
Nightly fitness batch. Automated Under_Watch triggers. Steward dashboard (Under Watch / Sunset / Due for Review). Deprecation impact graph.

### 4.7 Authentication and user management
Enterprise LDAP bind (ldaps://). Pending-first provisioning. Admin role assignment workflow. Auto-assign rules engine. In-app notifications only.

### 4.8 AI assistant
Conversational, RAG-backed, read-only. Session-scoped. Cites artifacts with links.

---

## 5. Non-functional requirements

| Requirement | Target |
|-------------|--------|
| Search P95 | < 500ms |
| Graph traversal P95 | < 300ms |
| AI synthesis P95 | < 5s |
| Studio canvas ops | < 100ms (local state) |
| Fitness batch | < 10 min |
| Page load P95 | < 2s |
| Concurrent users | 2,000 |
| Catalog scale (3yr) | 5,000 artifacts |
| Auth | Enterprise LDAP over TLS (ldaps) |
| Encryption at rest | AES-256 |
| Encryption in transit | TLS 1.3 minimum |
| Audit log | Immutable, append-only |
| Accessibility | WCAG 2.1 AA |
| Notifications | In-app only (no email, no Slack) |

---

## 6. Technology stack

### Backend
| Component | Technology |
|-----------|-----------|
| API framework | FastAPI (Python 3.12) |
| Graph database | FalkorDB |
| Search engine | Elasticsearch 8.x |
| Relational DB | PostgreSQL 16 |
| Cache / Queue | Redis 7 + Celery |
| LDAP | `ldap3` Python library |
| GitHub integration | `httpx` → GitHub Enterprise REST API |
| Embedding | text-embedding-3-large (OpenAI gateway) |
| LLM | GPT-4o or Claude (internal gateway) |

### Frontend
| Component | Technology |
|-----------|-----------|
| Framework | React 18 + Vite 5 + TypeScript 5 |
| State | Zustand + React Query |
| Canvas — Blueprint Studio | React Flow (editable) |
| Graph — Relationship viewer | React Flow (read-only, custom node types) |
| Rich text | TipTap |
| UI primitives | Radix UI |
| Styling | Tailwind CSS |
| Charts | Recharts |

---

## 7. Integrations

| System | Mechanism | Notes |
|--------|-----------|-------|
| Enterprise LDAP | ldaps:// bind, attribute fetch | No OIDC, no SAML |
| GitHub Enterprise | REST API, PAT or GitHub App | PR creation, merge, branch management |
| Technology Standards Register | REST API (nightly poll) | Triggers artifact staleness scan |
| LLM / Embeddings | OpenAI-compatible HTTP | Internal gateway |
| Notifications | In-app only | No email, no Slack |

---

## 8. Out of scope (v1)

- Email or Slack notifications
- Infrastructure automation / Terraform / Kubernetes manifests
- Mobile native app
- Real-time collaborative editing
- External EA tool integrations (Sparx, BiZZdesign)
- Public access

---

*End of Master PRD.*
