# SPEC-03 — Frontend: React + Vite, React Flow (Canvas + Graph)
**EA Knowledge Hub v2 · Claude Code implementation spec**

---

## 1. Project structure

```
ea-hub-frontend/
  src/
    main.tsx
    App.tsx
    router/index.tsx
    pages/
      Login/                     ← LDAP credential login form
      Home/
      Search/
      Explorer/
      ArtifactDetail/
      BlueprintStudio/
      Contribute/
      Governance/
      Lifecycle/
      Admin/                     ← user management + role assignment + auto-assign rules
      Profile/
      GuidedJourney/
      Pending/                   ← landing page for Pending users
    components/
      layout/
        AppShell.tsx
        Sidebar.tsx
        Topbar.tsx
        NotificationBell.tsx     ← SSE-connected, shows unread count badge
        PendingBanner.tsx        ← shown to Pending users
      artifacts/
        ArtifactCard.tsx
        ArtifactBadge.tsx
        FitnessBar.tsx
        ZachmanTag.tsx
        TOGAFTag.tsx
        RelationshipGraph.tsx    ← React Flow (read-only, interactive)
        VariantPanel.tsx
        ImpactPanel.tsx
      search/
        SearchBar.tsx
        FilterPanel.tsx
        DynamicFacets.tsx
        AISynthesis.tsx
        SearchResultItem.tsx
      studio/
        PatternPalette.tsx
        StudioCanvas.tsx         ← React Flow (editable, drag-and-drop)
        BlueprintManifest.tsx
        ValidationResults.tsx
        TechChoiceRow.tsx
      contributions/
        SubmissionForm.tsx
        WorkflowTracker.tsx
        ScorecardForm.tsx
        RFCCommentThread.tsx     ← displays GitHub PR comments (read from API)
        BoardVotePanel.tsx
      admin/
        PendingUsersPanel.tsx
        UserTable.tsx
        RoleAssignModal.tsx
        AutoAssignRuleList.tsx
        AutoAssignRuleForm.tsx
        RuleTester.tsx
      governance/
        LifecycleDashboard.tsx
        WaiverForm.tsx
        AnnualReviewPanel.tsx
      shared/
        Button.tsx
        Modal.tsx
        Badge.tsx
        Tabs.tsx
        Tooltip.tsx
        MarkdownRenderer.tsx
        DiffViewer.tsx
        Spinner.tsx
        EmptyState.tsx
        Pagination.tsx
        ConfirmDialog.tsx
        RichTextEditor.tsx
    hooks/
      useSearch.ts
      useArtifact.ts
      useContribution.ts
      useNotifications.ts        ← SSE connection + unread count
      usePersonaView.ts
    stores/
      authStore.ts
      searchStore.ts
      studioStore.ts
      notificationStore.ts
    api/
      client.ts
      artifacts.ts
      search.ts
      contributions.ts
      users.ts
      admin.ts
      governance.ts
    types/
      artifact.ts
      search.ts
      contribution.ts
      user.ts
      admin.ts
    styles/
      tokens.css
      global.css
    utils/
      slugify.ts
      formatDate.ts
      fitnessColor.ts
  index.html
  vite.config.ts
  tsconfig.json
  tailwind.config.js
```

---

## 2. Routing

```tsx
const routes = [
  { path: '/login',               element: <Login />,         public: true },
  { path: '/pending',             element: <Pending /> },      // Pending role landing
  { path: '/',                    element: <Home /> },
  { path: '/search',              element: <Search /> },
  { path: '/explore',             element: <Explorer /> },
  { path: '/explore/:setSlug',    element: <Explorer /> },
  { path: '/artifacts/:id',       element: <ArtifactDetail /> },
  { path: '/artifacts/:id/edit',  element: <Contribute mode="edit" /> },
  { path: '/contribute',          element: <Contribute mode="new" /> },
  { path: '/studio',              element: <BlueprintStudio /> },
  { path: '/studio/:id',          element: <BlueprintStudio /> },
  { path: '/journeys/:slug',       element: <GuidedJourney /> },
  { path: '/governance',          element: <Governance /> },
  { path: '/governance/board',    element: <BoardAgenda /> },
  { path: '/lifecycle',           element: <Lifecycle /> },
  { path: '/profile',             element: <Profile /> },
  { path: '/admin',               element: <Admin /> },
  { path: '/admin/users',         element: <AdminUsers /> },
  { path: '/admin/rules',         element: <AdminRules /> },
]
```

Route guards:
- `/login` — redirect to `/` if already authenticated
- `/pending` — shown only to Pending users; all other routes redirect Pending users here
- `/admin/*` — require Admin role
- `/governance`, `/lifecycle` — require DomainSteward or Admin
- `/governance/board` — require BoardMember or Admin
- All other authenticated routes — require any active role (not Pending)

---

## 3. LDAP login page

```tsx
// pages/Login/index.tsx
//
// Layout: centred card, max-width 400px
//
// Fields:
//   - Username (text input) — LDAP uid, not email
//   - Password (password input)
//   - "Sign in" button
//   - Below button: small text "Sign in with your corporate LDAP credentials"
//   - No "forgot password" link (password reset is handled externally via IT)
//   - No "register" link (accounts created automatically on first login)
//
// On submit:
//   1. POST /auth/login with {username, password}
//   2. On 200: store token in authStore, navigate to / (or redirect param)
//   3. On 401: show "Invalid username or password"
//   4. On 403 (suspended): show "Your account has been suspended. Contact your administrator."
//   5. If returned user.role === 'Pending': navigate to /pending
//
// Loading state: disable button, show spinner inside button
```

---

## 4. Pending user landing page

```tsx
// pages/Pending/index.tsx
//
// Shown when authenticated user has role = Pending.
// Cannot access any other page — router redirects here.
//
// Layout: centred, informational
//
// Content:
//   - Hub logo + "Welcome to the EA Knowledge Hub"
//   - Amber info box: "Your account is pending role assignment.
//     An administrator will review your account and assign you a role.
//     You can browse public certified content below while you wait."
//   - Link: "Browse certified patterns" → opens catalog in read-only mode
//   - User's LDAP details shown for reference: display_name, email, department, title
//   - "Sign out" button
//
// Pending users can access:
//   GET /artifacts?certification=Certified&certification=Mandated (read-only browse)
//   GET /artifacts/{id} (read-only)
//   GET /search (read-only results)
// All contribute/submit/rate actions are hidden from the UI.
```

---

## 5. Admin pages

### 5.1 Admin user management (`/admin/users`)

Layout: two-column — Pending Users panel (left, 380px) + full user table (right)

**Pending Users panel (`PendingUsersPanel`):**
```
┌─────────────────────────────────────────┐
│  Pending users (3)                       │
├─────────────────────────────────────────┤
│  Jane Doe                               │
│  jane.doe@corp.com                      │
│  Dept: Group Architecture               │
│  Title: Enterprise Architect            │
│  Joined: 2 hours ago                    │
│  [Assign role ↗]                        │
├─────────────────────────────────────────┤
│  Bob Chen ...                           │
└─────────────────────────────────────────┘
```

"Assign role" opens `RoleAssignModal`.

**`RoleAssignModal` fields:**
- User info header (name, email, dept, title, LDAP groups)
- Role dropdown: ReadOnly | Contributor | DomainReviewer | CrossDomainReviewer | PrincipalReviewer | DomainSteward | BoardMember
- TOGAF domains multi-select (shown when role is DomainReviewer, CrossDomainReviewer, DomainSteward)
- Reviewer level dropdown (shown when role is DomainReviewer/CrossDomainReviewer/PrincipalReviewer): Domain | CrossDomain | Principal
- Reason text field (optional, stored in audit log)
- [Cancel] [Assign role] buttons

On assign: `PUT /admin/users/{id}/role` → panel refreshes, user disappears from Pending list, success toast: "Role assigned — user has been notified"

**Full user table (`UserTable`):**
- Columns: Name, Email, Department, Role (badge), TOGAF Domains, Active, Last active, Actions
- Role badge colours: Pending=grey, Contributor=blue, DomainSteward=green, BoardMember=purple, Admin=red
- Filter bar: search name/email, filter by role, filter by active status
- Row actions: Change role | Suspend | Activate | View audit trail
- Pagination: 20 per page

### 5.2 Auto-assign rules (`/admin/rules`)

Layout: full-width

**Rule list (`AutoAssignRuleList`):**
```
Priority | Name                              | Condition                          | Assigns To | Enabled | Actions
1        | Architecture dept → Contributor   | department contains "Architecture" | Contributor| ✓       | Edit Disable Delete
2        | EA Group → Reviewer               | memberOf contains "CN=EA-Team..."  | DomainReviewer | ✓  | Edit Disable Delete
```

- Drag-handle for reordering (updates priority on server)
- "New rule" button → opens `AutoAssignRuleForm`
- "Test rules" button → opens `RuleTester`

**`AutoAssignRuleForm` fields:**
- Rule name (text)
- Condition field: dropdown [department | title | ldap_groups]
- Condition operator: dropdown [contains | equals | starts_with]
- Condition value: text
- Assigns role: dropdown (all roles except Pending and Admin)
- Enabled: toggle
- Preview: shows "If user's {field} {operator} '{value}' → assign {role}"
- [Cancel] [Save rule]

**`RuleTester`:**
- Look up user by ldap_uid (calls `POST /admin/auto-assign-rules/test` with fetched attrs)
- Or manually enter attributes: department, title, groups (textarea, one per line)
- Shows result: "Would match rule #{priority}: {name} → assign {role}" or "No rules match → would assign: Pending"
- Useful for previewing rule changes before saving

---

## 6. React Flow usage — two distinct contexts

The hub uses React Flow in two fundamentally different contexts. Both use the same library but different configurations:

### 6.1 Blueprint Studio canvas (editable — `StudioCanvas.tsx`)

```tsx
import ReactFlow, {
  addEdge, Background, Controls, MiniMap,
  useNodesState, useEdgesState, ReactFlowProvider
} from 'reactflow'
import 'reactflow/dist/style.css'

// Custom node type for pattern blocks
const PatternNode = ({ data }: { data: PatternNodeData }) => (
  <div className="pattern-node">
    <div className="pattern-node-header">
      <span className="cert-dot" data-tier={data.certificationTier} />
      <strong>{data.name}</strong>
    </div>
    {data.stalenessFlag && (
      <div className="staleness-warning">⚠ Source entering {data.stalenessState}</div>
    )}
    <input
      className="tech-annotation"
      placeholder="Technology (e.g. Kafka on GCP)"
      value={data.techAnnotation}
      onChange={e => data.onAnnotationChange(e.target.value)}
    />
  </div>
)

const nodeTypes = { patternNode: PatternNode }

// Edge types: typed relationships
const edgeTypes = {} // use default edges with custom labels

export const StudioCanvas = () => {
  const { canvasNodes, canvasEdges, updateNode, addEdge: storeAddEdge } = useStudioStore()
  const [nodes, setNodes, onNodesChange] = useNodesState(canvasNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(canvasEdges)

  const onConnect = useCallback((params) => {
    // Open edge type selector dialog before adding edge
    openEdgeTypeDialog(params)
  }, [])

  const onDrop = useCallback((event) => {
    const patternData = JSON.parse(event.dataTransfer.getData('pattern'))
    const position = reactFlowInstance.screenToFlowPosition({ x: event.clientX, y: event.clientY })
    const newNode = {
      id: `pattern-${patternData.id}`,
      type: 'patternNode',
      position,
      data: {
        ...patternData,
        techAnnotation: '',
        onAnnotationChange: (val) => updateNode(patternData.id, { techAnnotation: val })
      }
    }
    setNodes(nds => [...nds, newNode])
  }, [reactFlowInstance])

  return (
    <ReactFlowProvider>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={e => e.preventDefault()}
        nodeTypes={nodeTypes}
        fitView
        snapToGrid
        snapGrid={[16, 16]}
      >
        <Background variant="dots" gap={16} size={1} />
        <Controls />
        <MiniMap nodeColor={n => certColor(n.data?.certificationTier)} />
      </ReactFlow>
    </ReactFlowProvider>
  )
}

// Scope boundary: implemented as a ResizableBox positioned absolutely on the canvas
// (not a React Flow node, to avoid interfering with drop targets)
// Toggle via a toolbar button; stores position+size in studioStore
```

**Pattern palette drag integration:**
```tsx
// PatternPalette.tsx
const PatternItem = ({ pattern }) => (
  <div
    className="palette-item"
    draggable
    onDragStart={e => {
      e.dataTransfer.setData('pattern', JSON.stringify(pattern))
      e.dataTransfer.effectAllowed = 'copy'
    }}
  >
    <span className="pdot" style={{ background: domainColor(pattern.togafDomains[0]) }} />
    {pattern.name}
  </div>
)
```

**Edge type selector dialog:**
When a user connects two nodes, a small popover appears at the midpoint asking them to choose the relationship type: Integrates with / Pairs with / Depends on / Contrasts with / Enables / References. The selected type is stored as the edge label and in the manifest's relationship list.

### 6.2 Relationship graph viewer (read-only — `RelationshipGraph.tsx`)

```tsx
import ReactFlow, {
  Background, Controls,
  useNodesState, useEdgesState
} from 'reactflow'
import { useNavigate } from 'react-router-dom'

// Custom read-only node
const ArtifactNode = ({ data }) => {
  const navigate = useNavigate()
  return (
    <div
      className={`artifact-node artifact-node--${data.certificationTier.toLowerCase()}`}
      onClick={() => navigate(`/artifacts/${data.id}`)}
      style={{ cursor: 'pointer' }}
    >
      <div className="artifact-node-type">{data.contentType}</div>
      <div className="artifact-node-name">{data.name}</div>
      <div className="artifact-node-meta">
        <span className="cert-badge">{certBadge(data.certificationTier)}</span>
        <span className="fitness">{data.fitnessScore?.toFixed(0)}</span>
      </div>
    </div>
  )
}

// Central node (current artifact) — larger, highlighted
const CentralArtifactNode = ({ data }) => (
  <div className="artifact-node artifact-node--central">
    <div className="artifact-node-name">{data.name}</div>
  </div>
)

const nodeTypes = {
  artifactNode: ArtifactNode,
  centralNode: CentralArtifactNode
}

export const RelationshipGraph = ({ artifactId, relationships }) => {
  // Build nodes and edges from API response
  const { nodes, edges } = useMemo(() =>
    buildGraphData(artifactId, relationships), [artifactId, relationships])

  const [rfNodes] = useNodesState(nodes)
  const [rfEdges] = useEdgesState(edges)

  return (
    <div style={{ height: 340, border: '0.5px solid var(--color-border-tertiary)', borderRadius: 8 }}>
      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges}
        nodeTypes={nodeTypes}
        nodesDraggable={false}      // read-only: no dragging
        nodesConnectable={false}    // read-only: no new connections
        elementsSelectable={true}   // allow click-to-navigate
        panOnDrag={true}
        zoomOnScroll={true}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant="dots" gap={20} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}

// Graph data builder
function buildGraphData(centralId: string, relationships: ArtifactRelationship[]) {
  const nodes = [
    {
      id: centralId,
      type: 'centralNode',
      position: { x: 0, y: 0 },  // dagre will layout
      data: relationships.find(r => r.artifact.id === centralId)?.artifact || {}
    },
    ...relationships.map((rel, i) => ({
      id: rel.artifact.id,
      type: 'artifactNode',
      position: { x: 0, y: 0 },  // positioned by dagre
      data: rel.artifact
    }))
  ]

  const edges = relationships.map(rel => ({
    id: `${centralId}-${rel.artifact.id}-${rel.relType}`,
    source: centralId,
    target: rel.artifact.id,
    label: relTypeLabel(rel.relType),
    type: 'smoothstep',
    markerEnd: { type: 'arrowclosed' },
    style: { strokeWidth: 1.5, stroke: relTypeColor(rel.relType) },
    labelStyle: { fontSize: 10, fill: 'var(--color-text-tertiary)' },
    animated: rel.relType === 'SUPERSEDES'
  }))

  // Apply dagre layout
  return applyDagreLayout(nodes, edges, { rankdir: 'LR', ranksep: 80, nodesep: 40 })
}

// Dagre layout helper
function applyDagreLayout(nodes, edges, options) {
  const dagreGraph = new dagre.graphlib.Graph()
  dagreGraph.setDefaultEdgeLabel(() => ({}))
  dagreGraph.setGraph(options)
  nodes.forEach(n => dagreGraph.setNode(n.id, { width: 180, height: 70 }))
  edges.forEach(e => dagreGraph.setEdge(e.source, e.target))
  dagre.layout(dagreGraph)
  return {
    nodes: nodes.map(n => {
      const { x, y } = dagreGraph.node(n.id)
      return { ...n, position: { x: x - 90, y: y - 35 } }
    }),
    edges
  }
}
```

Install `dagre` for layout: `npm install dagre @types/dagre`

---

## 7. Notification system (in-app SSE)

```tsx
// hooks/useNotifications.ts
export function useNotifications() {
  const { user } = useAuthStore()
  const { unreadCount, setUnreadCount, addNotification } = useNotificationStore()

  useEffect(() => {
    if (!user) return
    // Connect to SSE endpoint
    const eventSource = new EventSource(
      `${API_URL}/users/me/notifications/stream`,
      { withCredentials: false }
    )
    // Must pass auth header — use EventSource polyfill that supports headers
    // or use fetchEventSource from @microsoft/fetch-event-source

    eventSource.onmessage = async (event) => {
      const data = JSON.parse(event.data)
      // Fetch the full notification
      const notif = await api.get(`/users/me/notifications/${data.notification_id}`)
      addNotification(notif.data)
      setUnreadCount(c => c + 1)
    }

    return () => eventSource.close()
  }, [user])

  return { unreadCount }
}
```

Use `@microsoft/fetch-event-source` instead of native `EventSource` to support the `Authorization` header:

```tsx
import { fetchEventSource } from '@microsoft/fetch-event-source'

fetchEventSource(`${API_URL}/users/me/notifications/stream`, {
  headers: { Authorization: `Bearer ${token}` },
  onmessage(event) { /* handle */ },
  onerror(err) { /* reconnect logic */ }
})
```

**`NotificationBell.tsx`:**
```tsx
// Shows bell icon in Topbar with red badge for unread count
// Click opens a Popover (Radix UI) with last 10 notifications
// Each notification: icon by event_type, message, relative timestamp, "Mark read" on click
// "Mark all read" button at top of popover
// "View all" link → /profile (Notifications tab)
```

---

## 8. Page specifications

### 8.1 Home

Sections:
1. Hero search bar (large, prominent)
2. Featured curated sets (horizontal scroll)
3. Recently certified (6-card grid)
4. Your domain (if user has togaf_domains — 3 most recent artifacts from those domains)
5. Lifecycle alerts (DomainSteward only) — amber banner: "N artifacts Under Watch in your domain"

### 8.2 Search

Left panel: `FilterPanel` with all filter dimensions + active filter chips.  
Main area: search bar → `AISynthesis` (collapsible) → `DynamicFacets` → result list → pagination.

### 8.3 Pattern Explorer (3-panel)

Left rail: TOGAF domain, ADM phase, Zachman lens, Curated Sets filters.  
Center: filter chips + card grid (comparison mode for 2–3 selected cards).  
Right: artifact detail pane with `RelationshipGraph` (React Flow read-only) embedded below the related artifacts list.

### 8.4 Artifact Detail

Header: title, badges, version, fitness, tags, action buttons.

Tabs:
1. Overview — summary, architecture diagram (if Blueprint — rendered React Flow canvas read-only from `architecture_diagram_json`)
2. Technology — technology choices table + source patterns
3. Variants — jurisdiction variant pills + content
4. Related — `RelationshipGraph` (React Flow, full-height)
5. Governance — status trail, participants, scorecard summary, certification signature
6. History — version list from GitHub API, `DiffViewer` on select
7. Issues — open/resolved issues, file issue button
8. Implementations — registered implementations, register button

For blueprints on Tab 1, render the saved `architecture_diagram_json` as a read-only React Flow canvas:
```tsx
<ReactFlow
  nodes={parsedNodes}
  edges={parsedEdges}
  nodesDraggable={false}
  nodesConnectable={false}
  fitView
  proOptions={{ hideAttribution: true }}
>
  <Background />
  <Controls showInteractive={false} />
</ReactFlow>
```

### 8.5 Blueprint Studio (3-panel)

Left: `PatternPalette` (searchable, draggable items, grouped by TOGAF domain).  
Center: `StudioCanvas` (React Flow editable) with toolbar (zoom, fit, scope boundary toggle, grid toggle), validation badge, export buttons.  
Right: `BlueprintManifest` form (all fields, `TechChoiceRow` per technology choice, `ValidationResults`).

### 8.6 Contribute (5-step wizard)

Step 1: Content type selection + New vs Amendment toggle.  
Step 2: Classification (TOGAF domain, ADM phase, Zachman, jurisdiction, regulatory tags).  
Step 3: Content authoring (TipTap editors per required section, word count enforcement on trade-offs section).  
Step 4: Relationships (link source patterns, ADRs, technology choices).  
Step 5: Preview + validation results + Submit.

After submit: `WorkflowTracker` showing gate pipeline with GitHub PR link.

RFC comments are fetched from GitHub API and displayed in `RFCCommentThread` (read-only in hub UI; contributors can comment directly in GitHub or via the hub's "Add comment" form which calls the GitHub PR comments API).

### 8.7 Lifecycle dashboard (DomainSteward)

Three-tab layout:
- Under Watch: table with artifact, fitness score, state badge, action button + right panel for review options
- Sunset: deprecation cards with countdown, dependency status (`ImpactPanel`), notification log
- Due for Review: table with annual review overdue artifacts, `AnnualReviewPanel` on row click

### 8.8 Admin: Users (`/admin/users`)

Two-column: `PendingUsersPanel` (left) + `UserTable` (right, with filters and pagination).

### 8.9 Admin: Auto-assign rules (`/admin/rules`)

Full-width: `AutoAssignRuleList` (drag-to-reorder) + "New rule" button + `RuleTester` panel (collapsible).

---

## 9. Auth store (Zustand)

```typescript
interface AuthStore {
  user: User | null
  token: string | null
  isPending: boolean          // user.role === 'Pending'
  personaView: ZachmanRow     // overrides user's default Zachman row
  login: (token: string, user: User) => void
  logout: () => void
  setPersonaView: (row: ZachmanRow) => void
  updateUser: (partial: Partial<User>) => void  // after role change
}
```

---

## 10. Notification store (Zustand)

```typescript
interface NotificationStore {
  notifications: Notification[]
  unreadCount: number
  addNotification: (n: Notification) => void
  markRead: (id: string) => void
  markAllRead: () => void
  setUnreadCount: (fn: (c: number) => number) => void
}
```

---

## 11. API client

```typescript
// api/client.ts
import axios from 'axios'
import { useAuthStore } from '../stores/authStore'

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  timeout: 30000,
})

client.interceptors.request.use(config => {
  const token = useAuthStore.getState().token
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

client.interceptors.response.use(
  res => res,
  async error => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout()
      window.location.href = '/login'
    }
    if (error.response?.status === 403 && error.response?.data?.error === 'ACCOUNT_PENDING') {
      window.location.href = '/pending'
    }
    return Promise.reject(error)
  }
)

export default client
```

---

## 12. Dependencies

```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "react-router-dom": "^6.23.0",
    "reactflow": "^11.11.0",
    "dagre": "^0.8.5",
    "@types/dagre": "^0.7.52",
    "zustand": "^4.5.0",
    "@tanstack/react-query": "^5.40.0",
    "axios": "^1.7.0",
    "@tiptap/react": "^2.4.0",
    "@tiptap/starter-kit": "^2.4.0",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-popover": "^1.0.7",
    "@radix-ui/react-tabs": "^1.0.4",
    "@radix-ui/react-tooltip": "^1.0.7",
    "@radix-ui/react-select": "^2.0.0",
    "@radix-ui/react-dropdown-menu": "^2.0.6",
    "@radix-ui/react-switch": "^1.0.3",
    "recharts": "^2.12.0",
    "react-markdown": "^9.0.0",
    "remark-gfm": "^4.0.0",
    "@microsoft/fetch-event-source": "^2.0.1",
    "react-diff-viewer-continued": "^4.0.0",
    "tailwindcss": "^3.4.0",
    "clsx": "^2.1.0",
    "date-fns": "^3.6.0",
    "python-slugify": "^0.0.1"
  },
  "devDependencies": {
    "typescript": "^5.4.0",
    "vite": "^5.2.0",
    "@vitejs/plugin-react": "^4.3.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0"
  }
}
```

---

## 13. Environment variables

```
VITE_API_URL=http://localhost:8000/api/v1
VITE_FEATURE_AI_SYNTHESIS=true
VITE_FEATURE_AI_ASSISTANT=true
VITE_GITHUB_WEB_URL=https://github.corp.example.com
```

No OIDC/SSO variables — authentication is LDAP-backed with hub-issued JWTs.

---

*End of SPEC-03*
