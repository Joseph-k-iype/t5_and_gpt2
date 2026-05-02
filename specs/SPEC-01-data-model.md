# SPEC-01 — Data Model, Graph Schema, and Search Mappings
**EA Knowledge Hub v2 · Claude Code implementation spec**

---

## 1. FalkorDB graph schema

### 1.1 Node types

```cypher
-- ARTIFACT (base; always combined with content-type label)
(:Artifact {
  id:                    STRING   -- UUID v4
  slug:                  STRING   -- URL-safe unique identifier
  name:                  STRING
  summary:               STRING   -- max 500 chars
  status:                STRING   -- Draft|RFC_Open|Reviewed|Accepted|Certified|Under_Watch|Sunset|Deprecated|Archived
  certification_tier:    STRING   -- Community|Reviewed|Certified|Mandated
  version:               STRING   -- semantic version e.g. "3.1.0"
  fitness_score:         FLOAT    -- 0.0–100.0, computed nightly
  adoption_count:        INTEGER
  avg_rating:            FLOAT
  rating_count:          INTEGER
  open_issues_count:     INTEGER
  togaf_domains:         [STRING] -- [Business|Data|Application|Technology|Intelligence]
  adm_phases:            [STRING] -- [Preliminary|A|B|C|D|E|F|G|H|Requirements_Management]
  zachman_rows:          [STRING] -- [Planner|Owner|Designer|Builder|Subcontractor|Enterprise]
  zachman_cols:          [STRING] -- [What|How|Where|Who|When|Why]
  zachman_primary_row:   STRING
  zachman_primary_col:   STRING
  applicable_jurisdictions: [STRING]
  regulatory_tags:       [STRING]
  technology_tags:       [STRING]
  author_id:             STRING   -- User.id
  domain_steward_id:     STRING   -- User.id
  created_at:            INTEGER  -- Unix ms
  updated_at:            INTEGER
  review_due_at:         INTEGER  -- annual review deadline
  sunset_at:             INTEGER  -- null unless Sunset state
  content_body:          STRING   -- full markdown body
  github_pr_number:      INTEGER  -- linked GitHub Enterprise PR number
  github_branch:         STRING   -- working branch name
  git_sha:               STRING   -- latest merged commit SHA
  embedding_vector:      VECTOR(1536)
})

-- PATTERN
(:Pattern:Artifact {
  problem_statement:      STRING
  forces:                 STRING  -- markdown
  solution_structure:     STRING  -- markdown
  consequences_gains:     STRING  -- markdown
  consequences_tradeoffs: STRING  -- markdown (required, min 50 words enforced)
  known_uses:             [STRING]
  not_for:                [STRING]
})

-- BLUEPRINT
(:Blueprint:Artifact {
  problem_statement:          STRING
  architecture_diagram_json:  STRING  -- React Flow nodes/edges JSON
  operational_sla:            STRING
  operational_dr:             STRING
  operational_scaling:        STRING
  operational_monitoring:     STRING
  not_for:                    [STRING]
  terraform_scaffold:         STRING  -- HCL template (optional)
  c4_dsl:                     STRING  -- Structurizr DSL (optional)
  archimate_xml:              STRING  -- ArchiMate export (optional)
})

-- ADR
(:ADR:Artifact {
  adr_number:               STRING   -- e.g. "ADR-0101"
  context:                  STRING   -- markdown
  decision_statement:       STRING
  consequences_positive:    STRING
  consequences_tradeoffs:   STRING
  supersession_conditions:  [STRING]
  status_trail:             STRING   -- JSON array [{status, date, actor_id, note}]
})

-- PRINCIPLE
(:Principle:Artifact {
  statement:           STRING
  rationale:           STRING
  implications:        STRING
  regulatory_drivers:  [STRING]
})

-- ANTI_PATTERN
(:AntiPattern:Artifact {
  problem_description: STRING
  symptoms:            [STRING]
  why_harmful:         STRING
  better_alternative:  STRING
})

-- CURATED_SET
(:CuratedSet {
  id:          STRING
  name:        STRING
  description: STRING
  steward_id:  STRING
  created_at:  INTEGER
  updated_at:  INTEGER
  is_featured: BOOLEAN
  display_order: INTEGER
})

-- JURISDICTION_VARIANT
(:JurisdictionVariant {
  id:               STRING
  artifact_id:      STRING
  jurisdiction:     STRING
  regulatory_basis: STRING
  changes:          STRING  -- markdown
  created_at:       INTEGER
  updated_at:       INTEGER
})

-- USER
(:User {
  id:                 STRING   -- UUID v4
  ldap_uid:           STRING   -- LDAP uid attribute (unique)
  display_name:       STRING   -- from LDAP cn
  email:              STRING   -- from LDAP mail
  department:         STRING   -- from LDAP department
  title:              STRING   -- from LDAP title
  manager_dn:         STRING   -- from LDAP manager
  ldap_groups:        [STRING] -- from LDAP memberOf (group DNs)
  role:               STRING   -- Pending|ReadOnly|Contributor|DomainReviewer|CrossDomainReviewer|PrincipalReviewer|DomainSteward|BoardMember|Admin
  active:             BOOLEAN  -- false = suspended
  togaf_domains:      [STRING] -- domain affiliations (set by Admin)
  reviewer_level:     STRING   -- Domain|CrossDomain|Principal|null
  reviews_completed:  INTEGER
  first_login_at:     INTEGER
  last_active_at:     INTEGER
  preferences:        STRING   -- JSON: {persona_view, jurisdiction_context}
})

-- AUTO_ASSIGN_RULE
(:AutoAssignRule {
  id:          STRING
  name:        STRING
  condition:   STRING   -- JSON: {field: "department", operator: "contains", value: "Architecture"}
  role:        STRING   -- role to assign if condition matches
  priority:    INTEGER  -- lower number = higher priority, first match wins
  enabled:     BOOLEAN
  created_by:  STRING   -- User.id
  created_at:  INTEGER
})

-- SUBMISSION
(:Submission {
  id:           STRING
  artifact_id:  STRING
  github_pr_number: INTEGER
  github_pr_url:    STRING
  gate:         INTEGER  -- 1-5
  submitted_at: INTEGER
  submitted_by: STRING
  rfc_opens_at: INTEGER
  rfc_closes_at: INTEGER
})

-- REVIEW_SCORECARD
(:ReviewScorecard {
  id:                  STRING
  submission_id:       STRING
  reviewer_id:         STRING
  correctness:         INTEGER
  completeness:        INTEGER
  consequence_honesty: INTEGER
  uniqueness:          INTEGER
  togaf_coherence:     INTEGER
  jurisdiction_coverage: INTEGER  -- null if waived
  jurisdiction_waived: BOOLEAN
  total_score:         INTEGER
  verdict:             STRING  -- Approve|ApproveWithChanges|Reject
  reviewer_note:       STRING
  completed_at:        INTEGER
})

-- OPEN_ISSUE
(:OpenIssue {
  id:          STRING
  artifact_id: STRING
  filed_by:    STRING
  title:       STRING
  description: STRING
  issue_type:  STRING  -- FactualError|StaleReference|MissingJurisdiction|ConflictWithExisting|Other
  status:      STRING  -- Open|UnderReview|Resolved|Dismissed
  resolution:  STRING
  filed_at:    INTEGER
  resolved_at: INTEGER
})

-- IMPLEMENTATION (Row 6 adoption record)
(:Implementation {
  id:            STRING
  artifact_id:   STRING
  domain:        STRING
  project_name:  STRING
  status:        STRING  -- Planned|InProgress|Live|Decommissioned
  registered_by: STRING
  registered_at: INTEGER
  notes:         STRING
})

-- NOTIFICATION (in-app only)
(:Notification {
  id:           STRING
  user_id:      STRING
  event_type:   STRING  -- RFCOpened|ReviewVerdict|BoardDecision|FitnessAlert|AnnualReviewDue|DependencyEnteringSunset|RoleAssigned|AccountPending
  artifact_id:  STRING  -- null for user-management events
  message:      STRING
  read:         BOOLEAN
  created_at:   INTEGER
})

-- WAIVER
(:Waiver {
  id:               STRING
  artifact_id:      STRING  -- Mandated artifact being deviated from
  filed_by:         STRING
  deviation:        STRING
  justification:    STRING
  time_box_months:  INTEGER
  status:           STRING  -- Pending|Granted|Rejected
  board_note:       STRING
  granted_at:       INTEGER
  expires_at:       INTEGER
})

-- STANDARDS_ENTRY
(:StandardsEntry {
  id:          STRING
  name:        STRING
  category:    STRING
  version:     STRING
  deprecated:  BOOLEAN
  deprecated_at: INTEGER
  replaced_by: STRING
})
```

### 1.2 Relationship types

```cypher
(a:Artifact)-[:PAIRS_WITH]->(b:Artifact)
(a:Artifact)-[:CONTRASTS_WITH]->(b:Artifact)
(a:Artifact)-[:SUPERSEDES]->(b:Artifact)
(a:Artifact)-[:DEPENDS_ON]->(b:Artifact)
(a:Artifact)-[:IMPLEMENTS]->(b:Artifact)
(a:Artifact)-[:REFERENCES]->(b:Artifact)
(a:Artifact)-[:ENABLES]->(b:Artifact)
(a:Blueprint)-[:SOURCED_FROM {role: STRING}]->(b:Pattern)
(a:ADR)-[:GOVERNS]->(b:Artifact)
(a:ADR)-[:CO_DECIDED_WITH]->(b:ADR)
(a:Blueprint)-[:USES_TECHNOLOGY {component: STRING}]->(s:StandardsEntry)
(a:Artifact)-[:IN_SET {order: INTEGER}]->(cs:CuratedSet)
(a:Artifact)-[:HAS_VARIANT]->(v:JurisdictionVariant)
(a:Artifact)-[:IMPLEMENTED_AS]->(i:Implementation)
(a:Artifact)-[:HAS_ISSUE]->(o:OpenIssue)
(u:User)-[:SUBMITTED]->(s:Submission)
(s:Submission)-[:FOR_ARTIFACT]->(a:Artifact)
(sc:ReviewScorecard)-[:FOR_SUBMISSION]->(s:Submission)
(u:User)-[:STEWARDS]->(a:Artifact)
(u:User)-[:AUTHORED]->(a:Artifact)
(u:User)-[:BOARD_APPROVED]->(a:Artifact)
(u:User)-[:RATED {score: INTEGER, note: STRING, rated_at: INTEGER}]->(a:Artifact)
(w:Waiver)-[:WAIVES_COMPLIANCE_WITH]->(a:Artifact)
```

### 1.3 Key Cypher queries

```cypher
-- Full artifact with graph context
MATCH (a:Artifact {id: $id})
OPTIONAL MATCH (a)-[r1]->(related:Artifact)
  WHERE related.status NOT IN ['Archived', 'Draft']
OPTIONAL MATCH (a)-[:HAS_VARIANT]->(v:JurisdictionVariant)
OPTIONAL MATCH (a)-[:IMPLEMENTED_AS]->(i:Implementation)
OPTIONAL MATCH (a)-[:HAS_ISSUE]->(o:OpenIssue {status: 'Open'})
RETURN a,
  collect(DISTINCT {rel_type: type(r1), artifact: related}) AS relationships,
  collect(DISTINCT v) AS variants,
  collect(DISTINCT i) AS implementations,
  count(DISTINCT o) AS open_issue_count

-- 2-hop relationship graph for React Flow viewer
MATCH (a:Artifact {id: $id})-[r*1..2]-(related:Artifact)
WHERE related.status NOT IN ['Archived', 'Draft']
RETURN DISTINCT related,
  type(r[0]) AS primary_rel,
  length(r) AS hops
ORDER BY hops ASC, related.fitness_score DESC
LIMIT 30

-- Deprecation impact traversal
MATCH (target:Artifact {id: $id})<-[r]-(dependent:Artifact)
RETURN dependent.id, dependent.name, dependent.certification_tier,
       type(r) AS rel, dependent.domain_steward_id
UNION
MATCH (target:Artifact {id: $id})<-[:GOVERNS]-(adr:ADR)
RETURN adr.id, adr.name, adr.certification_tier, 'GOVERNS' AS rel, adr.domain_steward_id

-- Fitness batch source query
MATCH (a:Artifact)
WHERE a.status IN ['Certified', 'Mandated']
OPTIONAL MATCH (a)-[:IMPLEMENTED_AS]->(i:Implementation {status: 'Live'})
OPTIONAL MATCH (a)-[:HAS_ISSUE]->(o:OpenIssue {status: 'Open'})
OPTIONAL MATCH (a)<-[:SUPERSEDES]-(newer:Artifact)
RETURN a.id, a.updated_at,
  count(DISTINCT i) AS live_count,
  a.avg_rating, a.rating_count,
  count(DISTINCT o) AS open_issues,
  count(DISTINCT newer) AS superseder_count

-- Semantic duplicate detection
MATCH (a:Artifact)
WHERE a.status NOT IN ['Draft', 'Archived']
  AND a.id <> $new_id
RETURN a.id, a.name,
  vector.similarity.cosine(a.embedding_vector, $embedding) AS similarity
ORDER BY similarity DESC
LIMIT 5

-- Pending users for admin dashboard
MATCH (u:User {role: 'Pending', active: true})
RETURN u ORDER BY u.first_login_at DESC
```

---

## 2. PostgreSQL schema

```sql
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ldap_uid        TEXT UNIQUE NOT NULL,
    display_name    TEXT NOT NULL,
    email           TEXT NOT NULL,
    department      TEXT,
    title           TEXT,
    manager_dn      TEXT,
    ldap_groups     TEXT[] DEFAULT '{}',
    role            TEXT NOT NULL DEFAULT 'Pending'
                    CHECK (role IN ('Pending','ReadOnly','Contributor','DomainReviewer',
                                    'CrossDomainReviewer','PrincipalReviewer',
                                    'DomainSteward','BoardMember','Admin')),
    active          BOOLEAN DEFAULT TRUE,
    togaf_domains   TEXT[] DEFAULT '{}',
    reviewer_level  TEXT CHECK (reviewer_level IN ('Domain','CrossDomain','Principal')),
    reviews_completed INTEGER DEFAULT 0,
    preferences     JSONB DEFAULT '{}',
    first_login_at  TIMESTAMPTZ DEFAULT NOW(),
    last_active_at  TIMESTAMPTZ
);

CREATE TABLE auto_assign_rules (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    condition   JSONB NOT NULL,  -- {field, operator, value} e.g. {field:"department",operator:"contains",value:"Architecture"}
    role        TEXT NOT NULL,
    priority    INTEGER NOT NULL,
    enabled     BOOLEAN DEFAULT TRUE,
    created_by  UUID REFERENCES users(id),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX auto_assign_priority_idx ON auto_assign_rules(priority) WHERE enabled = TRUE;

CREATE TABLE audit_log (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id    UUID REFERENCES users(id),
    action      TEXT NOT NULL,
    artifact_id TEXT,
    entity_type TEXT,
    entity_id   TEXT,
    payload     JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX audit_log_artifact_idx ON audit_log(artifact_id);
CREATE INDEX audit_log_created_idx  ON audit_log(created_at DESC);

CREATE TABLE notifications (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id),
    event_type  TEXT NOT NULL,
    artifact_id TEXT,
    message     TEXT NOT NULL,
    action_url  TEXT,
    read        BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX notifications_user_unread ON notifications(user_id, read, created_at DESC);

CREATE TABLE saved_searches (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id),
    name            TEXT NOT NULL,
    query           TEXT,
    filters         JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_run_at     TIMESTAMPTZ
);

CREATE TABLE search_events (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID REFERENCES users(id),
    query_text          TEXT,
    filters             JSONB DEFAULT '{}',
    result_count        INTEGER,
    no_results          BOOLEAN DEFAULT FALSE,
    clicked_artifact_id TEXT,
    session_id          TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX search_no_results_idx ON search_events(no_results, created_at DESC)
    WHERE no_results = TRUE;

CREATE TABLE certification_signatures (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id     TEXT NOT NULL,
    artifact_version TEXT NOT NULL,
    signed_by       UUID NOT NULL REFERENCES users(id),
    signature_data  TEXT NOT NULL,   -- RSA-4096 signature of content_hash
    content_hash    TEXT NOT NULL,   -- SHA-256 of canonical artifact JSON
    signed_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE board_votes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id   TEXT NOT NULL,
    voter_id        UUID NOT NULL REFERENCES users(id),
    vote            TEXT NOT NULL CHECK (vote IN ('Accept','Reject','Abstain')),
    rationale       TEXT,
    voted_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(submission_id, voter_id)
);

CREATE TABLE fitness_history (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id     TEXT NOT NULL,
    fitness_score   NUMERIC(5,2) NOT NULL,
    component_scores JSONB NOT NULL,
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX fitness_history_artifact_idx ON fitness_history(artifact_id, computed_at DESC);

CREATE TABLE ldap_sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id),
    jwt_jti     TEXT UNIQUE NOT NULL,   -- JWT ID for revocation
    issued_at   TIMESTAMPTZ DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL,
    revoked     BOOLEAN DEFAULT FALSE
);
CREATE INDEX ldap_sessions_jti ON ldap_sessions(jwt_jti);
CREATE INDEX ldap_sessions_user ON ldap_sessions(user_id);
```

---

## 3. Elasticsearch index: `ea_hub_artifacts`

```json
{
  "mappings": {
    "properties": {
      "id":                     { "type": "keyword" },
      "slug":                   { "type": "keyword" },
      "content_type":           { "type": "keyword" },
      "name":                   { "type": "text", "analyzer": "english",
                                  "fields": { "keyword": { "type": "keyword" } } },
      "summary":                { "type": "text", "analyzer": "english" },
      "content_body":           { "type": "text", "analyzer": "english" },
      "problem_statement":      { "type": "text", "analyzer": "english" },
      "decision_statement":     { "type": "text", "analyzer": "english" },
      "consequences_gains":     { "type": "text", "analyzer": "english" },
      "consequences_tradeoffs": { "type": "text", "analyzer": "english" },
      "status":                 { "type": "keyword" },
      "certification_tier":     { "type": "keyword" },
      "version":                { "type": "keyword" },
      "fitness_score":          { "type": "float" },
      "adoption_count":         { "type": "integer" },
      "avg_rating":             { "type": "float" },
      "open_issues_count":      { "type": "integer" },
      "togaf_domains":          { "type": "keyword" },
      "adm_phases":             { "type": "keyword" },
      "zachman_rows":           { "type": "keyword" },
      "zachman_cols":           { "type": "keyword" },
      "zachman_primary_row":    { "type": "keyword" },
      "zachman_primary_col":    { "type": "keyword" },
      "applicable_jurisdictions": { "type": "keyword" },
      "regulatory_tags":        { "type": "keyword" },
      "technology_tags":        { "type": "keyword" },
      "author_id":              { "type": "keyword" },
      "created_at":             { "type": "date" },
      "updated_at":             { "type": "date" },
      "embedding_vector": {
        "type": "dense_vector",
        "dims": 1536,
        "index": true,
        "similarity": "cosine"
      }
    }
  },
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1
  }
}
```

---

## 4. Redis key schema

```
session:{jti}                  → JSON user context           TTL: 8h
search_cache:{query_hash}      → JSON results                TTL: 5min
ratelimit:search:{user_id}     → count                       TTL: 60s (max 60/min)
ratelimit:api:{user_id}        → count                       TTL: 60s (max 300/min)
fitness_compute:lock           → 1                           TTL: 15min
ldap_attr_cache:{ldap_uid}     → JSON LDAP attributes        TTL: 1h
notifications:queue            → Redis Stream (XADD/XREAD)
embeddings:queue               → Redis Stream
```

---

## 5. GitHub Enterprise repository structure

```
ea-hub-content/                  ← repository in GitHub Enterprise
  principles/
    {slug}.md
  patterns/
    {domain}/
      {slug}.md
  blueprints/
    {domain}/
      {slug}/
        blueprint.md
        diagram.json             ← React Flow canvas state (nodes + edges JSON)
        variants/
          {jurisdiction}.md
  adrs/
    ADR-{number}-{slug}.md
  anti-patterns/
    {slug}.md
  curated-sets/
    {slug}.md
  .hub-schema-version
```

Each markdown file has YAML front-matter with all structured fields. The body is human-readable markdown. Backend parses front-matter on PR merge and syncs to FalkorDB + Elasticsearch.

---

## 6. Seed data requirements

Generate in `scripts/seed_data.py`:

- **5 Architecture Principles** — one per TOGAF domain + one cross-cutting regulatory
- **15 Patterns** — 3 per domain, all Certified
- **5 Blueprints** — Certified, spanning Payments, Data, Platform
- **5 ADRs** — Accepted, linked to blueprints above
- **3 Anti-patterns** — Certified, distributed systems
- **2 Curated Sets** — "Cloud Migration", "Regulatory Compliance"
- **Users:**
  - `admin` (Admin role) — ldap_uid: `admin`
  - `steward` (DomainSteward, domain: Application) — ldap_uid: `steward`
  - `architect` (Contributor) — ldap_uid: `architect`
  - `pending_user` (Pending) — to demonstrate the onboarding dashboard
- **1 Auto-assign rule** — department contains "Architecture" → Contributor
- **1 Under_Watch artifact** — fitness_score: 45, 2 open issues
- **1 Sunset artifact** — with 2 dependent blueprints and 1 referencing ADR

---

*End of SPEC-01*
