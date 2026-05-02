# SPEC-02 — Backend: FastAPI, LDAP Auth, GitHub Integration, Business Logic
**EA Knowledge Hub v2 · Claude Code implementation spec**

---

## 1. Project structure

```
ea-hub-backend/
  app/
    main.py
    config.py
    deps.py
    routers/
      auth.py           ← LDAP login, token refresh, logout
      artifacts.py
      search.py
      contributions.py
      reviews.py
      governance.py
      users.py          ← user profile, notifications
      admin.py          ← user management, role assignment, auto-assign rules
      analytics.py
      lifecycle.py
    services/
      artifact_service.py
      search_service.py
      contribution_service.py
      review_service.py
      fitness_service.py
      notification_service.py   ← in-app only; no email, no Slack
      github_service.py         ← GitHub Enterprise REST API
      embedding_service.py
      governance_service.py
      ldap_service.py           ← LDAP bind, attribute fetch, auto-assign
    models/
      artifact.py
      user.py
      contribution.py
      review.py
      search.py
      admin.py
    db/
      falkordb.py
      postgres.py
      elasticsearch.py
      redis.py
    core/
      auth.py           ← JWT issue/verify, LDAP-backed
      fitness.py
      validation.py
    tasks/
      fitness_task.py
      embedding_task.py
      notification_task.py
      standards_sync_task.py
  scripts/
    seed_data.py
    create_indexes.py
    migrate.py
  tests/
  Dockerfile
  pyproject.toml
  alembic.ini
```

---

## 2. Configuration (`app/config.py`)

```python
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "EA Knowledge Hub"
    DEBUG: bool = False
    SECRET_KEY: str                  # for JWT signing (HS256 for internal tokens)
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173"]

    # Databases
    FALKORDB_HOST: str = "localhost"
    FALKORDB_PORT: int = 6379
    FALKORDB_PASSWORD: str = ""
    POSTGRES_URL: str
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    REDIS_URL: str = "redis://localhost:6379"

    # LDAP (Enterprise)
    LDAP_SERVER: str              # ldaps://ldap.corp.example.com:636
    LDAP_BASE_DN: str             # DC=corp,DC=example,DC=com
    LDAP_BIND_DN: str             # CN=svc-ea-hub,OU=ServiceAccounts,DC=corp,DC=example,DC=com
    LDAP_BIND_PASSWORD: str       # service account password
    LDAP_USER_SEARCH_BASE: str    # OU=Users,DC=corp,DC=example,DC=com
    LDAP_USER_FILTER: str = "(uid={username})"
    LDAP_ATTR_UID: str = "uid"
    LDAP_ATTR_CN: str = "cn"
    LDAP_ATTR_MAIL: str = "mail"
    LDAP_ATTR_DEPT: str = "department"
    LDAP_ATTR_TITLE: str = "title"
    LDAP_ATTR_MANAGER: str = "manager"
    LDAP_ATTR_GROUPS: str = "memberOf"
    LDAP_USE_TLS: bool = True      # always True for production
    LDAP_CA_CERT_PATH: str = ""    # path to enterprise CA cert bundle

    # JWT
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 8

    # GitHub Enterprise
    GITHUB_BASE_URL: str          # https://github.corp.example.com/api/v3
    GITHUB_TOKEN: str             # PAT with repo scope, or GitHub App token
    GITHUB_REPO_OWNER: str        # org or user name
    GITHUB_REPO_NAME: str         # ea-hub-content
    GITHUB_DEFAULT_BRANCH: str = "main"

    # Embeddings / LLM
    OPENAI_API_BASE: str          # internal gateway base URL
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMS: int = 1536
    LLM_MODEL: str = "gpt-4o"

    # Fitness thresholds
    FITNESS_UNDER_WATCH_THRESHOLD: float = 60.0
    FITNESS_ESCALATE_THRESHOLD: float = 40.0
    RFC_DEFAULT_DAYS: int = 14
    SUNSET_DEFAULT_DAYS: int = 90

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 3. LDAP authentication (`app/services/ldap_service.py`)

```python
from ldap3 import Server, Connection, ALL, NTLM, SIMPLE, Tls
from ldap3.core.exceptions import LDAPBindError, LDAPException
import ssl

class LDAPService:

    def __init__(self):
        tls = Tls(
            validate=ssl.CERT_REQUIRED,
            ca_certs_file=settings.LDAP_CA_CERT_PATH or None
        ) if settings.LDAP_USE_TLS else None
        self.server = Server(settings.LDAP_SERVER, use_ssl=True, tls=tls, get_info=ALL)

    async def authenticate(self, username: str, password: str) -> dict | None:
        """
        Perform LDAP bind as the user.
        Returns LDAP attributes dict on success, None on failure.

        Steps:
        1. Search for the user DN using service account bind
        2. Attempt bind with found DN + supplied password
        3. If bind succeeds, return attribute dict
        """
        # Step 1: find user DN via service account
        svc_conn = Connection(
            self.server,
            user=settings.LDAP_BIND_DN,
            password=settings.LDAP_BIND_PASSWORD,
            authentication=SIMPLE,
            auto_bind=True
        )
        search_filter = settings.LDAP_USER_FILTER.format(username=username)
        svc_conn.search(
            search_base=settings.LDAP_USER_SEARCH_BASE,
            search_filter=search_filter,
            attributes=[
                settings.LDAP_ATTR_UID,
                settings.LDAP_ATTR_CN,
                settings.LDAP_ATTR_MAIL,
                settings.LDAP_ATTR_DEPT,
                settings.LDAP_ATTR_TITLE,
                settings.LDAP_ATTR_MANAGER,
                settings.LDAP_ATTR_GROUPS,
            ]
        )
        if not svc_conn.entries:
            return None
        entry = svc_conn.entries[0]
        user_dn = entry.entry_dn

        # Step 2: bind as the user to verify password
        try:
            user_conn = Connection(
                self.server,
                user=user_dn,
                password=password,
                authentication=SIMPLE,
                auto_bind=True
            )
        except (LDAPBindError, LDAPException):
            return None

        # Step 3: return attributes
        attrs = {
            'ldap_uid': str(entry[settings.LDAP_ATTR_UID]),
            'display_name': str(entry[settings.LDAP_ATTR_CN]),
            'email': str(entry[settings.LDAP_ATTR_MAIL]),
            'department': str(entry.get(settings.LDAP_ATTR_DEPT, '')),
            'title': str(entry.get(settings.LDAP_ATTR_TITLE, '')),
            'manager_dn': str(entry.get(settings.LDAP_ATTR_MANAGER, '')),
            'ldap_groups': list(entry[settings.LDAP_ATTR_GROUPS].values or []),
        }
        user_conn.unbind()
        svc_conn.unbind()
        return attrs

    async def refresh_attributes(self, ldap_uid: str) -> dict | None:
        """
        Fetch latest LDAP attributes for an existing user
        (called on login to keep department/title current).
        Uses service account bind only — no password needed.
        """
        # cache in Redis for 1h to avoid hammering LDAP on every request
        cached = await redis.get(f"ldap_attr_cache:{ldap_uid}")
        if cached:
            return json.loads(cached)

        svc_conn = Connection(self.server, user=settings.LDAP_BIND_DN,
                              password=settings.LDAP_BIND_PASSWORD,
                              authentication=SIMPLE, auto_bind=True)
        svc_conn.search(
            settings.LDAP_USER_SEARCH_BASE,
            settings.LDAP_USER_FILTER.format(username=ldap_uid),
            attributes=[settings.LDAP_ATTR_CN, settings.LDAP_ATTR_MAIL,
                        settings.LDAP_ATTR_DEPT, settings.LDAP_ATTR_TITLE,
                        settings.LDAP_ATTR_MANAGER, settings.LDAP_ATTR_GROUPS]
        )
        if not svc_conn.entries:
            return None
        attrs = self._parse_entry(svc_conn.entries[0])
        await redis.setex(f"ldap_attr_cache:{ldap_uid}", 3600, json.dumps(attrs))
        svc_conn.unbind()
        return attrs

    def evaluate_auto_assign_rules(self, ldap_attrs: dict,
                                    rules: list[AutoAssignRule]) -> str | None:
        """
        Evaluate ordered auto-assign rules against LDAP attributes.
        Returns the role string of the first matching rule, or None.

        Supported operators: contains, equals, starts_with, in
        Supported fields: department, title, ldap_groups (checks any group DN)
        """
        for rule in sorted(rules, key=lambda r: r.priority):
            if not rule.enabled:
                continue
            cond = rule.condition
            field_value = ldap_attrs.get(cond['field'], '')
            if isinstance(field_value, list):
                # e.g. ldap_groups
                match = any(
                    self._apply_operator(v, cond['operator'], cond['value'])
                    for v in field_value
                )
            else:
                match = self._apply_operator(field_value, cond['operator'], cond['value'])
            if match:
                return rule.role
        return None

    def _apply_operator(self, field_val: str, operator: str, cond_val: str) -> bool:
        fv = field_val.lower()
        cv = cond_val.lower()
        if operator == 'contains':     return cv in fv
        if operator == 'equals':       return fv == cv
        if operator == 'starts_with':  return fv.startswith(cv)
        if operator == 'in':           return fv in [v.lower() for v in cond_val.split(',')]
        return False
```

---

## 4. JWT auth (`app/core/auth.py`)

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
import jwt, uuid
from datetime import datetime, timedelta

security = HTTPBearer()

def issue_jwt(user_id: str, role: str) -> tuple[str, str]:
    """Issue JWT. Returns (token, jti)."""
    jti = str(uuid.uuid4())
    payload = {
        'sub': user_id,
        'role': role,
        'jti': jti,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRY_HOURS),
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, jti

async def get_current_user(token=Depends(security)) -> User:
    try:
        payload = jwt.decode(token.credentials, settings.SECRET_KEY,
                             algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

    # Check revocation in Postgres
    session = await postgres.fetchone(
        "SELECT revoked FROM ldap_sessions WHERE jwt_jti = $1", payload['jti']
    )
    if not session or session['revoked']:
        raise HTTPException(401, "Session revoked")

    user = await postgres.fetchone(
        "SELECT * FROM users WHERE id = $1 AND active = TRUE", payload['sub']
    )
    if not user:
        raise HTTPException(401, "User not found or suspended")

    return User(**user)

def require_role(*roles: str):
    async def dep(user: User = Depends(get_current_user)):
        if user.role not in roles:
            raise HTTPException(403, f"Role {user.role!r} cannot perform this action")
        return user
    return dep

def require_active():
    """Blocks Pending users from write operations."""
    async def dep(user: User = Depends(get_current_user)):
        if user.role == 'Pending':
            raise HTTPException(403, "Account pending role assignment")
        return user
    return dep
```

---

## 5. GitHub Enterprise integration (`app/services/github_service.py`)

```python
import httpx

class GitHubService:
    """
    All GitHub Enterprise interactions via REST API.
    Uses a PAT (Personal Access Token) with repo scope,
    or a GitHub App installation token.
    """

    BASE = settings.GITHUB_BASE_URL   # https://github.corp.example.com/api/v3
    OWNER = settings.GITHUB_REPO_OWNER
    REPO  = settings.GITHUB_REPO_NAME
    HEADERS = {
        "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    async def create_branch(self, branch_name: str, from_ref: str = "main") -> str:
        """Create a new branch from from_ref. Returns SHA of branch head."""
        async with httpx.AsyncClient() as c:
            # Get base SHA
            r = await c.get(f"{self.BASE}/repos/{self.OWNER}/{self.REPO}/git/ref/heads/{from_ref}",
                            headers=self.HEADERS)
            r.raise_for_status()
            sha = r.json()['object']['sha']
            # Create branch
            r = await c.post(f"{self.BASE}/repos/{self.OWNER}/{self.REPO}/git/refs",
                             headers=self.HEADERS,
                             json={"ref": f"refs/heads/{branch_name}", "sha": sha})
            r.raise_for_status()
            return sha

    async def upsert_file(self, branch: str, path: str,
                           content: str, message: str, sha: str = None) -> str:
        """Create or update a file on a branch. Returns new file SHA."""
        import base64
        payload = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch
        }
        if sha:
            payload["sha"] = sha
        async with httpx.AsyncClient() as c:
            r = await c.put(
                f"{self.BASE}/repos/{self.OWNER}/{self.REPO}/contents/{path}",
                headers=self.HEADERS, json=payload)
            r.raise_for_status()
            return r.json()['content']['sha']

    async def create_pr(self, title: str, body: str,
                         head_branch: str, base_branch: str = "main") -> dict:
        """Open a pull request. Returns PR dict with number and html_url."""
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{self.BASE}/repos/{self.OWNER}/{self.REPO}/pulls",
                headers=self.HEADERS,
                json={"title": title, "body": body,
                      "head": head_branch, "base": base_branch})
            r.raise_for_status()
            return r.json()

    async def add_pr_comment(self, pr_number: int, body: str):
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{self.BASE}/repos/{self.OWNER}/{self.REPO}/issues/{pr_number}/comments",
                headers=self.HEADERS, json={"body": body})
            r.raise_for_status()

    async def merge_pr(self, pr_number: int, merge_message: str) -> str:
        """Merge PR with squash merge. Returns merge commit SHA."""
        async with httpx.AsyncClient() as c:
            r = await c.put(
                f"{self.BASE}/repos/{self.OWNER}/{self.REPO}/pulls/{pr_number}/merge",
                headers=self.HEADERS,
                json={"merge_method": "squash", "commit_message": merge_message})
            r.raise_for_status()
            return r.json()['sha']

    async def get_pr_comments(self, pr_number: int) -> list[dict]:
        async with httpx.AsyncClient() as c:
            r = await c.get(
                f"{self.BASE}/repos/{self.OWNER}/{self.REPO}/issues/{pr_number}/comments",
                headers=self.HEADERS)
            r.raise_for_status()
            return r.json()
```

---

## 6. API endpoints

All endpoints prefixed `/api/v1`. All require `Authorization: Bearer {token}` unless marked [public].

### 6.1 Auth

```
POST   /auth/login               [public]  LDAP credential login → JWT
POST   /auth/logout              Revoke current JWT session
GET    /auth/me                  Current user from JWT
POST   /auth/refresh-attributes  Re-fetch LDAP attributes (department/title may change)
```

**POST /auth/login request:**
```json
{ "username": "jsmith", "password": "..." }
```

**POST /auth/login response:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 28800,
  "user": {
    "id": "...", "display_name": "John Smith",
    "role": "Contributor", "active": true,
    "togaf_domains": ["Application"],
    "pending": false
  }
}
```

**Login flow (server-side):**
1. Call `LDAPService.authenticate(username, password)`
2. If None → 401 Invalid credentials
3. If success: check if user exists in Postgres by `ldap_uid`
   - Not exists: fetch enabled auto-assign rules (ordered by priority), run `evaluate_auto_assign_rules`
     - If rule matches → role = matched role
     - If no rule matches → role = Pending
     - Insert user into Postgres + FalkorDB User node
     - If role = Pending: create in-app notification (AccountPending) for all Admins
   - Exists: update `ldap_groups`, `department`, `title`, `last_active_at` from fresh LDAP attrs
4. If user `active = false` → 403 Account suspended
5. Issue JWT (sub=user.id, role=user.role), store in `ldap_sessions`
6. Return token + user object

### 6.2 Admin — user management

```
GET    /admin/users/pending         List Pending users (Admin only)
GET    /admin/users                 List all users with filters (Admin)
GET    /admin/users/{id}            User detail with LDAP attributes
PUT    /admin/users/{id}/role       Assign role (Admin only)
PUT    /admin/users/{id}/suspend    Suspend user (Admin only)
PUT    /admin/users/{id}/activate   Reactivate suspended user (Admin only)
PUT    /admin/users/{id}/domains    Set togaf_domains for steward/reviewer
```

**PUT /admin/users/{id}/role request:**
```json
{
  "role": "DomainSteward",
  "togaf_domains": ["Application", "Data"],
  "reviewer_level": null,
  "reason": "Assigned as Application chapter lead Q3 2026"
}
```

**PUT /admin/users/{id}/role logic:**
1. Validate actor is Admin
2. Update `role`, `togaf_domains`, `reviewer_level` in Postgres + FalkorDB User node
3. If previous role was Pending: create Notification for user (event_type: RoleAssigned)
4. Write to audit_log: action=ROLE_ASSIGNED, payload={old_role, new_role, reason}
5. Revoke all existing active JWT sessions for the user (sets `revoked=true` in `ldap_sessions`) — next API call by that user will get 401 and they must re-login, at which point they get a new JWT with updated role

```
GET    /admin/auto-assign-rules     List all rules (Admin)
POST   /admin/auto-assign-rules     Create rule (Admin)
PUT    /admin/auto-assign-rules/{id} Update rule (Admin)
DELETE /admin/auto-assign-rules/{id} Delete rule (Admin)
POST   /admin/auto-assign-rules/reorder  Reorder priorities (Admin)
POST   /admin/auto-assign-rules/test     Test rules against a user's LDAP attributes
```

**POST /admin/auto-assign-rules request:**
```json
{
  "name": "Architecture department → Contributor",
  "condition": {
    "field": "department",
    "operator": "contains",
    "value": "Architecture"
  },
  "role": "Contributor",
  "priority": 10,
  "enabled": true
}
```

**POST /admin/auto-assign-rules/test request:**
```json
{
  "ldap_uid": "jsmith",
  "ldap_attrs": {
    "department": "Group Architecture",
    "title": "Senior Enterprise Architect",
    "ldap_groups": ["CN=EA-Team,OU=Groups,DC=corp,DC=example,DC=com"]
  }
}
```
Response: `{ "matched_rule": {...} | null, "would_assign_role": "Contributor" | "Pending" }`

### 6.3 Artifacts

```
GET    /artifacts                   List (paginated, filterable)
POST   /artifacts                   Create (Draft)
GET    /artifacts/{id}              Full detail with graph context
PUT    /artifacts/{id}              Update (author or steward)
DELETE /artifacts/{id}              Soft-delete → Archived (Admin)
GET    /artifacts/{id}/relationships Graph-derived relationships
GET    /artifacts/{id}/history       Version history from GitHub
GET    /artifacts/{id}/diff/{v1}/{v2} Diff between versions
GET    /artifacts/{id}/implementations List implementations
POST   /artifacts/{id}/implementations Register implementation
GET    /artifacts/{id}/variants     List jurisdiction variants
POST   /artifacts/{id}/variants     Add variant
PUT    /artifacts/{id}/variants/{jurisdiction} Update variant
GET    /artifacts/{id}/issues       List issues
POST   /artifacts/{id}/issues       File issue
PUT    /artifacts/{id}/issues/{issue_id} Resolve/dismiss issue
POST   /artifacts/{id}/rate         Submit rating 1-5
GET    /artifacts/{id}/waivers      List waivers (Mandated artifacts)
POST   /artifacts/{id}/waivers      File waiver request
```

### 6.4 Search

```
POST   /search                      Main search (NL + filters → ranked results + synthesis)
GET    /search/suggest              Typeahead suggestions
GET    /search/saved                List saved searches (current user)
POST   /search/saved                Save a search
DELETE /search/saved/{id}           Delete saved search
GET    /search/analytics/gaps       No-results gap report (Steward/Admin)
```

### 6.5 Contributions

```
POST   /contributions               Submit artifact for review (creates GitHub PR)
GET    /contributions/{id}          Submission status + gate detail
GET    /contributions/pending        All pending submissions (Steward view, own domains)
POST   /contributions/{id}/triage   Steward: approve triage, open RFC
POST   /contributions/{id}/assign-reviewer Steward: assign reviewer
POST   /contributions/{id}/scorecard Reviewer: submit scorecard
POST   /contributions/{id}/board-vote BoardMember: cast vote
POST   /contributions/{id}/certify  Steward: publish (after board accept)
POST   /contributions/{id}/reject   Steward/Board: reject with rationale
POST   /contributions/{id}/request-changes Steward: return to author
POST   /contributions/{id}/deprecation-proposal Initiate deprecation
GET    /contributions/deprecations   Active deprecations (Steward)
POST   /contributions/deprecations/{id}/board-decision Board accept/reject deprecation
```

### 6.6 Reviews and governance

```
GET    /reviews/qualified-reviewers  Eligible reviewers for a content type
GET    /reviews/{scorecard_id}       Scorecard detail
GET    /governance/board-agenda      Upcoming board items (BoardMember, Admin)
GET    /governance/waivers           All waivers (Admin, Steward)
PUT    /governance/waivers/{id}      Board decision on waiver
GET    /governance/standards         Standards register entries
POST   /governance/standards/sync    Trigger sync (Admin)
```

### 6.7 Lifecycle

```
GET    /lifecycle/dashboard          Steward domain health
GET    /lifecycle/under-watch        Under Watch list
GET    /lifecycle/sunset             Active deprecations
GET    /lifecycle/review-due         Annual review due list
POST   /lifecycle/{id}/annual-review Sign off annual review
POST   /lifecycle/{id}/initiate-sunset Start sunset
GET    /lifecycle/{id}/impact        Deprecation impact graph
```

### 6.8 Users and notifications

```
GET    /users/me                    Profile
PUT    /users/me                    Update preferences
GET    /users/me/notifications      List notifications (paginated, newest first)
PUT    /users/me/notifications/{id} Mark read
PUT    /users/me/notifications/mark-all-read Mark all read
GET    /users/me/notifications/unread-count Count of unread
```

---

## 7. Key service implementations

### 7.1 Notification service (in-app only)

```python
class NotificationService:
    """
    All notifications are in-app only.
    No email. No Slack. No webhooks.
    """

    async def create(self, user_id: str, event_type: str,
                      message: str, artifact_id: str = None,
                      action_url: str = None):
        """
        Insert notification into Postgres notifications table.
        Push to Redis Stream for real-time delivery via SSE.
        """
        notif_id = str(uuid.uuid4())
        await postgres.execute("""
            INSERT INTO notifications (id, user_id, event_type, artifact_id,
                                       message, action_url)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, notif_id, user_id, event_type, artifact_id, message, action_url)

        await redis.xadd("notifications:stream",
                          {'user_id': user_id, 'notification_id': notif_id})

    async def notify_all_admins(self, event_type: str, message: str,
                                 artifact_id: str = None, action_url: str = None):
        admins = await postgres.fetch(
            "SELECT id FROM users WHERE role = 'Admin' AND active = TRUE"
        )
        for admin in admins:
            await self.create(admin['id'], event_type, message, artifact_id, action_url)

    async def notify_domain_stewards(self, togaf_domains: list[str],
                                      event_type: str, message: str,
                                      artifact_id: str = None):
        stewards = await postgres.fetch("""
            SELECT id FROM users
            WHERE role = 'DomainSteward' AND active = TRUE
              AND togaf_domains && $1
        """, togaf_domains)
        for s in stewards:
            await self.create(s['id'], event_type, message, artifact_id)

    async def notify_artifact_subscribers(self, artifact_id: str,
                                           event_type: str, message: str):
        """Notify users who have saved searches matching this artifact."""
        # TODO: match saved search filters against artifact metadata
        pass
```

**SSE endpoint for real-time notifications:**
```python
@router.get("/users/me/notifications/stream")
async def notification_stream(user: User = Depends(get_current_user)):
    """Server-sent events stream for real-time in-app notifications."""
    async def event_generator():
        last_id = '$'
        while True:
            entries = await redis.xread({'notifications:stream': last_id},
                                         block=30000, count=10)
            if entries:
                for stream, msgs in entries:
                    for msg_id, data in msgs:
                        if data.get('user_id') == user.id:
                            yield f"data: {json.dumps({'notification_id': data['notification_id']})}\n\n"
                        last_id = msg_id
            else:
                yield ": keepalive\n\n"  # prevent proxy timeout

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### 7.2 Contribution service (GitHub-backed)

```python
class ContributionService:

    async def submit(self, artifact_id: str, submitter: User) -> Submission:
        """
        1. Run ValidationService — hard errors abort submission
        2. Get artifact from FalkorDB
        3. Serialize artifact to markdown with YAML front-matter
        4. Create GitHub branch: `submission/{artifact_slug}-{timestamp}`
        5. Upsert file on branch via GitHubService
        6. Create GitHub PR (title: "Submission: {artifact name}", body: validation summary)
        7. Create Submission node in FalkorDB with github_pr_number
        8. Update artifact: status = RFC_Open (pending triage), github_pr_number
        9. Notify domain stewards (NotificationService.notify_domain_stewards)
        """

    async def steward_triage(self, submission_id: str, steward: User,
                              decision: str, rfc_days: int = 14) -> Submission:
        """
        Validate steward has domain affiliation matching artifact.
        If approve:
          - Set rfc_opens_at, rfc_closes_at
          - Post GitHub PR comment: "RFC opened — community feedback welcome until {date}"
          - Advance submission gate to 2
        If reject:
          - Post PR comment with rationale
          - Close GitHub PR
          - Reset artifact to Draft
          - Notify author
        """

    async def certify(self, submission_id: str, steward: User) -> Artifact:
        """
        1. Validate board has accepted (quorum + majority Accept votes in Postgres)
        2. Squash-merge the GitHub PR via GitHubService.merge_pr
        3. Record merge commit SHA on artifact (git_sha field)
        4. Bump semantic version
        5. Generate RSA-4096 certification signature over canonical artifact JSON
        6. Store signature in certification_signatures Postgres table
        7. Update artifact status → Certified, certification_tier → Certified
        8. Re-index in Elasticsearch
        9. Queue embedding re-generation
        10. Notify domain subscribers (in-app)
        11. Notify owners of artifacts that the new content references (in-app)
        """
```

---

## 8. RBAC matrix

| Action | ReadOnly | Contributor | DomainReviewer | DomainSteward | BoardMember | Admin |
|--------|----------|-------------|----------------|---------------|-------------|-------|
| Read Certified content | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Create artifact (Draft) | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| Submit for review | — | ✓ (own) | ✓ (own) | ✓ | ✓ | ✓ |
| Triage submissions | — | — | — | ✓ (own domain) | — | ✓ |
| Submit scorecard | — | — | ✓ (qualified) | — | ✓ (qualified) | ✓ |
| Board vote | — | — | — | — | ✓ | ✓ |
| Certify (publish) | — | — | — | ✓ (own domain) | — | ✓ |
| Initiate deprecation | — | — | — | ✓ | — | ✓ |
| Manage waivers | — | — | — | ✓ | ✓ | ✓ |
| Assign user roles | — | — | — | — | — | ✓ |
| Manage auto-assign rules | — | — | — | — | — | ✓ |
| Suspend users | — | — | — | — | — | ✓ |

**Pending users:** all endpoints except `GET /artifacts` (Certified/Mandated only), `GET /auth/me`, `GET /users/me/notifications`, and `POST /auth/logout` return 403.

---

## 9. Async tasks (Celery)

```python
# Nightly fitness computation — 02:00 UTC daily
@celery.task(name='fitness.nightly')
async def run_fitness_nightly():
    await fitness_service.run_nightly_batch()

# Embedding generation — triggered after create/update
@celery.task(name='embeddings.generate', bind=True, max_retries=3)
async def generate_embedding(self, artifact_id: str):
    try:
        artifact = await artifact_service.get(artifact_id)
        text = f"{artifact.name}. {artifact.summary}. {artifact.content_body}"
        vector = await embedding_service.embed(text)
        await falkordb.update_embedding(artifact_id, vector)
        await elasticsearch.update_embedding(artifact_id, vector)
    except Exception as exc:
        self.retry(exc=exc, countdown=60 * (self.request.retries + 1))

# Notification dispatch — reads Redis Stream, writes to Postgres (done in-process)
# No email/Slack tasks required

# Standards register sync — 03:00 UTC daily
@celery.task(name='standards.sync')
async def sync_standards_register():
    """Poll firm standards register REST API.
    For each newly deprecated standard: scan artifact technology_tags.
    For Certified/Mandated matches: create Under_Watch trigger + in-app notification to steward.
    """
```

---

## 10. pyproject.toml dependencies

```toml
[project]
name = "ea-hub-backend"
version = "2.0.0"
requires-python = ">=3.12"

dependencies = [
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.2.0",
    "falkordb>=1.0.0",
    "asyncpg>=0.29.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "alembic>=1.13.0",
    "elasticsearch[async]>=8.13.0",
    "redis>=5.0.0",
    "celery[redis]>=5.3.0",
    "ldap3>=2.9.1",              # Enterprise LDAP
    "httpx>=0.27.0",             # GitHub Enterprise API calls
    "openai>=1.30.0",
    "python-jose[cryptography]>=3.3.0",
    "python-multipart>=0.0.9",
    "PyYAML>=6.0.1",
    "markdown>=3.6",
    "python-slugify>=8.0.3",
    "cryptography>=42.0.0",      # RSA signatures
    "prometheus-fastapi-instrumentator>=7.0.0",
    "structlog>=24.1.0",
]
```

---

## 11. Health check endpoints

```python
@router.get("/health")
async def liveness():
    return {"status": "ok"}

@router.get("/ready")
async def readiness():
    checks = {}
    for name, check_fn in [
        ('falkordb', lambda: falkordb.ping()),
        ('postgres', lambda: postgres.execute("SELECT 1")),
        ('elasticsearch', lambda: es.ping()),
        ('redis', lambda: redis.ping()),
    ]:
        try:
            await check_fn()
            checks[name] = 'ok'
        except Exception as e:
            checks[name] = str(e)
    ok = all(v == 'ok' for v in checks.values())
    return JSONResponse({'status': 'ok' if ok else 'degraded', 'checks': checks},
                        status_code=200 if ok else 503)
```

---

*End of SPEC-02*
