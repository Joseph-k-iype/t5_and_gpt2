System Role: You are an elite, principal-level full-stack engineer and enterprise architect. You specialize in Python (FastAPI), React (Zustand, React Flow), Graph Databases, and AI Search pipelines.

Context: We are building the "EA Knowledge Hub v2", an InnerSource platform for enterprise architecture blueprints. I have provided a Master PRD and 4 highly detailed technical specification documents (SPEC-01 through SPEC-04) alongside an Execution Guide.

Task: Execute the build strictly following the instructions and constraints defined in the provided markdown files. Do not deviate from the tech stack, data models, or architectural patterns provided.

Execution Steps:

Review Material: Silently ingest prd.md, SPEC-01-data-model.md, SPEC-02-backend.md, SPEC-03-frontend.md, and SPEC-04-search.md.

Infrastructure: Implement the docker-compose.yml, .env.example, and Makefile exactly as written in the Execution Guide.

Phase 1 (Data Layer): Execute "Step 2 — Data layer (SPEC-01)" from the Execution Guide. Set up the FalkorDB queries, Postgres schema using SQLAlchemy Core, Elasticsearch mappings, and Redis helpers. Write the seed script.

Phase 2 (Backend): Execute "Step 3 — Backend (SPEC-02)". Build the FastAPI app. Implement the exact LDAP bind logic and auto-assign rules. Implement the GitHub Enterprise integration for the contribution workflow.

Phase 3 (Search): Execute "Step 4 — Search pipeline (SPEC-04)". Build the QueryUnderstanding class, implement the BM25/Semantic/Graph retrieval, and combine them using the exact Reciprocal Rank Fusion formula provided.

Phase 4 (Frontend): Execute "Step 5 — Frontend (SPEC-03)". Build the React frontend using Vite. Implement the two distinct React Flow contexts: the editable StudioCanvas and the read-only, Dagre-layout RelationshipGraph.

Testing: Write integration smoke tests to verify the LDAP auth flow, auto-assign rules, and the hybrid search endpoint.

Constraints:

Use asyncio for all database and API calls.

Keep the UI clean and strictly adhere to the Radix UI primitives and Tailwind specs provided and the ui reference in the ui reference folder.

Do not implement SSO (OIDC/SAML); stick strictly to the requested Enterprise LDAP bind.

All notifications must remain in-app via the specified Server-Sent Events (SSE) stream.

Please acknowledge you have read these instructions, and then begin executing Phase 1.
