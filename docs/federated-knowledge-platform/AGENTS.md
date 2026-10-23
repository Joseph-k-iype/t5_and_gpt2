# Agent Instructions

Scope: Federated Knowledge Platform project. Preserve unrelated existing repository files.

Read ARCHITECTURE.md, SPEC.md, IMPLEMENTATION_PLAN.md and CODEX_INSTRUCTIONS.md before coding.

- Milestone 1 is ontology registry, SHACL validation, versioned mappings and FalkorDB derived projection.
- Maintain separate application and data architecture diagrams.
- Immutable RDF/OWL is authoritative; FalkorDB metadata is rebuildable.
- Prefer virtualization; no blanket ingestion of source data.
- Query frontends compile to a backend-neutral typed IR.
- Start with bounded Cypher and SQL subsets; reject unsupported semantics.
- Use seeded synthetic banking data only.
- Apply deny-by-default policy, source provenance, freshness and completeness.
- Keep secrets out of source and logs.
- Write and execute unit, contract and integration tests; disclose any environment blockers.
- Update API contracts and ADRs when behavior changes.
- Do not claim benchmark or test results without running them.

Suggested commands once implemented:
```bash
docker compose -f infra/docker-compose.yml up -d
python -m pytest tests/unit
python -m pytest tests/integration
ruff check .
```
