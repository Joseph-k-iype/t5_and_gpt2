# Codex Master Instructions

Act as Principal Architect, Staff Data Engineer and Knowledge Graph Engineer. Read ARCHITECTURE.md, SPEC.md, IMPLEMENTATION_PLAN.md and AGENTS.md. Inspect the repository before changing it; do not overwrite unrelated t5/gpt2 code.

Build an actual working, tested Federated Enterprise Knowledge Platform, not just documents. Maintain separate Application Architecture and Data Architecture Mermaid diagrams. Ontology registry is milestone 1. Use Python 3.12+, FastAPI, RDFLib, pySHACL, PostgreSQL and FalkorDB derived projection, with adapters for Trino and Qdrant later.

## Execute now
1. Inspect repository and create file-by-file M1 backlog.
2. Scaffold monorepo, locked dependencies, Docker Compose and CI.
3. Create synthetic banking ontology, SHACL shapes, positive/negative fixtures.
4. Implement immutable ontology versions and PostgreSQL migrations.
5. Implement parsing, validation, publication and concept lookup APIs.
6. Implement source capability registry and version-pinned semantic mappings.
7. Implement deterministic FalkorDB metadata projection rebuild.
8. Add authorization, audit and typed error contracts.
9. Run lint, type checks, unit and real integration tests where available.
10. Demonstrate register → validate → publish → map → discover using API calls.
11. Report exact test commands/results, limitations, files changed and M2 backlog.

## Rules
Prioritize correctness, explicit unsupported-feature errors and source policy enforcement. RDF/OWL remains semantic authority. Avoid unbounded traversals, fake connector responses, silent semantic approximations, hard-coded credentials, production data and premature V2 implementation. Maintain provenance, observed timestamps and completeness. Document decisions and never claim unexecuted tests or benchmarks passed.

## Next milestones
M2: typed graph-relational IR, restricted Cypher/SQL, bounded federated planner and executor.
M3: async durable jobs, vector retrieval, cache/materialization, benchmarking.
M4: RDF interoperability and additional query frontends.
M5: OWL reasoning and governed agent memory.
