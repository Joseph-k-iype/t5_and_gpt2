# Implementation Plan

## M1 — Ontology Registry (implement first)
| Ticket | Deliverable | Acceptance |
|---|---|---|
| M1-01 | Python monorepo, Docker Compose, CI | Local startup and smoke tests |
| M1-02 | Synthetic banking RDF/OWL and SHACL | Valid fixtures pass; invalid fixtures fail |
| M1-03 | PostgreSQL schema, migrations, immutable versions | Hash and conflict tests |
| M1-04 | FastAPI registration/validation/publish APIs | Contract and OpenAPI tests |
| M1-05 | Source catalog and semantic mappings | Customer IRI resolves to Trino view |
| M1-06 | FalkorDB metadata projection | Deterministic rebuild |
| M1-07 | Authorization, audit, idempotency | Negative policy tests |
| M1-08 | End-to-end demonstration | Register → validate → publish → map → discover |

## M2 — Query kernel
Typed IR; Cypher/SQL restricted parsers; Trino/FalkorDB adapters; bounded traversal; source-aware planner; batched joins; explain plan; provenance and query result envelopes. Gate: 100% passing golden cross-language equivalence fixtures on supported overlap, explicit unsupported-feature errors.

## M3 — Hybrid and async
Qdrant adapter; durable job API and cancellation; cache isolation; selective relationship materialization; freshness metadata; failure injection; reproducible benchmarks. Gate: completeness, policy and provenance preserved in all modes.

## M4 — Semantic interoperability
RDF import/export fidelity, ontology migration checks, selected SPARQL and GraphQL/Gremlin adapters. Gate: deterministic semantic fixture tests.

## M5 — Reasoning and agent context
Selected OWL 2 RL inference, inferred/asserted flags, versioned inference and governed memory. Gate: reasoning correctness and policy tests.

## Synthetic data
Seeded 10k, 100k and 1m entity tiers: customers, accounts, transactions, organizations, beneficial owners, KYC documents and risk events. Include degree skew, missing data, simulated latency and outages. Never use real banking data.

## Repository layout
```text
services/ontology-registry/
services/query-gateway/
packages/semantic-model/
packages/source-catalog/
packages/query-ir/
packages/planner/
packages/executor/
adapters/trino/
adapters/falkordb/
adapters/qdrant/
ontologies/banking/
datasets/synthetic/
tests/unit/
tests/contract/
tests/integration/
tests/benchmarks/
infra/docker-compose.yml
docs/
```

Definition of done: implemented code, actual test execution, lint/type checks, updated contracts, documented risks and demo commands. Never report unrun tests as passing.
