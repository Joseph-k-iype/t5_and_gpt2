# Federated Knowledge Platform — Technical Specification

## Goal
A Trino-like federation engine for enterprise knowledge graphs that virtualizes structured data through an existing data fabric and combines external graphs, documents and vector stores. Provide synchronous, asynchronous, cached and materialized execution. Ontology registry first.

## Reference stack
Python 3.12+, FastAPI, Pydantic, RDFLib, pySHACL, PostgreSQL, FalkorDB (derived metadata), Trino, Qdrant, Docker Compose, pytest, OpenTelemetry. Use replaceable connector interfaces.

## Milestone 1: Ontology registry
Store immutable RDF/OWL artifacts with SHA-256 hash, namespace, semantic version, creator and status. Lifecycle: draft → validated → published → deprecated. Validate syntax, SHACL shapes, ontology imports, mapping compatibility and access policies. Index concepts by IRI and version. Source catalog includes schemas, capabilities, batch lookup, pushdown, freshness and policy reference. Semantic mappings pin ontology version, source schema version, physical object, canonical identity and policy. FalkorDB projection must rebuild deterministically.

Synthetic banking ontology: Party, Customer, CorporateCustomer, Account, Transaction, Organization, Document, RiskEvent. Relationships: ownsAccount, hasTransaction, controlsOrganization, mentionsEntity, linkedToRiskEvent. Datatypes: customerId, accountId, amount, eventTime. CorporateCustomer subclass Customer.

### Registry APIs
| Method | Endpoint | Description |
|---|---|---|
| POST | /v1/ontologies | Register draft |
| POST | /v1/ontologies/{id}/versions/{version}/validate | SHACL and syntax validation |
| POST | /v1/ontologies/{id}/versions/{version}/publish | Publish immutable version |
| GET | /v1/concepts?iri=...&version=... | Resolve class or property |
| POST | /v1/sources | Register source capabilities |
| POST | /v1/mappings | Register semantic-to-source mapping |
| GET | /v1/concepts/sources?iri=... | Discover mapped physical sources |
| POST | /v1/projections/metadata-graph | Rebuild FalkorDB metadata |

Errors: INVALID_ONTOLOGY, SHACL_VIOLATION, VERSION_CONFLICT, MAPPING_INCOMPATIBLE, UNAUTHORIZED, SOURCE_UNAVAILABLE. Mutations require authorization, audit and idempotency where appropriate.

### Example mapping
```yaml
mapping_id: customer-trino-v1
ontology_version: 1.0.0
concept_iri: https://example.org/enterprise/ontology/Customer
source_id: enterprise-fabric
source_schema_version: 2026-10
physical_object: enterprise.customer_view
identity: {canonical_key: customerId, source_column: customer_id}
capabilities: {batch_lookup: true, filter_pushdown: true}
freshness: {mode: live}
security_policy_ref: customer-read-policy
```

## Milestone 2: Federated query engine
Cypher and SQL supported subsets compile into a typed Graph-Relational IR. Operators: ScanEntity, ExpandEdges, Filter, Project, Join, ResolveEntity, Aggregate, VectorTopK, Limit, Materialize. IR nodes include ontology version, typed schema, cardinality, source candidates, pushdown, policy, provenance and freshness. Start with indexed entity lookup, filters, deterministic joins, 1–3-hop traversal and limits. Planner uses filter/projection pushdown and batched source requests. Reject unsupported constructs explicitly.

## Milestone 3: Hybrid execution
LIVE reads sources; CACHED obeys policy scope and TTL; MATERIALIZED uses refreshable lineage-tagged relationship indexes; ASYNC uses durable jobs, cancellation, retries, expiry and idempotency. AUTO chooses only modes permitted by caller freshness and policy constraints. Query results include provenance, observed timestamps, completeness and execution-plan hash.

## Security and testing
OIDC authentication, deny-by-default entity/edge/attribute authorization, upstream row/column policies, scoped credentials, rate limits, query budgets and audit. Tests: deterministic RDF/SHACL fixtures, version conflicts, identity resolution, cross-language equivalence, source outages, cache isolation, partial-result status. Benchmarks compare federated/materialized/hybrid on identical synthetic source snapshots. Measure p50/p95/p99, throughput, memory, source calls, transfer, freshness and correctness. P95 under 2s for indexed bounded local queries is a proposed target, not measured performance.

## Later
RDF fidelity, SPARQL, GraphQL, Gremlin, selected OWL 2 RL inference, asserted/inferred provenance and governed agent memory.
