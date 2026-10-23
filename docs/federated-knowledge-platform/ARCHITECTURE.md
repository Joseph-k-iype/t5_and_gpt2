# Federated Knowledge Platform — Architecture

## Application Architecture

```mermaid
flowchart TB
 User[Users and AI Agents] --> Gateway[API Gateway / Auth]
 Gateway --> Frontends[Cypher SQL; later GraphQL Gremlin SPARQL]
 Frontends --> IR[Semantic Validation / Typed Graph-Relational IR]
 IR --> Planner[Federated Planner]
 Planner --> Executor[Sync and Async Executor]
 Executor --> Connectors[Source Adapters]
 Connectors --> Trino[Enterprise Data Fabric / Trino]
 Connectors --> Graph[Existing Graph Databases]
 Connectors --> Vector[Vector and Document Stores]
 Trino --> Records[Systems of Record]
 Executor --> Cache[Cache / Materialization]
 Executor --> Jobs[Durable Job Store]
 Ontology[Ontology Registry / SHACL] --> IR
 Ontology --> Mapping[Source Mapping Catalog]
 Mapping --> Planner
 Ontology --> Falkor[FalkorDB Metadata Projection]
```

The ontology registry is the control plane. Query gateway, planner and execution are the data plane. Trino owns structured federation; do not duplicate its connectors. FalkorDB stores a rebuildable metadata projection, not authoritative RDF/OWL semantics.

## Data Architecture

```mermaid
flowchart TB
 Artifacts[Immutable RDF OWL SHACL Artifacts] --> Concepts[Versioned Semantic Catalog]
 Concepts --> Mappings[Ontology-to-Source Mappings]
 Sources[Source Catalog / Capabilities / Policies] --> Mappings
 Mappings --> Logical[Logical Virtual Knowledge Graph]
 Logical --> Bindings[Typed Query IR Bindings]
 Logical --> Metadata[FalkorDB Derived Metadata]
 Bindings --> Live[Live Source Reads]
 Bindings --> Materialized[Materialized Relationship Index]
 Bindings --> Cached[Policy Scoped Cache]
 Live --> Fabric[Enterprise Data Fabric]
 Live --> Graphs[Graph Sources]
 Live --> Vectors[Vector / Document Sources]
 Live --> Provenance[Source Provenance and Observation Times]
 Materialized --> Provenance
 Cached --> Provenance
 Provenance --> Results[Results with Freshness and Completeness]
```

Authoritative ontology artifacts remain immutable. PostgreSQL holds registry metadata and versioned mappings. Source systems retain authoritative transactional records. Cache and materialized edges have explicit refresh, lineage and authorization policies. Do not claim globally consistent snapshots across unrelated sources.

## Decisions
1. RDF/OWL authoritative; FalkorDB metadata is derived.
2. Language-independent typed graph-relational IR.
3. Narrow bounded Cypher and SQL subset before other languages.
4. Live, cached, materialized and async execution modes.
5. Explicit unsupported-feature errors and partial-result states.
6. OWL reasoning added after core correctness and versioning.

## Risks
Cross-source fanout, entity identity collisions, ontology incompatibility, inconsistent snapshots, policy leaks via joins and cache, and distributed failure. Apply bounded traversal, batching, identity namespaces, pinned versions, source timestamps, policy-aware caches and resource budgets.
