/* Every code sample on the page, kept together so the registry, the contract
   and the queries can be checked against each other in one place. */

export const LABEL_YAML = `label: Customer
domain: party
owner: party-data-team
steward: j.doe
definition: >
  A legal or natural person with an active or historical contractual
  relationship with the firm. Excludes prospects (see: Prospect).

identity:
  strategy: minted            # minted | namespaced-natural
  key_property: entityId
  unique_constraint: true

properties:
  required:
    - key: entityId
      type: string
      classification: internal
    - key: customerType
      type: string
      classification: internal
      allowed_values: [INDIVIDUAL, ENTITY]
  optional:
    - key: legalName
      type: string
      classification: confidential
    - key: onboardedAt
      type: datetime
      classification: internal
  prohibited:
    - key: riskScore    # volatile; model as (:Customer)-[:HAS_ASSESSMENT]->(:RiskAssessment)

facets:                  # permitted secondary labels
  - Sanctioned
  - PoliticallyExposed

indexes:
  - type: range
    properties: [entityId]
  - type: range
    properties: [customerType]`;

export const REIFICATION = `// Before — cannot attach evidence, approver, or downstream events
(:Customer)-[:TRANSACTED_WITH {amount: 500, at: datetime()}]->(:Counterparty)

// After — the transaction is a first-class citizen
(:Customer)-[:INITIATED]->(:Transaction {entityId, amount, executedAt})-[:SETTLED_WITH]->(:Counterparty)
(:Transaction)-[:FLAGGED_BY]->(:Rule)
(:Transaction)-[:REVIEWED_BY]->(:Analyst)`;

export const IDEMPOTENT_WRITE = `UNWIND $batch AS row
MERGE (c:Customer {entityId: row.entityId})
ON CREATE SET c._ingestedAt = datetime()
SET c.legalName      = row.legalName,
    c.customerType   = row.customerType,
    c._contractId    = $contractId,
    c._sourceSystem  = $sourceSystem,
    c._sourceRecordId= row.sourceRecordId,
    c._pipelineVersion = $pipelineVersion,
    c._classification  = $classification,
    c._ingestedAt      = datetime()
RETURN count(c) AS written`;

export const NATURAL_KEY = `entityId = "<sourceSystem>:<labelSlug>:<naturalKey>"
// "corereg:customer:C-8842119"`;

export const MINTED_KEY = `entityId = "<labelSlug>:<ULID or UUIDv7>"
// "customer:01HQ2X8N4K7P9M3T"`;

export const KEY_SAMPLING = `MATCH (n:Customer) WITH n LIMIT 10000
UNWIND keys(n) AS k
RETURN DISTINCT k`;

export const Q_UNREGISTERED = `MATCH (t:Tenant)-[:DECLARES]->(l:LabelDef)
WHERE l._origin = 'observed' AND NOT EXISTS {
  MATCH (l)<-[:DECLARES]-(t) WHERE l._origin = 'declared'
}
RETURN t.graphKey, l.name, l.nodeCount
ORDER BY l.nodeCount DESC;`;

export const Q_BLAST = `MATCH (s:SourceSystem {name: $system})<-[:READS_FROM]-(c:IngestionContract)
      -[:WRITES_TO]->(t:Tenant)<-[:OWNS]-(team:Team)
RETURN t.graphKey, t.tier, collect(DISTINCT c.contractId), team.name;`;

export const Q_CONFIDENTIAL = `MATCH (p:PropertyKeyDef)-[:CLASSIFIED_AS]->(:Classification {level: 'confidential'})
MATCH (l:LabelDef)-[:HAS_PROPERTY]->(p)
MATCH (t:Tenant)-[:DECLARES]->(l)
RETURN p.name, collect(DISTINCT t.graphKey) AS tenants;`;

export const Q_FEDERATION = `MATCH (c:CanonicalConcept)<-[:ALIGNS_TO]-(l:LabelDef)<-[:DECLARES]-(t:Tenant)
WITH c, collect(DISTINCT t.graphKey) AS tenants
WHERE size(tenants) > 1
RETURN c.name, tenants;`;

export const Q_OWNERLESS = `MATCH (t:Tenant) WHERE NOT (t)<-[:OWNS]-(:Team)
RETURN t.graphKey, t.createdAt;`;

export const CONTRACT_YAML = `contractId: ing.kyc.corereg.v2
version: 2.1.0
status: active            # draft | active | deprecated | retired
targetGraph: prd.risk.kyc-network.v1

source:
  system: corereg
  object: dbo.customer_master
  mode: cdc
  cadence: PT15M
  freshnessSlo: PT1H

ownership:
  owningTeam: kyc-platform
  dataOwner: a.sharma
  steward: j.doe
  approvedBy: [arch-review-2026-03-11]

classification:
  level: confidential
  piiFields: [legal_name, date_of_birth, national_id]
  retention: P7Y
  deletionTrigger: source-record-deleted

writes:
  labels: [Customer]
  relationshipTypes: [IDENTIFIED_BY]

identity:
  Customer:
    strategy: minted
    resolver: identity-service/v3
    fallback: "corereg:customer:{customer_id}"

mapping:
  - source: customer_id
    target: { label: Customer, property: _sourceRecordId }
  - source: legal_name
    target: { label: Customer, property: legalName }
    transform: trim|normalise-whitespace
  - source: cust_type_cd
    target: { label: Customer, property: customerType }
    transform: lookup:ref/customer_type
    onUnmapped: reject

mergeSemantics: upsert
matchOn: [entityId]

validation:
  requiredProperties: [entityId, customerType]
  volumeTolerance: { min: 0.5, max: 2.0 }   # vs trailing 7-day mean
  onFailure: quarantine

indexRequirements:
  - { label: Customer, properties: [entityId], type: range }

lineage:
  upstream: [corereg.customer_master, ref.customer_type]`;

export const REL_YAML = `relationshipType: HOLDS_ACCOUNT
domain: party
owner: party-data-team
definition: >
  Asserts that a customer is a current or historical holder of an account.
  Historical holdings are retained with validTo set.

endpoints:
  - { source: Customer, target: TradeAccount }
  - { source: Customer, target: DepositAccount }

properties:
  required:
    - { key: validFrom, type: datetime }
  optional:
    - { key: validTo, type: datetime }
    - { key: holderRole, type: string, allowed_values: [PRIMARY, JOINT, BENEFICIAL] }

cardinality: many-to-many
parallelEdgesPermitted: true
parallelDisambiguator: [validFrom, holderRole]`;

export const D_STRUCTURE = `CALL db.labels();
CALL db.relationshipTypes();
CALL db.propertyKeys();
CALL db.indexes();
CALL db.constraints();`;

export const D_COUNTS = `MATCH (n:Customer) RETURN count(n);
MATCH ()-[r:HOLDS_ACCOUNT]->() RETURN count(r);`;

export const D_KEYS = `MATCH (n:Customer) WITH n LIMIT 10000
UNWIND keys(n) AS k
RETURN k, count(*) AS occurrences ORDER BY occurrences DESC;`;

export const D_GOVERNANCE = `MATCH (n:Customer)
RETURN
  count(n) AS total,
  sum(CASE WHEN n._contractId   IS NULL THEN 1 ELSE 0 END) AS missingContract,
  sum(CASE WHEN n._sourceSystem IS NULL THEN 1 ELSE 0 END) AS missingSource,
  sum(CASE WHEN n._ingestedAt   IS NULL THEN 1 ELSE 0 END) AS missingTimestamp;`;

export const D_PREFLIGHT = `MATCH (n:Customer)
WITH n.entityId AS id, count(*) AS c
WHERE c > 1 AND id IS NOT NULL
RETURN id, c ORDER BY c DESC;`;

export const D_HEALTH = `// Nodes missing an identity key
MATCH (n:Customer) WHERE n.entityId IS NULL RETURN count(n);

// Orphans
MATCH (n:Customer) WHERE NOT (n)--() RETURN count(n);

// Supernodes
MATCH (n:Customer) WITH n, count { (n)--() } AS degree
WHERE degree > 100000 RETURN n.entityId, degree ORDER BY degree DESC;`;

export const D_FRESHNESS = `MATCH (n) WHERE n._ingestedAt IS NOT NULL
RETURN n._contractId AS contract, max(n._ingestedAt) AS lastWrite
ORDER BY lastWrite ASC;`;
