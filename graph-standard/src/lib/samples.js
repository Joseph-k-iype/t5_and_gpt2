/* The registry and contract artefacts. The standard deliberately carries no
   queries: it states what must be true and what must be measured, and leaves
   the implementation to whoever runs the platform. */

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
