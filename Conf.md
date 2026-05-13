Beyond Permit and Prohibit: Provenance and Conflict Resolution Extensions for Generalising ODRL

Abstract
The Open Digital Rights Language (ODRL) 2.2 provides a robust core ontology for expressing policy rules—defining who may perform which actions on which assets. However, as its adoption accelerates beyond digital rights management (DRM) into general regulatory, cross-border, and supply-chain use cases, critical structural gaps have emerged. Chief among these are its simplistic three-valued conflict strategy (permit, prohibit, invalid) and the complete absence of native provenance vocabulary. Drawing on recent W3C Community Group and academic consensus, this paper argues that ODRL requires richer conflict-combining algorithms aligned with deontic legal principles (e.g., lex superior, lex specialis) and XACML 3.0 paradigms, as well as a first-class provenance layer linking rules to their derivation history and legal sources. We propose two thin, backward-compatible W3C profiles—odrl-prov and odrl-cr—that align ODRL with PROV-O and LegalRuleML. Furthermore, we outline a standardisation roadmap to transition these profiles into the ODRL 3.0 core, providing the auditability and defeasible reasoning required for modern industrial policy engines.

1. Introduction

The W3C Open Digital Rights Language (ODRL) 2.2 has achieved substantial success as an interoperable vocabulary for expressing machine-readable policies [1]. Originally rooted in digital rights management, ODRL’s core abstractions (Policy, Rule, Permission, Prohibition, Duty, Constraint, Party, Asset) are highly suited for expressing granular access and usage control. Consequently, industrial use cases beyond privacy now dominate the ODRL adoption curve. Notable domains include IPTC RightsML 2.0 for media licensing, JPEG Trust for image provenance, the International Data Spaces (IDS) Usage Control Language, Gaia-X Verifiable Credentials, the W3C Text and Data Mining (TDM) Reservation Protocol, and various financial trading licences.

The appeal of ODRL in these diverse environments stems from its simplicity and its foundation in semantic web technologies, which allow policies to be easily serialized (e.g., as JSON-LD or Turtle) and integrated into existing knowledge graphs. However, this simplicity comes at a cost when applied to highly regulated, multi-jurisdictional environments. The rules governing a media asset in IPTC RightsML are fundamentally different in complexity from the rules governing the cross-border transfer of sensitive financial data. The former often involves straightforward commercial agreements; the latter requires reconciling overlapping national laws, sector-specific regulations, and corporate contracts.

Despite this broad traction, the rules-engine ecosystem is increasingly routing around ODRL's normative semantics. Production deployments—ranging from the UBS Enterprise Data Mesh to the ODRL-PAP compiler [2]—routinely translate ODRL syntax into execution environments like Open Policy Agent (OPA)/Rego or Answer Set Programming (ASP) via InstAL [3]. Each proprietary execution layer injects its own conflict-resolution semantics and provenance tracking atop ODRL's underspecified base. For instance, when an ODRL policy is compiled into Rego, the compiler must make arbitrary decisions about how to handle conflicts that ODRL itself cannot express natively. As a result, ODRL is currently providing an interoperable syntax, but not interoperable policy semantics. If two different policy engines evaluate the same ODRL policy and arrive at different conclusions because they employ different, unstandardised conflict-resolution logic, the promise of semantic interoperability is broken.

Recent W3C Community Group activity indicates a readiness to address these limitations. As noted by the CG chairs in October 2025, the group "is seeking input from the broader communities about whether it is time to plan formal standardisation of the next versions of the specifications" [4]. This paper answers that call. We isolate two primary structural blockers—conflict resolution and provenance—and propose two modular, backward-compatible extension profiles (odrl-prov and odrl-cr) to resolve them. By addressing these gaps, we aim to ensure that ODRL can serve not just as a descriptive format, but as a robust foundation for automated, legally sound policy enforcement.

2. Structural Gaps in ODRL 2.2

2.1 The Conflict Resolution Deficit

The W3C ODRL Information Model 2.2 defines the conflict property as taking exactly one of three ConflictTerm values: perm, prohibit, or invalid (the default). This mechanism fires only when a Permission and a Prohibition target the exact same Action on the exact same Asset.

This design is structurally insufficient for complex regulatory reasoning. It is strictly weaker than XACML 3.0's combining algorithms (e.g., deny-overrides, permit-unless-deny, first-applicable) [5], which support ordered evaluation and indeterminate verdicts, allowing for a much more nuanced approach to resolving competing directives. Furthermore, ODRL provides no native mechanism for resolving:

Conflicts between two competing Permissions or two Prohibitions across merged policies. When multiple policies are combined (a common scenario in federated data spaces), ODRL cannot determine which permission takes precedence if they offer overlapping but slightly different rights.

Conflicts among Duties or Obligations. If one policy mandates data deletion after 30 days and another mandates retention for 7 years for tax purposes, ODRL provides no semantic vocabulary to resolve the conflicting obligations.

Conflicts arising from implicit action subsumption. (e.g., permit display versus prohibit print). As Steyskal & Polleres [6] identified, ODRL operates under the limiting assumption that actions are strictly disjoint, ignoring the real-world reality that granting permission to "edit" might implicitly conflict with a prohibition to "delete."

Jurisdictional, temporal, or specificity-based precedence. ODRL lacks the vocabulary to express that rules drawn from different legal instruments might have different weights based on their source (e.g., federal law vs. state law) or their enactment date.

2.1.1 Anatomy of the Conflict Gap: A Worked Example

Consider a request to transfer a customer dataset across borders, a common scenario in international banking. This action is governed concurrently by three policies:

An EU GDPR-derived Policy ($P_1$) prohibiting transfer to non-adequate countries.

A US CLOUD-Act-derived Policy ($P_2$) permitting disclosure upon lawful subpoena.

A corporate contract Policy ($P_3$) permitting transfer with explicit user consent.

All three target the identical Asset and Action (transfer).

Under ODRL 2.2, if we merge these into a single policy set ($P_1 \cup P_2 \cup P_3$):

Setting conflict=invalid (the default): This voids the entire merged policy upon detecting a conflict. The result is a "default deny" outcome that blocks perfectly legitimate, compliant transfers (e.g., a transfer where the user has consented).

Setting conflict=prohibit: This ensures the prohibition (GDPR) always overrides the permissions. This incorrectly blocks transfers where a user has explicitly consented, violating the intended operation of GDPR Article 49 derogations.

Setting conflict=perm: This ensures the most permissive rule always wins. This is a catastrophically incorrect setting for a regulated financial entity, as it would allow a corporate contract to override a statutory prohibition.

A legal reasoning engine must resolve this using established deontic principles:

Lex superior: Statutory law outweighs corporate contract.

Lex specialis: Specific derogations (like user consent) override general prohibitions.

Lex posterior: Newer regulations supersede older ones [7].

Currently, ODRL offers no normative vocabulary for these strategies, forcing systems to offload the reasoning to unstandardised external engines, thereby losing the benefits of a shared, semantic policy language.

2.2 The Provenance Deficit

The ODRL Information Model defines no normative properties for rule derivation. When an auditor asks, "Why was this transfer prohibited at 14:07 GMT on 12 May 2026, on the authority of which clause of which regulation, and which engineer signed off the derivation?", ODRL 2.2 cannot natively answer. This lack of traceability is a critical failure point for adoption in regulated industries where demonstrating compliance is as important as achieving it.

Implementers often attempt to retrofit Dublin Core terms (dc:creator, dct:source) or PAV properties onto their ODRL instances. However, these are generic metadata tags. They lack the relational expressivity required to model complex derivation chains and fail entirely to provide audit-trail provenance over runtime policy evaluation. The W3C ODRL Temporal Model Working Draft attempts to address versioning but remains unfinished and notably omits derivation chains.

LegalRuleML [8] demonstrates a superior, purpose-built pattern for this domain. Using dedicated metadata blocks (<lrml:LegalSources>, <lrml:Authorities>, <lrml:Context>), it links each rule explicitly to the exact textual provision it derives from, the authoring agent, its governing jurisdiction, and its temporal efficacy (e.g., when the rule enters into force versus when it becomes applicable). Without adopting a similar Linked Data approach, ODRL remains unsuitable for high-stakes compliance and smart-contract anchoring, where the exact provenance of a rule must be undeniable.

3. Proposed Architecture: Extension Profiles

Following the Data Privacy Vocabulary (DPV) 2.0 [9] pattern of layering domain-agnostic extensions atop a core ontology, we propose two thin, backward-compatible profiles. This avoids overloading the ODRL core with complexity that might not be needed for simpler DRM use cases, while providing essential industrial capabilities for regulatory environments.

3.1 The odrl-prov Profile (Provenance)

The odrl-prov profile imports concepts from prov: and pav:, mapping ODRL constructs directly into the PROV-O [10] hierarchy. We declare $\text{odrl:Policy} \sqsubseteq \text{prov:Entity}$ and $\text{odrl:Rule} \sqsubseteq \text{prov:Entity}$.

The profile adopts the following as standard ODRL properties, providing a normative vocabulary for traceability:

prov:wasDerivedFrom: Maps the specific ODRL rule back to its legal or contractual source document.

prov:wasAttributedTo: Identifies the authoring agent, system, or knowledge engineer responsible for formalising the rule.

prov:hadPrimarySource: Provides the original legal text URI, ideally using stable identifiers like an Akoma Ntoso or ELI (European Legislation Identifier).

pav:version and pav:previousVersion: Provides strict version control, essential for tracing changes in policy over time.

dct:hasJurisdiction: Bounds the rule's operational scope geographically or legally.

Crucially, the profile introduces $\text{odrl:EvaluationRecord} \sqsubseteq \text{prov:Entity}$ to record every Policy Decision Point (PDP) execution. This is a vital addition, as it unifies design-time metadata (where the rule came from) with runtime decision provenance (when and why the rule was applied to a specific request).

3.1.1 Provenance Fragment Example

A PROV-O-aligned extension natively represents the traceability chain in a machine-readable format:

ex:RulePolicy123 a odrl:Policy, prov:Entity ;
    prov:wasDerivedFrom legis:GDPR-Art-44 ;
    prov:wasGeneratedBy ex:DerivationActivity-789 ;
    prov:wasAttributedTo ex:DataArchitectAlice ;
    prov:hadPrimarySource <https://eur-lex.europa.eu/eli/reg/2016/679> ;
    pav:version "1.2.0" ;
    pav:previousVersion ex:RulePolicy122 ;
    dct:issued "2026-04-30T09:00:00Z"^^xsd:dateTime .

ex:DerivationActivity-789 a prov:Activity ;
    prov:startedAtTime "2026-04-30T08:42:00Z"^^xsd:dateTime ;
    prov:wasAssociatedWith ex:LegislationConverter-v2.3 ;
    prov:used <https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng/pdf> .

# Runtime evaluation provenance
ex:Decision-2026-05-12-1407 a prov:Entity, odrl:EvaluationRecord ;          
    prov:wasGeneratedBy ex:Evaluation-2026-05-12-1407 ;
    prov:wasInfluencedBy ex:RulePolicy123 .


This fragment delivers rule-to-source traceability at the article level, agent attribution, derivation timestamps, and runtime decision provenance. This is the exact unbroken chain of evidence required by legal auditors to verify automated compliance decisions.

3.2 The odrl-cr Profile (Conflict Resolution)

The odrl-cr profile fundamentally restructures how ODRL handles conflict. It generalises odrl:ConflictTerm from a closed enumeration of three simple values into an extensible class, permitting a rich hierarchy of conflict resolution strategies:

XACML Equivalents: cr:DenyOverrides, cr:PermitOverrides, cr:FirstApplicable, cr:OnlyOneApplicable. These are mapped directly to ODRL deontic vocabulary, allowing ODRL to match the expressive power of the industry-standard access control language.

cr:PriorityWeighted: Assigns a cr:hasWeight xsd:decimal property to each Rule, enabling deterministic, numerical ordering for simple priority-based resolution.

cr:LexSuperior: Resolves conflicts by authoritative-source rank. It traverses each Rule's prov:wasAttributedTo chain to assess a cr:authorityRank xsd:integer, ensuring statutory law automatically overrides lesser agreements.

cr:LexSpecialis: Resolves by specificity. This defeasible resolution strategy calculates specificity by counting the number of odrl:refinement constraints attached to a rule; the more refined (and therefore more specific) rule wins. This formalises the heuristic identified in defeasible deontic semantics [11], allowing exceptions to override general rules gracefully.

cr:LexPosterior: Resolves by temporal precedence. It reads the dct:issued value of the source legal instrument via the prov:hadPrimarySource linkage, ensuring the most recently enacted rule wins.

cr:JurisdictionScoped: Resolves conflicts by matching the dct:hasJurisdiction property on the rule against the contextual information of the request, following an explicitly defined precedence vector (e.g., [EU, UK, US]).

To guarantee auditability, the profile dictates that each strategy MUST emit a cr:ResolutionReport carrying the properties cr:winningRule, cr:overriddenRules, and cr:strategyApplied. This aligns perfectly with the Compliance Report Model [12] recently introduced to make the output of ODRL evaluations formally interoperable and transparent.

4. Integration and Comparative Position

4.1 Comparative Position against Legacy Languages

A comparison with predecessor policy languages such as Rei [13] and KAoS [14] is highly instructive. Both utilised semantic-web technologies (RDF-S and OWL-DL, respectively) for policy expression and, notably, offered significantly richer conflict-handling mechanisms than ODRL 2.2. KAoS, for example, pioneered "policy deconfliction" via an online description logic theorem prover to automatically detect and resolve conflicts across different abstraction levels.

However, neither language natively addressed rule provenance, nor did they achieve anything approaching ODRL’s massive industry adoption. Therefore, rather than abandoning ODRL for a theoretically purer alternative, the optimal path is to fold the defeasible logic strengths of Rei and KAoS, and the rich metadata models of LegalRuleML, into ODRL via our proposed profiles. This allows the community to leverage the vast existing ecosystem of ODRL tooling while upgrading its inferential capacity to meet modern demands.

4.2 Engine Integration: ODRE, Rego, and Smart Contracts

The shift towards these extended profiles is not merely a theoretical exercise; it directly supports and enhances the emerging landscape of runtime execution engines. The ODRE framework [15], for instance, provides a robust enforcement layer that explicitly embeds behavioural specification languages alongside ODRL's descriptive ontology. The proposed odrl-cr profile maps seamlessly into ODRE's enforcement algorithm. It functions as a discrete, formalised resolution stage that guarantees deterministic conflict handling prior to any action invocation.

Furthermore, the practical realities of enterprise compliance necessitate compiling declarative policies into executable code. Existing ODRL-to-Rego compilers [2] currently have to invent arbitrary resolution logic when compiling to Open Policy Agent (OPA). With standardisation, these compilers could natively digest odrl-cr instances, translating cr:LexSpecialis or cr:PriorityWeighted directives directly into Rego's native conflict-handling syntax. This formally closes the gap between "policy as data" (ODRL) and "policy as code" (OPA).

This integration becomes even more critical in the context of decentralised enforcement. In smart-contract applications, such as those pioneered by the International Data Spaces Association (IDSA) contract-offer pattern or the Eclipse Dataspace Components (EDC), on-chain enforcement decisions carry significant legal and financial weight. An on-chain decision to permit or deny a data transfer must be fully reconstructable from off-chain policy artefacts. If a dispute arises, auditors must verify the exact version of the policy evaluated and its chain of authority. The odrl-prov profile standardises the exact cryptographic linkages, derivation activities, and agent attributions required to anchor this reconstructability securely in the semantic web, enabling legally sound, automated decentralized enforcement.

5. Conclusion and Standardisation Roadmap

The W3C ODRL standard has successfully transcended its DRM origins, but its normative semantics have not scaled to meet the rigorous requirements of its new, highly regulated domains. Because ODRL currently lacks robust conflict resolution and provenance models, developers are forced to hardcode essential regulatory logic into external execution engines, effectively destroying the semantic interoperability the language was designed to provide.

To resolve this impasse, we strongly recommend the W3C ODRL Community Group pursue the following phased standardisation roadmap:

Near-term (2026 CG cycle): Adopt the odrl-prov profile as a Community Group Note, normatively recommending prov:wasDerivedFrom, prov:wasAttributedTo, prov:hadPrimarySource, and dct:hasJurisdiction as standard metadata requirements for any regulatory mapping.

Mid-term (ODRL 3.0 Design Phase): Generalise odrl:ConflictTerm from an enumeration to a full class hierarchy. Mandate that profiles MAY add specific ConflictTerm instances, and formalise odrl-cr to carry the XACML-equivalent combining set alongside the deontic lex-superior/specialis/posterior strategies.

Longer-term (ODRL 3.1): Tightly couple odrl-cr resolutions with the emerging Formal Semantics Compliance Report Model, ensuring that every ODRL evaluation naturally and mandatorily emits an auditable cr:ResolutionReport.

By embracing this modular profiling pattern and synthesising established deontic principles into its core operations, ODRL can fulfill its potential as the universal lingua franca for industrial, cross-border, and regulatory policy automation.

References

[1] W3C ODRL Community Group. ODRL Information Model 2.2. W3C Recommendation, 2018. Available: https://www.w3.org/TR/odrl-model/
[2] W. Stefan, et al., "ODRL-PAP: An ODRL Policy Administration Point Compiler," GitHub Repository, 2024. [Online]. Available: https://github.com/wistefan/odrl-pap
[3] M. De Vos, A. Kirrane, J. Padget, and K. Satoh, "ODRL Policy Formulation and Execution via InstAL," in Proc. of the 3rd International Joint Conference on Rules and Reasoning (RuleML+RR), LNCS 11784, pp. 36-50, 2019.
[4] R. Iannella, N. Fornara, and V. Rodríguez-Doncel, "W3C Standard ODRL Policy gaining industry adoption," W3C Blog, Oct. 2025.
[5] OASIS. eXtensible Access Control Markup Language (XACML) Version 3.0, OASIS Standard, 2013.
[6] S. Steyskal and A. Polleres, "Towards Formal Semantics for ODRL Policies," in RuleML, LNCS 9202, pp. 360-375, 2015.
[7] T. Olson, "A Defeasible Deontic Calculus for Resolving Norm Conflicts," arXiv preprint arXiv:2407.04869, 2024.
[8] OASIS. LegalRuleML Core Specification Version 1.0, OASIS Standard, Aug. 2021.
[9] H. Pandit, et al., "DPV 2.0: A Modular Ontology for Data Privacy," in Proc. of the 23rd International Semantic Web Conference (ISWC 2024), LNCS 15233, pp. 171–193, 2024.
[10] W3C. PROV-O: The PROV Ontology. W3C Recommendation, 2013.
[11] G. Governatori, A. Rotolo, S. Villata, and F. Gandon, "One License to Compose Them All," in Proc. of the 12th International Semantic Web Conference (ISWC 2013), LNCS 8218, pp. 151-166, 2013.
[12] D. Slabbinck, P. Rojas Meléndez, B. Esteves, P. Colpaert, and R. Verborgh, "Formalising ODRL Policy Evaluation," in Proc. of the 22nd Extended Semantic Web Conference (ESWC 2025), LNCS 15719, 2025.
[13] L. Kagal, T. Finin, and A. Joshi, "A Policy Language for a Pervasive Computing Environment," in Proc. of the 4th IEEE International Workshop on Policies for Distributed Systems and Networks (POLICY), pp. 63-74, 2003.
[14] A. Uszok, J. Bradshaw, et al., "New Developments in Ontology-Based Policy Management: Increasing the Practicality and Comprehensiveness of KAoS," in Proc. of the IEEE POLICY, 2008.
[15] A. Cimmino, J. Cano-Benito, and R. García-Castro, "ODRE: ODRL Enforcement Framework," Computers & Security, vol. 150, 104282, 2025.
