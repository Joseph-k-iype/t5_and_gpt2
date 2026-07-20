# Delivery Script — "Beyond Permit and Prohibit"
**W3C Workshop on the Future of ODRL · London · June 2026**
Joseph K Iype, Senior Data Scientist, HSBC

*This is a word-for-word script, ~15 minutes at a natural speaking pace. Stage directions are in [brackets]. Don't memorise it verbatim — read it aloud three or four times until the phrasing feels like yours, then speak from the slide headlines. The bolded sentences are the ones worth landing exactly as written.*

---

## Slide 1 — Title (≈1 min)

[Stand centred, wait for the room to settle, smile.]

Good morning, everyone. My name is Joseph Iype — I'm a Senior Data Scientist at HSBC, working in Data and AI Architecture out of Bangalore.

Before I start, a quick word on where this paper comes from. My day job is building machine-readable policy inside a heavily regulated, multi-jurisdictional bank. So everything I'm about to show you comes from production pain, not from theory. These are gaps we hit when we actually try to run ODRL against real regulatory obligations.

And the title really is the whole argument. ODRL today gives us "permit" and "prohibit." **Real regulatory reasoning needs a great deal more than permit and prohibit** — and I want to show you exactly what, and propose how we get there without breaking anything.

[Advance.]

## Slide 2 — ODRL has outgrown DRM (≈1 min)

Let me start from success, not criticism — because ODRL is genuinely succeeding.

The 2.2 model's abstractions are simple, they're semantic-web native, and they drop straight into knowledge graphs. And that's why, if you look at where ODRL is actually being adopted today, DRM is no longer the main story. It's RightsML in news media. JPEG Trust. The International Data Spaces. Gaia-X. The W3C TDM Reservation Protocol. Machine-readable licences for financial trading data. ODRL has quietly become the default policy language for data spaces and data governance.

But here's the pivot: **the character of the rules has changed underneath us.** A media licence is a commercial agreement between two parties. Cross-border transfer of sensitive financial data means reconciling overlapping statutes, sector regulation, and contracts — all at once. And the simplicity that was a virtue in DRM becomes a liability in that world.

[Advance.]

## Slide 3 — Syntax without semantics (≈1 min)

So here's the sharpest way I can frame the problem: **the rules-engine ecosystem is already routing around ODRL's normative semantics.**

What happens in practice? Policies get written in ODRL, and then compiled — through tools like ODRL-PAP — into engines like OPA and Rego. And every one of those compilers has to make arbitrary decisions about conflicts that ODRL simply cannot express.

Which means — and this is the punchline — [pause] you can take the same policy, run it through two different engines, and get two different answers. Because each engine injects its own conflict logic where the standard is silent.

So what we have today is an interoperable *syntax*, but not interoperable *policy semantics*. And that is precisely the failure a standard exists to prevent.

[Advance.]

## Slide 4 — The conflict-resolution deficit (≈1.5 min)

Let's get concrete about the first gap.

ODRL's entire conflict model is one property with three values: perm, prohibit, or invalid. And it only fires on an exact collision between a Permission and a Prohibition on the same action and asset.

For a benchmark, look at XACML 3.0 — a standard from **2013**. Thirteen years ago it already had rule-combining algorithms, ordered evaluation, and indeterminate verdicts. On conflict handling, ODRL today is strictly weaker than a thirteen-year-old standard.

And that leaves four blind spots. [Count them on your fingers.]

One — permission versus permission. Two merged policies both permit, under incompatible conditions. ODRL doesn't even see that as a conflict, and it's the everyday case in federated data spaces.

Two — conflicting duties. One policy says delete after thirty days; another says retain for seven years for tax. Both are obligations, so the conflict property never fires.

Three — action subsumption. ODRL treats actions as disjoint — Steyskal and Polleres pointed this out years ago. But "edit" plainly interacts with "delete."

And four — precedence. There is no vocabulary at all for jurisdictional, temporal, or specificity precedence. Lex superior, lex posterior, lex specialis — the things lawyers actually reason with — cannot be said.

[Advance.]

## Slide 5 — Worked example (≈2 min, slow down)

Let me make this painfully concrete with the example from the paper. [Slow your pace here.]

One request: transfer a customer dataset across borders. Three policies govern that same asset and that same action, at the same time.

P1 is derived from GDPR — it prohibits transfer to countries without an adequacy decision. P2 is derived from the US CLOUD Act — it permits disclosure in response to a lawful subpoena. And P3 is a corporate contract — it permits transfer where the customer has given explicit consent.

Now walk through every conflict setting ODRL offers.

Set it to *invalid* — the default — and the entire merged policy set is void. You're now in default-deny, and you're blocking transfers that are perfectly compliant and fully consented.

Set it to *prohibit*, and GDPR always wins. That sounds safe — until you remember that Article 49 derogations exist precisely to allow consented transfers. You've just made lawful business impossible.

Set it to *perm* — and this one, for a bank, is catastrophic — a corporate contract now overrides a statutory prohibition. [Pause. Let that sit.]

So here's the takeaway, and it's the most important sentence in the paper: **it's not that ODRL picks the wrong strategy — no expressible strategy is correct.** The right answer requires precedence, weights, and context that ODRL has no way to say.

[Advance.]

## Slide 6 — The provenance deficit (≈1.5 min)

The second gap. Let me read you the question that motivates it — this is the question an auditor actually asks:

[Read deliberately, almost verbatim from the slide.] *"Why was this transfer blocked at 14:07 GMT on the 12th of May 2026 — under which clause of which regulation — and which engineer approved the derivation?"*

ODRL 2.2 has no standards-based answer to that question. None.

Three quick points. First, the Information Model contains zero normative properties for derivation traceability — nothing links a rule to the legal text it came from.

Second, the retrofits don't work. Dublin Core and PAV give you flat metadata tags — no relational expressivity for derivation chains, and nothing at all about runtime evaluation. The Temporal Model draft is unfinished, and it omits derivation anyway.

Third — LegalRuleML already shows us the purpose-built pattern: every rule linked to the exact textual provision, the authoring agent, and its temporal efficacy. The pattern exists. ODRL just doesn't have it.

And for regulated industries, this is existential. **We don't just have to enforce compliance — we have to evidence it, with the same rigour.**

[Advance.]

## Slide 7 — The proposal: two thin profiles (≈1 min)

So, the proposal. And I want to lead with the design philosophy, because it's deliberate.

We follow the layering pattern that DPV 2.0 proved out: thin, modular profiles sitting on top of the core — not a heavier core. If you're doing media licensing, you never touch these profiles and nothing changes for you. If you're in a regulated domain, you switch them on.

Two profiles. `odrl-prov` answers the question: *where did this rule come from, and why did it fire?* And `odrl-cr` answers: *when rules collide, which one wins — and can we prove it?*

And let me stress this because it matters for adoption: **both profiles are strictly additive. Existing ODRL 2.2 policies are untouched.** The adoption cost for anyone who doesn't need them is effectively zero.

[Advance.]

## Slide 8 — odrl-prov (≈1 min)

Mechanically, `odrl-prov` is very simple — and that's the point.

We declare `odrl:Policy` and `odrl:Rule` as subclasses of `prov:Entity`, and we adopt four properties. `wasDerivedFrom` links a rule to its source document. `wasAttributedTo` names the authoring agent. `hadPrimarySource` pins the rule to the legal text itself — via ELI or Akoma Ntoso identifiers. And PAV's version and previousVersion give you change control.

The genuinely novel piece is the last one: **`odrl:EvaluationRecord`** — a record of every policy-decision-point execution. And what that does is unify design-time metadata with runtime decision provenance in one graph. That unification is the whole trick — it's what turns provenance from documentation into *evidence*.

[Advance.]

## Slide 9 — The code example (≈1 min)

Here's what that looks like in Turtle. Don't worry about reading every line — let me just trace the story it tells. [Gesture at the three blocks top to bottom.]

At the top: a policy derived from GDPR Article 44, attributed to a named architect, with its primary source pinned to the EUR-Lex ELI URI, and versioned with a link to its predecessor.

In the middle: the derivation activity itself — timestamped, and associated with a specific version of the converter tool. So even tool-assisted derivation is auditable.

And at the bottom: the runtime decision record — from 14:07 on the 12th of May. Which closes the loop on the auditor's question from three slides ago.

One machine-readable graph. All standard PROV-O. Nothing invented that didn't need to be.

[Advance.]

## Slide 10 — odrl-cr (≈2 min, slow down again)

Now the conflict-resolution profile — and there's really only one structural move here.

`ConflictTerm` stops being a closed three-value enumeration, and becomes an **extensible class**. Everything else follows from that.

First family of strategies: direct XACML equivalents. DenyOverrides, PermitOverrides, FirstApplicable, OnlyOneApplicable. That gives ODRL instant parity with the established industry standard for policy combination.

Second: `cr:PriorityWeighted`, with a `hasWeight` decimal on each rule. Deterministic ordering — and it's enough to express jurisdictional and specificity precedence. Statute outweighs contract. Specific provision outweighs general one.

And here is my favourite part as a practitioner: **auditability by construction.** Every strategy MUST emit a `cr:ResolutionReport` — the winning rule, the overridden rules, and the strategy that was applied — aligned with the Compliance Report Model. So the answer is never just "deny." The answer is: *deny, because this rule beat those rules, under this strategy.*

Tie it back to the worked example: DenyOverrides gives you the statutory baseline, weights express the Article 49 derogations, and you get a resolution report on every single decision. That scenario — which was unrepresentable — becomes routine.

[Advance.]

## Slide 11 — Why extend, not replace (≈1 min)

Now, some of you are thinking: why bolt this onto ODRL? Why not use something richer? Fair question — and we've actually run this experiment as a community.

Rei and KAoS, from the early 2000s, had far richer conflict handling than anything I've shown you. KAoS ran automated policy *deconfliction* through a description-logic theorem prover. Genuinely impressive.

But neither handled provenance natively. And neither achieved anything remotely close to ODRL's adoption. **Theoretical purity without an ecosystem loses. Every time.**

So the strategy here is deliberate: fold the defeasible-logic strengths of Rei and KAoS, and LegalRuleML's metadata model, *into* ODRL via profiles. Keep the tooling, keep the installed base — upgrade the inferential capacity.

[Advance.]

## Slide 12 — Conclusion & asks (≈1 min)

Let me land the plane. The thesis in one line: **ODRL is winning on adoption, but losing on semantics.** And two thin, backward-compatible profiles close that gap.

Three asks of this workshop.

One: adopt `odrl-prov` and `odrl-cr` as W3C profiles — they're thin, modular, and they break nothing.

Two: align, don't reinvent — PROV-O, PAV, LegalRuleML, XACML all exist; we should reuse them.

Three: incorporate this into ODRL 3.0 — so that auditability and defeasible reasoning become standard equipment for modern policy engines, not proprietary add-ons.

[Pause, look up from the slide.]

Let's make ODRL not just a descriptive format, but a robust foundation for automated, legally sound policy enforcement.

Thank you. [Beat.] I'd particularly welcome discussion on two open points — how strategies should compose when policy sets nest, and the design of the EvaluationRecord. Questions?

---

## Quick delivery reminders

- **Total ≈ 15 minutes.** If you're running long, compress slides 2 and 11 — never slides 5 and 10; those carry the argument.
- The two deliberate pauses: after "*a corporate contract overrides a statutory prohibition*" (slide 5) and after reading the auditor's question (slide 6). Silence does the work there.
- The auditor's question and the closing line are the only two things to deliver near-verbatim.
- If a question comes on nested-strategy composition, use the honest framing from the Q&A section of your talking-points doc: XACML's per-policy-set scoping is the natural path, and it's flagged for the community-group phase.
