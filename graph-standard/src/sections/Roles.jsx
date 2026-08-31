import { Band, SectionHead, Table } from "../components/Primitives.jsx";

const ROLES = [
  ["Graph Platform Owner", "The standard, the catalogue, capacity, the conformance report", "Tier promotion to production"],
  ["Domain Data Owner", "The data in their domain's tenants; classification", "Ingestion contracts touching their data"],
  [<>Graph Steward <span className="small">(per domain)</span></>, "Model quality, label and property registrations, drift resolution", "Label and relationship registrations"],
  ["Pipeline Engineer", "Contract implementation, idempotency, rejection handling", "Contract version increments"],
  ["Architecture Review", "Promotion decisions, accepting documented deviations", "Deviations from this standard"],
];

export default function Roles() {
  return (
    <Band id="roles" aud="biz ops eng sci">
      <SectionHead index="09 · Accountability" title="Roles" aud={["Everyone"]}>
        Registrations need a named human, not a distribution list. Ownership that belongs to everyone
        belongs to no one — and unowned graphs are how the current situation arose.
      </SectionHead>

      <Table head={["Role", "Accountable for", "Signs off on"]}>
        {ROLES.map((r, i) => (
          <tr key={i}>
            <td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td>
          </tr>
        ))}
      </Table>
    </Band>
  );
}
