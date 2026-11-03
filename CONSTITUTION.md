# CONSTITUTION.md

## Invariants

1. **Empirical Completion**: Prove every change with automated tests (`pytest`), build checks, or live HTTP/browser verification before declaring done.
2. **Real Execution**: Exercise real code paths and explicit test fixtures; keep security boundaries, permissions, and type checks intact.
3. **Aligned Architecture**: Record an ADR in [docs/adr/](file:///Users/andrestobelem/ws/at/danibook/docs/adr) and confirm with the user before shifting module boundaries, persistence engines, or runtime processes.
4. **Pointer Discipline**: Maintain progressive disclosure and a single source of truth across all Markdown files per [writing-for-agents](file:///Users/andrestobelem/ws/at/danibook/.agents/skills/writing-for-agents/SKILL.md).

## Escalation Protocol

When hitting contradictory specifications, missing credentials, or an invariant violation:
1. Stop execution immediately.
2. State the exact blocker and trade-off in one paragraph.
3. Wait for user direction.
