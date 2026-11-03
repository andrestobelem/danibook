# AGENTS.md

## Governance & Domain Pointers

- [GLOSSARY.md](file:///Users/andrestobelem/ws/at/danibook/GLOSSARY.md): **Canonical Language** — Consult before naming classes, endpoints, UI labels, or domain concepts.
- [CONSTITUTION.md](file:///Users/andrestobelem/ws/at/danibook/CONSTITUTION.md): **Invariants & Escalation** — Consult when hitting architectural boundaries, missing credentials, or conflicting specs.
- [VALUES.md](file:///Users/andrestobelem/ws/at/danibook/VALUES.md): **Core Tenets** — Consult when weighing speed against verification, or scope against refactoring.
- [MISSION.md](file:///Users/andrestobelem/ws/at/danibook/MISSION.md): **Litmus Test** — Consult to judge whether an intervention adds leverage or churn.
- [VISION.md](file:///Users/andrestobelem/ws/at/danibook/VISION.md): **BHAG** — Consult for long-horizon autonomy targets.
- [docs/adr/](file:///Users/andrestobelem/ws/at/danibook/docs/adr): **Architecture Decisions** — Consult before altering memory tiers, model lifecycle, or the Bun/Python seam.
- [docs/research/](file:///Users/andrestobelem/ws/at/danibook/docs/research): **Technical Spikes** — Consult or record primary-source benchmarks and hardware findings.

## Skills

- `design-principles`: [.agents/skills/design-principles/SKILL.md](file:///Users/andrestobelem/ws/at/danibook/.agents/skills/design-principles/SKILL.md) — Designing component boundaries, interfaces, or decoupling responsibilities.
- `domain-modeling`: [.agents/skills/domain-modeling/SKILL.md](file:///Users/andrestobelem/ws/at/danibook/.agents/skills/domain-modeling/SKILL.md) — Challenging terminology, updating `GLOSSARY.md`, or recording an ADR.
- `grilling`: [.agents/skills/grilling/SKILL.md](file:///Users/andrestobelem/ws/at/danibook/.agents/skills/grilling/SKILL.md) — Stress-testing plans, assumptions, and edge cases with the user.
- `writing-for-agents`: [.agents/skills/writing-for-agents/SKILL.md](file:///Users/andrestobelem/ws/at/danibook/.agents/skills/writing-for-agents/SKILL.md) — Authoring or pruning `AGENTS.md`, governance docs, or skills.

## Execution Loop

1. **Ground**: Read target files, `GLOSSARY.md`, and relevant ADRs; confirm every symbol and constraint exists in the environment.
2. **Cut Surgically**: Modify only the lines required by the task, preserving canonical domain terms and existing docstrings.
3. **Prove**: Run `/opt/homebrew/bin/uv run pytest tests/` and verify live HTTP/UI behavior; every check must go green before reporting completion.
