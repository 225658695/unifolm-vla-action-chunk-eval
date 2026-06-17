from __future__ import annotations

from schemas import PlannerOutput, SkillCommand


class SkillStateMachine:
    """Convert planner decisions into executable skill commands."""

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run

    def build_command(self, plan: PlannerOutput) -> SkillCommand:
        return SkillCommand(
            name=plan.next_skill,
            args=plan.skill_args,
            source_phase=plan.phase,
            dry_run=self.dry_run,
        )

