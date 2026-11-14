from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, Literal

from postgrest import AsyncPostgrestClient
from pydantic import TypeAdapter

from halo.chem.eval.objectives import objective_metric
from halo.chem.mol import IDENTITY_POLICY
from halo.chem.results import Failure, Metric, MetricResult, to_metric
from halo.core.errors import DataError
from halo.db.client import execute
from halo.llm.runtime import ModelCall
from halo.models.generation import (
    Candidate,
    Occurrence,
    PreparedCandidate,
    RGroupEdit,
    Run,
    RunConfig,
    RunOutcome,
    RunStart,
    RunStatus,
)
from halo.models.molecule import Descriptors, MoleculeProfile, Screening
from halo.models.usage import TokenUsage
from halo.types import GenerationRunId, MoleculeId, OptimizationTask, ProjectId, UserId

CANDIDATES = "candidates!candidates_run_fkey"
MODEL_CALLS = TypeAdapter(Sequence[ModelCall])
RUN_START = TypeAdapter(RunStart)
RUN_STATUS = TypeAdapter(RunStatus)
RUN_CONFIG = TypeAdapter(RunConfig)
SOURCE_PREDICTIONS = TypeAdapter(Mapping[OptimizationTask, Metric])
RGROUP_EDIT = TypeAdapter(RGroupEdit)
RUN_SELECT = (
    f"*, {CANDIDATES}("
    "ordinal, design, predictions, molecules(id, canonical_smiles, "
    "molecule_profiles(*)))"
)


def metric_json(result: MetricResult) -> dict[str, Any]:
    """Encode a property value or prediction failure."""
    if isinstance(result, Metric):
        return {
            "status": "computed",
            "value": result.value,
            "unit": result.unit,
            "method": result.method,
        }
    return {"status": "failed", "reason": result.reason}


def metric_from_json(value: Mapping[str, Any]) -> MetricResult:
    """Read a saved property value or prediction failure."""
    if not isinstance(value, Mapping):
        raise DataError("Stored metric must be an object.")
    status = value.get("status")
    if status == "computed":
        unit, method = value.get("unit"), value.get("method")
        if not isinstance(unit, str) or not unit.strip():
            raise DataError("Stored metric needs a unit.")
        if not isinstance(method, str) or not method.strip():
            raise DataError("Stored metric needs a method.")
        metric = to_metric(value.get("value"), unit=unit, method=method)
        if isinstance(metric, Failure):
            raise DataError("Stored metric needs a finite numeric value.")
        return metric
    if status == "failed":
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise DataError("Stored failure needs a reason.")
        return Failure(reason)
    raise DataError(f"Stored metric has an unknown status: {status!r}")


def profile_from_row(canonical_smiles: str, row: Mapping[str, Any]) -> MoleculeProfile:
    return MoleculeProfile(
        canonical_smiles=canonical_smiles,
        descriptors=Descriptors(
            molecular_weight_g_mol=row["molecular_weight_g_mol"],
            logp=row["logp"],
            tpsa_angstrom_sq=row["tpsa_angstrom_sq"],
            heavy_atom_count=row["heavy_atom_count"],
            h_bond_acceptors=row["h_bond_acceptors"],
            h_bond_donors=row["h_bond_donors"],
            rotatable_bonds=row["rotatable_bonds"],
            ring_count=row["ring_count"],
            aromatic_rings=row["aromatic_rings"],
        ),
        screening=Screening(
            qed=row["qed"],
            lipinski_violations=row["lipinski_violations"],
            sa_score=metric_from_json(row["sa_score"]),
            pains_alerts=tuple(row["pains_alerts"] or ()),
        ),
        predictions={
            name: metric_from_json(value) for name, value in row["predictions"].items()
        },
    )


def profile_fields(profile: MoleculeProfile) -> dict[str, Any]:
    """Prepare a profile for saving, preserving the first of each alert."""
    descriptors = profile.descriptors
    screening = profile.screening
    return {
        "molecular_weight_g_mol": descriptors.molecular_weight_g_mol,
        "logp": descriptors.logp,
        "tpsa_angstrom_sq": descriptors.tpsa_angstrom_sq,
        "heavy_atom_count": descriptors.heavy_atom_count,
        "h_bond_acceptors": descriptors.h_bond_acceptors,
        "h_bond_donors": descriptors.h_bond_donors,
        "rotatable_bonds": descriptors.rotatable_bonds,
        "ring_count": descriptors.ring_count,
        "aromatic_rings": descriptors.aromatic_rings,
        "qed": screening.qed,
        "lipinski_violations": screening.lipinski_violations,
        "sa_score": metric_json(screening.sa_score),
        "pains_alerts": list(dict.fromkeys(screening.pains_alerts)),
        "predictions": {
            name: metric_json(result) for name, result in profile.predictions.items()
        },
    }


def candidate_payload(
    candidates: Sequence[PreparedCandidate],
    criteria: Sequence[OptimizationTask] = (),
) -> list[dict[str, Any]]:
    """Prepare distinct candidates for saving in proposal order."""
    rows = []
    seen = set()
    for candidate in candidates:
        profile, design = candidate.profile, candidate.design
        if profile.canonical_smiles in seen:
            continue
        seen.add(profile.canonical_smiles)
        row = {
            "canonical_smiles": profile.canonical_smiles,
            "standardization_version": IDENTITY_POLICY,
            "ordinal": len(rows),
            "profile": profile_fields(profile),
        }
        if design:
            row["design"] = {
                "rationale": design.rationale,
                "evidence_ids": list(design.evidence_ids),
                "transformation": RGROUP_EDIT.dump_python(
                    design.transformation, mode="json"
                )
                if design.transformation
                else None,
            }
        if design or criteria:
            row["predictions"] = {
                key: metric_json(metric)
                for key in criteria
                if (metric := objective_metric(profile, key)) is not None
            }
        rows.append(row)
    return rows


def embedded_row(value: Any) -> Mapping[str, Any]:
    """Read one related PostgREST row from an object or singleton list."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    raise DataError("Expected one embedded database row.")


def run_config_from_row(row: Mapping[str, Any]) -> RunConfig:
    return RUN_CONFIG.validate_python(row["config"])


def run_from_row(row: Mapping[str, Any]) -> Run:
    """Read a run and order its candidates by ordinal."""
    candidates = []
    for entry in row.get("candidates") or ():
        molecule = embedded_row(entry["molecules"])
        profile = embedded_row(molecule["molecule_profiles"])
        design = entry.get("design") or {}
        predictions = entry.get("predictions")
        candidates.append(
            Candidate(
                molecule_id=MoleculeId(int(molecule["id"])),
                ordinal=int(entry["ordinal"]),
                profile=profile_from_row(molecule["canonical_smiles"], profile),
                transformation=RGROUP_EDIT.validate_python(design["transformation"])
                if design.get("transformation")
                else None,
                rationale=design.get("rationale"),
                evidence_ids=tuple(design.get("evidence_ids") or ()),
                run_predictions={
                    key: metric_from_json(result) for key, result in predictions.items()
                }
                if predictions is not None
                else None,
            )
        )
    candidates.sort(key=lambda candidate: candidate.ordinal)

    created_at = datetime.fromisoformat(row["created_at"])
    finished_at = row.get("finished_at")
    project_id = row.get("project_id")
    status = RUN_STATUS.validate_python(row["status"])
    return Run(
        id=GenerationRunId(str(row["id"])),
        user_id=UserId(str(row["user_id"])),
        source=Occurrence(
            MoleculeId(int(row["source_molecule_id"])),
            GenerationRunId(str(row["parent_run_id"]))
            if row.get("parent_run_id")
            else None,
        ),
        status=status,
        model=str(row["model"]),
        usage=TokenUsage(
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            reasoning_tokens=row["reasoning_tokens"],
            cached_tokens=row["cached_tokens"],
        ),
        failure_reason=row.get("failure_reason"),
        created_at=created_at,
        finished_at=datetime.fromisoformat(finished_at) if finished_at else None,
        candidates=tuple(candidates),
        project_id=ProjectId(str(project_id)) if project_id else None,
        config=run_config_from_row(row),
        request_id=row.get("request_id"),
        expires_at=created_at + timedelta(minutes=5)
        if project_id is None and status == "running"
        else None,
    )


class GenerationStore:
    def __init__(self, writer: AsyncPostgrestClient) -> None:
        self.writer = writer

    async def save_molecule(
        self, *, user_id: UserId, profile: MoleculeProfile
    ) -> MoleculeId:
        """Reuse the molecule ID when the canonical SMILES already exists."""
        result = await execute(
            self.writer.rpc(
                "save_molecule",
                {
                    "p_user_id": user_id,
                    "p_canonical_smiles": profile.canonical_smiles,
                    "p_standardization_version": IDENTITY_POLICY,
                    "p_profile": profile_fields(profile),
                },
            ).execute()
        )
        return MoleculeId(int(result.data))

    async def start_run(
        self,
        *,
        user_id: UserId,
        source_molecule_id: MoleculeId,
        model: str,
        config: RunConfig,
        request_id: str | None,
        request: dict | None,
    ) -> RunStart:
        result = await execute(
            self.writer.rpc(
                "start_run",
                {
                    "p_user_id": user_id,
                    "p_source_molecule_id": source_molecule_id,
                    "p_model": model,
                    "p_config": RUN_CONFIG.dump_python(
                        config, mode="json", exclude_none=True
                    ),
                    "p_request_id": request_id,
                    "p_request": request,
                },
            ).execute()
        )
        return RUN_START.validate_python(result.data)

    async def expire_run(self, user_id: UserId, run_id: GenerationRunId) -> bool:
        result = await execute(
            self.writer.rpc(
                "expire_runs", {"p_user_id": user_id, "p_run_id": run_id}
            ).execute()
        )
        return result.data

    async def finish_run(
        self,
        run_id: GenerationRunId,
        *,
        user_id: UserId,
        candidates: Sequence[PreparedCandidate],
        usage: TokenUsage,
        status: RunOutcome,
        criteria: Sequence[OptimizationTask] = (),
        source_predictions: Mapping[OptimizationTask, Metric] | None = None,
        calls: Sequence[ModelCall] = (),
    ) -> None:
        """Save candidates and the run outcome in one transaction."""
        await self._finish_run(
            run_id,
            status,
            usage,
            user_id=user_id,
            candidates=candidate_payload(candidates, criteria),
            source_predictions=source_predictions,
            calls=calls,
        )

    async def fail_run(
        self,
        run_id: GenerationRunId,
        *,
        user_id: UserId,
        reason: str,
        usage: TokenUsage,
        calls: Sequence[ModelCall] = (),
    ) -> None:
        """Mark the run failed if it is still running."""
        await self._finish_run(
            run_id, "failed", usage, user_id=user_id, failure_reason=reason, calls=calls
        )

    async def _finish_run(
        self,
        run_id: GenerationRunId,
        status: Literal[RunOutcome, "failed"],
        usage: TokenUsage,
        *,
        user_id: UserId,
        candidates: Sequence[dict[str, Any]] = (),
        failure_reason: str | None = None,
        source_predictions: Mapping[OptimizationTask, Metric] | None = None,
        calls: Sequence[ModelCall],
    ) -> None:
        await execute(
            self.writer.rpc(
                "finish_run",
                {
                    "p_user_id": user_id,
                    "p_run_id": run_id,
                    "p_status": status,
                    "p_candidates": candidates,
                    "p_usage": usage.model_dump(),
                    "p_failure_reason": failure_reason,
                    "p_source_predictions": SOURCE_PREDICTIONS.dump_python(
                        source_predictions, mode="json"
                    )
                    if source_predictions is not None
                    else None,
                    "p_calls": MODEL_CALLS.dump_python(calls, mode="json"),
                },
            ).execute()
        )
