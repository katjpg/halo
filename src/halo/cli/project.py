from typing import Annotated
from uuid import UUID

import typer

from halo.cli.format import console, print_step, print_table
from halo.cli.options import (
    BatchSize,
    Coverage,
    Decrease,
    Increase,
    LiveProgress,
    MinSimilarity,
    Preserve,
    client,
    design_criteria,
)
from halo.refs import molecule_label
from halo.types import DEFAULT_BATCH_SIZE, MoleculeId

app = typer.Typer(
    help="Work with projects on a running HALO service.", no_args_is_help=True
)
JSONOutput = Annotated[
    bool, typer.Option("--json", help="Print the saved data as JSON.")
]


@app.command()
def create(smiles: str) -> None:
    """Save a source molecule and print the new project ID."""
    with client() as halo:
        console.print(halo.projects.create(source_smiles=smiles).id)


@app.command("list")
def list_projects() -> None:
    """List your saved projects."""
    with client() as halo:
        projects = halo.projects.list()
    print_table(
        ("Project", "Source", "Created"),
        (
            (
                project.id,
                molecule_label(project.lead_id),
                project.created_at.isoformat(),
            )
            for project in projects
        ),
    )


@app.command()
def generate(
    project_id: UUID,
    increase: Increase = None,
    decrease: Decrease = None,
    preserve: Preserve = None,
    n: BatchSize = DEFAULT_BATCH_SIZE,
    instructions: Annotated[
        str | None, typer.Option(help="Design guidance for this batch.")
    ] = None,
    core: Annotated[str | None, typer.Option(help="Mapped core to preserve.")] = None,
    progress: LiveProgress = True,
) -> None:
    """Generate analogs from the active parent."""
    criteria = design_criteria(increase, decrease, preserve)
    with client() as halo:
        project = halo.projects.get(project_id, resume=True)
        run = project.generate_analogs(
            criteria=criteria,
            n=n,
            instructions=instructions,
            core_smiles=core,
            progress=print_step if progress else False,
        )
    console.print(f"Run {run.id}")
    console.print(str(run), soft_wrap=True)


@app.command()
def ask(
    project_id: UUID,
    message: str,
    progress: LiveProgress = True,
    as_json: JSONOutput = False,
) -> None:
    """Ask a question or give an instruction in the latest session."""
    with client() as halo:
        reply = halo.projects.get(project_id, resume=True).ask(
            message, progress=print_step if progress else False
        )
    console.print(
        reply.turn.model_dump_json(indent=2) if as_json else str(reply), soft_wrap=True
    )


@app.command()
def research(project_id: UUID, query: str, progress: LiveProgress = True) -> None:
    """Look up papers and patents related to the project source."""
    with client() as halo:
        response = halo.projects.get(project_id, resume=True).research(
            query, progress=print_step if progress else False
        )
    console.print(str(response), soft_wrap=True)


@app.command()
def remember(project_id: UUID, instruction: str) -> None:
    """Save design guidance such as 'Retain the fluoride in every round'."""
    with client() as halo:
        decision_id = halo.projects.get(project_id, resume=True).remember(instruction)
    console.print(f"Saved instruction {decision_id}.")


@app.command()
def history(
    project_id: UUID,
    session: Annotated[
        UUID | None, typer.Option(help="Session to read. Defaults to the latest.")
    ] = None,
    as_json: JSONOutput = False,
) -> None:
    """Print a conversation in turn order."""
    with client() as halo:
        history = halo.projects.get(project_id).history(session)
    console.print(
        history.model_dump_json(indent=2) if as_json else str(history), soft_wrap=True
    )


@app.command()
def runs(project_id: UUID) -> None:
    """List the project's saved generation runs."""
    with client() as halo:
        runs = halo.projects.get(project_id).runs()
    print_table(
        ("Run", "Source", "Analogs", "Status"),
        (
            (
                run.id,
                molecule_label(MoleculeId(run.source_molecule_id)),
                len(run.candidates),
                run.status.replace("_", " "),
            )
            for run in runs
        ),
    )


@app.command()
def inspect(
    project_id: UUID,
    run_id: UUID,
    min_similarity: MinSimilarity = 0.70,
    min_core_coverage: Coverage = 0.80,
    as_json: JSONOutput = False,
) -> None:
    """Review predictions, edits, structural clusters, and medchem annotations."""
    with client() as halo:
        review = halo.projects.get(project_id).inspect_run(
            run_id,
            min_similarity=min_similarity,
            min_core_coverage=min_core_coverage,
        )
    if as_json:
        console.print_json(data=review.tables())
    else:
        console.print(
            f"Run {review.run.id} · {review.run.status.replace('_', ' ')}\n"
            f"Baseline: {molecule_label(MoleculeId(review.run.source_molecule_id))}  {review.source_smiles}\n"
            f"Minimum similarity {review.min_similarity:.0%} · "
            f"Minimum core coverage {review.min_core_coverage:.0%}"
        )
        if review.run.failure_reason:
            console.print(review.run.failure_reason)
        for name, rows in review.tables().items():
            if rows:
                print_table(
                    tuple(rows[0]), (tuple(row.values()) for row in rows), title=name
                )


@app.command()
def set_parent(project_id: UUID, run_id: UUID, molecule: str) -> None:
    """Use a candidate, such as HALO-12, as the next design parent."""
    with client() as halo:
        project = halo.projects.get(project_id, resume=True)
        run = project.get_run(run_id)
        for candidate in run.candidates:
            label = molecule_label(MoleculeId(candidate.molecule_id))
            if molecule.upper() in (label, str(candidate.molecule_id)):
                project.set_parent(candidate, run=run)
                console.print(f"{label} is the active parent.")
                return
    raise typer.BadParameter("Choose a molecule listed in the run.")
