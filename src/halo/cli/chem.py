from pathlib import Path
from typing import Annotated

import typer

from halo import chem
from halo.chem.results import Failure
from halo.cli.format import console, print_table
from halo.cli.options import Coverage, Decrease, Increase, MinSimilarity, directions

app = typer.Typer(
    help="Profile, draw, and compare molecules locally.", no_args_is_help=True
)
DataPath = Annotated[
    Path,
    typer.Argument(
        exists=True,
        dir_okay=False,
        readable=True,
        help="SDF or CSV with a SMILES column.",
    ),
]
Column = Annotated[str, typer.Option(help="Name of the SMILES column.")]


@app.command()
def admet() -> None:
    """List predicted properties you can use as design criteria."""
    print_table(
        ("Criterion", "Type", "Unit", "Definition"),
        (
            (item.name, item.value_type, item.unit or "—", item.definition)
            for item in chem.admet_properties()
        ),
    )


@app.command()
def descriptors() -> None:
    """List the descriptors calculated by molecular profiling."""
    print_table(
        ("Descriptor", "Type", "Unit", "Definition"),
        (
            (item.name, item.value_type, item.unit, item.definition)
            for item in chem.descriptors()
        ),
    )


@app.command()
def profile(
    smiles: str,
    predict: Annotated[bool, typer.Option(help="Run ADMET-AI predictions.")] = False,
) -> None:
    """Profile one molecule from its SMILES."""
    if predict:
        from halo.chem.predict.admet_ai import ADMETAIPredictor

        result = chem.profile_molecule(smiles, predictors=(ADMETAIPredictor.load(),))
    else:
        result = chem.profile_molecule(smiles)
    console.print(result.canonical_smiles)
    print_table(
        ("Descriptor", "Value", "Unit"),
        (
            (item.name, f"{getattr(result.descriptors, item.name):.6g}", item.unit)
            for item in chem.descriptors()
        ),
        title="Calculated descriptors",
    )
    if predict:
        print_table(
            ("Property", "Prediction"),
            (
                (item.name, result.prediction(item.name))
                for item in chem.admet_properties()
            ),
            title="ADMET-AI predictions",
        )


@app.command()
def draw(smiles: str, output: Path) -> None:
    """Save a two-dimensional structure as a PNG."""
    chem.draw_molecule(smiles, path=output)
    console.print(output)


@app.command()
def fragment(smiles: str) -> None:
    """List pairs from single BRICS cuts of a molecule."""
    console.print(str(chem.fragment_molecule(smiles)), soft_wrap=True)


@app.command()
def cluster(
    data: DataPath, column: Column = "smiles", min_similarity: MinSimilarity = 0.70
) -> None:
    """Cluster molecules by Tanimoto similarity."""
    console.print(
        str(
            chem.cluster_molecules(
                chem.read_smiles(data, column=column), min_similarity=min_similarity
            )
        ),
        soft_wrap=True,
    )


@app.command()
def core(
    data: DataPath,
    column: Column = "smiles",
    min_coverage: Coverage = 1.0,
) -> None:
    """Find a maximum common substructure and report its coverage."""
    result = chem.find_mcs(
        chem.read_smiles(data, column=column),
        min_coverage=min_coverage,
    )
    if isinstance(result, Failure):
        typer.echo(result.reason, err=True)
        raise typer.Exit(1)
    console.print(str(result), soft_wrap=True)


@app.command()
def decompose(
    data: DataPath, core: str | None = None, column: Column = "smiles"
) -> None:
    """Show a shared core and the R-groups at each site."""
    molecules = chem.read_smiles(data, column=column)
    selected = core if core is not None else chem.find_mcs(molecules)
    if isinstance(selected, Failure):
        typer.echo(selected.reason, err=True)
        raise typer.Exit(1)
    console.print(str(chem.decompose_rgroups(molecules, selected)), soft_wrap=True)


@app.command()
def replace(
    smiles: str,
    core: Annotated[str, typer.Option(help="Core from the decomposition.")],
    site: Annotated[int, typer.Option(min=1, help="R-group number to replace.")],
    fragment: Annotated[
        str, typer.Option(help="Replacement with one attachment point.")
    ],
) -> None:
    """Replace one R-group and print the verified product."""
    console.print(
        str(
            chem.replace_rgroup(smiles=smiles, core=core, site=site, fragment=fragment)
        ),
        soft_wrap=True,
    )


@app.command()
def compare(
    data: DataPath,
    source: Annotated[str, typer.Option(help="Baseline SMILES.")],
    column: Column = "smiles",
    increase: Increase = None,
    decrease: Decrease = None,
) -> None:
    """Compare calculated properties and find the Pareto front."""
    comparison = chem.evaluate_pareto(
        source=source,
        candidates=chem.read_smiles(data, column=column),
        objectives=directions(increase, decrease),
    )
    console.print(str(comparison), soft_wrap=True)
