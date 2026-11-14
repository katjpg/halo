from os import environ
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from halo import HALOChemError
from halo.cli import chem, project
from halo.cli.format import console, print_step, stderr
from halo.cli.options import (
    BatchSize,
    Decrease,
    Increase,
    LiveProgress,
    Preserve,
    client,
    design_criteria,
)
from halo.core.errors import ApplicationError
from halo.types import DEFAULT_BATCH_SIZE

app = typer.Typer(
    help="Molecular analysis and design with HALO.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)
app.add_typer(chem.app, name="chem")
app.add_typer(project.app, name="project")


@app.command()
def optimize(
    smiles: str,
    increase: Increase = None,
    decrease: Decrease = None,
    preserve: Preserve = None,
    n: BatchSize = DEFAULT_BATCH_SIZE,
    progress: LiveProgress = True,
) -> None:
    """Generate analogs from a source molecule."""
    criteria = design_criteria(increase, decrease, preserve)
    with client() as halo:
        run = halo.optimize(
            smiles, criteria=criteria, n=n, progress=print_step if progress else False
        )
    console.print(f"Run {run.id}")
    console.print(str(run), soft_wrap=True)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Address to listen on.")] = "127.0.0.1",
    port: Annotated[
        int, typer.Option(min=1, max=65535, help="Port to listen on.")
    ] = 8000,
    config: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, readable=True, help="HALO YAML settings."
        ),
    ] = None,
) -> None:
    """Start the HALO API service."""
    import uvicorn

    if config is not None:
        environ["HALO_CONFIG_PATH"] = str(config.resolve())
    uvicorn.run("halo.app:app", host=host, port=port)


def main() -> None:
    try:
        app()
    except ValidationError as exc:
        error = exc.errors(include_input=False)[0]
        location = ".".join(map(str, error["loc"]))
        stderr.print(f"{location}: {error['msg']}" if location else error["msg"])
        raise SystemExit(1) from None
    except (ApplicationError, HALOChemError, OSError) as exc:
        stderr.print(str(exc))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
