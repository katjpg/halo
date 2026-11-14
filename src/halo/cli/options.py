from os import getenv
from typing import Annotated

import typer
from pydantic import TypeAdapter

from halo import HALOChem
from halo.types import MAX_BATCH, Criteria, CriterionDirection, DesignCriteria

Increase = Annotated[
    list[str] | None, typer.Option(help="Property to increase. Repeat for more.")
]
Decrease = Annotated[
    list[str] | None, typer.Option(help="Property to decrease. Repeat for more.")
]
Preserve = Annotated[
    list[str] | None,
    typer.Option(help="Property to keep close to the source. Repeat for more."),
]
BatchSize = Annotated[
    int, typer.Option(min=1, max=MAX_BATCH, help="Number of analogs to request.")
]
MinSimilarity = Annotated[
    float, typer.Option(min=0, max=1, help="Minimum Tanimoto similarity.")
]
Coverage = Annotated[
    float,
    typer.Option(
        min=0, max=1, help="Fraction that must share the core, greater than 0."
    ),
]
LiveProgress = Annotated[bool, typer.Option(help="Print live steps to stderr.")]


def client() -> HALOChem:
    access_token = getenv("HALO_ACCESS_TOKEN")
    if not access_token:
        raise typer.BadParameter("Set HALO_ACCESS_TOKEN to a user access token.")
    return HALOChem(
        base_url=getenv("HALO_BASE_URL", "http://localhost:8000"),
        access_token=access_token,
    )


def directions(
    increase: list[str] | None,
    decrease: list[str] | None,
    preserve: list[str] | None = None,
) -> dict[str, CriterionDirection]:
    objectives: dict[str, CriterionDirection] = {}
    for direction, names in (
        ("increase", increase),
        ("decrease", decrease),
        ("preserve", preserve),
    ):
        for name in names or ():
            if name in objectives and objectives[name] != direction:
                raise typer.BadParameter(f"Choose one direction for {name!r}.")
            objectives[name] = direction
    if not objectives:
        raise typer.BadParameter("Choose at least one property with a direction.")
    return objectives


def design_criteria(
    increase: list[str] | None, decrease: list[str] | None, preserve: list[str] | None
) -> Criteria:
    return TypeAdapter(DesignCriteria).validate_python(
        directions(increase, decrease, preserve)
    )
