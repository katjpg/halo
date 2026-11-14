from collections.abc import Iterable, Sequence

from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console(markup=False, highlight=False)
stderr = Console(stderr=True, markup=False, highlight=False)


def print_table(
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    title: str | None = None,
) -> None:
    table = Table(*columns, title=title)
    for row in rows:
        table.add_row(*(Text(str(value)) for value in row))
    console.print(table)


def print_step(step: str) -> None:
    stderr.print(step)
