from collections.abc import Sequence


def format_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Align columns in a plain-text table."""
    data = (columns, *rows)
    widths = [max(map(len, column)) for column in zip(*data, strict=True)]
    return "\n".join(
        "  ".join(
            value.ljust(width) for value, width in zip(row, widths, strict=True)
        ).rstrip()
        for row in data
    )
