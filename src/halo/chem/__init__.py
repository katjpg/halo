from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from halo.chem.operations import (
        cluster_molecules,
        decompose_rgroups,
        draw_molecule,
        evaluate_pareto,
        find_mcs,
        fragment_molecule,
        profile_molecule,
        profile_molecules,
        read_smiles,
        replace_rgroup,
    )
    from halo.chem.properties import admet_properties, descriptors

__all__ = (
    "admet_properties",
    "cluster_molecules",
    "decompose_rgroups",
    "descriptors",
    "draw_molecule",
    "evaluate_pareto",
    "find_mcs",
    "fragment_molecule",
    "profile_molecule",
    "profile_molecules",
    "read_smiles",
    "replace_rgroup",
)


def __getattr__(name: str) -> object:
    if name in ("admet_properties", "descriptors"):
        from halo.chem import properties

        return getattr(properties, name)
    if name in __all__:
        from halo.chem import operations

        return getattr(operations, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
