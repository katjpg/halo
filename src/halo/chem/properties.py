from dataclasses import dataclass
from typing import Literal

from halo.chem.eval.objectives import ADMET_UNITS
from halo.core.format import format_table
from halo.types import ADMETProperty


@dataclass(frozen=True)
class PropertyDefinition:
    """The meaning and units of a property exposed by HALO."""

    name: str
    definition: str
    value_type: Literal["probability", "float", "int"]
    unit: str | None
    method: str | None = None


class PropertyCatalog(tuple[PropertyDefinition, ...]):
    """An ordered, printable list of property definitions."""

    def __str__(self) -> str:
        return format_table(
            ("Property", "Type", "Unit", "Definition"),
            [
                (item.name, item.value_type, item.unit or "—", item.definition)
                for item in self
            ],
        )


ADMET_DEFINITIONS: dict[ADMETProperty, str] = {
    "bbbp": "Probability of crossing the blood–brain barrier.",
    "carcinogenicity": "Probability of a positive carcinogenicity label.",
    "dili": "Probability of a positive drug-induced liver injury label.",
    "herg": "Probability of hERG potassium channel inhibition.",
    "hia": "Probability of human intestinal absorption.",
    "mutagenicity": "Probability of a positive Ames mutagenicity label.",
    "pampa": "Probability of high permeability in the PAMPA artificial membrane assay.",
    "solubility": "Base-10 logarithm of aqueous solubility in mol/L.",
}

DESCRIPTORS = PropertyCatalog(
    (
        PropertyDefinition(
            "molecular_weight_g_mol",
            "Average molecular weight.",
            "float",
            "g/mol",
            "rdkit_mol_wt",
        ),
        PropertyDefinition(
            "logp",
            "Wildman–Crippen octanol/water partition coefficient (logP).",
            "float",
            "dimensionless",
            "rdkit_mol_logp",
        ),
        PropertyDefinition(
            "tpsa_angstrom_sq",
            "Ertl topological polar surface area from N and O.",
            "float",
            "Å²",
            "rdkit_calc_tpsa",
        ),
        PropertyDefinition(
            "heavy_atom_count",
            "Atoms other than hydrogen.",
            "int",
            "count",
            "rdkit_calc_num_heavy_atoms",
        ),
        PropertyDefinition(
            "h_bond_acceptors",
            "Hydrogen bond acceptors.",
            "int",
            "count",
            "rdkit_calc_num_hba",
        ),
        PropertyDefinition(
            "h_bond_donors",
            "Hydrogen bond donors.",
            "int",
            "count",
            "rdkit_calc_num_hbd",
        ),
        PropertyDefinition(
            "rotatable_bonds",
            "Rotatable bonds under RDKit's strict definition.",
            "int",
            "count",
            "rdkit_calc_num_rotatable_bonds",
        ),
        PropertyDefinition(
            "ring_count",
            "Rings perceived by RDKit.",
            "int",
            "count",
            "rdkit_calc_num_rings",
        ),
        PropertyDefinition(
            "aromatic_rings",
            "Rings classified as aromatic by RDKit.",
            "int",
            "count",
            "rdkit_calc_num_aromatic_rings",
        ),
    )
)


def admet_properties() -> PropertyCatalog:
    """List predicted properties accepted as design criteria."""
    return PropertyCatalog(
        PropertyDefinition(
            name,
            ADMET_DEFINITIONS[name],
            "probability" if unit == "probability" else "float",
            None if unit == "probability" else unit,
        )
        for name, unit in ADMET_UNITS.items()
    )


def descriptors() -> PropertyCatalog:
    """List descriptors calculated by molecular profiling."""
    return DESCRIPTORS
