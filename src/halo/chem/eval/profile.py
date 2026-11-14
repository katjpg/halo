from collections.abc import Callable, Sequence
from functools import cache
from typing import Protocol

from rdkit import Chem
from rdkit.Chem import Descriptors as RDKitDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from halo.chem.mol import canonical_structure, to_mol
from halo.chem.results import Failure, MetricResult, to_metric
from halo.core.errors import ParseError, ProviderError
from halo.models.molecule import Descriptors, MoleculeProfile, Screening
from halo.types import OptimizationTask

SA_SCORE_METHOD = "rdkit_contrib_sa_score"


class PropertyPredictor(Protocol):
    def predict_batch(
        self, smiles: Sequence[str]
    ) -> list[dict[OptimizationTask, MetricResult]]:
        """Return predictions by property for each canonical SMILES, in input order."""
        ...


def profile_molecules(
    smiles: Sequence[str], predictors: Sequence[PropertyPredictor] = ()
) -> list[MoleculeProfile]:
    """Compute descriptors, screening scores, and optional predictions.

    Args:
        smiles: Structures to profile.
        predictors: Predictors to run on each distinct canonical structure.

    Returns:
        One profile per input SMILES, in input order and including repeats.
    """
    if isinstance(smiles, str):
        raise ParseError("Provide a sequence of SMILES, such as ['CCO'].")
    key_by_text: dict[str, str] = {}
    structures: dict[str, Chem.Mol] = {}
    for text in dict.fromkeys(smiles):
        mol, key = canonical_structure(to_mol(text))
        key_by_text[text] = key
        structures.setdefault(key, mol)
    canonical = list(structures)
    predictions: list[dict[OptimizationTask, MetricResult]] = [{} for _ in canonical]
    for predictor in predictors:
        rows = predictor.predict_batch(canonical) if canonical else []
        if len(rows) != len(canonical):
            raise ProviderError(
                f"{type(predictor).__name__} returned {len(rows)} results "
                f"for {len(canonical)} molecules"
            )
        for merged, row in zip(predictions, rows, strict=True):
            if shared := merged.keys() & row.keys():
                raise ProviderError(f"predictors both returned {sorted(shared)}")
            merged.update(row)
    profiles = {}
    for (key, mol), row in zip(structures.items(), predictions, strict=True):
        descriptors = _descriptors(mol)
        profiles[key] = MoleculeProfile(
            canonical_smiles=key,
            descriptors=descriptors,
            screening=_screen_molecule(
                mol,
                molecular_weight_g_mol=descriptors.molecular_weight_g_mol,
                logp=descriptors.logp,
            ),
            predictions=row,
        )
    return [profiles[key_by_text[text]] for text in smiles]


def profile_molecule(
    smiles: str, predictors: Sequence[PropertyPredictor] = ()
) -> MoleculeProfile:
    """Compute descriptors and screening scores, with optional predictions."""
    return profile_molecules([smiles], predictors)[0]


def compute_descriptors(mol: Chem.Mol) -> Descriptors:
    """Compute physicochemical descriptors, ignoring ordinary hydrogens."""
    return _descriptors(Chem.RemoveHs(mol))


def _descriptors(mol: Chem.Mol) -> Descriptors:
    return Descriptors(
        molecular_weight_g_mol=float(RDKitDescriptors.MolWt(mol)),
        logp=float(RDKitDescriptors.MolLogP(mol)),
        tpsa_angstrom_sq=float(rdMolDescriptors.CalcTPSA(mol)),
        heavy_atom_count=rdMolDescriptors.CalcNumHeavyAtoms(mol),
        h_bond_acceptors=rdMolDescriptors.CalcNumHBA(mol),
        h_bond_donors=rdMolDescriptors.CalcNumHBD(mol),
        rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        ring_count=rdMolDescriptors.CalcNumRings(mol),
        aromatic_rings=int(rdMolDescriptors.CalcNumAromaticRings(mol)),
    )


def screen_molecule(mol: Chem.Mol) -> Screening:
    """Score drug-likeness and match PAINS alerts, ignoring ordinary hydrogens."""
    mol = Chem.RemoveHs(mol)
    return _screen_molecule(
        mol,
        molecular_weight_g_mol=RDKitDescriptors.MolWt(mol),
        logp=RDKitDescriptors.MolLogP(mol),
    )


def _screen_molecule(
    mol: Chem.Mol, *, molecular_weight_g_mol: float, logp: float
) -> Screening:
    alerts = (match.GetDescription() for match in _pains_catalog().GetMatches(mol))
    # The Lipinski counts use different donor and acceptor definitions.
    violations = (
        molecular_weight_g_mol > 500,
        logp > 5,
        rdMolDescriptors.CalcNumLipinskiHBD(mol) > 5,
        rdMolDescriptors.CalcNumLipinskiHBA(mol) > 10,
    )
    return Screening(
        qed=float(RDKitDescriptors.qed(mol)),
        lipinski_violations=sum(violations),
        sa_score=_score_synthetic_accessibility(mol),
        pains_alerts=tuple(dict.fromkeys(alerts)),
    )


def _score_synthetic_accessibility(mol: Chem.Mol) -> MetricResult:
    scorer = _sa_scorer()
    if scorer is None:
        return Failure("RDKit SA_Score module is unavailable")
    try:
        value = scorer(mol)
    except (ValueError, RuntimeError, ZeroDivisionError, OSError) as exc:
        return Failure(f"{SA_SCORE_METHOD} failed: {exc}")
    return to_metric(value, unit="dimensionless", method=SA_SCORE_METHOD)


@cache
def _pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


@cache
def _sa_scorer() -> Callable[[Chem.Mol], float] | None:
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError:
        return None
    return sascorer.calculateScore
