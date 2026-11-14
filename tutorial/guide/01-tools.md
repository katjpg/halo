# Molecular tools

Molecular descriptors summarize structural properties numerically, while substructure methods identify shared cores and variable fragments. Use these methods to profile a lead, compare analogs, and construct specific edits before evaluating their predicted properties.

## Choose a method

| Method | Use it to | HALO interface |
| --- | --- | --- |
| Molecular profiling | Inspect calculated descriptors and screening scores. | `chem.profile_molecule`, `chem.profile_molecules` |
| Maximum common substructure (MCS) | Find a core shared by a molecular series. | `chem.find_mcs` |
| R-group decomposition | Compare substituents at labeled sites on a core. | `chem.decompose_rgroups` |
| Fragment replacement | Construct an analog at a specified attachment site. | `chem.replace_rgroup` or `row.replace` |
| BRICS fragmentation | Inspect fragments obtained by cutting eligible bonds. | `chem.fragment_molecule` |
| Structural annotation | Identify functional groups and catalog alerts. | `StructuralAnnotator` |

A **mapped core** labels its attachment sites with dummy atoms such as `[*:1]`. An **R-group** is the fragment assigned to one of those sites. Structural alerts are matches to catalog patterns used during compound review.

## Tutorial

Profile a methoxy-to-ethoxy series, inspect a common core, and annotate the structures.

### 1. Profile the molecules

```python
from halo import chem

smiles = ["COc1ccc(F)cc1", "CCOc1ccc(F)cc1"]
profiles = chem.profile_molecules(smiles)
```

Each profile contains canonical SMILES, descriptors, and screening results. Inspect molecular weight in g/mol.

```python
for profile in profiles:
    mass_g_mol = profile.descriptors.molecular_weight_g_mol
    print(profile.canonical_smiles, round(mass_g_mol, 2))
```

```text
COc1ccc(F)cc1 126.13
CCOc1ccc(F)cc1 140.16
```

Other results include calculated logP, topological polar surface area (TPSA), QED, Lipinski rule violations, PAINS matches, and synthetic accessibility. Use `chem.profile_molecule(smiles)` for one structure. Batch profiling preserves input order and repeated structures.

List the available descriptors and their definitions, types, and units:

```python
print(chem.descriptors())
```

### 2. Find a common core

By default, the MCS must match every input molecule. Reduce `min_coverage` when a core may match only part of a larger series.

```python
core = chem.find_mcs(smiles, min_coverage=1.0)
groups = chem.decompose_rgroups(smiles, core)
print(groups)
```

```text
Core  Fc1ccc(OC[*:1])cc1
Index  SMILES          R1
0      COc1ccc(F)cc1   [H][*:1]
1      CCOc1ccc(F)cc1  C[*:1]
```

The inferred core includes `OC`, leaving hydrogen or methyl at R1. Supply a mapped core when you want to compare a particular substitution site. [R-group editing tutorial](../../examples/molecular-tools.md)

`find_mcs` records which inputs match the core. If no suitable core is found, it returns a `Failure` with the reason. [MCS reference](05-reference.md#find_mcs-and-decompose_rgroups)

### 3. Inspect possible fragments

BRICS fragmentation cuts each eligible bond separately. Each row shows the two fragments from one cut and retains their BRICS attachment labels.

```python
print(chem.fragment_molecule("CCOc1ccc(F)cc1"))
```

```text
Single BRICS cuts of CCOc1ccc(F)cc1
Fragment          Partner
[16*]c1ccc(F)cc1  [3*]OCC
[3*]Oc1ccc(F)cc1  [4*]CC
```

### 4. Annotate structural groups

Use `annotate` for one molecule or `annotate_many` for a series.

```python
from halo.chem.medchem import StructuralAnnotator

annotator = StructuralAnnotator()
annotation = annotator.annotate("CCOc1ccc(F)cc1")
annotations = annotator.annotate_many(smiles)
```

- `groups` contains matched groups and their atom indices.
- `alerts` contains catalog matches and their atom indices.
- Indices refer to the returned molblock for a single annotation and to canonical SMILES for a batch annotation.

Batch annotations preserve input order and repeats. [Annotation example](../../examples/molecular-tools.md#5-annotate-the-series)

## Compare a larger series

- **Cluster structures.** Load a CSV or SDF with `chem.read_smiles`, then group structures with `chem.cluster_molecules`. [Clustering tutorial](../../examples/molecular-screening.md)
- **Compare property goals.** Use `chem.evaluate_pareto` to calculate changes from a baseline and identify candidates on the Pareto front. [Property comparison](../../examples/molecular-screening.md#4-compare-property-trade-offs)
- **Use fingerprints directly.** `Fingerprints.from_mols` creates a collection for custom similarity analyses. Advanced operations include `compare_structures` for pair comparisons and `pick_diverse` for selecting varied fingerprints. [Fingerprint reference](05-reference.md#fingerprints) · [Pair comparisons](../../src/halo/chem/sar/comparison.py) · [Diversity selection](../../src/halo/chem/cluster.py)

To add predicted properties to a molecular profile, provide a predictor. [Molecular prediction](02-models.md#predict-molecular-properties)
