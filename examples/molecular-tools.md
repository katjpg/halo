# Decompose a series and replace an R-group

R-group decomposition separates a molecular series into a shared core and the substituents attached to it. Use it to compare analogs by substitution site or construct an analog by replacing one group.

## Key concepts

| Concept | Meaning |
| --- | --- |
| Core | The structure shared by the molecules being compared. |
| R-group | A fragment attached to a labeled site on the core. |
| Attachment label | A mapped dummy atom, such as `[*:1]`, that identifies where a fragment connects. |
| Structural annotation | Functional groups and structural alerts matched against medchem catalogs. |

## Tutorial

Decompose a fluoroaryl series, replace methoxy with ethoxy, and inspect the product.

### 1. Prepare the series and core

The mapped dummy `[*:1]` marks the substitution site to compare.

```python
from halo import chem

smiles = [
    "COc1ccc(F)cc1",
    "CCOc1ccc(F)cc1",
    "Oc1ccc(F)cc1",
]
core = "Fc1ccc([*:1])cc1"
```

### 2. Decompose the molecules

Each row associates a molecule with its fragments at the core's labeled sites.

```python
series = chem.decompose_rgroups(smiles, core=core)
print(series)
```

```text
Core  Fc1ccc([*:1])cc1
Index  SMILES          R1
0      COc1ccc(F)cc1   CO[*:1]
1      CCOc1ccc(F)cc1  CCO[*:1]
2      Oc1ccc(F)cc1    O[*:1]
```

### 3. Replace a fragment

Select the first matched molecule and replace its R1 fragment. The replacement uses the same attachment label.

```python
replacement = series.rows[0].replace(site=1, fragment="CCO[*:1]")
print(replacement)
```

```text
CCOc1ccc(F)cc1
R1  CO[*:1] → CCO[*:1]
```

`replacement.edit.reaction_smarts` provides the recorded structural transformation when available. [Replacement reference](../tutorial/guide/05-reference.md#replace_rgroup-and-draw_molecule)

### 4. Draw the product

Save the verified product as a PNG.

```python
chem.draw_molecule(replacement.product_smiles, path="ethoxy-analog.png")
```

### 5. Annotate the series

Use `annotate` for one molecule or `annotate_many` for a sequence. Preview the product's group names and count its catalog alerts.

```python
from halo.chem.medchem import StructuralAnnotator

annotator = StructuralAnnotator()
annotation = annotator.annotate(replacement.product_smiles)
annotations = annotator.annotate_many(smiles)
group_names = sorted({group.name for group in annotation.groups})
print(group_names[:3])
print("Alerts:", len(annotation.alerts))
```

```text
['Alkylarylethers', 'Arenes', 'Aromatic compounds']
Alerts: 0
```

Annotations contain matched groups, alerts, and their atom indices. Batch results preserve input order and repeats. [Annotation fields and atom order](../tutorial/guide/05-reference.md#structuralannotator)
