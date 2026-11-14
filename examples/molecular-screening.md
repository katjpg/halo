# Cluster and compare a molecular series

Structural clustering groups molecules by similarity so a compound library can be reviewed as smaller sets. Pareto analysis compares candidates across competing property goals. Use these methods together to inspect structural diversity and property trade-offs in a molecular series.

## Key concepts

- **Fingerprint:** a numerical representation of a molecule's structural features. HALO uses Morgan fingerprints for clustering.
- **Tanimoto similarity:** a comparison between fingerprints, from 0 to 1. Higher values indicate greater fingerprint similarity.
- **Cluster seed:** the molecule around which a Butina cluster is formed. The similarity threshold determines which molecules can join that seed.
- **Baseline:** the source molecule used to calculate property changes.
- **Pareto front:** the candidates that no other candidate dominates. One candidate dominates another when it performs at least as well on every objective and better on at least one.

## Tutorial

Load a small series, cluster its structures, and compare calculated logP and QED (a drug-likeness score) against a chosen baseline.

### 1. Load the series

Save the following structures as `series.csv`, or use your own CSV with a SMILES column.

```csv
smiles
COc1ccc(F)cc1
CCOc1ccc(F)cc1
Oc1ccc(F)cc1
```

```python
from halo import chem

smiles = chem.read_smiles("series.csv")
```

### 2. Cluster the structures

Set `min_similarity` to the minimum Tanimoto similarity required between a member and its cluster seed.

```python
clusters = chem.cluster_molecules(smiles, min_similarity=0.50)
print(clusters)
```

```text
Butina clusters (minimum Tanimoto similarity 0.5)
Cluster  Seed index  Member indices
1        1           0, 1
2        2           2
```

The indices refer to positions in the input list. Increase `min_similarity` when you require closer matches to each seed.

### 3. Retrieve the cluster seeds

Use each seed as a representative for inspection.

```python
representatives = [
    smiles[cluster.representative]
    for cluster in clusters.clusters
]
print(representatives)
```

```text
['CCOc1ccc(F)cc1', 'Oc1ccc(F)cc1']
```

### 4. Compare property trade-offs

Use the methoxy compound as the baseline and prefer **lower logP** and **higher QED**.

```python
comparison = chem.evaluate_pareto(
    source="COc1ccc(F)cc1",
    candidates=smiles,
    objectives={"logp": "decrease", "qed": "increase"},
)
print(comparison)
```

```text
Baseline  COc1ccc(F)cc1
SMILES          Δ logp (dimensionless)  Δ qed (dimensionless)  Pareto front
CCOc1ccc(F)cc1  +0.39                   +0.0542                yes
Oc1ccc(F)cc1    -0.303                  -0.0182                yes
```

Changes are **candidate minus baseline**. The ethoxy analog improves QED while increasing logP; the phenol analog lowers logP while lowering QED. Neither dominates the other, so both belong to the front. Use each candidate's `improved` flags to check the individual objectives against the baseline.

For custom similarity analyses, see the [fingerprint reference](../tutorial/guide/05-reference.md#fingerprints).
