# Generate analogs and research a project

Iterative molecular design combines analog generation, property comparison, and evidence retrieval. A saved project connects these activities so that later rounds can use the molecules and decisions from previous work.

Use this workflow when you want to generate analogs from a lead, choose a new parent, and investigate supporting papers or patents.

## Key concepts

| Concept | Role |
| --- | --- |
| Lead | The project's initial molecule. Research lookups use it by default. |
| Active parent | The molecule used as the source of the next generation run. |
| Run | A saved batch of verified candidates, their source, and their predictions. |
| Session | The conversation that records requests, responses, and the active parent. |

## Tutorial

Start the HALO service and set `HALO_ACCESS_TOKEN` to your Supabase user access token. These steps save a project and its results. [Service and sign-in setup](../README.md#api-credentials)

### 1. Open a project

Create the client and save the lead from its SMILES string.

```python
from os import environ

from halo import HALOChem

halo = HALOChem(
    base_url="http://localhost:8000",
    access_token=environ["HALO_ACCESS_TOKEN"],
)
project = halo.projects.create(source_smiles="COc1ccc(F)cc1")
```

### 2. Generate analogs

Request ADMET property changes relative to the active parent. The lead is the parent for this first run. Put design guidance in `instructions`.

```python
run = project.generate_analogs(
    criteria={"solubility": "increase", "herg": "decrease"},
    n=4,
    instructions="Retain the aryl fluoride in this batch.",
)
print(run)
```

*Recorded example output (test proposals scored by ADMET-AI 1.4.0).*

```text
HALO-77 · 2 of 4 analogs · completed
Analog   SMILES           Δ herg (probability)  herg goal  Δ solubility (log10 mol/L)  solubility goal
HALO-79  CCOc1ccc(F)cc1   -0.0635               met        -0.0834                     missed
HALO-78  CCCOc1ccc(F)cc1  -0.0119               met        -0.722                      missed
```

Changes are **candidate minus source**. Both recorded candidates meet the hERG criterion and miss the solubility criterion. Model-generated structures and predictions vary with the configured models.

The run may contain fewer candidates than requested. Use a mapped `core_smiles` when specific core atoms must be preserved. [Generation parameters](../tutorial/guide/05-reference.md#generate_analogs)

### 3. Choose the next parent

After reviewing the results, select a candidate for the next design round. This example uses the first returned candidate when one is available.

```python
if run.candidates:
    project.set_parent(run.candidates[0], run=run)
```

The next run uses that candidate as its source and retains its originating run as part of the lineage.

### 4. Retrieve research evidence

Research the project lead. Pass `molecule_id` explicitly to investigate a candidate instead.

```python
response = project.research("Find papers and patents about the source molecule.")
print(response)
```

*Example output when the sources checked have no match.*

```text
No matching papers or patents were found in the sources checked.
```

Findings include citations and source links. Use `response.findings` or `response.sources` for structured access; `response.turn` provides the saved turn. [Research results](../tutorial/guide/05-reference.md#research-and-ask)

### 5. Investigate a structural change

Name the change and property to search for matching sentences in CC BY full text.

```python
response = project.research(
    "Find papers that report how fluorination changes aqueous solubility."
)
print(response)
```

Quoted passages include a PMCID, available DOI and year, license URL, and location. Exact, analog, and topic evidence remain separate. Excluded sources and failed lookups are reported separately. [Full-text retrieval](../tutorial/guide/04-context.md#retrieve-full-text-passages)

### 6. Review the conversation

Print the saved turns, then close the client.

```python
history = project.history()
print(history)
halo.close()
```

The history uses `[USER]` and `[HALO]` labels. Access its ordered records through `history.turns`. Reopen the project by ID to retrieve saved runs. [Project reference](../tutorial/guide/05-reference.md#projects)
