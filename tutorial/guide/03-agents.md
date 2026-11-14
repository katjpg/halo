# Agents and workflows

Molecular design alternates between proposing analogs, evaluating their properties, and choosing what to investigate next. HALO's agents contribute to these tasks using the current structures, results, and scientific evidence. The workflow coordinates the tasks and saves their outcomes in the project.

## Agent roles

| Agent | Responsibility |
| --- | --- |
| `Orchestrator` | Identifies the task and scope of an open-ended message. |
| `Optimizer` | Proposes molecular structures or fragment edits. |
| `Researcher` | Synthesizes findings from scientific evidence and cited records. |
| `Analyst` | Interprets molecular properties and prediction results. |
| `Strategist` | Proposes experiments or selects candidates for further investigation. |

Each agent returns a structured result. HALO performs the calculations, validates proposals and citations, and saves the completed turn.

## Design criteria and baselines

Design criteria specify which ADMET properties to **increase**, **decrease**, or **preserve** relative to the active parent. A `preserve` criterion favors a smaller absolute change.

- A new comparison of the **latest batch** uses that run's source molecule.
- A new comparison of **other molecules** uses the session's active parent.
- Changes saved in a run use the source and candidate predictions recorded for that run.

One candidate dominates another if it performs at least as well on every objective and better on at least one. The **Pareto front** contains candidates that no other candidate dominates. HALO groups these candidates by structural similarity and provides their results to the `Strategist`, which can recommend up to two candidates.

## Tutorial

Open `project` using the [project tutorial](../../examples/basic-agent.md), then generate and review a batch.

### 1. Generate molecular candidates

Put design guidance in `instructions` and use `criteria` for the requested property changes.

```python
run = project.generate_analogs(
    criteria={"solubility": "increase", "herg": "decrease", "hia": "preserve"},
    n=4,
    instructions="Retain the aryl fluoride in this batch.",
)
print(run)
```

With a mapped core, the `Optimizer` proposes replacements at one site per candidate. Otherwise, it proposes whole molecules. HALO checks the protected structure, including stereochemistry, removes repeated structures, and evaluates the verified candidates. A run can contain fewer candidates than requested. If no candidates are verified, it records `limit_exhausted`.

Use `project.remember("Retain the aryl fluoride in every round.")` for guidance that should remain available in later runs. [Saved decisions](04-context.md#3-record-and-recall-decisions)

### 2. Review the batch

```python
review = project.inspect_run(
    run,
    min_similarity=0.70,
    min_core_coverage=0.80,
)
print(review)
```

| Review | What to inspect |
| --- | --- |
| Candidates | Prediction changes and individual goal outcomes relative to the run's source. |
| Edits | Replaced fragments and recorded transformations. |
| Clusters and R-groups | Cluster medoids, common-core coverage, and verified decompositions. |
| Medchem annotations | Structural groups and alerts added or removed relative to the source. |

`min_similarity` controls similarity to each Butina seed. `min_core_coverage` is the fraction of cluster members that must match the MCS. The review uses a **medoid** (the member with the smallest total fingerprint distance to the others) as each cluster's representative.

For a `preserve` goal, an unchanged value is `met`. A changed value is `unassessed` because no tolerance was specified. [Goal outcomes](05-reference.md#generate_analogs)

### 3. Select the next parent

Choose a candidate after reviewing its results. This example uses the first available candidate.

```python
if run.candidates:
    project.set_parent(run.candidates[0], run=run)
```

The next generation run starts from that molecule and records its originating run. [Local Pareto comparison](../../examples/molecular-screening.md#4-compare-property-trade-offs)

### 4. Investigate the evidence

`project.research` looks up papers and patents for the lead by default. Pass `molecule_id` to investigate an analog.

```python
response = project.research("Find papers and patents about the source molecule.")
print(response)
```

For saved assay measurements, use `project.ask` so the workflow can retrieve those records. The `Strategist` can also propose an experiment; an accepted comparison and its reason become a project decision. [Evidence and decisions](04-context.md)

### 5. Ask an open-ended question

`project.ask` uses the `Orchestrator` to identify the requested task from the message and current project context. Direct methods such as `generate_analogs` already identify the task.

```python
reply = project.ask("What should I compare next?", progress=True)
print(reply)
print(project.history())
```

HALO asks for clarification when the request cannot be resolved. `reply.turn` provides the saved turn; history shows the conversation in order with `[USER]` and `[HALO]` labels. [Request context](04-context.md#build-context-for-a-task) · [Execution limits](05-reference.md#execution-limits)
