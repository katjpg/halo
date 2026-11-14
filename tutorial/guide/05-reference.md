# API reference

Use `halo.chem` for local molecular calculations and structural operations.
Use `HALOChem` to generate analogs, retrieve evidence, and work with saved
projects on a HALO service. Advanced interfaces provide direct access to
algorithms and model providers.

## Public Python API

### Molecular operations

Import these functions from `halo.chem`.

#### `profile_molecule` and `profile_molecules`

```python
def profile_molecule(
    smiles: str,
    predictors: Sequence[PropertyPredictor] = (),
) -> MoleculeProfile: ...

def profile_molecules(
    smiles: Sequence[str],
    predictors: Sequence[PropertyPredictor] = (),
) -> list[MoleculeProfile]: ...
```

| Parameter | Meaning | Default |
| --- | --- | --- |
| `smiles` | One SMILES string for `profile_molecule`, or a sequence for `profile_molecules`. | Required |
| `predictors` | Property predictors to run alongside local calculations. | `()` |

Compute local descriptors and screening results. Each profile contains
`canonical_smiles`, `descriptors`, `screening`, and `predictions`. Predictors
receive each distinct canonical structure once. Batch results follow the
input order and include repeats.

Invalid structures raise `ParseError`. A predictor raises `ProviderError` if
it returns the wrong number of rows or provides a property key already returned
by another predictor. [Profiling tutorial](01-tools.md#1-profile-the-molecules) · [Source](../../src/halo/chem/eval/profile.py)

Use `profile_molecule` for one SMILES string. `profile_molecules` requires a
sequence, such as `[smiles]`, even when you want to profile only one molecule.
An empty sequence returns an empty list.

#### `read_smiles` and `cluster_molecules`

```python
def read_smiles(path: str | Path, *, column: str = "smiles") -> list[str]: ...

def cluster_molecules(
    smiles: Sequence[str],
    *,
    min_similarity: float = 0.70,
    include_chirality: bool = True,
) -> ClusterResult: ...
```

| Parameter | Meaning | Default |
| --- | --- | --- |
| `path` | Path to a UTF-8 CSV or SDF. | Required |
| `column` | Column containing SMILES strings. | `"smiles"` |
| `smiles` | Structures to cluster, in input order. | Required |
| `min_similarity` | Minimum Tanimoto similarity to the Butina seed, from 0 to 1. | `0.70` |
| `include_chirality` | Include chirality in the Morgan fingerprints. | `True` |

`read_smiles` preserves the input order and includes repeated structures.
It raises `ParseError` if the requested CSV column is missing or repeated,
if a SMILES cell is blank, or if an SDF record is invalid. SDF entries are
converted to canonical isomeric SMILES.

Clustering validates the structures and returns each cluster's member and
seed indices, the fingerprint method, and the distance cutoff
(`1 - min_similarity`). Print the result to inspect the clusters.
[Tutorial](../../examples/molecular-screening.md)

#### `evaluate_pareto`

```python
def evaluate_pareto(
    *,
    source: str | MoleculeProfile,
    candidates: Sequence[str | MoleculeProfile],
    objectives: Mapping[str, Direction],
) -> ParetoComparison: ...
```

| Parameter | Meaning |
| --- | --- |
| `source` | Structure or profile used as the baseline for property changes. |
| `candidates` | Structures or profiles to compare, in input order. |
| `objectives` | Descriptor, screening, or prediction names mapped to `"increase"` or `"decrease"`. |

When you provide SMILES strings, the function computes their profiles locally.
To compare predicted properties, provide profiles that already contain those
predictions. Missing, failed, nonfinite, or incompatible objective values
raise `ParseError`. Predictions can be compared only when they share a unit
and method.

| Result field | Meaning |
| --- | --- |
| `source_smiles` | Canonical SMILES of the comparison source. |
| `source_metrics` | Source value, unit, and method for each objective. |
| `objectives` | Property names and preferred directions. |
| `candidates` | Distinct candidates in input order, excluding the source. |

Each candidate records its `smiles` and property `values`. For each objective,
`delta` is the candidate's value minus the source's value, and `improved`
indicates whether that change is in the preferred direction. The remaining
flags report whether the candidate is on the Pareto front (`pareto_optimal`)
and whether it dominates the source (`dominates_source`).

The Pareto front compares candidates with each other. Use `improved` to check
each objective against `source`. Candidate values and deltas use the units
recorded in `source_metrics`. Use `print(comparison)` to see the baseline
and changes.
[Example](../../examples/molecular-screening.md) · [Source](../../src/halo/chem/operations.py)

#### `find_mcs` and `decompose_rgroups`

```python
def find_mcs(
    molecules: Sequence[str | Chem.Mol], *, min_coverage: float = 1.0,
) -> CommonCore | Failure: ...

def decompose_rgroups(
    smiles: Sequence[str], core: str | Chem.Mol | CommonCore,
) -> RGroupDecomposition: ...
```

| Parameter | Meaning | Default |
| --- | --- | --- |
| `molecules` | At least two structures to search for a common core. | Required |
| `min_coverage` | Fraction of input molecules that must match, in (0, 1]. | `1.0` |
| `smiles` | Molecular series to decompose. | Required |
| `core` | Shared SMILES, SMARTS query, RDKit molecule, or result of `find_mcs`. | Required |

`find_mcs` searches for a common core of at least two molecules. The result
contains `smarts`, `atom_count`, `matched_indices`, and the actual `coverage`.
It returns a `Failure` if no core meets the coverage or the search times out.

Print the result of `decompose_rgroups` to see its mapped `core_smiles` and a
table of fragments. The `rows` field contains the verified decompositions,
each with its zero-based index in the input. The `unmatched` field lists the
indices of molecules that could not be matched and rebuilt. The function
raises `ParseError` if no rows pass verification or if the rows require
different mapped cores. Use `row.replace(site=1, fragment="CCO*")` to edit
one of the verified rows. [Example](../../examples/molecular-tools.md)

#### `fragment_molecule`

```python
def fragment_molecule(value: str | Chem.Mol) -> Fragmentation: ...
```

Cut each eligible BRICS bond in a connected structure separately and list the
distinct pairs of fragments. The `cuts` field records which fragments came
from the same cut, while `fragments` lists the distinct fragments across
all cuts. The labels identify BRICS attachment classes. A structure with
no eligible bonds returns an empty result.

#### `replace_rgroup` and `draw_molecule`

```python
def replace_rgroup(
    *, smiles: str, core: str, site: int, fragment: str
) -> RGroupReplacement: ...

def draw_molecule(
    value: str | Chem.Mol | CommonCore,
    *,
    width: int = 300,
    height: int = 300,
    path: str | Path | None = None,
) -> bytes: ...
```

`replace_rgroup` decomposes the source using a mapped core, replaces the
fragment at `site`, and verifies the resulting structure. Provide a connected
fragment with one attachment point (for example, `CCO*`). The result contains
`product_smiles` and the recorded `edit`. Invalid edits raise `ParseError`.

`draw_molecule` accepts SMILES, InChI, molfile text, a common core, or an
RDKit molecule. It returns PNG bytes at the requested pixel dimensions.
If `path` is provided, it writes the PNG there and replaces any existing
file. [Example](../../examples/molecular-tools.md)

#### Prediction results

```python
def admet_properties() -> PropertyCatalog: ...
def descriptors() -> PropertyCatalog: ...
```

List ADMET properties accepted as criteria, or list calculated descriptors.
Print the catalog to read its table, or iterate over its definitions. Each
definition provides `name`, `definition`, `value_type`, and `unit`.
Classifiers return probabilities from zero to one.

Use `profile.prediction(name)` to inspect a property on a local or saved
profile. To inspect the prediction recorded when a candidate was generated,
use `candidate.run_prediction(name)`. That recorded prediction is the value
used to compute the candidate's saved change from the source. Print either
result to see its value, unit, and method. A failed prediction prints its
reason; an absent prediction prints a message that it is unavailable.

| Field | Computed | Failed | Unavailable |
| --- | --- | --- | --- |
| `status` | `"computed"` | `"failed"` | `"unavailable"` |
| `value` | Numerical value, including zero | `None` | `None` |
| `unit`, `method` | Recorded unit and method | `None` | `None` |
| `error` | `None` | Failure reason | `None` |

Stored `computed` metrics require a finite value and nonempty unit and method
strings. Stored `failed` metrics require a reason. When reading saved metrics,
HALO converts finite numeric strings to numbers; booleans, nonfinite values,
and incomplete records raise `DataError`.

### Projects

#### `HALOChem`

```python
class HALOChem:
    def __init__(
        self, *, base_url: str, access_token: str, timeout_s: float = 180.0
    ) -> None: ...
```

Import `HALOChem` from `halo`. Use it as a context manager or call `close()`
when finished.

| Parameter | Meaning | Default |
| --- | --- | --- |
| `base_url` | URL of the running HALO service. | Required |
| `access_token` | Access token from the user's Supabase Auth session. | Required |
| `timeout_s` | Timeout in seconds per HTTP network phase. | `180.0` |

The client raises `HALOChemError` for a failed request or turn. The
exception provides `code` and, when available, `status_code` or `turn_id`.
Check the project before repeating a write after a connection failure.
[Service and sign-in setup](../../README.md#api-credentials) · [Source](../../src/halo/clients/sdk.py)

#### Open a project

These methods are available on `halo.projects` after creating a `HALOChem` instance.

```python
def create(self, *, source_smiles: str) -> Project: ...
def get(self, project_id: str | UUID, *, resume: bool = False) -> Project: ...
def list(self) -> list[ProjectRecord]: ...
```

`create` profiles and saves the molecule identified by `source_smiles`.
`get` opens the project identified by `project_id`. The returned `Project`
provides the project's `id` and the lead's `source_molecule_id`.

The `Project` object opens a session on its first turn and reuses that
session for later turns. A new session starts with the lead as its active
parent. Set `resume=True` to continue the latest session instead. Once a
session is open, its identifier is available as `session_id`. Saved runs
and decisions remain available across sessions.

#### `generate_analogs`

```python
def generate_analogs(
    self,
    *,
    criteria: Criteria,
    n: int = 4,
    instructions: str | None = None,
    core_smiles: str | None = None,
    progress: bool | Progress = False,
    request_id: str | UUID | None = None,
) -> GenerationRun: ...
```

| Parameter | Meaning | Default |
| --- | --- | --- |
| `criteria` | ADMET properties mapped to `"increase"`, `"decrease"`, or `"preserve"` relative to the active parent. | Required |
| `n` | Number of requested analogs, from 1 to 6. | `4` |
| `instructions` | Natural-language guidance for this run, up to 3,000 characters. | `None` |
| `core_smiles` | Mapped core for fragment replacement. When omitted, the workflow uses the parent's saved core if available. | `None` |
| `progress` | Print live steps with `True`, or receive each step in a callback. | `False` |
| `request_id` | Reuse this UUID to recover the same request. | Generated |

Return a saved run with verified candidates, possibly fewer than requested.
`run.source_molecule_id` identifies the parent used for this run. A `preserve`
criterion favors predictions close to that parent's value.

For each candidate, `prediction_changes` records **candidate minus source**
in the recorded unit. A change is `None` when comparable predictions are
unavailable. `goal_outcomes` reports whether each criterion was met.

| Outcome | Meaning |
| --- | --- |
| `met` | The prediction moved in the requested direction, stayed equal for `preserve`, or reached a sampling target. |
| `missed` | The prediction failed to move in the requested direction or reach the sampling target. |
| `unavailable` | The saved results cannot support the comparison. |
| `unassessed` | The prediction changed for `preserve`, which has no tolerance. |

Use `print(run)` to see prediction changes and goal outcomes beside each
candidate. For structured output, use the run's fields or `run.model_dump_json()`.

For a run outside a project, call:

```python
def optimize(
    self, source_smiles: str, *, criteria: Criteria, n: int = 4,
    progress: bool | Progress = False,
    request_id: str | UUID | None = None,
) -> GenerationRun: ...
```

`HALOChem.optimize` saves the source molecule and generation run without
creating a project or session. It validates `criteria` and `n` before saving
the source or sending an HTTP request. Pydantic raises `ValidationError` if
`criteria` is empty or contains an unsupported property or direction.
The value of `n` must be an integer from 1 to 6; booleans and numeric strings
are not accepted.

#### `set_parent`

```python
def set_parent(
    self, candidate: CandidateResponse, *, run: GenerationRunResponse,
    request_id: str | UUID | None = None,
) -> None: ...
```

| Parameter | Meaning |
| --- | --- |
| `candidate` | A candidate from `run.candidates`. |
| `run` | The saved generation run in this project that produced the candidate. |

Makes the candidate the session's active parent and sole selection. A later
run records the candidate's molecule ID as `source_molecule_id` and its
originating run as `parent_generation_run_id`. A run from another project or
a candidate missing from the given run raises `ParseError`.

#### `research` and `ask`

```python
def research(
    self, query: str, *, molecule_id: int | None = None,
    progress: bool | Progress = False,
    request_id: str | UUID | None = None,
) -> ResearchResult: ...

def ask(
    self, message: str, *, progress: bool | Progress = False,
    request_id: str | UUID | None = None,
) -> Reply: ...
```

| Parameter | Meaning | Default |
| --- | --- | --- |
| `query` | Natural-language request guiding the paper and patent lookup. | Required |
| `molecule_id` | Project molecule to investigate. | `None`, which uses the project lead |
| `message` | Open-ended request for the `Orchestrator` to route. | Required |
| `progress` | Print live steps with `True`, or receive each step in a callback. | `False` |
| `request_id` | Reuse this UUID to recover the same request. | Generated |

- `research` saves the lookup as a session turn and returns a `ResearchResult`.
  Use `print(response)` to read its findings and separate exact, analog, and
  topic sources. Use `response.sources.for_scope("exact")` or `"analog"` to
  select a group. `sources.unavailable` lists failed lookups;
  `sources.exclusions` lists publication IDs and exclusion reasons.
  `turn` holds the saved record and trace.
- `ask` returns a printable `Reply`. Read `reply.turn` for the full saved
  turn, `reply.result` for the structured result, and `reply.run` for any
  saved generation run created by the turn.

#### Research passages

`ResearchResult.sources.passages` contains `Passage` records. Focused queries
about a structural change and a property add verified CC BY full-text passages
to the abstract and patent results. Full-text retrieval examines at most five
papers and retains up to three matching sentences per paper, within the
synthesis context limit. [Retrieval method](04-context.md#retrieve-full-text-passages)

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | `str` | Citation ID. Full-text IDs use `PAPER/{pmcid}/full-text-{paragraph}-{sentence}`. |
| `publication_id` | `str` | Identifier of the source publication. |
| `source_type` | `Literal["paper", "patent"]` | Publication type. |
| `title` | `str` | Publication title. |
| `text` | `str` | Original full-text sentence or the selected abstract excerpt. |
| `url` | `str` | Publication link. |
| `link` | `SourceLink` | Exact, analog, or topic relationship and its supporting database reference. |
| `pmcid` | `str \| None` | Europe PMC full-text identifier, when available. |
| `doi` | `str \| None` | DOI, when available. |
| `year` | `int \| None` | Publication year, when available. |
| `license_url` | `str \| None` | CC BY URL verified in the XML permissions for a full-text quote. |
| `locator` | `str \| None` | Paragraph and sentence position in the extracted main text. |

`SourceLink` records how a publication was connected to the molecule:

| `relation` | Evidence group | Link |
| --- | --- | --- |
| `exact_activity`, `exact_pubchem` | Exact | A ChEMBL activity or PubChem record for the requested structure. |
| `neighbor_activity`, `neighbor_pubchem` | Analog | A record for a similar structure. |
| `topic` | Topic | A text match without a verified structure link. |

`source_ref` and `matched_smiles` identify a linked database record and structure.
Analog similarity is recorded in `morgan_tanimoto` with its `fingerprint_method`.

The locator uses paragraph order after excluded content has been removed.
These records, including their citation IDs, are saved in `TurnResult.passages`.
`TurnResult.source_exclusions` preserves exclusion reasons as
`SourceExclusion(publication_id, reason)` records. Neither field requires a
separate database table.

#### Recover an interrupted request

```python
def get_turn(self, turn_id: int, *, session_id: str | UUID | None = None) -> TurnResponse: ...
def get_run(self, run_id: str | UUID) -> GenerationRun: ...
```

After an interruption, use `Project.get_turn` or `HALOChem.get_run` with the
saved `turn_id` or `run_id` provided by `HALOChemError`. If only `request_id`
is available, repeat the same request with that UUID. Project turns must be
replayed in the same session.
A replay while a turn is running raises `HALOChemError` with code `running`
and the saved turn ID. A standalone run replay returns its current status.

#### `history` and `remember`

```python
def history(self, session_id: str | UUID | None = None) -> History: ...
def remember(self, instruction: str, *, request_id: str | UUID | None = None) -> DecisionId: ...
```

`history` reads the active session, or the latest session if none is active.
Print the returned `History` to read the conversation. Use `history.turns`
to access its ordered records, or `history.model_dump_json()` to serialize
them.

`remember` saves design guidance for the next batch that saves candidates.
Use wording such as “in every round” to make the guidance apply to later
batches as well.

#### `runs` and `inspect_run`

```python
def runs(self) -> list[GenerationRun]: ...
def get_run(self, run_id: str | UUID) -> GenerationRun: ...
def inspect_run(
    self, run: GenerationRun | str | UUID, *, min_similarity: float = 0.70,
    min_core_coverage: float = 0.80,
) -> RunReview: ...
```

Use `runs` to list saved runs in creation order and `get_run` to fetch a run
by ID. `inspect_run` reviews the recorded predictions, goal outcomes, and
Pareto membership. It also annotates structures, clusters the candidates with
Butina, and searches for an MCS in clusters with at least two members.

`min_similarity` is the minimum Tanimoto similarity to each Butina seed, from
0 to 1. `min_core_coverage` is the fraction of each cluster that must match its
core, in (0, 1]. Cluster representatives in this review are medoids. Print the
review to read its tables, or use `review.tables()` to access their rows.

## Command line

Use `halo --help` or `uv run halo --help` from the repository. Local
commands run without credentials. Remote commands use `HALO_ACCESS_TOKEN`
and `HALO_BASE_URL` (default `http://localhost:8000`).

```bash
uv run halo chem profile 'COc1ccc(F)cc1'
uv run halo chem fragment 'CCOc1ccc(F)cc1'
uv run halo chem admet
uv run halo chem descriptors
```

Create a project, save an instruction, and generate analogs:

```bash
PROJECT_ID=$(uv run halo project create 'COc1ccc(F)cc1')
uv run halo project remember "$PROJECT_ID" "Retain the aryl fluoride in every round."
uv run halo project generate "$PROJECT_ID" --increase solubility --decrease herg --n 4
uv run halo project history "$PROJECT_ID"
```

Use `project inspect PROJECT_ID RUN_ID --min-core-coverage 0.9` with the
printed run ID to review the batch. Remote generation, research, and chat
show live steps on stderr by default. Use `--no-progress` to hide them.
Use `--json` on `ask`, `history`, or `inspect` for structured output.

## Advanced Python API

### Molecular structures

#### `to_mol`

```python
def to_mol(
    value: str | Chem.Mol,
    *,
    notation: Notation = "smiles",
    allow_dummies: bool = False,
) -> Chem.Mol: ...
```

Parse a structure or copy an RDKit molecule while preserving atom order and
explicit hydrogens. [Source](../../src/halo/chem/mol.py)

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `value` | `str \| Chem.Mol` | One structure string or an RDKit molecule. | Required |
| `notation` | `Notation` | One of `smiles`, `smarts`, `molblock`, or `inchi`. | `"smiles"` |
| `allow_dummies` | `bool` | Permit attachment atoms and hydrogen-only fragments. | `False` |

Returns a `Chem.Mol`. Raises `ParseError` for invalid input or disallowed atoms.

#### `decompose_molecule`

```python
def decompose_molecule(
    smiles: str,
    core: str,
    *,
    core_notation: Literal["smiles", "smarts"] = "smiles",
) -> Decomposition: ...
```

Split a molecule on a provided core and verify that the fragments reconstruct
the source stereochemistry. [Source](../../src/halo/chem/sar/rgroup.py)

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `smiles` | `str` | Source molecule. | Required |
| `core` | `str` | Core structure or query. Use mapped dummies to specify sites. | Required |
| `core_notation` | `Literal["smiles", "smarts"]` | Notation used by `core`. | `"smiles"` |

Returns a `Decomposition` with `molecule_index`, `core_smiles`, and `groups`
keyed by site label. Raises `ParseError` if the source cannot be decomposed
and rebuilt on the core.

#### `halo.chem.sar.rgroup.replace_rgroup`

```python
def replace_rgroup(
    source: Decomposition,
    *,
    site: int,
    fragment_smiles: str,
    label_attachment: bool = False,
) -> tuple[str, RGroupEdit]: ...
```

Replace one fragment and verify the structure of the assembled product.
[Source](../../src/halo/chem/sar/rgroup.py)

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source` | `Decomposition` | Core and fragments for the parent molecule. | Required |
| `site` | `int` | Attachment label present in the source. | Required |
| `fragment_smiles` | `str` | A connected fragment with one dummy atom whose attachment label matches the site. | Required |
| `label_attachment` | `bool` | Assign `site` to a lone dummy atom that has no map or isotope label. | `False` |

Returns the canonical product SMILES and an `RGroupEdit` with `site`,
`from_fragment_smiles`, `to_fragment_smiles`, and optional `reaction_smarts`.
Raises `ParseError` for invalid edits, unchanged structures, or changes to
the protected attachments or core.

### Molecular evaluation

#### `Fingerprints`

Import from `halo.chem.fingerprints`.

```python
class Fingerprints:
    @classmethod
    def from_mols(
        cls,
        mols: Sequence[Chem.Mol],
        *,
        radius: int = 2,
        n_bits: int = 2048,
        include_chirality: bool = False,
        prepared: bool = False,
    ) -> Self: ...
```

Compute Morgan fingerprints of the same length, in the order of the input
molecules. Set `prepared=True` only if those molecules are already sanitized
and have no ordinary hydrogens or atom maps.

| Access | Result |
| --- | --- |
| `len(fps)` | Number of fingerprints. |
| `for fp in fps` | Fingerprints in input order. |
| `fps[i]` | One fingerprint; negative indices count from the end. |
| `fps[start:stop]` | A tuple of fingerprints. |

An out-of-range index raises `IndexError`. An empty collection supports
iteration and slicing. [Clustering tutorial](../../examples/molecular-screening.md)

#### `cluster_butina`

```python
def cluster_butina(fps: Fingerprints, *, cutoff: float) -> ClusterResult: ...
```

Cluster binary fingerprints using a Tanimoto distance cutoff in `[0, 1]`.
Distances equal to the cutoff are included. Create the fingerprints with
`Fingerprints.from_mols`. The result records the cutoff and fingerprint
method. Each cluster contains the indices of its members and the index of
its Butina seed. [Source](../../src/halo/chem/cluster.py)

#### `halo.chem.eval.pareto.evaluate_pareto`

```python
def evaluate_pareto(
    *,
    baseline: MoleculeValues,
    candidates: Sequence[MoleculeValues],
    objectives: Mapping[str, Direction],
) -> list[ParetoEvaluation]: ...
```

Compare candidates using the provided objective values.
[Source](../../src/halo/chem/eval/pareto.py)

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `baseline` | `MoleculeValues` | Source ID and property values used to compute changes. | Required |
| `candidates` | `Sequence[MoleculeValues]` | Candidate IDs and values for the same properties. | Required |
| `objectives` | `Mapping[str, Direction]` | Property names mapped to `increase` or `decrease`. | Required |

Returns one evaluation per distinct candidate ID, in input order. Each row
contains the candidate's values and their differences from the baseline
(candidate minus baseline). It also records whether each objective improved,
whether the candidate is on the Pareto front, and whether it dominates the
baseline. The Pareto front is computed among the candidates.

The function raises `ParseError` if the objectives are empty or invalid, if
values are missing or nonfinite, or if repeated IDs have conflicting values.

[Pareto evaluation example](../../examples/molecular-screening.md)

### Inference

#### `OpenAIProvider` and `ModelRunner`

```python
class OpenAIProvider:
    def __init__(
        self,
        settings: OpenAISettings | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None: ...

    async def generate(self, request: ModelRequest) -> ModelResponse[str]: ...

    async def generate_structured(
        self, request: ModelRequest, schema: type[T]
    ) -> ModelResponse[T]: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, kw_only=True)
class ModelRunner:
    provider: ModelProvider
    settings: ModelSettings
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    roles: Mapping[AgentRole, ModelSettings] = field(default_factory=dict)

    def new_trace(self) -> ModelTrace: ...
```

The provider calls the model service. The runner combines that provider with
shared model settings and optional role settings. Use `ModelSettings` to set
the model name and token limit. Settings are immutable; construct new settings
when changing a configuration.

If you omit provider settings, `OpenAIProvider` reads `OPENAI_API_KEY` and
`OPENAI_BASE_URL` from the environment or `.env`. Pass `api_key` or `base_url`
to override them. Close the provider with `aclose()` when finished. The official
endpoint is recorded as `openai`; custom Responses API endpoints are recorded
as `openai_compatible`.
[Provider](../../src/halo/llm/providers/openai.py)

| Runner parameter | Type | Meaning | Default |
| --- | --- | --- | --- |
| `provider` | `ModelProvider` | Model adapter with text generation, structured generation, and cleanup. | Required |
| `settings` | `ModelSettings` | Shared model name, token limit, reasoning effort, and temperature. | Required |
| `runtime` | `RuntimeSettings` | Limits on model calls and context size, plus optional prompt capture. | `RuntimeSettings()` |
| `roles` | `Mapping[AgentRole, ModelSettings]` | Complete settings for individual roles. | `{}` |

`ModelRequest` holds the resolved inputs for one provider call:

| Field | Type | Default |
| --- | --- | --- |
| `model` | `str` | Required |
| `messages` | `tuple[Message, ...]` | Required; each message has `role` and `content` |
| `max_tokens` | `int \| None` | `None` |
| `temperature` | `float \| None` | `None` |
| `reasoning_effort` | `ReasoningEffort \| None` | `None` |
| `cache_namespace` | `str \| None` | `None` |

Requests and messages are immutable. `ModelResponse.output` contains either
text or the parsed schema instance. Its other fields are `usage`, `model`
(the returned model identifier), and `provider`.

Create one trace per operation with `new_trace()`. `ModelRunner.run()` and
`run_agent()` return `(output, call_index)`. A caller that validates output
itself uses that index with `record_validation()`. Agents with their own
validator record its outcome before returning.
[Runner](../../src/halo/llm/runtime.py)

#### Execution limits

| Limit | Default |
| --- | --- |
| Model calls per operation (`runtime.max_model_calls`) | 5, shared across routing, planning, synthesis, and retries. |
| Model context (`runtime.max_context_bytes`) | 12,000 UTF-8 bytes for the complete JSON user message. |
| Traced tool calls per turn | 4. |
| Turn dispatch timeout | 130 seconds. |
| Validation retries for selected agents | 1, within the shared call budget. |

`ModelTrace` records calls in start order, including concurrent samples.
Reported usage from failed calls remains in the total. Cached tokens are
included in input tokens; reasoning tokens are included in output tokens.
Run pricing uses the configured model name, while call traces retain the
returned model name. Unknown pricing remains unavailable.

Saved calls record `validation_status` (`accepted`, `partial`, or `rejected`)
and `validation_issues`. The status is empty when validation was not performed.
Calculate an issue rate from validated calls with nonempty issue lists, divided
by all calls with a validation status. Report unvalidated calls separately.

Prompt capture is off by default. Enable it in the YAML configuration to save
messages, settings, and the output schema in `model_calls.request_json`:

```yaml
runtime:
  capture_prompts: true
```

The invocation digest identifies the serialized request. [Runtime configuration](02-models.md#configure-language-models)

#### `respond`

```python
async def respond(
    task: ModelRunner,
    question: str,
    *,
    molecules: Mapping[MoleculeId, str | None],
    evidence: Sequence[Citation],
    results: Sequence[ResultValue] = (),
    trace: ModelTrace,
    role: Literal["researcher", "analyst"] = "researcher",
    evidence_origin: EvidenceOrigin = "application_record",
) -> Response: ...
```

Synthesize findings from provided records and verify their references.
[Source](../../src/halo/llm/agents/researcher.py)

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `task` | `ModelRunner` | Inference configuration. | Required |
| `question` | `str` | Question to answer. | Required |
| `molecules` | `Mapping[MoleculeId, str \| None]` | Allowed molecule IDs and their SMILES, when available. | Required |
| `evidence` | `Sequence[Citation]` | Text records with distinct citation IDs. | Required |
| `results` | `Sequence[ResultValue]` | Numeric records with distinct IDs and provenance. | `()` |
| `trace` | `ModelTrace` | Call budget and trace for this operation. | Required |
| `role` | `Literal["researcher", "analyst"]` | Instructions and schema validator to use. | `"researcher"` |
| `evidence_origin` | `EvidenceOrigin` | `application_record`, `external_document`, or `saved_record`. | `"application_record"` |

Returns a `Response` containing `findings` and `limitations`. Each finding
contains `text`, `molecule_ids`, `evidence_ids`, and `result_ids`.
`ProviderError` covers upstream failures and unusable model responses.
`LimitError` identifies local call or context limits. When research synthesis
reaches a local limit, the workflow retains its retrieved passages and reports
the limit in the response.

#### `EuropePMCClient`

Import from `halo.clients.chem.europe_pmc`.

```python
class EuropePMCClient:
    async def article(self, identifier: str) -> Article | None: ...
    async def search(self, query: str, *, limit: int = 12) -> tuple[Article, ...]: ...
    async def search_full_text(self, query: str, *, limit: int = 5) -> tuple[Article, ...]: ...
    async def full_text(self, pmcid: str) -> FullText: ...
```

`article()` accepts a DOI, PMID, or PMCID. Conventional `DOI:`, `PMID:`, and
`PMCID:` prefixes are supported, as are DOI URLs. It returns metadata even
when no abstract is available. A missing record returns `None`; malformed
identifiers raise `ParseError`, and service failures or mismatched records
raise a provider error.

`search()` returns records with abstracts. `search_full_text()` finds up to
five open-access candidates indexed as CC BY. `full_text()` parses an article's
XML, permissions, and main-text paragraphs. The research workflow verifies
license eligibility before selecting quoted sentences. Use the client as an
asynchronous context manager.

Look up metadata when you already know a publication identifier:

```python
import asyncio

from halo.clients.chem.europe_pmc import EuropePMCClient


async def lookup_article():
    async with EuropePMCClient() as papers:
        return await papers.article("DOI:10.1039/d5sc07348k")


article = asyncio.run(lookup_article())
print(article.title if article else "No matching article.")
```

[Source](../../src/halo/clients/chem/europe_pmc.py)

#### `SemanticScholarClient`

Import from `halo.clients.chem.s2`. Use this client for standalone paper and
passage searches. Pass a Semantic Scholar API key through `api_key`.

```python
class SemanticScholarClient:
    def __init__(
        self,
        http: HttpClient | None = None,
        *,
        api_key: str | None = None,
        requests_per_second: float = DEFAULT_INTEGRATIONS.semantic_scholar_requests_per_second,
    ) -> None: ...

    async def search_papers(
        self,
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        fields_of_study: Sequence[str] = (),
        year: str | None = None,
    ) -> PaperSearch: ...

    async def search_snippets(
        self, query: str, *, limit: int = 10
    ) -> tuple[Snippet, ...]: ...
```

For example, use the key exported as `SEMANTIC_SCHOLAR_API_KEY`:

```python
import asyncio
from os import environ

from halo.clients.chem.s2 import SemanticScholarClient


async def search_papers():
    async with SemanticScholarClient(
        api_key=environ["SEMANTIC_SCHOLAR_API_KEY"],
    ) as papers:
        return await papers.search_papers("matched molecular pairs", limit=3)


result = asyncio.run(search_papers())
for paper in result.papers:
    print(paper.title, paper.url)
```

`PaperSearch.papers` contains the returned metadata records.
[Source](../../src/halo/clients/chem/s2.py)

#### `ADMETAIPredictor`

```python
class ADMETAIPredictor:
    @classmethod
    def load(
        cls,
        *,
        oracle_data_dir: Path = DEFAULT_INTEGRATIONS.oracle_data_dir,
    ) -> "ADMETAIPredictor": ...

    def predict(self, smiles: str) -> dict[OptimizationTask, MetricResult]: ...

    def predict_batch(
        self,
        smiles: Sequence[str],
    ) -> list[dict[OptimizationTask, MetricResult]]: ...
```

`load` returns an `ADMETAIPredictor` with the ADMET-AI model and configured
PyTDC oracles. `oracle_data_dir` defaults to `Path("oracle")` and must end
in `oracle`.

Use `predict("CCO")` to predict properties for one molecule. For a batch,
use `predict_batch(["CCO"])`. The batch method returns one mapping from
property names to results for each input, in the same order and including
repeats. It sends each distinct canonical structure to each prediction
source once. An empty sequence returns an empty list. Passing a single
SMILES string to `predict_batch` raises `ParseError`.

Each `MetricResult` is either a successful `Metric` or a `Failure` with a
reason. A failed prediction does not discard successful predictions for
other properties. Invalid SMILES raise `ParseError`.
[Source](../../src/halo/chem/predict/admet_ai.py)

[Inference configuration](02-models.md)

#### `StructuralAnnotator`

Import from `halo.chem.medchem`.

```python
class StructuralAnnotator:
    def annotate(self, value: str | Chem.Mol) -> StructuralAnnotation: ...

    def annotate_many(
        self,
        values: Sequence[str | Chem.Mol],
        *,
        include_molblock: bool = False,
    ) -> list[StructuralAnnotation]: ...
```

Return the matched structural groups and catalog alerts. For `annotate`,
the result includes a molblock with the same atom order as the input.
For `annotate_many`, atom indices refer to the canonical SMILES. The batch
returns one result per input, including repeats, and preserves their order.
An empty sequence returns an empty list. Passing a single SMILES string
to `annotate_many` raises `ParseError`.
[Example](../../examples/molecular-tools.md#5-annotate-the-series)

## HTTP API

Before starting the server, install HALO, apply the database schema, and configure the credentials for the service. [Service setup](../../README.md#api-credentials)

The running server exposes its OpenAPI specification at `/openapi.json` and
interactive documentation at `/docs`. To authenticate an `/api` request,
send the user's Supabase access token in the
`Authorization: Bearer <access_token>` header.

`GET /health` returns `{"status": "ok"}` without authentication.
`GET /ready` also checks that startup has finished and that the service
can query the project database. It returns `{"status": "ready"}` when
both checks pass, or HTTP 503 with `{"status": "not_ready"}` otherwise.

### `create_app`

```python
def create_app(
    settings: Settings | None = None,
    *,
    predictor: PropertyPredictor | None = None,
    provider: ModelProvider | None = None,
) -> FastAPI: ...
```

Construct the service for an ASGI host. Pass `settings` to provide the
configuration explicitly, or use the environment and YAML defaults.
During startup, the application initializes `ADMETAIPredictor` and
`OpenAIProvider` unless you provide substitutes.
[Source](../../src/halo/app.py)

### Endpoints

| Method | Path | Request | Response |
| --- | --- | --- | --- |
| GET | `/health` | | Service status |
| GET | `/ready` | | Predictor and database readiness |
| POST | `/api/molecules` | `MoleculeRequest` | `ProfileResponse` |
| GET | `/api/molecules/{molecule_id}` | | `ProfileResponse` |
| POST | `/api/generation-runs` | `OptimizationRequest` or `SamplingRequest` | `GenerationRunResponse` |
| GET | `/api/generation-runs/{run_id}` | | `GenerationRunResponse` |
| POST | `/api/research` | `ResearchRequest` | `ResearchResponse` |
| POST | `/api/projects` | `ProjectRequest` | `Project` |
| GET | `/api/projects` | | Projects, newest first |
| GET | `/api/projects/{project_id}` | | `Project` |
| POST | `/api/projects/{project_id}/assay-results` | `AssayResultRequest` | `AssayResult` |
| GET | `/api/projects/{project_id}/assay-results` | Optional `molecule_id`, `assay`, and `endpoint` query filters | Assay results, newest first |
| GET | `/api/projects/{project_id}/generation-runs` | | Generation runs in creation order |
| POST | `/api/projects/{project_id}/sessions` | | `Session` |
| GET | `/api/projects/{project_id}/sessions` | | Sessions, newest first |
| GET | `/api/sessions/{session_id}` | | `Session` |
| POST | `/api/sessions/{session_id}/turns` | `TurnRequest` | `TurnResponse` |
| GET | `/api/sessions/{session_id}/turns` | | Turn summaries in conversation order |
| GET | `/api/sessions/{session_id}/turns/{turn_id}` | | `TurnResponse` with trace |

List endpoints return the matching records in the documented order.
The server handles batching internally. [Routes](../../src/halo/api/router.py) · [Wire schemas](../../src/halo/schemas/)

The POST endpoints that create turns and generation runs also support streaming
responses. To request one, send `Accept: application/x-ndjson`. After saving
the initial record, the server sends a line with a `started` object. That
object provides the `turn_id` or `run_id` and a `location` where the client
can retrieve the record.

The following lines report progress as `step` messages. The final line
contains the saved `result` or an `error`; an error includes `status_code`.
If the connection drops, the server continues the operation. The client can
retrieve its saved record at `location`.

### Request contracts

| Request | Fields and constraints |
| --- | --- |
| `MoleculeRequest` | `smiles` is a nonblank structure string, up to 2,000 characters. |
| `ProjectRequest` | `source_molecule_id` identifies a saved molecule owned by the caller. |
| `OptimizationRequest` | `type: "design"`, `source_molecule_id`, and a nonempty `criteria` map. `n` defaults to 4 and accepts 1–6. `request_id` is an optional UUID. |
| `SamplingRequest` | `type: "sample"`, `source_molecule_id`, and nonempty `tasks`. Optional `targets` are finite values keyed by a requested task. Sample count comes from configuration. `request_id` is an optional UUID. |
| `ResearchRequest` | `molecule_id` and a nonblank `question` of up to 3,000 characters. |
| `AssayResultRequest` | `molecule_id`, `assay_id`, `endpoint`, finite `value`, `unit`, `conditions`, and `source_ref`. Conditions are a JSON object up to 1,024 bytes. `relation` defaults to `=` and also accepts `<`, `<=`, `>`, and `>=`. An optional `decision_id` links the result to an accepted comparison. |
| `TurnRequest` | Exactly one of `message` or a typed `action`. A `research` action takes the project `molecule_id` and `question` fields. `request_id` is an optional UUID. |

If `type` is omitted, the generation endpoint infers it from the provided
fields. A request with `criteria` uses the design schema. A request with
`tasks` uses the sample schema.

For example, submit this body to `POST /api/generation-runs` using an existing
source molecule ID.

```json
{
  "type": "design",
  "source_molecule_id": 1,
  "criteria": {"solubility": "increase", "herg": "decrease"},
  "n": 4
}
```

`GenerationRunResponse` includes the candidates in order, the source's
recorded predictions (`source_predictions`), and model usage. For each
candidate, `prediction_changes` records its predicted value minus the
source's predicted value, using the saved unit. A null change means that
the values cannot be compared. The `goal_outcomes` field reports which
criteria or targets the saved results meet.

To create a session turn, send either a message or a typed action. The
following body records guidance for the next design run.

```json
{
  "action": {
    "type": "set_preference",
    "text": "Retain the aryl fluoride.",
    "scope": "next_run"
  }
}
```

Repeating a request with the same request ID and body returns the saved run
or turn. Reusing that ID with a different body returns HTTP 409.

For assay results, `source_ref` identifies a measurement within the project.
Repeating that reference with the same contents returns the saved result.
A conflicting measurement for that reference returns HTTP 409.

`TurnResponse.result` contains the task's output, or the saved error if the
turn failed. An HTTP response with status 200 can therefore contain a turn
whose status is `failed`. Inspect the turn's status as well as the HTTP status.

Turn statuses are `running`, `completed`, `failed`, and `cancelled`. At most
one turn can run in a project at a time. A turn still running after five
minutes is marked failed when a later request starts a turn or reads that turn.

### Error responses

Ordinary errors return `ErrorResponse(code, message)`. Server faults hide their
internal details. Authentication failures include `WWW-Authenticate: Bearer`.

| HTTP status | Code | Meaning |
| --- | --- | --- |
| 400 | `parse_error` | Invalid operation input. |
| 401 | `unauthenticated` | Missing or invalid bearer token. |
| 404 | `not_found` | Record absent or inaccessible to the caller. |
| 409 | `conflict` | Conflicting request or active operation. |
| 422 | `validation_error`, `limit_exceeded` | Invalid request fields or a local execution limit. |
| 502 | `provider_error` | Upstream service failure or unusable model output. |
| 500 | `internal_error` | Unexpected server failure. |

A saved turn can report a failed operation in its result while retaining the
turn record. Streaming errors carry the same code and message, plus
`status_code`. Cancellation ends the stream with a `cancelled` error.
