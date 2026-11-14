# Models and inference

Molecular property prediction estimates endpoints such as aqueous solubility from a structure. Use predictions to compare analogs against the same property criteria. Language models have a different role in HALO: they propose structures, interpret supplied evidence, and suggest subsequent investigations.

## Types of results

| Result | Origin | Example |
| --- | --- | --- |
| Descriptor | Calculation from molecular structure | RDKit molecular weight in g/mol |
| Prediction | Estimate from a molecular model | ADMET-AI solubility in log₁₀ mol/L |
| Measurement | Reported experimental result | An assay value with its conditions and source |

Predictions are comparable when they share an endpoint, unit, and method. A design run saves the predictions used for both its source and its candidates.

## Predict molecular properties

### 1. List the supported properties

The catalog lists the ADMET properties accepted as design criteria, with definitions and value types.

```python
from halo import chem

print(chem.admet_properties())
```

### 2. Load the predictor

`ADMETAIPredictor` combines ADMET-AI predictions with configured PyTDC scoring functions.

```python
from halo.chem.predict.admet_ai import ADMETAIPredictor

predictor = ADMETAIPredictor.load()
```

### 3. Predict one molecule

```python
predictions = predictor.predict("CCO")
print(predictions["solubility"])
```

*Recorded example output (ADMET-AI 1.4.0).*

```text
1.263 log10 mol/L (admet_ai:Solubility_AqSolDB@1.4.0)
```

The printed result includes its value, unit, and method. Failed predictions include a reason; successful predictions for other properties remain available.

### 4. Predict a series

```python
smiles = ["CCO", "CCN", "CCO"]
batch_predictions = predictor.predict_batch(smiles)
```

Results follow the input order and include repeats. Use `predict` for a single SMILES string and `predict_batch` for a sequence.

To compute descriptors and predictions together, pass the loaded predictor to `chem.profile_molecule` or `chem.profile_molecules` through `predictors=(predictor,)`.

| Prediction source | Output |
| --- | --- |
| ADMET-AI classifiers | Probabilities for endpoints such as hERG inhibition and human intestinal absorption. |
| ADMET-AI solubility model | Aqueous logS in log₁₀ mol/L. |
| PyTDC scoring functions | DRD2 activity score and penalized logP. |

[Property definitions](https://github.com/swansonk14/admet_ai/blob/main/admet_ai/resources/data/admet.csv) · [Predictor reference](05-reference.md#admetaipredictor)

## Configure language models

Each agent uses instructions and an output schema for its task. HALO validates the returned structures, references, and selections before using them. Choose a model for each role according to the task it performs.

| Configuration | Default | Purpose |
| --- | --- | --- |
| `agents.model` | `gpt-6-luna` | Routing, synthesis, and strategy. |
| `agents.max_tokens` | `1800` | Output token limit for these roles. |
| `agents.roles.optimizer.model` | `gpt-6.1-terra` | Molecular proposals in the repository configuration. |
| `agents.roles.optimizer.max_tokens` | `2048` | Output token limit for molecular proposals in that configuration. |

### Set role-specific options

Add a role under `agents.roles` in the YAML configuration. Available roles are `orchestrator`, `optimizer`, `researcher`, `analyst`, and `strategist`.

```yaml
agents:
  roles:
    researcher:
      model: gpt-6-luna
      max_tokens: 2400
      reasoning_effort: low
```

Save this configuration as `settings.yaml` and start the service:

```bash
uv run halo serve --config settings.yaml
```

Set `reasoning_effort` to `none` when specifying a temperature. Configure credentials through `OPENAI_API_KEY`; use `OPENAI_BASE_URL` for a server supporting the Responses API and structured output. [Service setup](../../README.md#api-credentials)

For provider interfaces, execution limits, usage, and prompt capture, see the [inference reference](05-reference.md#inference).
