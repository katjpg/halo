import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import chdir
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from halo.chem.eval.objectives import ADMET_UNITS, is_current
from halo.chem.mol import canonicalize_smiles
from halo.chem.results import Failure, Metric, MetricResult, to_metric
from halo.config.settings import DEFAULT_INTEGRATIONS
from halo.core.errors import ConfigurationError, ParseError, ProviderError
from halo.types import ADMETProperty, OptimizationTask

if TYPE_CHECKING:
    from pandas import DataFrame

logger = logging.getLogger(__name__)

WARMUP_SMILES = "CCO"


ADMET_PROPERTIES: dict[ADMETProperty, str] = {
    "pampa": "PAMPA_NCATS",
    "bbbp": "BBB_Martins",
    "carcinogenicity": "Carcinogens_Lagunin",
    "herg": "hERG",
    "hia": "HIA_Hou",
    "dili": "DILI",
    "mutagenicity": "AMES",
    "solubility": "Solubility_AqSolDB",
}

ORACLES: dict[OptimizationTask, str] = {"drd2": "DRD2", "penalized_logp": "LogP"}

assert ADMET_PROPERTIES.keys() == ADMET_UNITS.keys()
assert ADMET_PROPERTIES.keys().isdisjoint(ORACLES)


OraclePredictor = Callable[[list[str]], Sequence[float]]


class ADMETBatchModel(Protocol):
    def predict(self, smiles: list[str]) -> "DataFrame | dict[str, float]":
        """Return a table indexed by SMILES for a batch."""
        ...


def _admet_failures(reason: str) -> dict[OptimizationTask, MetricResult]:
    return {name: Failure(reason) for name in ADMET_PROPERTIES}


class ADMETAIPredictor:
    """Molecular property predictions from ADMET-AI and TDC oracles.

    ADMET endpoints report probabilities or aqueous logS.
    """

    def __init__(
        self,
        model: ADMETBatchModel,
        oracles: Mapping[OptimizationTask, OraclePredictor] | None = None,
    ) -> None:
        self._model = model
        self._oracles = dict(oracles or {})
        model_version = version("admet-ai")
        self.methods: Mapping[ADMETProperty, str] = {
            key: f"admet_ai:{column}@{model_version}"
            for key, column in ADMET_PROPERTIES.items()
        }

    @classmethod
    def load(
        cls, *, oracle_data_dir: Path = DEFAULT_INTEGRATIONS.oracle_data_dir
    ) -> "ADMETAIPredictor":
        """Load the molecular prediction models during single-threaded startup.

        PyTDC reads a folder named oracle under the process working
        directory, which changes briefly during this call.
        """
        oracle_data_dir = oracle_data_dir.expanduser().absolute()
        if oracle_data_dir.name != "oracle":
            raise ConfigurationError(
                "integrations.oracle_data_dir must end in 'oracle'"
            )

        from admet_ai import ADMETModel
        from tdc import Oracle

        try:
            oracle_data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigurationError(
                f"Cannot create oracle directory {oracle_data_dir}."
            ) from exc
        oracles: dict[OptimizationTask, OraclePredictor] = {}
        with chdir(oracle_data_dir.parent):
            for name, oracle in ORACLES.items():
                try:
                    built = Oracle(name=oracle)
                    built([WARMUP_SMILES])
                except (Exception, SystemExit) as exc:
                    # PyTDC exits when an oracle file is corrupt.
                    raise ProviderError(
                        f"tdc oracle {oracle!r} is unavailable: {exc}"
                    ) from exc
                oracles[name] = built
        return cls(ADMETModel(), oracles)

    def warmup(self) -> None:
        """Run one ADMET-AI prediction before serving requests."""
        (values,) = self._predict_admet([WARMUP_SMILES])
        if not any(isinstance(value, Metric) for value in values.values()):
            raise ProviderError("ADMET-AI warmup returned no usable predictions.")

    def predict(self, smiles: str) -> dict[OptimizationTask, MetricResult]:
        """Predict the configured properties of one molecule."""
        return self.predict_batch([smiles])[0]

    def predict_batch(
        self, smiles: Sequence[str]
    ) -> list[dict[OptimizationTask, MetricResult]]:
        """Predict configured properties for each SMILES in input order.

        Repeated structures share predictions. A failed property does not
        discard successful results from other sources.
        """
        if isinstance(smiles, str):
            raise ParseError("Provide a sequence of SMILES, such as ['CCO'].")
        if not smiles:
            return []
        keys = {text: canonicalize_smiles(text) for text in dict.fromkeys(smiles)}
        distinct = list(dict.fromkeys(keys.values()))
        rows = self._predict_admet(distinct)
        for name in self._oracles:
            for row, value in zip(
                rows, self._predict_oracle(name, distinct), strict=True
            ):
                row[name] = value
        by_smiles = dict(zip(distinct, rows, strict=True))
        return [by_smiles[keys[text]].copy() for text in smiles]

    def _predict_admet(
        self, smiles: list[str]
    ) -> list[dict[OptimizationTask, MetricResult]]:
        try:
            frame = self._model.predict(smiles=smiles)
            if isinstance(frame, dict):
                raise ValueError("expected a table indexed by SMILES")
            rows = frame.to_dict(orient="index")
        except Exception as exc:
            logger.warning("admet_ai batch of %d failed: %s", len(smiles), exc)
            return [_admet_failures(f"admet_ai failed: {exc}") for _ in smiles]

        predictions: list[dict[OptimizationTask, MetricResult]] = []
        for key in smiles:
            row = rows.get(key)
            if row is None:
                predictions.append(_admet_failures("admet_ai returned no prediction"))
                continue
            values: dict[OptimizationTask, MetricResult] = {}
            for name, column in ADMET_PROPERTIES.items():
                method = self.methods[name]
                result = to_metric(
                    row.get(column), unit=ADMET_UNITS[name], method=method
                )
                if isinstance(result, Metric) and not is_current(name, result):
                    result = Failure(f"{method} returned an invalid probability")
                values[name] = result
            predictions.append(values)
        return predictions

    def _predict_oracle(
        self, name: OptimizationTask, smiles: list[str]
    ) -> list[MetricResult]:
        method = f"tdc:{ORACLES[name]}"
        try:
            values = self._oracles[name](smiles)
        except Exception as exc:
            logger.warning("%s batch of %d failed: %s", method, len(smiles), exc)
            return [Failure(f"{method} failed: {exc}") for _ in smiles]
        if len(values) != len(smiles):
            reason = f"{method} returned {len(values)} of {len(smiles)} values"
            logger.warning("%s", reason)
            return [Failure(reason) for _ in smiles]
        return [
            to_metric(value, unit="dimensionless", method=method) for value in values
        ]
