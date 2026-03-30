from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


EVENT_TYPES = ("out", "walk_hbp", "single", "double", "triple", "home_run")
BASE_OUT_STATE_COUNT = 24
DEFAULT_SIMULATION_COUNT = 20_000
DEFAULT_GAME_INNINGS = 9
DEFAULT_STARTER_INNINGS = 5
DEFAULT_MAX_PLATE_APPEARANCES_PER_INNING = 60

_FIRST_BASE = np.uint8(1)
_SECOND_BASE = np.uint8(2)
_THIRD_BASE = np.uint8(4)


@dataclass(frozen=True, slots=True)
class EventProbabilityProfile:
    out: float
    walk_hbp: float
    single: float
    double: float
    triple: float
    home_run: float

    def as_dict(self) -> dict[str, float]:
        return {
            "out": float(self.out),
            "walk_hbp": float(self.walk_hbp),
            "single": float(self.single),
            "double": float(self.double),
            "triple": float(self.triple),
            "home_run": float(self.home_run),
        }

    def cumulative_probabilities(self) -> np.ndarray:
        return np.cumsum(np.asarray([self.as_dict()[name] for name in EVENT_TYPES], dtype=float))


@dataclass(frozen=True, slots=True)
class HalfInningSimulationResult:
    runs: np.ndarray
    mean_plate_appearances: float
    event_counts: dict[str, int]
    truncated_half_innings: int


@dataclass(frozen=True, slots=True)
class AwayGameSimulationResult:
    support: np.ndarray
    pmf: np.ndarray
    simulated_runs: np.ndarray
    expected_runs: float
    shutout_probability: float
    tail_probabilities: dict[str, float]
    quantiles: dict[str, float]
    shape_summary: dict[str, float]
    diagnostics: dict[str, object]


def state_index(*, outs: int, bases: int) -> int:
    resolved_outs = int(outs)
    resolved_bases = int(bases)
    if resolved_outs < 0 or resolved_outs > 2:
        raise ValueError("outs must be between 0 and 2 for a live base-out state")
    if resolved_bases < 0 or resolved_bases > 7:
        raise ValueError("bases must be between 0 and 7")
    return (resolved_outs * 8) + resolved_bases


def decode_state_index(index: int) -> tuple[int, int]:
    resolved_index = int(index)
    if resolved_index < 0 or resolved_index >= BASE_OUT_STATE_COUNT:
        raise ValueError("state index must be between 0 and 23")
    return resolved_index // 8, resolved_index % 8


def normalize_event_probabilities(probabilities: Mapping[str, float]) -> EventProbabilityProfile:
    raw = np.asarray(
        [max(float(probabilities.get(name, 0.0)), 0.0) for name in EVENT_TYPES],
        dtype=float,
    )
    if not np.any(raw > 0.0):
        raise ValueError("at least one event probability must be positive")
    normalized = raw / raw.sum()
    return EventProbabilityProfile(
        out=float(normalized[0]),
        walk_hbp=float(normalized[1]),
        single=float(normalized[2]),
        double=float(normalized[3]),
        triple=float(normalized[4]),
        home_run=float(normalized[5]),
    )


def expected_runs_per_half_inning(
    profile: EventProbabilityProfile | Mapping[str, float],
) -> float:
    resolved_profile = (
        profile if isinstance(profile, EventProbabilityProfile) else normalize_event_probabilities(profile)
    )
    transition_matrix = np.zeros((BASE_OUT_STATE_COUNT, BASE_OUT_STATE_COUNT), dtype=float)
    expected_immediate_runs = np.zeros(BASE_OUT_STATE_COUNT, dtype=float)

    for outs in range(3):
        for bases in range(8):
            current_state = state_index(outs=outs, bases=bases)
            for event_name, event_probability in resolved_profile.as_dict().items():
                next_outs, next_bases, runs_scored, inning_over = apply_event_to_state(
                    outs=outs,
                    bases=bases,
                    event=event_name,
                )
                expected_immediate_runs[current_state] += float(event_probability) * float(runs_scored)
                if inning_over:
                    continue
                next_state = state_index(outs=next_outs, bases=next_bases)
                transition_matrix[current_state, next_state] += float(event_probability)

    expected_future_runs = np.linalg.solve(
        np.eye(BASE_OUT_STATE_COUNT, dtype=float) - transition_matrix,
        expected_immediate_runs,
    )
    return float(expected_future_runs[state_index(outs=0, bases=0)])


def expected_runs_for_game(
    *,
    starter_profile: EventProbabilityProfile | Mapping[str, float],
    bullpen_profile: EventProbabilityProfile | Mapping[str, float],
    innings: int = DEFAULT_GAME_INNINGS,
    starter_innings: int = DEFAULT_STARTER_INNINGS,
) -> float:
    resolved_innings = max(1, int(innings))
    resolved_starter_innings = min(max(0, int(starter_innings)), resolved_innings)
    starter_expected = expected_runs_per_half_inning(starter_profile)
    bullpen_expected = expected_runs_per_half_inning(bullpen_profile)
    return float(
        (resolved_starter_innings * starter_expected)
        + ((resolved_innings - resolved_starter_innings) * bullpen_expected)
    )


def apply_event_to_state(*, outs: int, bases: int, event: str) -> tuple[int, int, int, bool]:
    resolved_outs = int(outs)
    resolved_bases = int(bases)
    if resolved_outs < 0 or resolved_outs > 2:
        raise ValueError("outs must be between 0 and 2")
    if resolved_bases < 0 or resolved_bases > 7:
        raise ValueError("bases must be between 0 and 7")
    if event not in EVENT_TYPES:
        raise ValueError(f"unsupported event: {event}")

    on_first = bool(resolved_bases & int(_FIRST_BASE))
    on_second = bool(resolved_bases & int(_SECOND_BASE))
    on_third = bool(resolved_bases & int(_THIRD_BASE))

    if event == "out":
        next_outs = resolved_outs + 1
        inning_over = next_outs >= 3
        return min(next_outs, 3), 0 if inning_over else resolved_bases, 0, inning_over

    if event == "walk_hbp":
        runs = 1 if (on_first and on_second and on_third) else 0
        next_bases = (
            int(_FIRST_BASE)
            | (int(on_first or on_second) << 1)
            | (int(on_third or (on_first and on_second)) << 2)
        )
        return resolved_outs, next_bases, runs, False

    if event == "single":
        runs = int(on_third)
        next_bases = int(_FIRST_BASE) | (int(on_first) << 1) | (int(on_second) << 2)
        return resolved_outs, next_bases, runs, False

    if event == "double":
        runs = int(on_second) + int(on_third)
        next_bases = int(_SECOND_BASE) | (int(on_first) << 2)
        return resolved_outs, next_bases, runs, False

    if event == "triple":
        runs = int(on_first) + int(on_second) + int(on_third)
        return resolved_outs, int(_THIRD_BASE), runs, False

    runs = 1 + int(on_first) + int(on_second) + int(on_third)
    return resolved_outs, 0, runs, False


def simulate_half_inning(
    profile: EventProbabilityProfile | Mapping[str, float],
    *,
    simulations: int,
    rng: np.random.Generator,
    max_plate_appearances: int = DEFAULT_MAX_PLATE_APPEARANCES_PER_INNING,
) -> HalfInningSimulationResult:
    resolved_profile = (
        profile if isinstance(profile, EventProbabilityProfile) else normalize_event_probabilities(profile)
    )
    resolved_simulations = max(1, int(simulations))
    resolved_max_pa = max(3, int(max_plate_appearances))

    runs = np.zeros(resolved_simulations, dtype=np.int16)
    outs = np.zeros(resolved_simulations, dtype=np.int8)
    bases = np.zeros(resolved_simulations, dtype=np.uint8)
    active = np.ones(resolved_simulations, dtype=bool)
    event_counts = {name: 0 for name in EVENT_TYPES}
    total_plate_appearances = 0
    truncated_half_innings = 0
    cumulative = resolved_profile.cumulative_probabilities()

    plate_appearance_index = 0
    while bool(np.any(active)):
        if plate_appearance_index >= resolved_max_pa:
            truncated_half_innings = int(np.count_nonzero(active))
            break

        active_indices = np.flatnonzero(active)
        draws = rng.random(len(active_indices))
        sampled = np.searchsorted(cumulative, draws, side="right")
        total_plate_appearances += len(active_indices)
        plate_appearance_index += 1

        for event_index, event_name in enumerate(EVENT_TYPES):
            event_mask = sampled == event_index
            if not bool(np.any(event_mask)):
                continue

            target_indices = active_indices[event_mask]
            event_counts[event_name] += int(len(target_indices))

            if event_name == "out":
                outs[target_indices] = outs[target_indices] + 1
                inning_over = outs[target_indices] >= 3
                if bool(np.any(inning_over)):
                    finished_indices = target_indices[inning_over]
                    active[finished_indices] = False
                    bases[finished_indices] = 0
                continue

            event_bases = bases[target_indices]
            on_first = (event_bases & _FIRST_BASE) > 0
            on_second = (event_bases & _SECOND_BASE) > 0
            on_third = (event_bases & _THIRD_BASE) > 0

            if event_name == "walk_hbp":
                runs[target_indices] = runs[target_indices] + (on_first & on_second & on_third).astype(np.int16)
                new_bases = (
                    _FIRST_BASE
                    | ((on_first | on_second).astype(np.uint8) << 1)
                    | ((on_third | (on_first & on_second)).astype(np.uint8) << 2)
                )
                bases[target_indices] = new_bases
                continue

            if event_name == "single":
                runs[target_indices] = runs[target_indices] + on_third.astype(np.int16)
                bases[target_indices] = (
                    _FIRST_BASE
                    | (on_first.astype(np.uint8) << 1)
                    | (on_second.astype(np.uint8) << 2)
                )
                continue

            if event_name == "double":
                runs[target_indices] = (
                    runs[target_indices]
                    + on_second.astype(np.int16)
                    + on_third.astype(np.int16)
                )
                bases[target_indices] = _SECOND_BASE | (on_first.astype(np.uint8) << 2)
                continue

            if event_name == "triple":
                runs[target_indices] = (
                    runs[target_indices]
                    + on_first.astype(np.int16)
                    + on_second.astype(np.int16)
                    + on_third.astype(np.int16)
                )
                bases[target_indices] = np.full(len(target_indices), _THIRD_BASE, dtype=np.uint8)
                continue

            runs[target_indices] = (
                runs[target_indices]
                + 1
                + on_first.astype(np.int16)
                + on_second.astype(np.int16)
                + on_third.astype(np.int16)
            )
            bases[target_indices] = np.zeros(len(target_indices), dtype=np.uint8)

    if truncated_half_innings > 0:
        active[active] = False
        bases[:] = 0

    return HalfInningSimulationResult(
        runs=runs,
        mean_plate_appearances=float(total_plate_appearances / resolved_simulations),
        event_counts=event_counts,
        truncated_half_innings=truncated_half_innings,
    )


def simulate_away_game_distribution(
    *,
    starter_profile: EventProbabilityProfile | Mapping[str, float],
    bullpen_profile: EventProbabilityProfile | Mapping[str, float],
    simulations: int = DEFAULT_SIMULATION_COUNT,
    innings: int = DEFAULT_GAME_INNINGS,
    starter_innings: int = DEFAULT_STARTER_INNINGS,
    seed: int | None = None,
    max_plate_appearances_per_inning: int = DEFAULT_MAX_PLATE_APPEARANCES_PER_INNING,
) -> AwayGameSimulationResult:
    resolved_simulations = max(1, int(simulations))
    resolved_innings = max(1, int(innings))
    resolved_starter_innings = min(max(0, int(starter_innings)), resolved_innings)
    rng = np.random.default_rng(seed)

    starter = starter_profile if isinstance(starter_profile, EventProbabilityProfile) else normalize_event_probabilities(starter_profile)
    bullpen = bullpen_profile if isinstance(bullpen_profile, EventProbabilityProfile) else normalize_event_probabilities(bullpen_profile)

    simulated_runs = np.zeros(resolved_simulations, dtype=np.int16)
    mean_runs_by_inning: list[float] = []
    mean_plate_appearances_by_inning: list[float] = []
    total_event_counts = {name: 0 for name in EVENT_TYPES}
    truncated_half_innings = 0

    for inning_number in range(1, resolved_innings + 1):
        active_profile = starter if inning_number <= resolved_starter_innings else bullpen
        inning_result = simulate_half_inning(
            active_profile,
            simulations=resolved_simulations,
            rng=rng,
            max_plate_appearances=max_plate_appearances_per_inning,
        )
        simulated_runs = simulated_runs + inning_result.runs
        mean_runs_by_inning.append(float(np.mean(inning_result.runs)))
        mean_plate_appearances_by_inning.append(inning_result.mean_plate_appearances)
        truncated_half_innings += int(inning_result.truncated_half_innings)
        for event_name, count in inning_result.event_counts.items():
            total_event_counts[event_name] += int(count)

    support = np.arange(int(np.max(simulated_runs, initial=0)) + 1, dtype=int)
    counts = np.bincount(simulated_runs, minlength=len(support))
    pmf = counts.astype(float) / float(resolved_simulations)
    total_plate_appearances = int(sum(total_event_counts.values()))
    tail_probabilities = {
        "p_ge_1": float(np.mean(simulated_runs >= 1)),
        "p_ge_3": float(np.mean(simulated_runs >= 3)),
        "p_ge_5": float(np.mean(simulated_runs >= 5)),
        "p_ge_10": float(np.mean(simulated_runs >= 10)),
    }
    quantiles = distribution_quantiles(support, pmf, quantiles=(0.25, 0.50, 0.75))
    shape_summary = summarize_distribution_shape(
        support,
        pmf,
        quantiles=quantiles,
    )

    return AwayGameSimulationResult(
        support=support,
        pmf=pmf,
        simulated_runs=simulated_runs,
        expected_runs=float(np.mean(simulated_runs)),
        shutout_probability=float(pmf[0]) if len(pmf) > 0 else 0.0,
        tail_probabilities=tail_probabilities,
        quantiles=quantiles,
        shape_summary=shape_summary,
        diagnostics={
            "simulation_count": resolved_simulations,
            "innings": resolved_innings,
            "starter_innings": resolved_starter_innings,
            "seed": None if seed is None else int(seed),
            "max_simulated_runs": int(np.max(simulated_runs, initial=0)),
            "mean_runs_by_inning": mean_runs_by_inning,
            "mean_plate_appearances_by_inning": mean_plate_appearances_by_inning,
            "mean_plate_appearances_per_game": float(total_plate_appearances / resolved_simulations),
            "event_counts": total_event_counts,
            "event_share_by_plate_appearance": {
                name: (float(count) / float(total_plate_appearances) if total_plate_appearances > 0 else 0.0)
                for name, count in total_event_counts.items()
            },
            "truncated_half_innings": int(truncated_half_innings),
        },
    )


def pad_probability_vector(
    probabilities: Sequence[float] | np.ndarray,
    *,
    support_max: int,
) -> np.ndarray:
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    resolved_support_max = max(int(support_max), len(array) - 1)
    padded = np.zeros(resolved_support_max + 1, dtype=float)
    padded[: len(array)] = array
    total = float(padded.sum())
    if total <= 0.0:
        raise ValueError("probabilities must contain positive mass")
    return padded / total


def distribution_quantiles(
    support: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    quantiles: Sequence[float] = (0.25, 0.50, 0.75),
) -> dict[str, float]:
    resolved_support = np.asarray(list(support), dtype=float)
    normalized = _normalize_probability_array(probabilities)
    if len(resolved_support) != len(normalized):
        raise ValueError("support and probabilities must have matching lengths")

    cdf = np.cumsum(normalized)
    result: dict[str, float] = {}
    for quantile in quantiles:
        resolved_quantile = float(quantile)
        if resolved_quantile < 0.0 or resolved_quantile > 1.0:
            raise ValueError("quantiles must be between 0 and 1")
        index = int(np.argmax(cdf >= resolved_quantile))
        result[f"p{int(round(resolved_quantile * 100.0))}"] = float(resolved_support[index])
    return result


def summarize_distribution_shape(
    support: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    quantiles: Mapping[str, float] | None = None,
) -> dict[str, float]:
    resolved_support = np.asarray(list(support), dtype=float)
    normalized = _normalize_probability_array(probabilities)
    if len(resolved_support) != len(normalized):
        raise ValueError("support and probabilities must have matching lengths")

    expected_value = float(np.dot(resolved_support, normalized))
    centered = resolved_support - expected_value
    variance = float(np.dot(centered**2, normalized))
    resolved_quantiles = (
        dict(quantiles)
        if quantiles is not None
        else distribution_quantiles(resolved_support, normalized, quantiles=(0.25, 0.50, 0.75))
    )
    entropy = float(
        -np.sum(
            normalized[normalized > 0.0] * np.log(normalized[normalized > 0.0])
        )
    )
    p25 = float(resolved_quantiles.get("p25", expected_value))
    p75 = float(resolved_quantiles.get("p75", expected_value))
    return {
        "variance": variance,
        "stddev": float(np.sqrt(max(variance, 0.0))),
        "entropy": entropy,
        "iqr": float(p75 - p25),
    }


def _normalize_probability_array(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    clipped = np.clip(array, 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        raise ValueError("probabilities must contain positive mass")
    return clipped / total


__all__ = [
    "AwayGameSimulationResult",
    "BASE_OUT_STATE_COUNT",
    "DEFAULT_GAME_INNINGS",
    "DEFAULT_MAX_PLATE_APPEARANCES_PER_INNING",
    "DEFAULT_SIMULATION_COUNT",
    "DEFAULT_STARTER_INNINGS",
    "EVENT_TYPES",
    "EventProbabilityProfile",
    "HalfInningSimulationResult",
    "apply_event_to_state",
    "decode_state_index",
    "distribution_quantiles",
    "expected_runs_for_game",
    "expected_runs_per_half_inning",
    "normalize_event_probabilities",
    "pad_probability_vector",
    "simulate_away_game_distribution",
    "simulate_half_inning",
    "summarize_distribution_shape",
    "state_index",
]
