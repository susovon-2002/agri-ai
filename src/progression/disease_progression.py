"""
AgriVision AI
Disease Progression Engine
============================================

Purpose
-------
Analyze repeated observations of the same plant and estimate:

1. Disease consistency
2. Affected-area progression
3. Progression velocity
4. Linear trend
5. Relative percentage change
6. Severity transition
7. Confidence consistency
8. Overall progression state
9. Risk flags
10. Structured JSON + CSV reports

Input structure
---------------
data/progression/
    plant_001/
        day_01.json
        day_04.json
        day_07.json

OR

data/progression/
    plant_001/
        day_01.jpg
        day_04.jpg
        day_07.jpg

The preferred production workflow is to first run the existing
AgriVision inference pipeline on each image and save its structured
result as JSON.

Example observation JSON:

{
    "plant_id": "plant_001",
    "observation_date": "2026-09-01",
    "disease": "Tomato_Late_blight",
    "crop": "Tomato",
    "confidence": 0.97,
    "affected_area_percent": 15.2,
    "severity": "Moderate"
}

This engine does NOT claim biological disease progression from
image percentage alone. It reports image-derived progression
signals and flags them for decision support.

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "progression_images"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "progression"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_OBSERVATIONS = 2

STABLE_CHANGE_THRESHOLD = 2.0
INCREASING_CHANGE_THRESHOLD = 5.0

LOW_CONFIDENCE_THRESHOLD = 0.70
HIGH_CONFIDENCE_THRESHOLD = 0.90

SEVERITY_ORDER = {
    "Healthy": 0,
    "Low": 1,
    "Moderate": 2,
    "High": 3,
}


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class Observation:
    plant_id: str
    observation_date: datetime

    disease: str
    crop: str

    confidence: float
    affected_area_percent: float
    severity: str

    source_file: str = ""


# ============================================================
# SAFE NUMERIC HELPERS
# ============================================================

def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(
        minimum,
        min(maximum, value),
    )


def safe_float(
    value,
    default=0.0,
):

    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except (
        TypeError,
        ValueError,
    ):

        return default


# ============================================================
# DATE PARSER
# ============================================================

def parse_date(value):

    if isinstance(
        value,
        datetime,
    ):
        return value

    text = str(value).strip()

    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]

    for fmt in formats:

        try:
            return datetime.strptime(
                text,
                fmt,
            )
        except ValueError:
            pass

    raise ValueError(
        f"Unsupported date format: {value}"
    )


# ============================================================
# OBSERVATION VALIDATION
# ============================================================

def validate_observation(
    data: dict,
):

    required = [
        "observation_date",
        "disease",
        "crop",
        "confidence",
        "affected_area_percent",
        "severity",
    ]

    missing = [
        key
        for key in required
        if key not in data
    ]

    if missing:

        raise ValueError(
            "Missing required fields: "
            + ", ".join(missing)
        )


# ============================================================
# LOAD JSON OBSERVATIONS
# ============================================================

def load_json_observations():

    observations = []

    if not INPUT_DIR.exists():

        return observations

    json_files = sorted(
        INPUT_DIR.rglob("*.json")
    )

    for path in json_files:

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            if isinstance(
                data,
                list,
            ):

                records = data

            elif isinstance(
                data,
                dict,
            ):

                records = [data]

            else:

                print(
                    f"Skipping invalid JSON: "
                    f"{path}"
                )

                continue

            default_plant_id = (
                path.parent.name
            )

            for record in records:

                if not isinstance(
                    record,
                    dict,
                ):
                    continue

                try:

                    validate_observation(
                        record
                    )

                    plant_id = str(
                        record.get(
                            "plant_id",
                            default_plant_id,
                        )
                    )

                    observation = Observation(
                        plant_id=plant_id,

                        observation_date=parse_date(
                            record[
                                "observation_date"
                            ]
                        ),

                        disease=str(
                            record[
                                "disease"
                            ]
                        ),

                        crop=str(
                            record[
                                "crop"
                            ]
                        ),

                        confidence=clamp(
                            safe_float(
                                record[
                                    "confidence"
                                ]
                            ),
                            0.0,
                            1.0,
                        ),

                        affected_area_percent=clamp(
                            safe_float(
                                record[
                                    "affected_area_percent"
                                ]
                            ),
                            0.0,
                            100.0,
                        ),

                        severity=str(
                            record[
                                "severity"
                            ]
                        ),

                        source_file=str(
                            path
                        ),
                    )

                    observations.append(
                        observation
                    )

                except Exception as exc:

                    print(
                        f"Skipping record "
                        f"in {path}: {exc}"
                    )

        except Exception as exc:

            print(
                f"Could not read "
                f"{path}: {exc}"
            )

    return observations


# ============================================================
# SORT OBSERVATIONS
# ============================================================

def sort_observations(
    observations,
):

    return sorted(
        observations,
        key=lambda item: (
            item.plant_id,
            item.observation_date,
        ),
    )


# ============================================================
# LINEAR REGRESSION
# ============================================================

def linear_trend(
    days,
    values,
):

    x = np.asarray(
        days,
        dtype=float,
    )

    y = np.asarray(
        values,
        dtype=float,
    )

    # Guard: < 2 points, or all observations on the same day (x range == 0)
    if len(x) < 2 or np.ptp(x) == 0.0:

        return {
            "slope_per_day": 0.0,
            "intercept": float(np.mean(y)) if len(y) else 0.0,
            "r_squared": 0.0,
        }

    try:
        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )

        predicted = slope * x + intercept
        ss_res = np.sum((y - predicted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = 0.0 if ss_tot <= 1e-12 else max(0.0, 1.0 - ss_res / ss_tot)

        return {
            "slope_per_day": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
        }

    except Exception:

        return {
            "slope_per_day": 0.0,
            "intercept": float(np.mean(y)) if len(y) else 0.0,
            "r_squared": 0.0,
        }


# ============================================================
# DISEASE CONSISTENCY
# ============================================================

def disease_consistency(
    observations,
):

    diseases = [
        item.disease
        for item in observations
    ]

    if not diseases:

        return {
            "consistent": False,
            "dominant_disease": None,
            "consistency_percent": 0.0,
        }

    counts = {}

    for disease in diseases:

        counts[disease] = (
            counts.get(
                disease,
                0,
            )
            + 1
        )

    dominant = max(
        counts,
        key=counts.get,
    )

    consistency = (
        counts[dominant]
        / len(diseases)
        * 100
    )

    return {
        "consistent": (
            len(counts) == 1
        ),
        "dominant_disease": dominant,
        "consistency_percent": float(
            consistency
        ),
    }


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

def confidence_analysis(
    observations,
):

    values = np.array(
        [
            item.confidence
            for item in observations
        ],
        dtype=float,
    )

    return {
        "mean_confidence": float(
            values.mean()
        ),
        "minimum_confidence": float(
            values.min()
        ),
        "maximum_confidence": float(
            values.max()
        ),
        "confidence_std": float(
            values.std()
        ),
        "low_confidence_observations": int(
            np.sum(
                values
                < LOW_CONFIDENCE_THRESHOLD
            )
        ),
    }


# ============================================================
# AFFECTED AREA ANALYSIS
# ============================================================

def affected_area_analysis(
    observations,
):

    first = observations[0]
    last = observations[-1]

    start_area = (
        first.affected_area_percent
    )

    end_area = (
        last.affected_area_percent
    )

    absolute_change = (
        end_area
        - start_area
    )

    if start_area > 0:

        relative_change = (
            absolute_change
            / start_area
            * 100
        )

    else:

        relative_change = (
            float("inf")
            if end_area > 0
            else 0.0
        )

    days = [
        (
            item.observation_date
            - first.observation_date
        ).total_seconds()
        / 86400.0
        for item in observations
    ]

    values = [
        item.affected_area_percent
        for item in observations
    ]

    trend = linear_trend(
        days,
        values,
    )

    if len(observations) >= 2:

        total_days = days[-1]

        if total_days > 0:

            velocity = (
                absolute_change
                / total_days
            )

        else:

            velocity = 0.0

    else:

        velocity = 0.0

    return {
        "initial_affected_area_percent": float(
            start_area
        ),
        "latest_affected_area_percent": float(
            end_area
        ),
        "absolute_change_percentage_points": float(
            absolute_change
        ),
        "relative_change_percent": float(
            relative_change
        ),
        "progression_velocity_percent_per_day": float(
            velocity
        ),
        "trend_slope_percent_per_day": float(
            trend[
                "slope_per_day"
            ]
        ),
        "trend_r_squared": float(
            trend[
                "r_squared"
            ]
        ),
    }


# ============================================================
# SEVERITY ANALYSIS
# ============================================================

def severity_analysis(
    observations,
):

    first = observations[0]
    last = observations[-1]

    first_level = SEVERITY_ORDER.get(
        first.severity,
        -1,
    )

    last_level = SEVERITY_ORDER.get(
        last.severity,
        -1,
    )

    return {
        "initial_severity": first.severity,
        "latest_severity": last.severity,
        "severity_level_change": (
            last_level
            - first_level
            if first_level >= 0
            and last_level >= 0
            else None
        ),
        "severity_increased": (
            last_level > first_level
            if first_level >= 0
            and last_level >= 0
            else False
        ),
        "severity_decreased": (
            last_level < first_level
            if first_level >= 0
            and last_level >= 0
            else False
        ),
    }


# ============================================================
# PROGRESSION CLASSIFICATION
# ============================================================

def classify_progression(
    area_analysis,
    severity_result,
    consistency,
):

    change = (
        area_analysis[
            "absolute_change_percentage_points"
        ]
    )

    slope = (
        area_analysis[
            "trend_slope_percent_per_day"
        ]
    )

    severity_up = (
        severity_result[
            "severity_increased"
        ]
    )

    severity_down = (
        severity_result[
            "severity_decreased"
        ]
    )

    if not consistency[
        "consistent"
    ]:

        state = (
            "DISEASE_LABEL_UNCERTAIN"
        )

    elif (
        change
        >= INCREASING_CHANGE_THRESHOLD
        or slope
        >= 1.0
        or severity_up
    ):

        state = "INCREASING"

    elif (
        change
        <= -INCREASING_CHANGE_THRESHOLD
        or slope
        <= -1.0
        or severity_down
    ):

        state = "DECREASING"

    elif (
        abs(change)
        <= STABLE_CHANGE_THRESHOLD
    ):

        state = "STABLE"

    else:

        state = "UNCERTAIN"

    return state


# ============================================================
# RISK FLAGS
# ============================================================

def generate_risk_flags(
    observations,
    area_analysis,
    severity_result,
    confidence_result,
    consistency,
):

    flags = []

    if not consistency[
        "consistent"
    ]:

        flags.append(
            "Disease label changed across observations."
        )

    if (
        consistency[
            "consistency_percent"
        ]
        < 80
    ):

        flags.append(
            "Disease classification consistency is below 80%."
        )

    if (
        confidence_result[
            "minimum_confidence"
        ]
        < LOW_CONFIDENCE_THRESHOLD
    ):

        flags.append(
            "At least one observation has low model confidence."
        )

    if (
        area_analysis[
            "trend_r_squared"
        ]
        < 0.50
    ):

        flags.append(
            "Affected-area trend has weak linear fit."
        )

    if (
        area_analysis[
            "initial_affected_area_percent"
        ]
        == 0
        and area_analysis[
            "latest_affected_area_percent"
        ]
        > 0
    ):

        flags.append(
            "Disease-associated affected area appeared after the first observation."
        )

    if (
        severity_result[
            "severity_increased"
        ]
    ):

        flags.append(
            "Estimated severity category increased."
        )

    if not flags:

        flags.append(
            "No major progression-quality flags detected."
        )

    return flags


# ============================================================
# SINGLE PLANT ANALYSIS
# ============================================================

def analyze_plant(
    observations,
):

    observations = sorted(
        observations,
        key=lambda item: item.observation_date,
    )

    if len(observations) < MIN_OBSERVATIONS:

        return {
            "plant_id": observations[0].plant_id,
            "status": "INSUFFICIENT_OBSERVATIONS",
            "observations": [
                asdict(item)
                for item in observations
            ],
        }

    consistency = (
        disease_consistency(
            observations
        )
    )

    confidence = (
        confidence_analysis(
            observations
        )
    )

    area = (
        affected_area_analysis(
            observations
        )
    )

    severity = (
        severity_analysis(
            observations
        )
    )

    progression_state = (
        classify_progression(
            area,
            severity,
            consistency,
        )
    )

    flags = generate_risk_flags(
        observations,
        area,
        severity,
        confidence,
        consistency,
    )

    total_days = (
        observations[-1].observation_date
        - observations[0].observation_date
    ).total_seconds() / 86400.0

    return {
        "plant_id": observations[0].plant_id,
        "crop": observations[0].crop,
        "status": "ANALYZED",
        "observation_count": len(
            observations
        ),
        "monitoring_period_days": float(
            total_days
        ),
        "progression_state": progression_state,
        "disease_consistency": consistency,
        "confidence_analysis": confidence,
        "affected_area_analysis": area,
        "severity_analysis": severity,
        "risk_flags": flags,
        "observations": [
            {
                **asdict(item),
                "observation_date": (
                    item.observation_date
                    .isoformat()
                ),
            }
            for item in observations
        ],
    }


# ============================================================
# FLATTEN RESULT FOR CSV
# ============================================================

def flatten_result(
    result,
):

    if result[
        "status"
    ] != "ANALYZED":

        return {
            "plant_id": result[
                "plant_id"
            ],
            "status": result[
                "status"
            ],
        }

    area = result[
        "affected_area_analysis"
    ]

    consistency = result[
        "disease_consistency"
    ]

    confidence = result[
        "confidence_analysis"
    ]

    severity = result[
        "severity_analysis"
    ]

    return {
        "plant_id": result[
            "plant_id"
        ],
        "crop": result[
            "crop"
        ],
        "progression_state": result[
            "progression_state"
        ],
        "observation_count": result[
            "observation_count"
        ],
        "monitoring_period_days": result[
            "monitoring_period_days"
        ],
        "dominant_disease": consistency[
            "dominant_disease"
        ],
        "disease_consistency_percent": consistency[
            "consistency_percent"
        ],
        "mean_confidence": confidence[
            "mean_confidence"
        ],
        "initial_affected_area_percent": area[
            "initial_affected_area_percent"
        ],
        "latest_affected_area_percent": area[
            "latest_affected_area_percent"
        ],
        "absolute_change_percentage_points": area[
            "absolute_change_percentage_points"
        ],
        "relative_change_percent": area[
            "relative_change_percent"
        ],
        "progression_velocity_percent_per_day": area[
            "progression_velocity_percent_per_day"
        ],
        "trend_slope_percent_per_day": area[
            "trend_slope_percent_per_day"
        ],
        "trend_r_squared": area[
            "trend_r_squared"
        ],
        "initial_severity": severity[
            "initial_severity"
        ],
        "latest_severity": severity[
            "latest_severity"
        ],
        "severity_level_change": severity[
            "severity_level_change"
        ],
        "risk_flag_count": len(
            result[
                "risk_flags"
            ]
        ),
    }


# ============================================================
# SAVE REPORT
# ============================================================

def save_reports(
    results,
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = (
        OUTPUT_DIR
        / "disease_progression_results.json"
    )

    csv_path = (
        OUTPUT_DIR
        / "disease_progression_summary.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "progression_summary.json"
    )

    report_path = (
        OUTPUT_DIR
        / "DISEASE_PROGRESSION_REPORT.txt"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            default=str,
        )

    rows = [
        flatten_result(result)
        for result in results
    ]

    pd.DataFrame(
        rows
    ).to_csv(
        csv_path,
        index=False,
    )

    analyzed = [
        result
        for result in results
        if result.get(
            "status"
        )
        == "ANALYZED"
    ]

    states = {}

    for result in analyzed:

        state = result[
            "progression_state"
        ]

        states[state] = (
            states.get(
                state,
                0,
            )
            + 1
        )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_plants": len(
            results
        ),
        "analyzed_plants": len(
            analyzed
        ),
        "insufficient_observation_plants": (
            len(results)
            - len(analyzed)
        ),
        "progression_state_distribution": states,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "AGRIVISION AI\n"
        )

        file.write(
            "DISEASE PROGRESSION ENGINE\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            f"Generated: {timestamp}\n"
        )

        file.write(
            f"Total plants: {len(results)}\n"
        )

        file.write(
            f"Analyzed plants: {len(analyzed)}\n\n"
        )

        file.write(
            "Progression Distribution\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        for state, count in sorted(
            states.items()
        ):

            file.write(
                f"{state}: {count}\n"
            )

        file.write(
            "\nIMPORTANT:\n"
        )

        file.write(
            "Progression status is an image-derived "
            "decision-support signal. It does not by "
            "itself establish biological disease "
            "progression or replace field inspection "
            "by an agricultural professional.\n"
        )

    print()
    print(
        "Reports saved:"
    )

    print(
        json_path
    )

    print(
        csv_path
    )

    print(
        summary_path
    )

    print(
        report_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "AgriVision AI"
    )

    print(
        "Disease Progression Engine"
    )

    print("=" * 70)

    print()

    print(
        f"Input directory:\n{INPUT_DIR}"
    )

    print()

    observations = (
        load_json_observations()
    )

    print(
        f"Observations loaded: "
        f"{len(observations)}"
    )

    if not observations:

        print()
        print(
            "NO PROGRESSION OBSERVATIONS FOUND."
        )

        print()
        print(
            "Create observation JSON files such as:"
        )

        print()

        print(
            "data\\progression\\plant_001\\day_01.json"
        )

        print(
            "data\\progression\\plant_001\\day_04.json"
        )

        print(
            "data\\progression\\plant_001\\day_07.json"
        )

        print()

        print(
            "Each JSON should contain:"
        )

        print(
            "plant_id"
        )

        print(
            "observation_date"
        )

        print(
            "disease"
        )

        print(
            "crop"
        )

        print(
            "confidence"
        )

        print(
            "affected_area_percent"
        )

        print(
            "severity"
        )

        return

    observations = (
        sort_observations(
            observations
        )
    )

    grouped = {}

    for observation in observations:

        grouped.setdefault(
            observation.plant_id,
            [],
        ).append(
            observation
        )

    results = []

    print()

    print(
        f"Plants detected: "
        f"{len(grouped)}"
    )

    print()

    for plant_id, items in grouped.items():

        print(
            f"Analyzing {plant_id}: "
            f"{len(items)} observations"
        )

        result = analyze_plant(
            items
        )

        results.append(
            result
        )

    save_reports(
        results
    )

    print()

    print("=" * 70)

    print(
        "DISEASE PROGRESSION ANALYSIS COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()

