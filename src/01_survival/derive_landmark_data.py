#!/usr/bin/env python3
"""Deriva as populações de sobrevivência e binária para o landmark no dia 7."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LANDMARK_DAY = 7
HORIZON_2Y_FROM_TRANSPLANT = 730


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    time = pd.to_numeric(df["followup_days_fullfu"], errors="raise")
    event = pd.to_numeric(df["event_death_fullfu"], errors="raise").astype(int)
    # Restringe a análise a participantes vivos e ainda sob risco após o dia 7.
    eligible = time.gt(LANDMARK_DAY)
    survival = df.loc[eligible].copy()
    # O relógio da análise de Cox é reiniciado no marco temporal.
    survival["landmark_followup_days"] = time.loc[eligible] - LANDMARK_DAY
    survival["event_after_landmark"] = event.loc[eligible]
    if len(survival) != 186 or int(survival["event_after_landmark"].sum()) != 20:
        raise ValueError("Landmark eligible population differs from 186/20")
    survival.to_csv(args.output_dir / "base_s_landmark_day7_fullfu.csv", index=False)

    # O horizonte binário permanece contado desde a data do transplante.
    status = np.where(
        event.loc[eligible].eq(1) & time.loc[eligible].le(HORIZON_2Y_FROM_TRANSPLANT),
        1,
        np.where(
            time.loc[eligible].ge(HORIZON_2Y_FROM_TRANSPLANT)
            | (event.loc[eligible].eq(1) & time.loc[eligible].gt(HORIZON_2Y_FROM_TRANSPLANT)),
            0,
            np.nan,
        ),
    )
    binary = survival.copy()
    binary["y2_landmark_status"] = status
    binary["y2_landmark_observed"] = pd.notna(status).astype(int)
    observed = binary.loc[binary["y2_landmark_observed"].eq(1)].copy()
    if len(observed) != 115 or int(observed["y2_landmark_status"].sum()) != 16:
        raise ValueError("Landmark two-year observable population differs from 115/16")
    binary.to_csv(args.output_dir / "base_b_2y_landmark_day7_all.csv", index=False)
    observed.to_csv(args.output_dir / "base_b_2y_landmark_day7_observed.csv", index=False)


if __name__ == "__main__":
    main()
