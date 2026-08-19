"""Deriva desfechos binários observáveis nos horizontes definidos no estudo."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(os.environ.get("TCC_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "base_s_survival_fullfu.csv"
BASE_B_5Y_FALLBACK_PATH = PROJECT_ROOT / "data" / "processed" / "base_b_5y_all.csv"

BASE_B_1Y_ALL_PATH = PROJECT_ROOT / "data" / "processed" / "base_b_1y_all.csv"
BASE_B_1Y_OBSERVED_PATH = PROJECT_ROOT / "data" / "processed" / "base_b_1y_observed.csv"
BASE_B_2Y_ALL_PATH = PROJECT_ROOT / "data" / "processed" / "base_b_2y_all.csv"
BASE_B_2Y_OBSERVED_PATH = PROJECT_ROOT / "data" / "processed" / "base_b_2y_observed.csv"

FEASIBILITY_CSV_PATH = PROJECT_ROOT / "outputs" / "binary_horizon_feasibility.csv"
FEASIBILITY_REPORT_PATH = PROJECT_ROOT / "outputs" / "binary_horizon_feasibility_report.md"
QC_REPORT_PATH = PROJECT_ROOT / "outputs" / "binary_horizon_qc_report.md"

REQUIRED_COLUMNS = [
    "id_original",
    "followup_days_fullfu",
    "event_death_fullfu",
    "date_tx",
    "end_date_fullfu",
    "date_death",
    "age_gt40_main",
    "dgf_main",
    "sex_male",
    "race3_clean",
    "donor_deceased",
    "hiv_positive",
    "hla_high_mismatch_group",
    "cit_hours",
]

CORE_MINIMAL_COLUMNS = [
    "age_gt40_main",
    "dgf_main",
    "sex_male",
    "race3_clean",
    "donor_deceased",
]

HORIZON_CONFIG = [
    {
        "label": "1_year",
        "days": 365,
        "status_col": "y1_status",
        "observed_col": "y1_observed",
        "recommendation": "candidate_sensitivity_binary_horizon",
    },
    {
        "label": "2_years",
        "days": 730,
        "status_col": "y2_status",
        "observed_col": "y2_observed",
        "recommendation": "candidate_main_binary_exploratory_horizon",
    },
    {
        "label": "5_years",
        "days": 1825,
        "status_col": "y5_status",
        "observed_col": "y5_observed",
        "recommendation": "not_recommended_as_main_binary_model_due_to_low_observed_n",
    },
]


def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Colunas obrigatorias ausentes em {INPUT_PATH}: {missing_text}")


def validate_followup_days(df: pd.DataFrame) -> None:
    followup = pd.to_numeric(df["followup_days_fullfu"], errors="coerce")
    if followup.isna().any():
        missing_n = int(followup.isna().sum())
        raise ValueError(
            "A coluna `followup_days_fullfu` contem valores ausentes ou nao numericos: "
            f"{missing_n} linha(s)."
        )
    negative_n = int(followup.lt(0).sum())
    if negative_n:
        raise ValueError(
            "A coluna `followup_days_fullfu` contem valores negativos: "
            f"{negative_n} linha(s)."
        )


def validate_event_column(df: pd.DataFrame) -> None:
    events = pd.to_numeric(df["event_death_fullfu"], errors="coerce")
    if events.isna().any():
        invalid_n = int(events.isna().sum())
        raise ValueError(
            "A coluna `event_death_fullfu` contem valores ausentes ou nao numericos: "
            f"{invalid_n} linha(s)."
        )
    invalid_mask = ~events.isin([0, 1])
    if invalid_mask.any():
        invalid_n = int(invalid_mask.sum())
        raise ValueError(
            "A coluna `event_death_fullfu` deve conter apenas 0 ou 1. "
            f"Foram encontrados {invalid_n} valor(es) invalido(s)."
        )


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["followup_days_fullfu"] = pd.to_numeric(
        prepared["followup_days_fullfu"], errors="raise"
    )
    prepared["event_death_fullfu"] = pd.to_numeric(
        prepared["event_death_fullfu"], errors="raise"
    ).astype("Int64")
    return prepared


def derive_binary_horizon(df: pd.DataFrame, horizon_days: int, status_col: str, observed_col: str) -> pd.DataFrame:
    """Distingue evento, vida confirmada e censura antes do horizonte."""
    derived = df.copy()
    followup = derived["followup_days_fullfu"]
    event = derived["event_death_fullfu"]

    derived[status_col] = pd.Series(pd.NA, index=derived.index, dtype="Int64")

    death_by_horizon = event.eq(1) & followup.le(horizon_days)
    alive_observed_past_horizon = event.eq(0) & followup.ge(horizon_days)
    death_after_horizon = event.eq(1) & followup.gt(horizon_days)
    censored_before_horizon = event.eq(0) & followup.lt(horizon_days)

    # O desfecho binario e "morte ate o horizonte"; mortes posteriores contam como vivo no horizonte.
    derived.loc[death_by_horizon, status_col] = 1
    derived.loc[alive_observed_past_horizon | death_after_horizon, status_col] = 0

    # Censura antes do horizonte impede classificar o paciente como vivo nesse marco temporal.
    derived.loc[censored_before_horizon, status_col] = pd.NA
    derived[observed_col] = derived[status_col].notna().astype("Int64")

    return derived


def validate_status_column(df: pd.DataFrame, status_col: str, observed_col: str) -> None:
    observed = pd.to_numeric(df[observed_col], errors="coerce")
    if observed.isna().any() or (~observed.isin([0, 1])).any():
        raise ValueError(
            f"A coluna `{observed_col}` deve conter apenas 0 ou 1 sem missing."
        )

    observed_status = pd.to_numeric(df.loc[df[observed_col].eq(1), status_col], errors="coerce")
    if observed_status.isna().any():
        raise ValueError(
            f"A coluna `{status_col}` contem valores NA em linhas marcadas como observadas."
        )
    if (~observed_status.isin([0, 1])).any():
        raise ValueError(
            f"A coluna `{status_col}` deve conter apenas 0 ou 1 nas linhas observadas."
        )


def load_5y_dataset(base_df: pd.DataFrame) -> pd.DataFrame:
    if {"y5_status", "y5_observed"}.issubset(base_df.columns):
        return base_df.copy()

    if not BASE_B_5Y_FALLBACK_PATH.exists():
        raise FileNotFoundError(
            "As colunas `y5_status` e `y5_observed` nao existem na base principal "
            f"e o arquivo de fallback nao foi encontrado: {BASE_B_5Y_FALLBACK_PATH}"
        )

    fallback = load_csv(BASE_B_5Y_FALLBACK_PATH)
    missing = [column for column in ["y5_status", "y5_observed"] if column not in fallback.columns]
    if missing:
        raise ValueError(
            "O arquivo de fallback de 5 anos nao contem as colunas esperadas: "
            + ", ".join(missing)
        )
    return fallback


def build_observed_dataset(df: pd.DataFrame, observed_col: str) -> pd.DataFrame:
    """Seleciona outcomes observáveis antes da aplicação de complete case nos modelos."""
    return df.loc[df[observed_col].eq(1)].copy()


def validate_output_sizes(all_df: pd.DataFrame, observed_df: pd.DataFrame, original_n: int, observed_col: str) -> None:
    if len(all_df) != original_n:
        raise ValueError(
            f"A base `_all` perdeu linhas: esperado {original_n}, encontrado {len(all_df)}."
        )
    expected_observed_n = int(all_df[observed_col].sum())
    if len(observed_df) != expected_observed_n:
        raise ValueError(
            f"A base `_observed` diverge do numero de observados em `{observed_col}`: "
            f"esperado {expected_observed_n}, encontrado {len(observed_df)}."
        )


def create_feasibility_row(
    df: pd.DataFrame,
    horizon_label: str,
    horizon_days: int,
    status_col: str,
    observed_col: str,
    recommendation: str,
) -> dict[str, object]:
    n_total = len(df)
    n_observed = int(df[observed_col].sum())
    n_events = int(df[status_col].eq(1).sum())
    n_non_events = int(df[status_col].eq(0).sum())
    n_non_observed = n_total - n_observed
    event_rate_observed = round(n_events / n_observed, 6) if n_observed else pd.NA

    core_complete = df[df[status_col].notna()].copy()
    for column in CORE_MINIMAL_COLUMNS:
        core_complete = core_complete[core_complete[column].notna()]

    n_core = len(core_complete)
    n_core_events = int(core_complete[status_col].eq(1).sum())
    n_core_non_events = int(core_complete[status_col].eq(0).sum())
    event_rate_core = round(n_core_events / n_core, 6) if n_core else pd.NA

    return {
        "horizon_label": horizon_label,
        "horizon_days": horizon_days,
        "n_total": n_total,
        "n_observed": n_observed,
        "n_events": n_events,
        "n_non_events": n_non_events,
        "n_non_observed": n_non_observed,
        "event_rate_observed": event_rate_observed,
        "n_core_minimal_completecase": n_core,
        "n_events_core_minimal_completecase": n_core_events,
        "n_non_events_core_minimal_completecase": n_core_non_events,
        "event_rate_core_minimal": event_rate_core,
        "methodological_recommendation": recommendation,
    }


def create_feasibility_table(base_1y: pd.DataFrame, base_2y: pd.DataFrame, base_5y: pd.DataFrame) -> pd.DataFrame:
    """Resume a viabilidade amostral dos horizontes sem ajustar modelos."""
    datasets = {
        "1_year": base_1y,
        "2_years": base_2y,
        "5_years": base_5y,
    }

    rows = []
    for config in HORIZON_CONFIG:
        rows.append(
            create_feasibility_row(
                df=datasets[config["label"]],
                horizon_label=config["label"],
                horizon_days=config["days"],
                status_col=config["status_col"],
                observed_col=config["observed_col"],
                recommendation=config["recommendation"],
            )
        )
    return pd.DataFrame(rows)


def validate_feasibility_statuses(feasibility_df: pd.DataFrame) -> None:
    for status_col in ["y1_status", "y2_status", "y5_status"]:
        if status_col not in feasibility_df.attrs:
            continue


def markdown_table(df: pd.DataFrame) -> str:
    display_df = df.copy()
    for column in ["event_rate_observed", "event_rate_core_minimal"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: "NA" if pd.isna(value) else f"{float(value):.3f}"
            )
    return display_df.to_markdown(index=False)


def count_total_inconsistencies(base_1y: pd.DataFrame, base_2y: pd.DataFrame, base_5y: pd.DataFrame) -> int:
    inconsistencies = 0
    inconsistencies += int(
        (base_1y["event_death_fullfu"].eq(0) & base_1y["followup_days_fullfu"].lt(365) & base_1y["y1_status"].notna()).sum()
    )
    inconsistencies += int(
        (base_2y["event_death_fullfu"].eq(0) & base_2y["followup_days_fullfu"].lt(730) & base_2y["y2_status"].notna()).sum()
    )
    inconsistencies += int(
        (base_1y["event_death_fullfu"].eq(1) & base_1y["followup_days_fullfu"].gt(365) & base_1y["y1_status"].ne(0)).sum()
    )
    inconsistencies += int(
        (base_2y["event_death_fullfu"].eq(1) & base_2y["followup_days_fullfu"].gt(730) & base_2y["y2_status"].ne(0)).sum()
    )
    if "y5_status" in base_5y.columns and "y5_observed" in base_5y.columns:
        observed_status = pd.to_numeric(
            base_5y.loc[base_5y["y5_observed"].eq(1), "y5_status"], errors="coerce"
        )
        inconsistencies += int(observed_status.isna().sum())
        inconsistencies += int((~observed_status.isin([0, 1])).sum())
    return inconsistencies


def build_feasibility_report(feasibility_df: pd.DataFrame) -> str:
    recommendations = "\n".join(
        f"- `{row.horizon_label}`: `{row.methodological_recommendation}`"
        for row in feasibility_df.itertuples(index=False)
    )
    table_md = markdown_table(feasibility_df)

    return f"""# Viabilidade de horizontes binarios fixos

## Objetivo

- Construir e resumir a viabilidade amostral de desfechos binarios de mortalidade em 1, 2 e 5 anos a partir de `base_s_survival_fullfu.csv`.
- Manter a analise principal do projeto em sobrevivencia/Cox, deixando esta etapa restrita a bases, QC e avaliacao de viabilidade.

## Regra de construcao

- `status = 1` quando a morte ocorreu ate o horizonte.
- `status = 0` quando o paciente foi observado vivo ate o horizonte.
- `status = 0` quando a morte ocorreu apenas apos o horizonte, pois o desfecho e mortalidade ate o horizonte fixo.
- `status = NA` quando houve censura antes do horizonte.

## Alerta metodologico

- Pacientes censurados antes do horizonte nao sao classificados como vivos.
- Nenhum paciente com observacao insuficiente antes do marco temporal e tratado como nao evento.

## Tabela resumida

{table_md}

## Recomendacao metodologica automatica

{recommendations}

## Observacao final

- Nenhum modelo preditivo foi rodado nesta etapa.
"""


def build_qc_report(
    input_df: pd.DataFrame,
    base_1y: pd.DataFrame,
    base_2y: pd.DataFrame,
    base_5y: pd.DataFrame,
    used_5y_fallback: bool,
    inconsistency_count: int,
) -> str:
    deaths_total = int(input_df["event_death_fullfu"].sum())
    deaths_1y = int(base_1y["y1_status"].eq(1).sum())
    deaths_2y = int(base_2y["y2_status"].eq(1).sum())
    deaths_5y = int(base_5y["y5_status"].eq(1).sum())

    dead_after_1y_as_non_event = int(
        (base_1y["event_death_fullfu"].eq(1) & base_1y["followup_days_fullfu"].gt(365) & base_1y["y1_status"].eq(0)).sum()
    )
    dead_after_2y_as_non_event = int(
        (base_2y["event_death_fullfu"].eq(1) & base_2y["followup_days_fullfu"].gt(730) & base_2y["y2_status"].eq(0)).sum()
    )
    censored_before_1y_as_na = int(
        (base_1y["event_death_fullfu"].eq(0) & base_1y["followup_days_fullfu"].lt(365) & base_1y["y1_status"].isna()).sum()
    )
    censored_before_2y_as_na = int(
        (base_2y["event_death_fullfu"].eq(0) & base_2y["followup_days_fullfu"].lt(730) & base_2y["y2_status"].isna()).sum()
    )

    required_columns_text = "\n".join(f"- `{column}`: OK" for column in REQUIRED_COLUMNS)
    five_year_source = (
        f"`{BASE_B_5Y_FALLBACK_PATH}`"
        if used_5y_fallback
        else "`data/processed/base_s_survival_fullfu.csv`"
    )

    return f"""# QC das bases binarias 1y e 2y

## Entrada

- Arquivo de entrada principal: `{INPUT_PATH}`
- Numero de linhas lidas: {len(input_df)}
- Fonte usada para 5 anos: {five_year_source}

## Colunas obrigatorias

{required_columns_text}

## Checagens e contagens

- Contagem de inconsistencias: {inconsistency_count}
- Numero de mortes totais em `event_death_fullfu`: {deaths_total}
- Numero de mortes ate 1 ano: {deaths_1y}
- Numero de mortes ate 2 anos: {deaths_2y}
- Numero de mortes ate 5 anos: {deaths_5y}
- Numero de pacientes mortos apos 1 ano classificados como nao evento em 1 ano: {dead_after_1y_as_non_event}
- Numero de pacientes mortos apos 2 anos classificados como nao evento em 2 anos: {dead_after_2y_as_non_event}
- Numero de pacientes censurados antes de 1 ano classificados como NA em 1 ano: {censored_before_1y_as_na}
- Numero de pacientes censurados antes de 2 anos classificados como NA em 2 anos: {censored_before_2y_as_na}
"""


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def print_summary(generated_paths: list[Path], feasibility_df: pd.DataFrame) -> None:
    print("Arquivos gerados:")
    for path in generated_paths:
        print(path)
    print()
    print("Tabela resumida de viabilidade:")
    print(feasibility_df.to_string(index=False))
    print()
    print("Bases binarias de 1 e 2 anos geradas. Nenhum modelo preditivo foi executado.")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {INPUT_PATH}")

    input_df = load_csv(INPUT_PATH)
    validate_required_columns(input_df, REQUIRED_COLUMNS)
    validate_followup_days(input_df)
    validate_event_column(input_df)

    prepared_df = prepare_input(input_df)
    base_1y_all = derive_binary_horizon(prepared_df, 365, "y1_status", "y1_observed")
    base_2y_all = derive_binary_horizon(prepared_df, 730, "y2_status", "y2_observed")

    validate_status_column(base_1y_all, "y1_status", "y1_observed")
    validate_status_column(base_2y_all, "y2_status", "y2_observed")

    base_1y_observed = build_observed_dataset(base_1y_all, "y1_observed")
    base_2y_observed = build_observed_dataset(base_2y_all, "y2_observed")

    validate_output_sizes(base_1y_all, base_1y_observed, len(prepared_df), "y1_observed")
    validate_output_sizes(base_2y_all, base_2y_observed, len(prepared_df), "y2_observed")

    used_5y_fallback = not {"y5_status", "y5_observed"}.issubset(prepared_df.columns)
    base_5y_all = load_5y_dataset(prepared_df)
    validate_status_column(base_5y_all, "y5_status", "y5_observed")
    if len(base_5y_all) != len(prepared_df):
        raise ValueError(
            "A base de 5 anos usada na viabilidade nao tem o mesmo numero de linhas da base principal."
        )

    feasibility_df = create_feasibility_table(base_1y_all, base_2y_all, base_5y_all)
    feasibility_report = build_feasibility_report(feasibility_df)
    inconsistency_count = count_total_inconsistencies(base_1y_all, base_2y_all, base_5y_all)
    qc_report = build_qc_report(
        input_df=prepared_df,
        base_1y=base_1y_all,
        base_2y=base_2y_all,
        base_5y=base_5y_all,
        used_5y_fallback=used_5y_fallback,
        inconsistency_count=inconsistency_count,
    )

    save_dataframe(base_1y_all, BASE_B_1Y_ALL_PATH)
    save_dataframe(base_1y_observed, BASE_B_1Y_OBSERVED_PATH)
    save_dataframe(base_2y_all, BASE_B_2Y_ALL_PATH)
    save_dataframe(base_2y_observed, BASE_B_2Y_OBSERVED_PATH)
    save_dataframe(feasibility_df, FEASIBILITY_CSV_PATH)
    save_text(feasibility_report, FEASIBILITY_REPORT_PATH)
    save_text(qc_report, QC_REPORT_PATH)

    generated_paths = [
        BASE_B_1Y_ALL_PATH,
        BASE_B_1Y_OBSERVED_PATH,
        BASE_B_2Y_ALL_PATH,
        BASE_B_2Y_OBSERVED_PATH,
        FEASIBILITY_CSV_PATH,
        FEASIBILITY_REPORT_PATH,
        QC_REPORT_PATH,
    ]
    print_summary(generated_paths, feasibility_df)


if __name__ == "__main__":
    main()
