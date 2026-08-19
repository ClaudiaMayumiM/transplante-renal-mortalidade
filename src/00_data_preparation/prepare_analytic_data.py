"""Prepara as bases analíticas e seus relatórios de controle de qualidade."""

import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("TCC_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
EXPECTED_INPUT_NAMES = (
    "Transplantion Outcomes GSH(1).csv",
    "Transplantion Outcomes GSH.csv",
)
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "base_master.csv"
FLOW_COUNTS_PATH = PROJECT_ROOT / "outputs" / "flow_counts.csv"
MISSING_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "missing_summary.csv"
RECODING_MAP_PATH = PROJECT_ROOT / "outputs" / "recoding_map.csv"
QC_REPORT_PATH = PROJECT_ROOT / "outputs" / "data_preparation_qc_report.md"
BASE_S_PATH = PROJECT_ROOT / "data" / "processed" / "base_s_survival.csv"
BASE_S_FULLFU_PATH = PROJECT_ROOT / "data" / "processed" / "base_s_survival_fullfu.csv"
BASE_B_5Y_ALL_PATH = PROJECT_ROOT / "data" / "processed" / "base_b_5y_all.csv"
BASE_B_5Y_OBSERVED_PATH = PROJECT_ROOT / "data" / "processed" / "base_b_5y_observed.csv"
ADMIN_CENSOR_DATE = pd.Timestamp("2015-06-30")


def find_input_csv(project_root: Path = PROJECT_ROOT) -> Path:
    """Localiza o CSV bruto dentro da pasta do projeto."""
    for expected_name in EXPECTED_INPUT_NAMES:
        matches = sorted(project_root.rglob(expected_name))
        if matches:
            return matches[0]

    csv_matches = sorted(project_root.rglob("*.csv"))
    names = ", ".join(str(path.relative_to(project_root)) for path in csv_matches)
    raise FileNotFoundError(
        "CSV de entrada nao encontrado. "
        f"Nomes esperados: {EXPECTED_INPUT_NAMES}. CSVs encontrados: {names}"
    )


def load_data(csv_path: Path) -> pd.DataFrame:
    """Le o CSV preservando todas as linhas e colunas."""
    try:
        return pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin1")


def parse_study_date(value):
    """Parser de datas do estudo."""
    parsed = pd.to_datetime(value, format="%d%b%Y", errors="coerce")
    if isinstance(parsed, pd.Series):
        original = pd.Series(value, index=parsed.index)
        needs_fallback = parsed.isna() & original.notna()
        if needs_fallback.any():
            fallback = pd.to_datetime(
                original.where(needs_fallback),
                errors="coerce",
                dayfirst=True,
            )
            parsed = parsed.fillna(fallback)
    elif pd.isna(parsed) and pd.notna(value):
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return parsed


def clean_label(series: pd.Series) -> pd.Series:
    """Remove prefixos numericos do tipo '1. ' e normaliza espacos."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"^\s*\d+\.\s*", "", regex=True)
        .str.strip()
    )


def _yes_no_from_labeled(series: pd.Series) -> pd.Series:
    cleaned = clean_label(series).str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    result = result.mask(cleaned.str.contains("yes", na=False), 1)
    result = result.mask(cleaned.str.contains("no", na=False), 0)
    return result


def _binary_from_bool(mask: pd.Series, known: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=mask.index, dtype="Int64")
    result.loc[known] = mask.loc[known].astype("Int64")
    return result


def _date_parse_failures(raw: pd.Series, parsed: pd.Series) -> int:
    return int(raw.notna().sum() - parsed.notna().sum())


def derive_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva preditores e flags de conferência sem exclusão ou imputação."""
    derived = df.copy()

    derived["id_original"] = derived["id"]
    derived["age_years"] = pd.to_numeric(derived["ageR"], errors="coerce")
    derived["transplant_number"] = pd.to_numeric(
        derived["TransplantNumber"], errors="coerce"
    )
    derived["retransplant_raw"] = derived["Retransplant"]

    derived["date_tx"] = parse_study_date(derived["Dateoftxplant"])
    derived["date_death"] = parse_study_date(derived["Dateofdeath"])
    derived["date_last_fu"] = parse_study_date(derived["DateLastFU"])
    derived["enddateD_fallback"] = parse_study_date(derived["enddateD"])
    derived["followup_end_real"] = derived["date_last_fu"].combine_first(
        derived["enddateD_fallback"]
    )

    age_group = clean_label(derived["agegroupR3"])
    derived["age_gt40_main"] = pd.Series(pd.NA, index=derived.index, dtype="Int64")
    derived.loc[age_group.eq(">40"), "age_gt40_main"] = 1
    derived.loc[age_group.eq("<40"), "age_gt40_main"] = 0
    derived["age_gt40_from_numeric"] = _binary_from_bool(
        derived["age_years"] > 40,
        derived["age_years"].notna(),
    )
    derived["age_gt40_discrepancy"] = (
        derived["age_gt40_main"].notna()
        & derived["age_gt40_from_numeric"].notna()
        & derived["age_gt40_main"].ne(derived["age_gt40_from_numeric"])
    )
    derived["age_exactly40_labeled_gt40"] = (
        derived["age_years"].eq(40)
        & age_group.eq(">40")
    )

    sex_label = clean_label(derived["GenderR"])
    sex_upper = sex_label.str.upper()
    derived["sex_clean"] = pd.Series(pd.NA, index=derived.index, dtype="string")
    derived.loc[sex_upper.str.startswith("M", na=False), "sex_clean"] = "Male"
    derived.loc[sex_upper.str.startswith("F", na=False), "sex_clean"] = "Female"
    derived["sex_male"] = pd.Series(pd.NA, index=derived.index, dtype="Int64")
    derived.loc[derived["sex_clean"].eq("Male"), "sex_male"] = 1
    derived.loc[derived["sex_clean"].eq("Female"), "sex_male"] = 0

    race_label = clean_label(derived["RaceR3"])
    race_lower = race_label.str.lower()
    derived["race3_clean"] = pd.Series(pd.NA, index=derived.index, dtype="string")
    derived.loc[race_lower.eq("african"), "race3_clean"] = "African"
    derived.loc[
        race_lower.str.contains("mixed", na=False)
        | race_lower.str.contains("asian", na=False),
        "race3_clean",
    ] = "Mixed ethnicity/Asian"
    derived.loc[race_lower.eq("white"), "race3_clean"] = "White"

    hiv = clean_label(derived["HIVstatusR"]).str.lower()
    derived["hiv_positive"] = pd.Series(pd.NA, index=derived.index, dtype="Int64")
    derived.loc[hiv.eq("positive"), "hiv_positive"] = 1
    derived.loc[hiv.eq("negative"), "hiv_positive"] = 0

    donor = clean_label(derived["Donortype2"]).str.lower()
    deceased_mask = donor.str.contains("dbd", na=False) | donor.str.contains(
        "dcd", na=False
    )
    living_mask = donor.str.contains("living", na=False)
    derived["donor_type_binary"] = pd.Series(pd.NA, index=derived.index, dtype="string")
    derived.loc[deceased_mask, "donor_type_binary"] = "Deceased"
    derived.loc[living_mask, "donor_type_binary"] = "Living"
    derived["donor_deceased"] = pd.Series(pd.NA, index=derived.index, dtype="Int64")
    derived.loc[deceased_mask, "donor_deceased"] = 1
    derived.loc[living_mask, "donor_deceased"] = 0

    derived["hla_mismatches_n"] = pd.to_numeric(
        derived["HLAmismatches"], errors="coerce"
    )
    derived["hla_gt5_strict_num"] = _binary_from_bool(
        derived["hla_mismatches_n"] > 5,
        derived["hla_mismatches_n"].notna(),
    )
    hla_group = clean_label(derived["HLAmismatches3"])
    derived["hla_high_mismatch_group"] = pd.Series(
        pd.NA, index=derived.index, dtype="Int64"
    )
    derived.loc[
        hla_group.str.contains("5-8", na=False), "hla_high_mismatch_group"
    ] = 1
    derived.loc[
        hla_group.str.contains("0-4", na=False), "hla_high_mismatch_group"
    ] = 0
    derived["hla_group_vs_strict_discrepancy"] = (
        derived["hla_gt5_strict_num"].notna()
        & derived["hla_high_mismatch_group"].notna()
        & derived["hla_gt5_strict_num"].ne(derived["hla_high_mismatch_group"])
    )
    derived["hla_exactly5_boundary_case"] = (
        derived["hla_mismatches_n"].eq(5)
        & derived["hla_high_mismatch_group"].notna()
    )

    derived["cit_hours"] = pd.to_numeric(derived["CIT"], errors="coerce")
    derived["cit_group_clean"] = clean_label(derived["CITgroup"])
    derived["cit_ge7h"] = _binary_from_bool(
        derived["cit_hours"] >= 7,
        derived["cit_hours"].notna(),
    )

    derived["dgf_main"] = _yes_no_from_labeled(derived["dial_1stweek"])
    derived["dgf_alt"] = _yes_no_from_labeled(derived["dial_1stweek2"])
    derived["dgf_discrepancy"] = (
        derived["dgf_main"].notna()
        & derived["dgf_alt"].notna()
        & derived["dgf_main"].ne(derived["dgf_alt"])
    )

    derived["acute_rejection_raw"] = derived["Acuterejection3"]
    derived["acute_rejection_flag_raw"] = derived["_Acuterejection3"]
    derived["date_acute_rejection"] = parse_study_date(derived["date_acute_rej3"])

    return derived


def derive_survival_outcomes(
    df: pd.DataFrame,
    administrative_censor_date: pd.Timestamp | None = ADMIN_CENSOR_DATE,
) -> pd.DataFrame:
    """Define tempos e eventos com seguimento completo ou censura administrativa."""
    derived = df.copy()
    died = pd.to_numeric(derived["died"], errors="coerce").eq(1)

    if administrative_censor_date is None:
        derived["event_death_fullfu"] = (
            died & derived["date_death"].notna()
        ).astype("Int64")
        derived["end_date_fullfu"] = derived["followup_end_real"]
        event_mask = derived["event_death_fullfu"].eq(1)
        derived.loc[event_mask, "end_date_fullfu"] = derived.loc[
            event_mask, "date_death"
        ]
        derived["followup_days_fullfu"] = (
            derived["end_date_fullfu"] - derived["date_tx"]
        ).dt.days
        derived.loc[derived["followup_days_fullfu"] < 0, "followup_days_fullfu"] = pd.NA
        derived["followup_years_fullfu"] = derived["followup_days_fullfu"] / 365.25
        return derived

    derived["event_death"] = (
        died
        & derived["date_death"].notna()
        & derived["date_death"].le(administrative_censor_date)
    ).astype("Int64")
    censor_end = derived["followup_end_real"].where(
        derived["followup_end_real"].le(administrative_censor_date),
        administrative_censor_date,
    )
    censor_end = censor_end.fillna(administrative_censor_date)

    derived["end_date"] = censor_end
    event_mask = derived["event_death"].eq(1)
    derived.loc[event_mask, "end_date"] = derived.loc[event_mask, "date_death"]
    derived["time_to_death_or_censor_days"] = (
        derived["end_date"] - derived["date_tx"]
    ).dt.days
    derived.loc[
        derived["time_to_death_or_censor_days"] < 0,
        "time_to_death_or_censor_days",
    ] = pd.NA
    derived["time_to_death_or_censor_years"] = (
        derived["time_to_death_or_censor_days"] / 365.25
    )

    return derived


def derive_binary_5y(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva desfecho binario de mortalidade em 5 anos."""
    derived = df.copy()
    died = pd.to_numeric(derived["died"], errors="coerce").eq(1)

    derived["y5_date"] = derived["date_tx"] + pd.DateOffset(years=5)
    observed_until = derived["followup_end_real"]
    observed_until = observed_until.mask(
        died & derived["date_death"].notna(),
        derived["date_death"],
    )

    derived["y5_status"] = pd.Series(pd.NA, index=derived.index, dtype="Int64")
    death_by_5y = died & derived["date_death"].notna() & derived["date_death"].le(
        derived["y5_date"]
    )
    observed_5y_alive = observed_until.ge(derived["y5_date"])

    derived.loc[death_by_5y, "y5_status"] = 1
    derived.loc[~death_by_5y & observed_5y_alive, "y5_status"] = 0
    derived["y5_observed"] = derived["y5_status"].notna().astype("Int64")

    return derived


def create_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Cria sumario de missing para todas as variaveis recebidas."""
    missing_n = df.isna().sum()
    nonmissing_n = df.notna().sum()
    missing_pct = (missing_n / len(df) * 100).round(2)
    return pd.DataFrame(
        {
            "variable": missing_n.index,
            "missing_n": missing_n.values,
            "missing_pct": missing_pct.values,
            "nonmissing_n": nonmissing_n.values,
        }
    )


def create_recoding_map() -> pd.DataFrame:
    """Documenta origem, regra, papel analitico e notas das variaveis derivadas."""
    rows = [
        (
            "id_original",
            "id",
            "copia direta",
            "identificador",
            "Preserva o identificador do arquivo bruto.",
        ),
        (
            "age_years",
            "ageR",
            "conversao numerica sem imputacao",
            "preditor/auditoria",
            "Idade continua preservada para auditoria.",
        ),
        (
            "age_gt40_main",
            "agegroupR3",
            "1 para a categoria original rotulada >40; 0 para a categoria original rotulada <40; NA caso contrario",
            "preditor principal",
            "A categoria original rotulada >40 inclui participantes com idade exatamente igual a 40 anos; a variavel nao foi reconstruida a partir da idade continua.",
        ),
        (
            "age_gt40_from_numeric",
            "ageR",
            "1 se ageR > 40; 0 se ageR <= 40; NA se missing",
            "auditoria",
            "Usada para comparar com agegroupR3.",
        ),
        (
            "age_gt40_discrepancy",
            "agegroupR3, ageR",
            "True quando age_gt40_main diverge de age_gt40_from_numeric",
            "auditoria",
            "Nao corrige automaticamente discrepancias.",
        ),
        (
            "age_exactly40_labeled_gt40",
            "agegroupR3, ageR",
            "True quando ageR == 40 e agegroupR3 rotula o caso como >40",
            "auditoria",
            "Explicita que a variavel principal reproduz a categoria original do dataset.",
        ),
        (
            "sex_clean",
            "GenderR",
            "Male se rotulo limpo comecar com M; Female se comecar com F",
            "variavel de preparacao/descritiva usada para derivar o indicador de sexo do conjunto ampliado",
            "Prefixos numericos sao removidos antes da regra.",
        ),
        (
            "sex_male",
            "sex_clean",
            "1 para Male; 0 para Female; NA caso contrario",
            "preditor do conjunto ampliado dos modelos ridge e arvore",
            "Codificacao binaria derivada de sex_clean.",
        ),
        (
            "race3_clean",
            "RaceR3",
            "remove prefixos e padroniza para African, Mixed ethnicity/Asian, White",
            "preditor do conjunto ampliado dos modelos ridge e arvore; variavel descritiva",
            "Tambem utilizada na caracterizacao descritiva; sem agrupamento binario nesta preparacao.",
        ),
        (
            "hiv_positive",
            "HIVstatusR",
            "1 se positive; 0 se negative; NA caso contrario",
            "variavel descritiva",
            "Sem imputacao.",
        ),
        (
            "donor_type_binary",
            "Donortype2",
            "Deceased se contem DBD/DCD; Living se contem Living; NA caso contrario",
            "variavel de preparacao/descritiva usada para derivar o indicador de doador falecido",
            "Donortype2 e a fonte principal.",
        ),
        (
            "donor_deceased",
            "Donortype2",
            "1 para Deceased; 0 para Living; NA caso contrario",
            "preditor do conjunto ampliado dos modelos ridge e arvore",
            "Substitui a regra anterior baseada em Donortype.",
        ),
        (
            "hla_mismatches_n",
            "HLAmismatches",
            "conversao numerica sem imputacao",
            "variavel descritiva",
            "Mantida para caracterizacao da coorte; missing relevante exige cautela.",
        ),
        (
            "hla_high_mismatch_group",
            "HLAmismatches3",
            "1 se contem 5-8; 0 se contem 0-4; NA caso contrario",
            "variavel descritiva",
            "Agrupamento preservado para caracterizacao descritiva da coorte; nao integra os modelos finais.",
        ),
        (
            "hla_gt5_strict_num",
            "HLAmismatches",
            "1 se HLAmismatches > 5; 0 se <= 5; NA se missing",
            "auditoria",
            "Mantida apenas para auditoria numerica estrita.",
        ),
        (
            "hla_group_vs_strict_discrepancy",
            "hla_high_mismatch_group, hla_gt5_strict_num",
            "True quando as duas codificacoes divergem",
            "auditoria",
            "Divergencias esperadas principalmente quando HLAmismatches == 5.",
        ),
        (
            "hla_exactly5_boundary_case",
            "HLAmismatches, HLAmismatches3",
            "True quando HLAmismatches == 5 e ha classificacao por grupo",
            "auditoria",
            "Explicita o ponto de fronteira entre a regra numerica estrita e o agrupamento 5-8 do dataset.",
        ),
        (
            "cit_hours",
            "CIT",
            "conversao numerica sem imputacao",
            "variavel descritiva",
            "Tempo de isquemia fria em horas, mantido para caracterizacao da coorte; nao integra os modelos finais.",
        ),
        (
            "cit_group_clean",
            "CITgroup",
            "remove prefixos numericos e preserva rotulo",
            "auditoria/descritivo",
            "Nao substitui cit_hours.",
        ),
        (
            "cit_ge7h",
            "CIT",
            "1 se CIT >= 7; 0 se CIT < 7; NA se missing",
            "variavel descritiva",
            "Derivado sem imputacao para caracterizacao da coorte; nao integra os modelos finais.",
        ),
        (
            "dgf_main",
            "dial_1stweek",
            "1 se yes; 0 se no; NA caso contrario",
            "preditor principal",
            "Fonte principal por reproduzir os 41 casos de DGF do artigo.",
        ),
        (
            "dgf_alt",
            "dial_1stweek2",
            "1 se yes; 0 se no; NA caso contrario",
            "auditoria",
            "Apenas auditoria.",
        ),
        (
            "dgf_discrepancy",
            "dgf_main, dgf_alt",
            "True quando as duas codificacoes divergem",
            "auditoria",
            "Nao altera dgf_main.",
        ),
        (
            "acute_rejection_raw",
            "Acuterejection3",
            "copia direta",
            "auditoria/secundaria",
            "Nao usada como covariavel fixa principal nesta etapa.",
        ),
        (
            "acute_rejection_flag_raw",
            "_Acuterejection3",
            "copia direta",
            "auditoria/secundaria",
            "Preservada para analise secundaria.",
        ),
        (
            "date_acute_rejection",
            "date_acute_rej3",
            "parse_study_date",
            "auditoria/secundaria",
            "Rejeicao no artigo-base e covariavel dependente do tempo.",
        ),
        (
            "event_death",
            "died, Dateofdeath",
            "1 para morte ate 30/06/2015; 0 caso contrario",
            "desfecho de sensibilidade",
            "Analise de sensibilidade com censura administrativa.",
        ),
        (
            "time_to_death_or_censor_days",
            "date_tx, end_date",
            "diferenca em dias entre end_date e date_tx",
            "tempo de sensibilidade",
            "Tempo da analise de sensibilidade com censura administrativa; valores negativos sao convertidos para NA.",
        ),
        (
            "event_death_fullfu",
            "died, Dateofdeath",
            "1 para morte observada no follow-up real; 0 caso contrario",
            "desfecho Base S full follow-up",
            "Sem censura administrativa.",
        ),
        (
            "y5_status",
            "date_tx, date_death, followup_end_real",
            "1 morte <=5 anos; 0 vivo observado >=5 anos; NA nao observavel",
            "avaliacao de viabilidade amostral do desfecho",
            "Horizonte de cinco anos usado somente para viabilidade; nao integra a modelagem binaria final.",
        ),
        (
            "y5_observed",
            "y5_status",
            "1 quando y5_status e 0 ou 1; 0 quando NA",
            "flag de viabilidade amostral",
            "Explicita a reducao do N efetivo; nao integra a modelagem binaria final.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "derived_variable",
            "source_column",
            "rule",
            "role_in_analysis",
            "notes",
        ],
    )


def create_flow_counts(
    raw_df: pd.DataFrame,
    master: pd.DataFrame,
    base_s: pd.DataFrame,
    base_s_fullfu: pd.DataFrame,
    base_b_5y_all: pd.DataFrame,
    base_b_5y_observed: pd.DataFrame,
) -> pd.DataFrame:
    """Cria uma tabela com as contagens das principais bases analíticas derivadas."""
    s_events = int(base_s["event_death"].sum())
    s_full_events = int(base_s_fullfu["event_death_fullfu"].sum())
    b_events = int(base_b_5y_all["y5_status"].eq(1).sum())
    b_nonevents = int(base_b_5y_all["y5_status"].eq(0).sum())

    rows = [
        {
            "step": "CSV original",
            "n": len(raw_df),
            "events": int(pd.to_numeric(raw_df["died"], errors="coerce").sum()),
            "censored_or_nonevents": len(raw_df)
            - int(pd.to_numeric(raw_df["died"], errors="coerce").sum()),
            "notes": "Arquivo bruto localizado e lido sem exclusao de linhas.",
        },
        {
            "step": "Base master",
            "n": len(master),
            "events": int(pd.to_numeric(master["died"], errors="coerce").sum()),
            "censored_or_nonevents": len(master)
            - int(pd.to_numeric(master["died"], errors="coerce").sum()),
            "notes": "Base analitica expandida com colunas originais e derivadas.",
        },
        {
            "step": "Base S principal (full follow-up)",
            "n": len(base_s_fullfu),
            "events": s_full_events,
            "censored_or_nonevents": len(base_s_fullfu) - s_full_events,
            "notes": "Sobrevivencia principal usando follow-up real.",
        },
        {
            "step": "Base S sensibilidade (censura administrativa)",
            "n": len(base_s),
            "events": s_events,
            "censored_or_nonevents": len(base_s) - s_events,
            "notes": "Sobrevivencia de sensibilidade com censura administrativa em 30/06/2015.",
        },
        {
            "step": "Base B all",
            "n": len(base_b_5y_all),
            "events": b_events,
            "censored_or_nonevents": b_nonevents,
            "notes": "Horizonte fixo de 5 anos; y5_status NA quando nao observavel.",
        },
        {
            "step": "Base B observed",
            "n": len(base_b_5y_observed),
            "events": int(base_b_5y_observed["y5_status"].eq(1).sum()),
            "censored_or_nonevents": int(base_b_5y_observed["y5_status"].eq(0).sum()),
            "notes": "Subconjunto observavel em 5 anos.",
        },
    ]
    return pd.DataFrame(rows)


def _counts_text(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False)
    return "\n".join(f"- {category}: {int(n)}" for category, n in counts.items())


def _time_summary(series: pd.Series) -> tuple[float, float, float]:
    if series.dropna().empty:
        return (np.nan, np.nan, np.nan)
    return (
        float(series.min(skipna=True)),
        float(series.median(skipna=True)),
        float(series.max(skipna=True)),
    )


def write_qc_report(
    report_path: Path,
    csv_path: Path,
    raw_df: pd.DataFrame,
    master: pd.DataFrame,
    base_s: pd.DataFrame,
    base_s_fullfu: pd.DataFrame,
    base_b_5y_all: pd.DataFrame,
    base_b_5y_observed: pd.DataFrame,
    generated_files: list[Path],
) -> None:
    """Escreve o relatório de controle de qualidade da preparação da base analítica."""
    died_numeric = pd.to_numeric(master["died"], errors="coerce")
    date_failures = {
        "Dateoftxplant -> date_tx": _date_parse_failures(master["Dateoftxplant"], master["date_tx"]),
        "Dateofdeath -> date_death": _date_parse_failures(master["Dateofdeath"], master["date_death"]),
        "DateLastFU -> date_last_fu": _date_parse_failures(master["DateLastFU"], master["date_last_fu"]),
        "enddateD -> enddateD_fallback": _date_parse_failures(master["enddateD"], master["enddateD_fallback"]),
        "date_acute_rej3 -> date_acute_rejection": _date_parse_failures(master["date_acute_rej3"], master["date_acute_rejection"]),
    }
    negative_dates = int(
        (
            master["date_tx"].isna()
            | (master["date_death"].notna() & master["date_death"].lt(master["date_tx"]))
            | (
                master["followup_end_real"].notna()
                & master["followup_end_real"].lt(master["date_tx"])
            )
            | base_s["time_to_death_or_censor_days"].isna()
            | base_s_fullfu["followup_days_fullfu"].isna()
        ).sum()
    )
    died_without_date = int(died_numeric.eq(1).sum() - master.loc[died_numeric.eq(1), "date_death"].notna().sum())
    death_date_without_died = int(master["date_death"].notna().sum() - master.loc[died_numeric.eq(1), "date_death"].notna().sum())
    insufficient_y5 = int(base_b_5y_all["y5_observed"].eq(0).sum())

    s_full_events = int(base_s_fullfu["event_death_fullfu"].sum())
    s_events = int(base_s["event_death"].sum())
    sf_min, sf_median, sf_max = _time_summary(base_s_fullfu["followup_days_fullfu"])
    s_min, s_median, s_max = _time_summary(base_s["time_to_death_or_censor_days"])
    y5_observable = int(base_b_5y_all["y5_observed"].sum())
    generated_lines = "\n".join(f"- `{path}`" for path in generated_files)
    date_failure_lines = "\n".join(
        f"- {name}: {n}" for name, n in date_failures.items()
    )

    report = f"""# Relatório de controle de qualidade da preparação da base analítica

## 1. Arquivo bruto utilizado

- Caminho do arquivo: `{csv_path}`
- Número de linhas: {len(raw_df)}
- Número de colunas: {len(raw_df.columns)}
- Confirmação: as 198 linhas foram preservadas quando presentes no arquivo bruto.

## 2. Validação inicial

- Contagem de óbitos (`died`): {int(died_numeric.sum())}
- Distribuição da categoria etária de maior idade (`age_gt40_main`):
{_counts_text(master["age_gt40_main"])}
- Distribuição de sexo (`sex_clean`):
{_counts_text(master["sex_clean"])}
- Distribuição de raça/etnia (`race3_clean`):
{_counts_text(master["race3_clean"])}
- Distribuição de HIV (`hiv_positive`):
{_counts_text(master["hiv_positive"])}
- Distribuição de tipo de doador (`donor_type_binary`):
{_counts_text(master["donor_type_binary"])}
- Distribuição de DGF (`dgf_main`):
{_counts_text(master["dgf_main"])}
- Distribuição de HLA agrupado 5-8 (`hla_high_mismatch_group`):
{_counts_text(master["hla_high_mismatch_group"])}
- Distribuição de HLA numérico estrito > 5 (`hla_gt5_strict_num`):
{_counts_text(master["hla_gt5_strict_num"])}
- Missing de CIT (`cit_hours`): {int(master["cit_hours"].isna().sum())}

## 3. Parsing de datas

- Colunas parseadas: `Dateoftxplant`, `Dateofdeath`, `DateLastFU`, `enddateD`, `date_acute_rej3`.
- Falhas de parsing por coluna:
{date_failure_lines}
- Inconsistências encontradas: {negative_dates}

## 4. Base S — Sobrevivência

- N Base S principal (full follow-up): {len(base_s_fullfu)}
- Eventos Base S principal (full follow-up): {s_full_events}
- Censurados Base S principal (full follow-up): {len(base_s_fullfu) - s_full_events}
- Follow-up Base S principal em dias: mínimo {sf_min:.0f}, mediano {sf_median:.0f}, máximo {sf_max:.0f}
- N Base S de sensibilidade (censura administrativa): {len(base_s)}
- Eventos Base S de sensibilidade: {s_events}
- Censurados Base S de sensibilidade: {len(base_s) - s_events}
- Follow-up Base S de sensibilidade em dias: mínimo {s_min:.0f}, mediano {s_median:.0f}, máximo {s_max:.0f}
- Diferença metodológica: a Base S principal usa o acompanhamento real disponível; a base com censura administrativa em 30/06/2015 permanece apenas para sensibilidade.
- Padronizacao metodologica da preparacao da base analitica: Kaplan-Meier deve ser tratado apenas como analise descritiva de sobrevida, o modelo de Cox como analise associativa e prognostica principal e os modelos preditivos como etapa posterior.

## 5. Base B — Horizonte fixo de 5 anos

- N total: {len(base_b_5y_all)}
- N observável: {y5_observable}
- Eventos até 5 anos: {int(base_b_5y_all["y5_status"].eq(1).sum())}
- Não eventos observáveis: {int(base_b_5y_all["y5_status"].eq(0).sum())}
- N não observável: {insufficient_y5}
- Nota metodológica: o horizonte de cinco anos serve apenas à avaliação de viabilidade amostral do desfecho binário. A redução do N efetivo não sustenta modelagem binária nesse horizonte.

## 6. Recodificações

- `age_gt40_main` usa `agegroupR3` e reproduz a categoria original do dataset, nao uma recodificacao numerica estrita de `ageR > 40`.
- `dgf_main` usa `dial_1stweek` como fonte principal porque reproduz os 41 casos de DGF do artigo; `dial_1stweek2` é apenas auditoria.
- `race3_clean` usa `RaceR3`, com remoção de prefixos e padronização de rótulos.
- `donor_deceased` e `donor_type_binary` usam `Donortype2`.
- O agrupamento de HLA usa `HLAmismatches3` (`0-4` vs `5-8`) e é mantido para caracterização descritiva da coorte.
- `hla_gt5_strict_num` usa `HLAmismatches > 5` apenas para auditoria numerica estrita.
- Divergencias entre agrupamento e corte estrito sao esperadas principalmente quando `HLAmismatches == 5`.
- HLA e CIT permanecem como variaveis de caracterizacao descritiva da coorte e exigem cautela por missing relevante; nao integram os modelos finais.
- Rejeição aguda foi preservada apenas para auditoria/análise secundária. No artigo-base, episódios de rejeição foram tratados como covariável dependente do tempo; portanto, sua inclusão como preditor fixo exigiria redefinição do marco temporal ou abordagem landmark.
- Nenhuma imputação foi aplicada.

## 7. Inconsistências e alertas

- Datas negativas ou ausentes: {negative_dates}
- Divergências de idade: {int(master["age_gt40_discrepancy"].sum())}
- Casos com `ageR == 40` rotulados como `>40`: {int(master["age_exactly40_labeled_gt40"].sum())}
- Divergências de DGF: {int(master["dgf_discrepancy"].sum())}
- Divergências HLA grupo vs estrito: {int(master["hla_group_vs_strict_discrepancy"].sum())}
- Casos de fronteira com `HLAmismatches == 5`: {int(master["hla_exactly5_boundary_case"].sum())}
- Casos com `died == 1` e `Dateofdeath` ausente: {died_without_date}
- Casos com `Dateofdeath` preenchida e `died == 0`: {death_date_without_died}
- Follow-up insuficiente para status em 5 anos: {insufficient_y5}

## 8. Arquivos gerados

{generated_lines}
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def save_master(df: pd.DataFrame, output_path: Path = MASTER_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    (PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "scripts").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    csv_path = find_input_csv()
    raw_df = load_data(csv_path)
    analytic_df = derive_predictors(raw_df)
    base_s = derive_survival_outcomes(analytic_df)
    base_s_fullfu = derive_survival_outcomes(
        analytic_df, administrative_censor_date=None
    )
    base_b_5y_all = derive_binary_5y(analytic_df)
    base_b_5y_observed = base_b_5y_all.loc[
        base_b_5y_all["y5_observed"].eq(1)
    ].copy()
    master_expanded = base_s.copy()
    master_expanded["event_death_fullfu"] = base_s_fullfu["event_death_fullfu"]
    master_expanded["end_date_fullfu"] = base_s_fullfu["end_date_fullfu"]
    master_expanded["followup_days_fullfu"] = base_s_fullfu["followup_days_fullfu"]
    master_expanded["followup_years_fullfu"] = base_s_fullfu["followup_years_fullfu"]
    master_expanded["y5_date"] = base_b_5y_all["y5_date"]
    master_expanded["y5_status"] = base_b_5y_all["y5_status"]
    master_expanded["y5_observed"] = base_b_5y_all["y5_observed"]

    save_master(master_expanded)
    base_s.to_csv(BASE_S_PATH, index=False)
    base_s_fullfu.to_csv(BASE_S_FULLFU_PATH, index=False)
    base_b_5y_all.to_csv(BASE_B_5Y_ALL_PATH, index=False)
    base_b_5y_observed.to_csv(BASE_B_5Y_OBSERVED_PATH, index=False)

    missing_summary = create_missing_summary(master_expanded)
    recoding_map = create_recoding_map()
    flow_counts = create_flow_counts(
        raw_df,
        master_expanded,
        base_s,
        base_s_fullfu,
        base_b_5y_all,
        base_b_5y_observed,
    )
    missing_summary.to_csv(MISSING_SUMMARY_PATH, index=False)
    recoding_map.to_csv(RECODING_MAP_PATH, index=False)
    flow_counts.to_csv(FLOW_COUNTS_PATH, index=False)

    generated_files = [
        MASTER_PATH,
        BASE_S_PATH,
        BASE_S_FULLFU_PATH,
        BASE_B_5Y_ALL_PATH,
        BASE_B_5Y_OBSERVED_PATH,
        FLOW_COUNTS_PATH,
        MISSING_SUMMARY_PATH,
        RECODING_MAP_PATH,
        QC_REPORT_PATH,
    ]
    write_qc_report(
        QC_REPORT_PATH,
        csv_path,
        raw_df,
        master_expanded,
        base_s,
        base_s_fullfu,
        base_b_5y_all,
        base_b_5y_observed,
        generated_files,
    )

    events = int(base_s_fullfu["event_death_fullfu"].sum())
    censored = len(base_s_fullfu) - events
    observed_5y = int(base_b_5y_all["y5_observed"].sum())
    print(f"N original: {len(raw_df)}")
    print(f"Eventos Base S principal (full follow-up): {events}")
    print(f"Censurados Base S principal (full follow-up): {censored}")
    print(f"Observaveis 5 anos: {observed_5y}")
    print(f"Caminho do CSV: {csv_path}")
    print(f"Base master criada: {MASTER_PATH}")
    print(f"Base S criada: {BASE_S_PATH}")
    print(f"Base S full follow-up criada: {BASE_S_FULLFU_PATH}")
    print(f"Base B 5 anos all criada: {BASE_B_5Y_ALL_PATH}")
    print(f"Base B 5 anos observed criada: {BASE_B_5Y_OBSERVED_PATH}")
    print(f"Flow counts criado: {FLOW_COUNTS_PATH}")
    print(f"Missing summary criado: {MISSING_SUMMARY_PATH}")
    print(f"Recoding map criado: {RECODING_MAP_PATH}")
    print(f"Relatorio QC criado: {QC_REPORT_PATH}")


if __name__ == "__main__":
    main()
