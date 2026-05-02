import pyshare as ps
import polars as pl

from utils import (
    ENDS_MEET,
    CANCER,
    COMPUTER_SKILLS,
    GENDER,
    HEALTH_LITERACY_HELP,
    ISCED_1997,
    clean_share_missing,
    recode_with,
)


df: pl.DataFrame = ps.read_share_wave(
    wave=9,
    path="../data-share",
    modules=[
        "cv_r",
        "dn",
        "ph",
        "hc",
        "it",
        "hh",
        "co",
        "gv_isced",
        "gv_health",
        "gv_weights",
    ],
)

df_clean = (
    df.select(
        "mergeid",
        "country",
        "hhid9",
        "gender",
        "dn003_",
        "ph006d10",
        "hc886_",
        "hc887_",
        "hc876_",
        "hc877_",
        "hc889_",
        "it003_",
        "it004_",
        "hh017e",
        "co007_",
        "isced1997_r",
        "isced2011_r",
        "adl",
        "bmi",
        "cchw_w9",
    )
    .rename(
        {
            "dn003_": "year_of_birth",
            "ph006d10": "cancer",
            "hc886_": "mammogram",
            "hc887_": "colon_cancer_screening",
            "hc876_": "gp_contacts",
            "hc877_": "specialist_contacts",
            "hc889_": "health_literacy_help",
            "it003_": "computer_skills",
            "it004_": "internet_past_7_days",
            "hh017e": "hh_monthly_income",
            "co007_": "make_ends_meet",
            "isced1997_r": "isced1997",
            "isced2011_r": "isced2011",
            "cchw_w9": "cross_sectional_weight",
        }
    )
    .with_columns(
        country=pl.col("country").replace_strict(ps.MAP_ID_TO_COUNTRY),
        gender=clean_share_missing(pl.col("gender")).replace_strict(
            GENDER, default=None
        ),
        age=(2022 - clean_share_missing(pl.col("year_of_birth"))),
        cancer=recode_with("cancer", with_map=CANCER),
        mammogram=recode_with("mammogram"),
        colon_cancer_screening=recode_with("colon_cancer_screening"),
        internet_past_7_days=recode_with("internet_past_7_days"),
        computer_skills=clean_share_missing(pl.col("computer_skills")).replace_strict(
            COMPUTER_SKILLS, default=None
        ),
        health_literacy_help=clean_share_missing(
            pl.col("health_literacy_help")
        ).replace_strict(HEALTH_LITERACY_HELP, default=None),
        make_ends_meet=clean_share_missing(pl.col("make_ends_meet")).replace_strict(
            ENDS_MEET, default=None
        ),
        isced1997=clean_share_missing(pl.col("isced1997")),
        isced1997_label=clean_share_missing(pl.col("isced1997")).replace_strict(
            ISCED_1997, default=None
        ),
        isced2011=clean_share_missing(pl.col("isced2011")),
        gp_contacts=clean_share_missing(pl.col("gp_contacts")),
        specialist_contacts=clean_share_missing(pl.col("specialist_contacts")),
        hh_monthly_income=clean_share_missing(
            pl.col("hh_monthly_income"), financial=True
        ),
        adl=clean_share_missing(pl.col("adl")),
        bmi=clean_share_missing(pl.col("bmi")),
        cross_sectional_weight=clean_share_missing(pl.col("cross_sectional_weight")),
    )
    .with_columns(
        low_computer_skill=pl.col("computer_skills").is_in(
            ["Poor", "Never used a computer"]
        ),
        no_recent_internet=pl.col("internet_past_7_days").eq(False),
        digitally_excluded=(
            pl.col("computer_skills").is_in(["Poor", "Never used a computer"])
            | pl.col("internet_past_7_days").eq(False)
        ),
        tertiary_educated=pl.col("isced1997").is_in([5, 6]),
        log_hh_monthly_income=pl.col("hh_monthly_income").log1p(),
        any_screening=pl.any_horizontal(
            pl.col("mammogram").eq(True), pl.col("colon_cancer_screening").eq(True)
        ),
    )
    .with_columns(
        eligible_colon_screening=pl.col("age").is_between(50, 74),
        eligible_mammogram=(
            pl.col("gender").eq("Female") & pl.col("age").is_between(50, 69)
        ),
    )
    .with_columns(
        missed_colon_screening=(
            pl.col("eligible_colon_screening")
            & pl.col("colon_cancer_screening").eq(False)
        ),
        missed_mammogram=(pl.col("eligible_mammogram") & pl.col("mammogram").eq(False)),
    )
    .filter(pl.col("age") >= 50)
)

df_clean.write_csv("src/digital-exclusion-and-cancer/data-clean.csv")
