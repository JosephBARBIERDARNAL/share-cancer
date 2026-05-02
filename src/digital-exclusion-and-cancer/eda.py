import polars as pl

df = pl.read_csv("src/digital-exclusion-and-cancer/data-clean.csv")

# Overview
df.select(
    n=pl.len(),
    n_countries=pl.col("country").n_unique(),
    age_min=pl.col("age").min(),
    age_max=pl.col("age").max(),
    age_mean=pl.col("age").mean(),
    share_female=(pl.col("gender") == "Female").mean(),
    share_digitally_excluded=pl.col("digitally_excluded").mean(),
    share_low_computer_skill=pl.col("low_computer_skill").mean(),
    share_no_recent_internet=pl.col("no_recent_internet").mean(),
    share_colon_screened=pl.col("colon_cancer_screening").mean(),
    share_mammogram=pl.col("mammogram").mean(),
    share_cancer=pl.col("cancer").mean(),
)

# Screening by digital
screening_by_digital = (
    df.group_by("digitally_excluded")
    .agg(
        n=pl.len(),
        mean_age=pl.col("age").mean(),
        female_share=(pl.col("gender") == "Female").mean(),
        tertiary_share=pl.col("tertiary_educated").mean(),
        median_income=pl.col("hh_monthly_income").median(),
        mean_gp_contacts=pl.col("gp_contacts").mean(),
        mean_specialist_contacts=pl.col("specialist_contacts").mean(),
        colon_screened=pl.col("colon_cancer_screening").mean(),
        mammogram=pl.col("mammogram").mean(),
        missed_colon_screening=pl.col("missed_colon_screening").mean(),
        missed_mammogram=pl.col("missed_mammogram").mean(),
        cancer_prevalence=pl.col("cancer").mean(),
    )
    .sort("digitally_excluded")
)
screening_by_digital

# Colon by digital
colon_eligible = df.filter(pl.col("eligible_colon_screening"))
colon_by_digital = (
    colon_eligible.group_by("digitally_excluded")
    .agg(
        n=pl.len(),
        screened=pl.col("colon_cancer_screening").mean(),
        missed=pl.col("missed_colon_screening").mean(),
        mean_age=pl.col("age").mean(),
        female_share=(pl.col("gender") == "Female").mean(),
        tertiary_share=pl.col("tertiary_educated").mean(),
        median_income=pl.col("hh_monthly_income").median(),
        mean_gp_contacts=pl.col("gp_contacts").mean(),
        mean_specialist_contacts=pl.col("specialist_contacts").mean(),
    )
    .sort("digitally_excluded")
)
colon_by_digital

# Mammogram digital
mammogram_eligible = df.filter(pl.col("eligible_mammogram"))
mammogram_by_digital = (
    mammogram_eligible.group_by("digitally_excluded")
    .agg(
        n=pl.len(),
        screened=pl.col("mammogram").mean(),
        missed=pl.col("missed_mammogram").mean(),
        mean_age=pl.col("age").mean(),
        tertiary_share=pl.col("tertiary_educated").mean(),
        median_income=pl.col("hh_monthly_income").median(),
        mean_gp_contacts=pl.col("gp_contacts").mean(),
        mean_specialist_contacts=pl.col("specialist_contacts").mean(),
    )
    .sort("digitally_excluded")
)
mammogram_by_digital


# Screening gap
def digital_gap_table(df: pl.DataFrame, outcome: str) -> pl.DataFrame:
    tab = (
        df.group_by("digitally_excluded")
        .agg(
            n=pl.len(),
            rate=pl.col(outcome).mean(),
        )
        .sort("digitally_excluded")
    )

    wide = tab.select(
        n_not_excluded=pl.col("n").filter(~pl.col("digitally_excluded")).first(),
        n_excluded=pl.col("n").filter(pl.col("digitally_excluded")).first(),
        rate_not_excluded=pl.col("rate").filter(~pl.col("digitally_excluded")).first(),
        rate_excluded=pl.col("rate").filter(pl.col("digitally_excluded")).first(),
    ).with_columns(
        absolute_gap=pl.col("rate_excluded") - pl.col("rate_not_excluded"),
        relative_gap=pl.col("rate_excluded") / pl.col("rate_not_excluded"),
    )

    return wide


digital_gap_table(colon_eligible, "missed_colon_screening")
digital_gap_table(mammogram_eligible, "missed_mammogram")
