import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df_model = pd.read_csv("src/digital-exclusion-and-cancer/data-clean.csv")


def is_boolean_like(series):
    values = series.dropna()

    if values.empty:
        return False

    if pd.api.types.is_bool_dtype(values):
        return True

    if not pd.api.types.is_object_dtype(values):
        return False

    return values.astype(str).str.lower().isin(["true", "false"]).all()


def boolean_like_to_int(series):
    return series.astype(str).str.lower().map({"true": 1, "false": 0}).astype(int)


def prepare_model_data(
    df,
    outcome,
    predictors,
    eligibility_col,
):
    cols = [outcome, eligibility_col] + predictors

    d = df.loc[df[eligibility_col], cols].dropna().copy()

    for c in d.columns:
        if is_boolean_like(d[c]):
            d[c] = boolean_like_to_int(d[c])

    d[outcome] = d[outcome].astype(int)

    return d


def logit_table(model):
    conf = model.conf_int()

    return pd.DataFrame(
        {
            "term": model.params.index,
            "odds_ratio": np.exp(model.params.values),
            "ci_low": np.exp(conf[0].values),
            "ci_high": np.exp(conf[1].values),
            "p_value": model.pvalues.values,
        }
    ).sort_values("p_value")


colon_base_predictors = ["digitally_excluded"]

colon_age_gender_predictors = ["digitally_excluded", "age", "gender"]

colon_ses_predictors = [
    "digitally_excluded",
    "age",
    "gender",
    "isced1997",
    "log_hh_monthly_income",
]

colon_full_predictors = [
    "digitally_excluded",
    "age",
    "gender",
    "isced1997",
    "log_hh_monthly_income",
    "gp_contacts",
    "specialist_contacts",
    "health_literacy_help",
    "adl",
    "bmi",
    "country",
]

colon_models = {}

for name, predictors in {
    "unadjusted": colon_base_predictors,
    "age_gender": colon_age_gender_predictors,
    "ses_adjusted": colon_ses_predictors,
    "fully_adjusted": colon_full_predictors,
}.items():
    d = prepare_model_data(
        df_model,
        outcome="missed_colon_screening",
        predictors=predictors,
        eligibility_col="eligible_colon_screening",
    )

    formula = (
        "missed_colon_screening ~ digitally_excluded"
        if name == "unadjusted"
        else "missed_colon_screening ~ "
        + " + ".join(
            [
                f"C({x})" if x in ["gender", "health_literacy_help", "country"] else x
                for x in predictors
            ]
        )
    )

    model = smf.logit(formula, data=d).fit(disp=False)
    colon_models[name] = {
        "model": model,
        "n": len(d),
        "table": logit_table(model),
    }

colon_digital_results = pd.concat(
    [
        result["table"]
        .query("term == 'digitally_excluded'")
        .assign(model=name, n=result["n"])
        for name, result in colon_models.items()
    ],
    ignore_index=True,
)

colon_digital_results

mammogram_base_predictors = ["digitally_excluded"]
mammogram_age_predictors = ["digitally_excluded", "age"]
mammogram_ses_predictors = [
    "digitally_excluded",
    "age",
    "isced1997",
    "log_hh_monthly_income",
]
mammogram_full_predictors = [
    "digitally_excluded",
    "age",
    "isced1997",
    "log_hh_monthly_income",
    "gp_contacts",
    "specialist_contacts",
    "health_literacy_help",
    "adl",
    "bmi",
    "country",
]

mammogram_models = {}

for name, predictors in {
    "unadjusted": mammogram_base_predictors,
    "age_adjusted": mammogram_age_predictors,
    "ses_adjusted": mammogram_ses_predictors,
    "fully_adjusted": mammogram_full_predictors,
}.items():
    d = prepare_model_data(
        df_model,
        outcome="missed_mammogram",
        predictors=predictors,
        eligibility_col="eligible_mammogram",
    )

    formula = (
        "missed_mammogram ~ digitally_excluded"
        if name == "unadjusted"
        else "missed_mammogram ~ "
        + " + ".join(
            [
                f"C({x})" if x in ["health_literacy_help", "country"] else x
                for x in predictors
            ]
        )
    )

    model = smf.logit(formula, data=d).fit(disp=False)
    mammogram_models[name] = {
        "model": model,
        "n": len(d),
        "table": logit_table(model),
    }

mammogram_digital_results = pd.concat(
    [
        result["table"]
        .query("term == 'digitally_excluded'")
        .assign(model=name, n=result["n"])
        for name, result in mammogram_models.items()
    ],
    ignore_index=True,
)

mammogram_digital_results

digital_results = pd.concat(
    [
        colon_digital_results.assign(outcome="Missed colon cancer screening"),
        mammogram_digital_results.assign(outcome="Missed mammogram"),
    ],
    ignore_index=True,
)

digital_results = digital_results[
    ["outcome", "model", "n", "odds_ratio", "ci_low", "ci_high", "p_value"]
].round({"odds_ratio": 3, "ci_low": 3, "ci_high": 3, "p_value": 4})

digital_results
