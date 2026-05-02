import polars as pl

YES_NO = {1: True, 5: False}

GENDER = {1: "Male", 2: "Female"}

COMPUTER_SKILLS = {
    1: "Excellent",
    2: "Very good",
    3: "Good",
    4: "Fair",
    5: "Poor",
    6: "Never used a computer",
}

HEALTH_LITERACY_HELP = {
    1: "Always",
    2: "Often",
    3: "Sometimes",
    4: "Rarely",
    5: "Never",
}

ENDS_MEET = {
    1: "With great difficulty",
    2: "With some difficulty",
    3: "Fairly easily",
    4: "Easily",
}

ISCED_1997 = {
    0: "Pre-primary",
    1: "Primary",
    2: "Lower secondary",
    3: "Upper secondary",
    4: "Post-secondary non-tertiary",
    5: "First stage tertiary",
    6: "Second stage tertiary",
}

SHARE_MISSING_CODES = [-1, -2, -3, -4, -5, -7, -9]
SHARE_FINANCIAL_MISSING_CODES = [-9999991, -9999992]


def clean_share_missing(expr: pl.Expr, *, financial: bool = False) -> pl.Expr:
    missing_codes = SHARE_MISSING_CODES + (
        SHARE_FINANCIAL_MISSING_CODES if financial else []
    )
    return pl.when(expr.is_in(missing_codes)).then(None).otherwise(expr)


def recode_yes_no(col: str) -> pl.Expr:
    return clean_share_missing(pl.col(col)).replace_strict(
        YES_NO, default=None, return_dtype=pl.Boolean
    )
