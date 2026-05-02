import polars as pl
from pyshare import (
    MAP_SHARE_FINANCIAL_MISSING_CODES,
    MAP_SHARE_MISSING_CODES,
    MAP_YES_NO,
)


def clean_share_missing(expr: pl.Expr, *, financial: bool = False) -> pl.Expr:
    missing_codes = MAP_SHARE_MISSING_CODES + (
        MAP_SHARE_FINANCIAL_MISSING_CODES if financial else []
    )
    return pl.when(expr.is_in(missing_codes)).then(None).otherwise(expr)


def recode_with(col: str, with_map: dict = MAP_YES_NO) -> pl.Expr:
    return clean_share_missing(pl.col(col)).replace_strict(
        with_map, default=None, return_dtype=pl.Boolean
    )
