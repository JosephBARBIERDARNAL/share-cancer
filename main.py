import pyshare as ps
import polars as pl
import morethemes as mt

mt.set_theme("minimal")


df = ps.read_share_wave(
    wave=9, path="../data-share", modules=["cv_r", "ph", "dn", "ep"]
)

df_clean = df.select(
    "mergeid",
    "country",
    "gender",
    "hhid9",
    "exrate",
    receive_recognition="ep032_",
    year_of_birth="dn003_",
    cancer="ph006d10",
).with_columns(
    country=pl.col("country").replace_strict(
        ps.MAP_ID_TO_COUNTRY, return_dtype=pl.Utf8
    ),
    age=(2022 - pl.col("year_of_birth")),
    receive_recognition=pl.col("receive_recognition").replace_strict(
        {
            1: "Strongly agree",
            2: "Agree",
            3: "Disagree",
            4: "Strongly disagree",
        }
    ),
)
