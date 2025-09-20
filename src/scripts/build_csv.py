from asyncio import run
from datetime import datetime
from io import StringIO
from pathlib import Path
from re import sub
from zipfile import ZipFile

from aiofiles import open as aopen
from httpx import AsyncClient
from polars import DataFrame, col, concat, datatypes, lit, read_csv, struct
from polars import all as pl_all
from rich.progress import track
from typer import Typer

BASE_PATH = Path(__file__).parents[2] / "data"

IN_PATH = BASE_PATH / "in"
OUT_PATH = BASE_PATH / "out"
RAW_PATH = BASE_PATH / "raw"
BASE_URL = "https://portal.inmet.gov.br"
SCHEMA: dict[str, datatypes.DataTypeClass] = {
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": datatypes.Float64,
    "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)": datatypes.Float64,
    "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)": datatypes.Float64,
    "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)": datatypes.Float64,
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": datatypes.Float64,
    "TEMPERATURA DO PONTO DE ORVALHO (°C)": datatypes.Float64,
    "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)": datatypes.Float64,
    "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)": datatypes.Float64,
    "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)": datatypes.Float64,
    "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)": datatypes.Float64,
    "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)": datatypes.Int64,
    "UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)": datatypes.Int64,
    "UMIDADE RELATIVA DO AR, HORARIA (%)": datatypes.Int64,
    "VENTO, DIREÇÃO HORARIA (gr) (° (gr))": datatypes.Int64,
    "VENTO, RAJADA MAXIMA (m/s)": datatypes.Float64,
    "VENTO, VELOCIDADE HORARIA (m/s)": datatypes.Float64,
}

app = Typer()


async def download_data_by_year(year: int):
    file_name = f"{year}.zip"
    if (RAW_PATH / file_name).exists():
        print(f"File for the year {year} already downloaded")
        return
    now = datetime.now()
    current_year = now.year
    if year > current_year:
        raise IndexError(f"Max allowed year is {current_year}")

    async with AsyncClient(base_url=BASE_URL, timeout=None) as client:
        res = await client.get(f"/uploads/dadoshistoricos/{file_name}")
    async with aopen(RAW_PATH / file_name, "wb") as file:
        await file.write(res.content)


def unzip_file(file_path: Path, out_path: Path):
    with ZipFile(file_path, "r") as f:
        f.extractall(out_path)


def convert_datetime(date: str, time_df: str, field: str):
    time = list(time_df.split()[0])
    hour = "".join(time[:2])
    minute = "".join(time[2:])
    dt = date + "T" + f"{hour}:{minute}:00Z"
    return getattr(datetime.strptime(dt, "%Y/%m/%dT%H:%M:%SZ"), field)


def get_temp_mean(min_: float, max_: float):
    return sum([min_, max_]) / 2


async def load_data(file_path: Path):
    async with aopen(file_path, "r") as f:
        all_lines = await f.readlines()
        head = all_lines[4:7]
        data = all_lines[8:]
    return head, data


def build_csv(head: list[str], data: list[str]):
    extra_columns = {
        "LATITUDE": 0.0,
        "LONGITUDE": 0.0,
        "ALTITUDE": 0.0,
    }
    merge_columns = ("Data", "Hora UTC")
    temp_columns = (
        "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)",
        "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)",
    )

    for h in head:
        key, h_value = h.split(":")
        h_value = h_value[1:-1]
        extra_columns[key] = float(sub(r",", ".", h_value))

    df = read_csv(
        StringIO("".join(data)),
        encoding="latin-1",
        separator=";",
        decimal_comma=True,
        infer_schema=False,
        # ignore_errors=True,
    )
    df = df.drop("RADIACAO GLOBAL (Kj/m²)", "")
    df = df.drop_nulls()
    df = df.select(pl_all().map_elements(lambda x: sub(r",", ".", x)))
    for key, e_value in extra_columns.items():
        df = df.with_columns(lit(e_value).alias(key))

    df = df.with_columns(
        struct(*merge_columns)
        .map_elements(
            lambda x: convert_datetime(x["Data"], x["Hora UTC"], "hour"),
            datatypes.Int64,
        )
        .alias("Hora")
    )
    df = df.with_columns(
        struct(*merge_columns)
        .map_elements(
            lambda x: convert_datetime(x["Data"], x["Hora UTC"], "month"),
            datatypes.Int64,
        )
        .alias("Mes")
    )
    df = df.with_columns(
        struct(*merge_columns)
        .map_elements(
            lambda x: convert_datetime(x["Data"], x["Hora UTC"], "day"), datatypes.Int64
        )
        .alias("Dia")
    )
    for key, value in SCHEMA.items():
        df = df.with_columns(col(key).cast(value))
    df = df.with_columns(
        struct(*temp_columns)
        .map_elements(
            lambda x: get_temp_mean(
                x["TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)"],
                x["TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)"],
            ),
            datatypes.Float64,
        )
        .alias("temp_mean")
    )
    return df.drop(*merge_columns)


async def download_and_process(year: int):
    await download_data_by_year(year)
    file_csv = f"{year}.csv"
    if (OUT_PATH / file_csv).exists():
        print("File already created")
        return
    unpack_path = IN_PATH / str(year)
    for path in (IN_PATH, OUT_PATH, RAW_PATH, unpack_path):
        path.mkdir(parents=True, exist_ok=True)
    file_zip = f"{year}.zip"
    unzip_file(RAW_PATH / file_zip, unpack_path)
    df = DataFrame()
    for file in track(list(unpack_path.iterdir())):
        head, data = await load_data(file)
        tmp = build_csv(head, data)
        df = concat((df, tmp))

    df.write_csv(OUT_PATH / file_csv)


@app.command()
def main(year: int):
    run(download_and_process(year))


if __name__ == "__main__":
    app()
