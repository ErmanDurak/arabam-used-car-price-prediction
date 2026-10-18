"""
src/cleaner.py: Data cleaning and preprocessing pipeline for arabam.com car dataset.
Contains modular functions for feature-specific cleaning and formatting.

"""

import os
import re
import pandas as pd
import numpy as np

def clean_price(val) -> float:
    """
    Cleans Turkish Lira price strings into float numbers.
    If discounted (e.g. '895.000 TL880.000 TL'), selects the final current price.
    
    """
    
    if pd.isna(val):
        return np.nan

    val_str = str(val).strip()

    price_matches = re.findall(r"(\d+(?:\.\d+)*)", val_str)

    if not price_matches:
        return np.nan

    final_price_str = price_matches[-1].replace(".", "")

    try:
        return float(final_price_str)
    except ValueError:
        return np.nan

def clean_km(val) -> float:
    """
    Cleans kilometre strings into float numbers.
    Example: '98.800 KM' -> 98800.0
    
    """

    if pd.isna(val):
        return np.nan

    val_str = str(val).strip()

    cleaned_str = re.sub(r"[^0-9]", "", val_str)

    if not cleaned_str:
        return np.nan

    try:
        return float(cleaned_str)
    except ValueError:
        return np.nan

def clean_engine_size(val) -> float:
    """
    Cleans engine displacement strings into float (cc).
    Handles exact values ('1598 CC') and ranges ('1201 - 1400 cm3' -> midpoint).

    """

    if pd.isna(val):
        return np.nan

    val_str = str(val).strip()

    numbers = [float(n) for n in re.findall(r"\d+", val_str)]

    if not numbers:
        return np.nan

    if len(numbers) >= 2:
        return (numbers[0] + numbers[1]) / 2

    return numbers[0]

def clean_engine_power(val) -> float:

    if pd.isna(val):
        return np.nan

    val_str = str(val).strip()

    numbers = [float(n) for n in re.findall(r"\d+", val_str)]

    if not numbers:
        return np.nan

    if len(numbers) >= 2:
        return (numbers[0] + numbers[1]) / 2

    return numbers[0]

def clean_year(val) -> float:

    if pd.isna(val):
        return np.nan

    try:
        year_num = float(val)
        if 1970 <= year_num <= 2026:
            return year_num
        return np.nan
    except (ValueError, TypeError):
        return np.nan

def clean_fuel_consumption(val) -> float:

    if pd.isna(val):
        return np.nan

    val_str = str(val).strip()

    cleaned_str = val_str.replace(",", ".").replace("lt", "").replace("LT", "").strip()

    try:
        cons_val = float(cleaned_str)
        if 0 < cons_val < 30:
            return cons_val
        return np.nan
    except ValueError:
        return np.nan

def parse_damage_status(val) -> dict:
    """
    Parses 'boya-değişen' text into 4 structured numerical features:
    painted_count, local_painted_count, changed_count, is_original.

    """

    if pd.isna(val) or str(val).strip() == "Belirtilmemiş":
        return {
            "painted_count" : np.nan,
            "local_painted_count": np.nan,
            "changed_count": np.nan,
            "is_original": 0
        }

    val_str = str(val).strip().lower()

    if "tamamı orjinal" in val_str or "tamamı orijinal" in val_str:
        return {
            "painted_count": 0.0,
            "local_painted_count": 0.0,
            "changed_count": 0.0,
            "is_original": 1
        }

    local_match = re.search(r"(\d+)\s*lokal\s*boyalı", val_str)
    local_cnt = float(local_match.group(1)) if local_match else 0

    val_without_local = re.sub(r"\d+\s*lokal\s*boyalı", "", val_str)
    painted_match = re.search(r"(\d+)\s*boyalı", val_without_local)
    painted_cnt = float(painted_match.group(1)) if painted_match else 0

    changed_match = re.search(r"(\d+)\s*değişen", val_str)
    changed_cnt = float(changed_match.group(1)) if changed_match else 0

    return {
        "painted_count" : painted_cnt,
        "local_painted_count" : local_cnt,
        "changed_count" : changed_cnt,
        "is_original" : 0
    }

def clean_category(val) -> str:

    if pd.isna(val):
        return np.nan

    val_str = str(val).strip()

    if val_str in ["-", "", "Belirtilmemiş"]:
        return np.nan

    return val_str.title()

def clean_arabam_data(raw_csv_path: str, output_csv_path: str = None) -> pd.DataFrame:
    """
    Main pipeline function that ingests raw arabam.com data,
    applies feature-specific cleaning routines, standardizes column names to English,
    and exports the cleaned dataset.
    """

    print(f"[INFO] Ingesting raw dataset from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)

    df["price"] = df["price"].apply(clean_price)
    df["kilometre"] = df["kilometre"].apply(clean_km)
    df["motor hacmi"] = df["motor hacmi"].apply(clean_engine_size)
    df["motor gücü"] = df["motor gücü"].apply(clean_engine_power)
    df["yıl"] = df["yıl"].apply(clean_year)
    df["ort. yakıt tüketimi"] = df["ort. yakıt tüketimi"].apply(clean_fuel_consumption)

    damage_df = df["boya-değişen"].apply(parse_damage_status).apply(pd.Series)
    df = pd.concat([df, damage_df], axis=1)

    categorical_cols = ["marka", "seri", "model", "vites tipi", "yakıt tipi", "kasa tipi", "çekiş", "renk"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_category)

    column_mapping = {
        "price" : "price",
        "marka" : "brand",
        "seri" : "series",
        "model" : "model",
        "yıl" : "year",
        "kilometre" : "kilometer",
        "vites tipi" : "transmission",
        "yakıt tipi" : "fuel_type",
        "kasa tipi" : "body_type",
        "renk" : "color",
        "motor hacmi" : "engine_displacement_cc",
        "motor gücü" : "engine_power_hp",
        "çekiş" : "drivetrain",
        "ort. yakıt tüketimi" : "avg_fuel_consumption",
        "painted_count": "painted_count",
        "local_painted_count": "local_painted_count",
        "changed_count": "changed_count",
        "is_original": "is_original"
    }

    cleaned_df = df[list(column_mapping.keys())].rename(columns = column_mapping)

    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        cleaned_df.to_csv(output_csv_path, index=False)
        print(f"[SUCCESS] Cleaned dataset saved to: {output_csv_path} ({cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns)")

    return cleaned_df

if __name__ == "__main__":
    raw_path = "data/raw/raw_arabam.csv"
    processed_path = "data/processed/cleaned_arabam.csv"

    clean_arabam_data(raw_csv_path=raw_path, output_csv_path=processed_path)

