from sqlalchemy.engine import Engine, Connection
from sqlalchemy import create_engine
from os import getenv
from typing import Optional
import pandas as pd
from pandas import DataFrame, Series, Timestamp, Index
from datetime import datetime
import numpy as np
from numpy import ndarray


def db_connection() -> Engine:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except:
        print('No ".env" file or python-dotenv not installed... Using default env variables...')
    dbname: Optional[str] = getenv('POSTGRES_DB_NAME')
    host: Optional[str] = getenv('POSTGRES_HOST')
    user: Optional[str] = getenv('POSTGRES_USERNAME')
    password: Optional[str] = getenv('POSTGRES_PASSWORD')
    port: Optional[str] = getenv('POSTGRES_PORT')

    postgres_str: str = f'postgresql://{user}:{password}@{host}:{port}/{dbname}'

    engine: Engine = create_engine(postgres_str)

    return engine


def group_hourly(df: DataFrame) -> DataFrame:
    df: DataFrame = df.copy()
    df['day']: Series = df['start_date_utc'].dt.year.astype('str') + '-' + df['start_date_utc'].dt.month.astype(
        'str') + '-' + df[
                            'start_date_utc'].dt.day.astype('str')
    df['day']: Series = pd.to_datetime(df['day'], infer_datetime_format=True)
    grouped: DataFrame = df.groupby(['plant_name_up', 'day', df.start_date_utc.dt.hour]).agg(
        {'kwh': 'mean'})
    grouped: DataFrame = grouped.reset_index(drop=False).rename(columns={'start_date_utc': 'time'})
    #     grouped: DataFrame = grouped.sort_values(by=['plant_name_up', 'day', 'time'], ascending=True, ignore_index=True)
    grouped['time'] = grouped['day'].astype('str') + ' ' + grouped['time'].astype('str') + ':00:00'
    grouped['time'] = grouped['time'].astype('datetime64[ns, UTC]')
    grouped: DataFrame = grouped.sort_values(by=['plant_name_up', 'time'], ascending=True, ignore_index=True)
    grouped.drop('day', axis=1, inplace=True)

    return grouped


def extract_weather(weather_sql: str, engine: Engine) -> DataFrame:
    weather_df: DataFrame = pd.read_sql_query(weather_sql, con=engine)
    weather_df['wind_gusts_100m_1h_ms'] = weather_df['wind_gusts_100m_1h_ms'].astype('float64')
    weather_df['wind_gusts_100m_ms'] = weather_df['wind_gusts_100m_ms'].astype('float64')
    weather_df: DataFrame = weather_df.sort_values(by=['timestamp_utc'], ascending=True, ignore_index=True)
    # pivot weather df
    weather_df_piv = weather_df.pivot(index='timestamp_utc', columns=['plant_name_up'],
                                      values=list(weather_df.columns[3:]))
    new_index = Index([i[0] + '_' + i[1] for i in weather_df_piv.columns])
    weather_df_piv.columns = new_index
    weather_df_piv.reset_index(drop=False, inplace=True)

    return weather_df_piv


def merge_df(energy: DataFrame, weather: DataFrame) -> DataFrame:
    df: DataFrame = energy.merge(weather, left_on='time',
                                 right_on='timestamp_utc')
    df.drop(['timestamp_utc'], axis=1, inplace=True)
    df = df.sort_values(by=['time'], ascending=True, ignore_index=True)

    return df


def extract_energy(engine: Engine) -> DataFrame:
    sql_energy: str = "SELECT * FROM sorgenia_energy"
    energy_df: DataFrame = pd.read_sql_query(sql_energy, con=engine)
    energy_grouped: DataFrame = group_hourly(energy_df)
    # pivot energy grouped
    energy_grouped_piv: DataFrame = energy_grouped.pivot(index='time', columns='plant_name_up', values='kwh')
    energy_grouped_piv.columns.name = None
    energy_grouped_piv.reset_index(drop=False, inplace=True)

    return energy_grouped_piv


def preprocess_sorgenia(weather_source: str):
    engine: Engine = db_connection()
    # sql_energy: str = "SELECT * FROM sorgenia_energy"
    # energy_df: DataFrame = pd.read_sql_query(sql_energy, con=engine)
    energy_grouped: DataFrame = extract_energy(engine)
    if weather_source == 'copernicus':
        weather_df: DataFrame = extract_weather(f"SELECT * FROM sorgenia_weather_{weather_source}", engine)
    else:
        weather_df: DataFrame = extract_weather(f"SELECT * FROM sorgenia_weather", engine)
    df: DataFrame = merge_df(energy_grouped, weather_df)
    # adding features
    timestamp_s: Series = df['time'].map(datetime.timestamp)
    day: int = 24 * 60 * 60
    year: float = 365.2425 * day

    df['Day sin']: Series = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos']: Series = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin']: Series = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos']: Series = np.cos(timestamp_s * (2 * np.pi / year))

    earliest_time: Timestamp = df.time.min()
    df['t']: Series = (df['time'] - earliest_time).dt.seconds / 60 / 60 + (df['time'] - earliest_time).dt.days * 24
    df['days_from_start']: Series = (df['time'] - earliest_time).dt.days
    # df["id"] = df["plant_name_up"]
    df['hour']: Series = df["time"].dt.hour
    df['day']: Series = df["time"].dt.day
    df['day_of_week']: Series = df["time"].dt.dayofweek
    df['month']: Series = df["time"].dt.month
    # df['categorical_id']: Series = df['id'].copy()
    df['hours_from_start']: Series = df['t']
    df['categorical_day_of_week']: Series = df['day_of_week'].copy()
    df['categorical_hour']: Series = df['hour'].copy()

    # save df to csv file
    df.to_csv(f'data/sorgenia_wind_{weather_source}.csv', index=False)
    print(f'Saved in data/sorgenia_wind_{weather_source}.csv')
    print('Done.')


if __name__ == "__main__":
    preprocess_sorgenia('copernicus')
