from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from datetime import timedelta

# Define the main entity - represents a unique sticker/product combination
sticker_entity = Entity(
    name="sticker",
    join_keys=["sticker_id"],
    description="Sticker product entity - combination of country, store, and product"
)

# Define data source for features
# This points to the parquet file we'll create from training data
sticker_features_source = FileSource(
    name="sticker_features_source",
    path="data/feature_data.parquet",
    timestamp_field="event_timestamp"
)

# Define feature view for date-based features
# These are extracted from the date column using FastAI's add_datepart
date_features_view = FeatureView(
    name="date_features",
    entities=[sticker_entity],
    ttl=timedelta(days=365),  # Features valid for 1 year
    schema=[
        Field(name="Year", dtype=Int64),
        Field(name="Month", dtype=Int64),
        Field(name="Week", dtype=Int64),
        Field(name="Day", dtype=Int64),
        Field(name="Dayofweek", dtype=Int64),
        Field(name="Dayofyear", dtype=Int64),
        Field(name="Is_month_end", dtype=Int64),
        Field(name="Is_month_start", dtype=Int64),
        Field(name="Is_quarter_end", dtype=Int64),
        Field(name="Is_quarter_start", dtype=Int64),
        Field(name="Is_year_end", dtype=Int64),
        Field(name="Is_year_start", dtype=Int64),
    ],
    source=sticker_features_source,
    tags={"team": "ml_team", "version": "v1"}
)

# Define feature view for categorical features
# These are encoded versions of country, store, and product
categorical_features_view = FeatureView(
    name="categorical_features",
    entities=[sticker_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="country_encoded", dtype=Int64),
        Field(name="store_encoded", dtype=Int64),
        Field(name="product_encoded", dtype=Int64),
    ],
    source=sticker_features_source,
    tags={"team": "ml_team", "version": "v1"}
)