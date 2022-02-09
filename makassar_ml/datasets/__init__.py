from .csv_timeseries import CsvTimeseriesDataset # Must be first import due to circular import issue.
from .beijing_pm25 import BeijingPM25Dataset
from .timeseries_forecast_wrapper import TimeseriesForecastDatasetWrapper

# Load PyTorch Lightning modules if installed.
try:
    import pytorch_lightning
    from .lt_beijing_pm25 import BeijingPM25LightningDataModule
    del pytorch_lightning # Cleanup namespace.
except ImportError:
    pass