import pandas as pd

from common.config_loader import load_config
from common.logger import get_logger

logger = get_logger(__name__)


def load_raw_data() -> pd.DataFrame:
    config = load_config()

    raw_path = config["data"]["raw_path"]
    drop_columns = config["data"]["drop_columns"]
    target_col = config["data"]["target_col"]

    df_raw = pd.read_csv(raw_path)
    df = df_raw.drop(columns=drop_columns)

    logger.info("Arquivo carregado: %s", raw_path)
    logger.info("df_raw shape: %s", df_raw.shape)
    logger.info("df shape após remoção de identificadores: %s", df.shape)
    logger.info("Distribuição do target:\n%s", df[target_col].value_counts().to_string())

    return df
