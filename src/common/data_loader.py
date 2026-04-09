import pandas as pd

from common.config_loader import load_config
from common.logger import get_logger

logger = get_logger("common.data_loader")


def load_raw_data() -> pd.DataFrame:
    config = load_config()

    raw_path = config["data"]["raw_path"]
    target_col = config["data"]["target_col"]

    raw_dataframe = pd.read_csv(raw_path)

    logger.info("Arquivo carregado: %s", raw_path)
    logger.info("Shape do dado bruto: %s", raw_dataframe.shape)
    logger.info(
        "Distribuição do target:\n%s",
        raw_dataframe[target_col].value_counts().to_string(),
    )

    return raw_dataframe
