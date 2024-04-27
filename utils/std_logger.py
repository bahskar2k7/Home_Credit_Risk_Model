import os
from pathlib import Path
import logging
from datetime import datetime
# from utils.config_reader import cfg


HOME_PATH = os.getenv("USERPROFILE")
PROJECT_NAME = "home-credit"
LOG_FILE_NAME = PROJECT_NAME + "_" + datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_DIR_PATH = Path(HOME_PATH).joinpath("data", "logs", PROJECT_NAME)
LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR_PATH.joinpath(LOG_FILE_NAME+".log")
date_fmt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s -> %(funcName)s: %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOG_PATH)],
)
logger = logging.getLogger()
logging.getLogger("chardet.charsetprober").disabled = True
