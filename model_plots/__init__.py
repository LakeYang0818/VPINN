import logging

logging.basicConfig(
    format="%(levelname)-8s %(module)-14s %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)

# Set matplotlib backend globally in order to avoid potential issues from
# people forgetting to set this
import matplotlib

matplotlib.use("Agg")
# Custom DAG operations
import pandas as pd

from utopya.eval import register_operation

register_operation(name="pd.Index", func=pd.Index)
