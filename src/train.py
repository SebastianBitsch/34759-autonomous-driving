
import logging
import torch
from models.cnn_encoder import CNNEncoder as Encoder



logging.basicConfig(
    encoding="utf-8",
    filename="Encoder.log",
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

logger = logging.getLogger("train")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Encoder().to(device)
logger.info("Model loaded successfully to device: %s", device)



