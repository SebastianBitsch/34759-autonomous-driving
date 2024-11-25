
import logging
import torch
from models.cnn_encoder import CNNEncoder as Encoder
from data import get_data_loader



logging.basicConfig(
    encoding="utf-8",
    filename="Encoder.log",
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

logger = logging.getLogger("train")

EPOCH = 5
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Encoder().to(device)
logger.info("Model loaded successfully to device: %s", device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_loader, test_loader = get_data_loader()
for epoch in range(EPOCH):
    running_train_loss = 0.0
    last_train_loss = 0.0
    validation_loss = 0.0
    
    
    
    for batch_idx, batch in enumerate(train_loader):
        model.train()
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        output = model(images)
        
        optimizer.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        last_train_loss = loss.item()
        running_train_loss += last_train_loss
        
        optimizer.step()
        
        
        
