from transformers import BertConfig, BertModel
import torch

print("torch.cuda.is_available()",torch.cuda.is_available())
print("torch.cuda.current_device()", torch.cuda.current_device())

# Download model and configuration from S3 and cache.
model = BertModel.from_pretrained('bert-base-uncased')
