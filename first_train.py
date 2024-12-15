import sys

sys.path.insert(0, r'N:\VersionControl\thirdparty\Windowspython3_packages_nlp')
sys.path.insert(0, r'P:\Pipeline')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from snow.common.lib import yaml_rw as yaml

# กำหนด tags ที่ใช้ในโมเดล
tags_list = ["AI", "Machine Learning", "Programming", "NLP", "Deep Learning"]

# สร้างฟังก์ชันเพื่อแปลงผลลัพธ์ tensor เป็น tag ที่มนุษย์เข้าใจ
def convert_to_tags(predicted_tensor, threshold=0.05):  # ลด threshold
    predicted_tags = []

    # แปลง tensor เป็น numpy array
    predicted_tensor = predicted_tensor.cpu().numpy()

    for idx, value in enumerate(predicted_tensor[0]):
        if value >= threshold:
            if idx < len(tags_list):
                predicted_tags.append(tags_list[idx])

    return predicted_tags


# ---------- ขั้นตอนที่ 1: โหลดข้อมูล ----------
# เพิ่ม path และโหลด YAML
data = yaml.read(r"./train_data/data1.yaml", 'safe')
train_data = data["train"]
validation_data = data["validation"]

print("Train Data:", train_data)
print("Validation Data:", validation_data)

# ---------- ขั้นตอนที่ 2: สร้าง Tokenizer ----------
train_sentences = [item["sentence"] for item in train_data]

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=1000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)
tokenizer.train_from_iterator(train_sentences, trainer)

example_sentence = "I love machine learning."
output = tokenizer.encode(example_sentence)
print("Tokens:", output.tokens)
print("IDs:", output.ids)


# ---------- ขั้นตอนที่ 3: แปลงข้อมูลและสร้าง DataLoader ----------
def tokenize_data(data):
    tokenized_data = []
    for item in data:
        encoded = tokenizer.encode(item["sentence"])
        tokenized_data.append({
            "input_ids": encoded.ids,
            "tags": item["tags"]
        })
    return tokenized_data


tokenized_train_data = tokenize_data(train_data)
tokenized_validation_data = tokenize_data(validation_data)


def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    tags = [item["tags"] for item in batch]
    return {"input_ids": input_ids, "tags": tags}


train_loader = DataLoader(tokenized_train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(tokenized_validation_data, batch_size=2, collate_fn=collate_fn)


# ---------- ขั้นตอนที่ 4: สร้างโมเดล ----------
class SimpleNLPTagger(nn.Module):
    def __init__(self, vocab_size, embed_size, num_tags):
        super(SimpleNLPTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, num_tags)
        self.softmax = nn.Sigmoid()

    def forward(self, input_ids):
        x = self.embedding(input_ids).mean(dim=1)
        return self.softmax(self.fc(x))


vocab_size = tokenizer.get_vocab_size()
embed_size = 128
num_tags = len({tag for item in train_data for tag in item["tags"]})

model = SimpleNLPTagger(vocab_size, embed_size, num_tags)
print("Model created successfully!")

# ---------- ขั้นตอนที่ 5: ฝึกโมเดล ----------
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 1500
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"]
        tags = batch["tags"]

        tag_tensor = torch.zeros((len(tags), num_tags))
        for i, tag_list in enumerate(tags):
            for tag in tag_list:
                tag_idx = list({tag for item in train_data for tag in item["tags"]}).index(tag)
                tag_tensor[i, tag_idx] = 1.0

        outputs = model(input_ids)
        loss = criterion(outputs, tag_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

# ---------- ขั้นตอนที่ 6: ทดสอบโมเดล ----------
model.eval()
with torch.no_grad():
    for batch in validation_loader:
        input_ids = batch["input_ids"]
        tags = batch["tags"]

        # ส่งข้อมูลเข้าโมเดล
        outputs = model(input_ids)

        # แปลงผลลัพธ์ tensor เป็น tags ที่มนุษย์เข้าใจ
        predicted_tags = convert_to_tags(outputs, threshold=0.05)
        print("Predicted Tags:", predicted_tags)
        print("True Tags:", tags)
        break
