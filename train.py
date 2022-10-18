import argparse
from tqdm import tqdm
import copy
from transformers import GPT2Tokenizer, OPTForCausalLM, get_scheduler, default_data_collator
from datasets import load_dataset

from itertools import chain

from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator

accelerator = Accelerator()

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', default = 2048, type = int)
parser.add_argument('--batch_size', default = 4, type = int)
parser.add_argument('--num_proc', default = 16, type = int)
parser.add_argument('--gradient_accumulation_steps', default = 1, type = int)
parser.add_argument('--epochs', default = 1, type = int)
parser.add_argument('--save_every', default = 1000, type = int)
parser.add_argument("--resume_from_checkpoint", default = True, type = bool)
parser.add_argument('--resume_step', default = 0, type = int)
parser.add_argument('--model_checkpoint', default = None, type = str)
args = parser.parse_args()

# Constants

EPOCHS = args.epochs
SEQ_LEN = args.seq_len
gradient_accumulation_steps = args.gradient_accumulation_steps
save_every = args.save_every
BATCH_SIZE = args.batch_size
NUM_PROC = args.num_proc
resume_from_checkpoint = args.resume_from_checkpoint
RESUME_STEP = args.resume_step
model_checkpoint = args.model_checkpoint

model = OPTForCausalLM.from_pretrained("facebook/opt-350m")

if resume_from_checkpoint:
    model = OPTForCausalLM.from_pretrained(model_checkpoint)

optimizer = AdamW(model.parameters(), lr=3e-5)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

# Load dataset
load_train_dataset = load_dataset('conceptofmind/code-train-dedup')
# load_eval_dataset = load_dataset('conceptofmind/code-valid-dedup')

# Tokenizer
def tokenize(examples):
    seq_length = SEQ_LEN
    examples = tokenizer(examples['content'])
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= seq_length:
        total_length = (total_length // seq_length) * seq_length

    result = {
        k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
        for k, t in concatenated_examples.items()
    }

    result["labels"] = copy.deepcopy(result["input_ids"])

    return result

with accelerator.main_process_first():
    tokenized_train_dataset = load_train_dataset.map(
        tokenize, 
        batched = True, 
        num_proc = NUM_PROC, 
        remove_columns = 'content'
        )

    # tokenized_eval_dataset = load_eval_dataset.map(
    #     tokenize, 
    #     batched = True, 
    #     num_proc = NUM_PROC, 
    #     remove_columns = 'content'
    #     )

pytorch_train_dataset = tokenized_train_dataset.with_format('torch')
# pytorch_eval_dataset = tokenized_eval_dataset.with_format('torch')

# Create dataloader
train_dataloader = DataLoader(
    pytorch_train_dataset['train'], 
    shuffle = True, 
    drop_last = True, 
    collate_fn = default_data_collator, 
    batch_size = BATCH_SIZE
    )

# eval_dataloader = DataLoader(
#     pytorch_eval_dataset['train'], 
#     shuffle = False, 
#     drop_last=True, 
#     collate_fn = default_data_collator, 
#     batch_size = BATCH_SIZE
#     )

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=1000,
    num_training_steps = (len(train_dataloader) * EPOCHS) // gradient_accumulation_steps
)

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

progress_bar = tqdm(range(EPOCHS * len(train_dataloader)), disable=not accelerator.is_main_process)

model.train()
for epoch in range(EPOCHS):
    for step, batch in enumerate(train_dataloader, start=1):
        if resume_from_checkpoint:
            if RESUME_STEP is not None and step < RESUME_STEP:
                continue

        # Do training
        loss = model(**batch).loss
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)

        accelerator.clip_grad_norm_(model.parameters(), 1.0)

        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)

        if step % save_every == 0:
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.push_to_hub("code-350-model", commit_message=f"step {step}")

if accelerator.is_main_process:
    unwrapped_final_model = accelerator.unwrap_model(model)
    unwrapped_final_model.push_to_hub("code-model-final-350", commit_message=f"step {step}")  