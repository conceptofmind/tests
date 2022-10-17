import transformers
from tqdm import tqdm
import copy
from transformers import GPT2Tokenizer, OPTForCausalLM, get_scheduler, default_data_collator
from datasets import load_dataset

from itertools import chain

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator

accelerator = Accelerator()

# Constants

EPOCHS = 1
SEQ_LEN = 2048
gradient_accumulation_steps = 1
BATCH_SIZE = 4
NUM_PROC = 16
RESUME_STEP = 0

model = OPTForCausalLM.from_pretrained("facebook/opt-300m")

optimizer = AdamW(model.parameters(), lr=3e-5)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-300m")

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

# load in the weights and states from a previous save
# if args.resume_from_checkpoint:
#     if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
#         accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
#         accelerator.load_state(args.resume_from_checkpoint)
#         path = os.path.basename(args.resume_from_checkpoint)
#     else:
#         # Get the most recent checkpoint
#         dirs = [f.name for f in os.scandir(args.save_dir) if f.is_dir() and "step" in str(f)]
#         dirs.sort(key=os.path.getctime)
#         path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
#     # Extract the step of the checkpoint to continue from there
#     training_difference = os.path.splitext(path)[0]
#     resume_step = int(training_difference.replace("step_", ""))

progress_bar = tqdm(range(EPOCHS * len(train_dataloader)), disable=not accelerator.is_main_process)

model.train()
for epoch in range(EPOCHS):
    for step, batch in enumerate(train_dataloader, start=1):
        # if args.resume_from_checkpoint:
        #     if RESUME_STEP is not None and step < RESUME_STEP:
        #         continue

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

        if step % 1000 == 0:
            if accelerator.is_main_process:
                model.push_to_hub("code-model", commit_message=f"step {step}") 