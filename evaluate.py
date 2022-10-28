import copy
import torch
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from itertools import chain

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, default_data_collator

NUM_PROC = 16
SEQ_LEN = 2048
seed = 42
BATCH_SIZE = 4
max_eval_steps = -1

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("conceptofmind/code-model-final-350")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

load_eval_dataset = load_dataset('conceptofmind/code-valid-dedup')

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

tokenized_eval_dataset = load_eval_dataset.map(
    tokenize, 
    batched = True, 
    num_proc = NUM_PROC, 
    remove_columns = 'content'
    )

pytorch_eval_dataset = tokenized_eval_dataset.with_format('torch')

eval_dataloader = DataLoader(
    pytorch_eval_dataset['train'], 
    shuffle = False, 
    drop_last=True, 
    collate_fn = default_data_collator, 
    batch_size = BATCH_SIZE
    )

# Setup Accelerator
accelerator = Accelerator()

# Parse configuration

set_seed(seed)

# Prepare everything with our `accelerator`.
model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

model.eval()
losses = []
for step, batch in enumerate(eval_dataloader):
    with torch.no_grad():
        outputs = model(**batch)
        print(outputs.loss)
    loss = outputs.loss.repeat(BATCH_SIZE)
    losses.append(accelerator.gather(loss))

    if max_eval_steps > 0 and step >= max_eval_steps:
        break
loss = torch.mean(torch.cat(losses))
try:
    perplexity = torch.exp(loss)
except OverflowError:
    perplexity = float("inf")

# Evaluate and save the last checkpoint
print("Evaluating and saving model after training")
# eval_loss, perplexity = evaluate()
print(f"loss/eval: {loss.item()}, perplexity: {perplexity.item()}")