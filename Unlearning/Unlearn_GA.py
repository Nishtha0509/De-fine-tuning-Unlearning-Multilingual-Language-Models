import logging
import os
import time
import torch
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Suppress verbose logs
import transformers
from datasets import logging as ds_logging, disable_progress_bar
transformers.logging.set_verbosity_error()
ds_logging.set_verbosity_error()
disable_progress_bar()

# Enhanced logger setup
def setup_logger(log_file: str = 'unlearning.log') -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s'))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

logger = setup_logger()

@dataclass
class ModelConfig:
    name: str
    path: str
    adapter_base: Optional[str] = None

defaults = {
    'languages': ['en'],
    'datasets': {'en': 'eng_forget01.json', 'en_retain': 'eng_retain99.json'},
    'batch_size': 4,
    'max_length': 512,
    'ga_lr': 8e-6,
    'ga_iters': 10,
    'threshold': 2.0
}

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cuda'
def load_model(cfg: ModelConfig):
    start = time.time()
    try:
        model_load_path = cfg.adapter_base or cfg.path
        logger.debug(f"Loading model from {model_load_path}")
        model = AutoModelForCausalLM.from_pretrained(model_load_path)
        tokenizer = AutoTokenizer.from_pretrained(model_load_path)
        #DEVICE = 'cuda'
        model = model.half()
        model.to(DEVICE)
        logger.info(f"DEVICE = {DEVICE}, model parameters on {next(model.parameters()).device}")
        if torch.cuda.is_available():
            logger.info(
              f"CUDA Mem Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB; "
              f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB"
            )
        logger.info(f"Model '{cfg.name}' loaded in {time.time()-start:.2f}s; config: {type(model).__name__}")
        return model, tokenizer
    except Exception:
        logger.exception("Failed to load model or tokenizer")
        raise

def prepare_loader(tokenizer, path, batch_size, max_length):
    try:
        logger.debug(f"Loading dataset from {path}")
        ds = load_dataset('json', data_files=path, split='train')
        logger.info(f"Loaded dataset '{os.path.basename(path)}' ({len(ds)} samples)")
    except Exception:
        logger.exception(f"Failed to load dataset {path}")
        raise

    def tokenize(batch):
        texts = batch.get('text') or [f"Q:{q} A:{a}" for q, a in zip(batch['question'], batch['answer'])]
        out = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
        out['labels'] = out['input_ids'].copy()
        return out

    start_map = time.time()
    tok = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    logger.info(f"Tokenized {len(tok)} samples in {time.time()-start_map:.2f}s for '{os.path.basename(path)}'")
    tok.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    loader = DataLoader(tok, batch_size=batch_size)
    logger.debug(f"DataLoader with {len(loader)} batches ready")
    return loader

# Compute average loss with per-batch timing logs
def avg_loss(model, loader):
    logger.info(f"Starting avg_loss on {len(loader)} batches")
    model.eval()
    total, count = 0.0, 0
    try:
        with torch.no_grad():                                  # <<< ADD THIS
            for i, batch in enumerate(loader, 1):
                batch_start = time.time()
                outputs = model(
                    input_ids=batch['input_ids'].to(DEVICE),
                    attention_mask=batch['attention_mask'].to(DEVICE),
                    labels=batch['labels'].to(DEVICE),
                )
                loss_val = outputs.loss.item()
                total += loss_val
                count += 1
                batch_time = time.time() - batch_start
                if i % 50 == 0 or i == len(loader):
                    logger.debug(f"Batch {i}/{len(loader)}: loss={loss_val:.4f}, time={batch_time:.2f}s")
    except Exception:
        logger.exception("Error during avg_loss loop")
    finally:
        model.train()
    avg = total / count if count else float('inf')
    logger.info(f"Computed avg_loss: {avg:.4f} over {count} batches")
    return avg


def gradient_ascent(model, loader, lr, iters):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    logger.info(f"Gradient Ascent start: lr={lr}, its={iters}")
    for it in range(iters):
        start_it = time.time()
        for batch in loader:
            outputs = model(
                input_ids=batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
                labels=batch['labels'].to(DEVICE)
            )
            (-outputs.loss).backward()
            optimizer.step()
            optimizer.zero_grad()
        logger.debug(f"GA iteration {it+1} took {time.time()-start_it:.2f}s")
    logger.info("Gradient Ascent complete")
    return model

def unlearn_model(cfg: ModelConfig, defaults: dict):
    logger.info(f"=== Unlearning '{cfg.name}' ===")
    try:
        model, tokenizer = load_model(cfg)
        lang = defaults['languages'][0]
        forget = prepare_loader(tokenizer, defaults['datasets'][lang], defaults['batch_size'], defaults['max_length'])
        retain = prepare_loader(tokenizer, defaults['datasets'][f"{lang}_retain"], defaults['batch_size'], defaults['max_length'])

        logger.info("Calculating initial retain loss...")
        init_ret = avg_loss(model, retain)

        logger.info("Calculating forget loss before GA...")
        bef = avg_loss(model, forget)

        model = gradient_ascent(model, forget, defaults['ga_lr'], defaults['ga_iters'])

        logger.info("Calculating forget loss after GA...")
        aft = avg_loss(model, forget)

        logger.info("Calculating final retain loss...")
        fin_ret = avg_loss(model, retain)

        logger.info(f"Losses -> retain_init: {init_ret:.4f}, forget_before: {bef:.4f}, forget_after: {aft:.4f}, retain_final: {fin_ret:.4f}")

        out = f"./unlearned/{cfg.name}"
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
    except Exception:
        logger.exception(f"Unlearning failed for {cfg.name}")
        raise

if __name__ == '__main__':
    start = time.time()
    print(torch.cuda.is_available(), torch.version.cuda)
    configs = [ModelConfig(name='Gemma', path='/data/courses/2025/class_cse576spring2025_vgupt140/Axolotls/FineTuning/TOFU_Gemma-3-4B-it/Full_TOFU_Gemma-3-4B-it_ENG'), ModelConfig(name='Llama', path='/data/courses/2025/class_cse576spring2025_vgupt140/Axolotls/FineTuning/TOFU_Llamas-3.2-3B/Full_TOFU_Llamas-3.2-3B_ENG'), ModelConfig(name='Qwen', path='/data/courses/2025/class_cse576spring2025_vgupt140/Axolotls/FineTuning/TOFU_Qwen2.5-7B-Instruct/Full_TOFU_Qwen2.5-7B-Instruct_ENG')]
    for c in configs:
        unlearn_model(c, defaults)
    logger.info(f"Total time: {time.time()-start:.1f}s")