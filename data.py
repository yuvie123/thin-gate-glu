"""
Data for Experiment B: FineWeb, already tokenized with the GPT-2 tokenizer.

Source: the Hugging Face dataset `kjj0/fineweb10B-gpt2` (the shards used by the
modded-nanogpt project). Each shard is ~200 MB and holds 100M tokens as uint16,
after a header of 256 int32 values (magic number 20240520, version 1, n_tokens).

Download on the GPU machine only:
    python data.py --num_train_shards 6     # 600M training tokens + the validation shard (~1.4 GB)
"""

import argparse
import os

import numpy as np
import torch

REPO_ID = "kjj0/fineweb10B-gpt2"
HEADER_BYTES = 256 * 4
MAGIC = 20240520


def download(data_dir, num_train_shards):
    from huggingface_hub import hf_hub_download
    names = ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, num_train_shards + 1)]
    for name in names:
        if not os.path.exists(os.path.join(data_dir, name)):
            print("downloading", name)
            hf_hub_download(repo_id=REPO_ID, filename=name, repo_type="dataset", local_dir=data_dir)
    print("data ready in", data_dir)


def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == MAGIC, f"bad magic number in {path}"
    assert header[1] == 1, f"unsupported shard version in {path}"
    n_tokens = int(header[2])
    # memmap: the file stays on disk and only the pages we index are read into RAM
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES, shape=(n_tokens,))


class TokenData:
    """Random-window sampler over token shards.

    The sequence of batches depends only on `seed`, so every arm of the
    experiment that uses the same seed sees exactly the same data in the same order.
    """

    def __init__(self, data_dir, seq_len, seed, num_train_shards=None, synthetic_vocab=None):
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        if synthetic_vocab is not None:
            # --smoke mode: random tokens, no files needed (for shape/typo checks only)
            fake = np.random.default_rng(0).integers(0, synthetic_vocab, size=200_000).astype(np.uint16)
            self.train, self.val = [fake], fake[:50_000]
            return
        files = sorted(f for f in os.listdir(data_dir) if f.startswith("fineweb_train_"))
        assert files, f"no training shards in {data_dir}; run `python data.py` first"
        if num_train_shards:
            files = files[:num_train_shards]
        self.train = [load_shard(os.path.join(data_dir, f)) for f in files]
        self.val = load_shard(os.path.join(data_dir, "fineweb_val_000000.bin"))

    @property
    def train_tokens_available(self):
        return sum(len(s) for s in self.train)

    def _windows(self, tokens, starts):
        T = self.seq_len
        buf = np.stack([np.asarray(tokens[s:s + T + 1]) for s in starts]).astype(np.int64)
        buf = torch.from_numpy(buf)
        return buf[:, :-1], buf[:, 1:]

    def train_batch(self, batch_size):
        # One (shard, start) draw per sequence, so the stream of training sequences is identical
        # for every arm and does not depend on --micro_bs.
        T, rows = self.seq_len, []
        for _ in range(batch_size):
            shard = self.train[self.rng.integers(len(self.train))]
            s = self.rng.integers(0, len(shard) - T - 1)
            rows.append(np.asarray(shard[s:s + T + 1]))
        buf = torch.from_numpy(np.stack(rows).astype(np.int64))
        return buf[:, :-1], buf[:, 1:]

    def val_batches(self, batch_size, num_batches):
        """The same fixed validation windows every time (non-overlapping, from the start of the val shard)."""
        T = self.seq_len
        for b in range(num_batches):
            starts = [(b * batch_size + i) * T for i in range(batch_size)]
            if starts[-1] + T + 1 > len(self.val):
                return
            yield self._windows(self.val, starts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/fineweb10B")
    ap.add_argument("--num_train_shards", type=int, default=6, help="each shard = 100M tokens, ~200 MB")
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    download(args.data_dir, args.num_train_shards)
