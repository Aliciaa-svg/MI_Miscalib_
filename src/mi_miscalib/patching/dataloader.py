from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from mi_miscalib.data.load_bucket import ConfidenceRecord
from mi_miscalib.models.adapter import apply_chat_template


class TSLDDataset(Dataset):
    def __init__(self, records: list[ConfidenceRecord], tokenizer: object) -> None:
        self.items: list[tuple[str, str, list[int]]] = []
        self.skipped = 0
        self.n_length_mismatch = 0

        for rec in records:
            if not rec.clean_conversation or not rec.corrupt_conversation:
                self.skipped += 1
                continue
            try:
                clean_str = apply_chat_template(tokenizer, rec.clean_conversation)
                corrupt_str = apply_chat_template(tokenizer, rec.corrupt_conversation)
            except Exception:
                self.skipped += 1
                continue
            if len(tokenizer.encode(clean_str)) != len(tokenizer.encode(corrupt_str)):
                self.n_length_mismatch += 1
                self.skipped += 1
                continue
            self.items.append((clean_str, corrupt_str, [0, 0]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, str, list[int]]:
        return self.items[idx]


class RetrievalDataset(Dataset):
    def __init__(self, records: list[ConfidenceRecord], tokenizer: object) -> None:
        self.items: list[tuple[str, str, list[int]]] = []
        self.skipped = 0

        for rec in records:
            if not rec.clean_conversation or not rec.corrupt_conversation:
                self.skipped += 1
                continue
            try:
                clean_str = apply_chat_template(tokenizer, rec.clean_conversation)
                corrupt_str = apply_chat_template(tokenizer, rec.corrupt_conversation)
            except Exception:
                self.skipped += 1
                continue
            clean_target_ids = tokenizer(rec.replacement_answer, add_special_tokens=False)["input_ids"]
            corrupt_target_ids = tokenizer(rec.corrupted_answer, add_special_tokens=False)["input_ids"]
            if not clean_target_ids or not corrupt_target_ids:
                self.skipped += 1
                continue
            self.items.append((clean_str, corrupt_str, [clean_target_ids[0], corrupt_target_ids[0]]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, str, list[int]]:
        return self.items[idx]


def collate_eap(
    batch: list[tuple[str, str, list[int]]],
) -> tuple[list[str], list[str], tuple]:
    clean, corrupted, labels = zip(*batch)
    return list(clean), list(corrupted), labels


def _make_dataloader(dataset: Dataset, n_records: int, batch_size: int) -> DataLoader:
    if dataset.skipped > 0:
        pct = 100 * dataset.skipped / max(n_records, 1)
        mismatch = getattr(dataset, "n_length_mismatch", 0)
        print(
            f"[dataloader] Skipped {dataset.skipped}/{n_records} records ({pct:.1f}%), "
            f"of which {mismatch} have clean/corrupted length mismatch."
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_eap)


def make_eap_dataloader(
    records: list[ConfidenceRecord],
    tokenizer: object,
    batch_size: int = 1,
) -> DataLoader:
    return _make_dataloader(TSLDDataset(records, tokenizer), len(records), batch_size)


def make_retrieval_dataloader(
    records: list[ConfidenceRecord],
    tokenizer: object,
    batch_size: int = 1,
) -> DataLoader:
    return _make_dataloader(RetrievalDataset(records, tokenizer), len(records), batch_size)
