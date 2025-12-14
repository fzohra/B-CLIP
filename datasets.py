import argparse
from collections import defaultdict, Counter
import json
import re
from itertools import groupby
import os
import pickle
import zipfile
import pandas as pd
import csv as _csv

import numpy as np
from PIL import Image, ImageFile

import torch
from torchvision import transforms
from torchvision import datasets as t_datasets

import utils
from pathlib import Path


ImageFile.LOAD_TRUNCATED_IMAGES = True

import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def deduplicate_sentence_fast(text, max_substring_len=20):
    """
    Ultra-fast deduplication using regex for substring patterns.
    
    Handles both word-level and character-level repetitions (like Thai script).
    
    Examples:
        >>> deduplicate_sentence_fast("hello hello world world")
        "hello world"
        >>> deduplicate_sentence_fast("บริการบริการบริการ")
        "บริการ"
    """
    if not text or len(text) < 2:
        return text
    
    # Use regex to remove repeated substrings
    # Pattern: (.+?)\1+ matches any substring repeated 2+ times consecutively
    # (.+?) = non-greedy capture group (the pattern to find)
    # \1+ = that same pattern repeated 1+ more times (total 2+)
    
    # Apply character-level regex only for longer sequences (with limits)
    if len(text) > 100:
        # Single pass with limited replacements to avoid hangs
        for min_len in [10, 5, 3]:
            pattern = rf'(.{{{min_len},{max_substring_len}}}?)\1\1+'  # 3+ reps minimum
            text = re.sub(pattern, r'\1', text, count=10)  # Limit to 10 replacements max
    
    # Also handle word-level deduplication with groupby (very fast)
    words = text.split()
    if len(words) >= 2:
        deduped_words = [word for word, _ in groupby(words)]
        text = ' '.join(deduped_words)
    
    return text.strip()

def yfcc_loader(root, index):
    index = format(index, "0>8d")
    repo = index[:2]
    z = index[2: 5]
    file_img = index[5:] + '.jpg'
    path_zip = os.path.join(root, 'images', repo, z) + '.zip'
    with zipfile.ZipFile(path_zip, 'r') as myzip:
        img = Image.open(myzip.open(file_img))
    return img.convert('RGB')


def _parse_mixture(argval):
    """
    Turn 'cc3m+datacomp' into (['cc3m', 'datacomp'], None)
          'cc3m+datacomp@0.2' → (['cc3m', 'datacomp'], [0.2, 0.8])
          'cc3m+datacomp+cc12m@0.1,0.5,0.4' → ...
    """
    if "@" in argval:
        ds_part, prob_part = argval.split("@", 1)
        probs = [float(p) for p in prob_part.split(",")]
    else:
        ds_part, probs = argval, None
    datasets = ds_part.split("+")
    return datasets, probs

import random

class MixedDataset(torch.utils.data.Dataset):
    """
    Sample from N constituent datasets with user-defined probabilities.

    Args
    ----
    datasets : list[torch.utils.data.Dataset]
        The individual datasets (must yield the *same* tuple structure).
    probs : list[float]  (sums to 1.0)
        Sampling probabilities, e.g. [0.2, 0.8] for 20 % CC3M, 80 % DataComp.
    """
    def __init__(self, datasets, probs):
        assert len(datasets) == len(probs), "One prob per dataset"
        self.datasets = datasets
        self.cum_probs = np.cumsum(np.array(probs, dtype=np.float64))
        self.max_len = max(len(d) for d in datasets)  # length used by sampler
        print(f"MixedDataset: datasets={datasets}")
        print(f"{len(datasets)} datasets, max_len={self.max_len}")
        print(f"length={len(datasets[0])}, {len(datasets[1])}")
        print(f"probs={probs}")
        print(f"cum_probs={self.cum_probs}")
    def __len__(self):
        # Makes sure one *epoch* walks the longest dataset once.
        return self.max_len

    def __getitem__(self, idx):
        r = random.random()
        chosen = np.searchsorted(self.cum_probs, r)
        ds = self.datasets[chosen]
        # Re-index into the chosen dataset to avoid IndexError.
        sample = ds[idx % len(ds)]
        return sample


class ImageCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata):
        self.dataset = dataset
        self.root = root
        if self.dataset == 'yfcc15m':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        elif self.dataset == 'coco':
            samples = defaultdict(list)
            with open(metadata) as f:
                annotations = json.load(f)['annotations']
            for ann in annotations:
                samples[ann['image_id']].append(ann['caption'])
            self.samples = [(k, v) for k, v in samples.items()]
        elif self.dataset == 'cc12m' or self.dataset == 'cc3m':
            if metadata.endswith('.npy'):
                self.samples = np.load(metadata, allow_pickle=True)
            elif metadata.endswith('.csv'):
                # read the entire two-column CSV (very fast C engine)
                df = pd.read_csv(metadata, sep='\t', engine='c')
                # group captions by image_path, so captions becomes a list
                grouped = (
                    df
                    .groupby("image_path", sort=False)["caption"]
                    .agg(list)
                    .reset_index()
                )
                # now self.samples is exactly [(img_path, [cap1,cap2,…]), …]
                self.samples = list(grouped.itertuples(index=False, name=None))
        # ───────────── DataComp ─────────────
        elif self.dataset == "datacomp":
            if metadata.endswith(".npy"):
                self.samples = np.load(metadata, allow_pickle=True)
            elif metadata.endswith(".csv"):
                # DataComp metadata is **comma-separated**
                df = pd.read_csv(
                    metadata,
                    sep=",",
                    engine="c",
                    quoting=_csv.QUOTE_MINIMAL,  # keep commas inside captions
                )
                grouped = (
                    df.groupby("image_path", sort=False)["caption"]
                    .agg(list)
                    .reset_index()
                )
                self.samples = list(grouped.itertuples(index=False, name=None))
        # ───────────── ShareGPT4V CSVs (LAION & long-COCO) ─────────────
        elif self.dataset in {"laion", "coco_long"}:
            # identical layout to DataComp: comma-sep with quoted captions
            df = pd.read_csv(
                metadata,
                sep=",",
                engine="c",
                quoting=_csv.QUOTE_MINIMAL,
            )
            grouped = (
                df.groupby("image_path", sort=False)["caption"]
                  .agg(list)
                  .reset_index()
            )
            self.samples = list(grouped.itertuples(index=False, name=None))
        elif self.dataset == 'redcaps':
            with open(metadata) as f:
                annotations = json.load(f)
            self.samples = [(ann['image_id'], ann['subreddit'], ann['caption']) for ann in annotations]

    def get_raw_item(self, i):
        if self.dataset == 'yfcc15m':
            index, title, desc = self.samples[i]
            caption = np.random.choice([title, desc])
            img = yfcc_loader(self.root, index)
        elif self.dataset == 'coco':
            index, captions = self.samples[i]
            path = os.path.join(self.root, 'train2017', '{:012d}.jpg'.format(index))
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'cc3m' or self.dataset == 'datacomp':
            path, captions = self.samples[i]
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'cc12m':
            path, captions = self.samples[i]
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset in {'laion', 'coco_long'}:
            path, captions = self.samples[i]
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset == 'redcaps':
            image_id, subreddit, caption = self.samples[i]
            path = os.path.join(self.root, subreddit, f"{image_id}.jpg")
            img = pil_loader(path)

        return img, caption

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class ImageCaptionDatasetCLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None, tokenizer=None, use_text_concepts=False, use_text_tokens=False):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.tokenizer = tokenizer
        self.use_text_concepts = use_text_concepts
        self.use_text_tokens = use_text_tokens
        self.spacy_concepts_extractor = spacy_concepts_extractor()

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            image = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            if self.use_text_concepts:
                concepts = self.spacy_concepts_extractor.concepts_minimal(caption)
                concepts = self.tokenizer(concepts)
            elif self.use_text_tokens:
                concepts = self.spacy_concepts_extractor.concepts_words(caption)
                concepts = self.tokenizer(concepts)
            caption = self.tokenizer(caption)

        if self.use_text_concepts or self.use_text_tokens:
            return image, caption, concepts
        else:
            return image, caption

class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(images)
        self.labels = np.load(labels)

    def __getitem__(self, index):
        img = pil_loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

import subprocess

def automatic_download_dataset(name, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if name == "food101":
        from torchvision.datasets import Food101
        print(f"[INFO] Downloading Food101 via torchvision to {root}...")
        Food101(root=root, download=True)
        images_dir = root / "images"
        meta_dir = root / "meta"

        def prepare_split(split):
            target_dir = root / split
            target_dir.mkdir(parents=True, exist_ok=True)

            with open(meta_dir / f"{split}.txt") as f:
                lines = f.read().splitlines()
            for line in lines:
                class_name, img_base = line.split("/")
                img_name = f"{img_base}.jpg"
                class_dir = target_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                src = images_dir / class_name / img_name
                dst = class_dir / img_name
                if not dst.exists():
                    os.symlink(src.resolve(), dst)

        print("[INFO] Organizing Food101 into ImageFolder format...")
        prepare_split("train")
        prepare_split("test")
    elif name == "dtd":
        from torchvision.datasets import DTD
        print("[INFO] Downloading DTD via torchvision...")
        DTD(root=root, download=True, split="train")
    elif name == "flowers":
        from torchvision.datasets import Flowers102
        print("[INFO] Downloading Flowers102 via torchvision...")
        Flowers102(root=root, download=True)
    elif name == "pets":
        from torchvision.datasets import OxfordIIITPet
        print("[INFO] Downloading OxfordIIITPet via torchvision...")
        OxfordIIITPet(root=root, download=True)
    elif name == "caltech101":
        from torchvision.datasets import Caltech101
        print("[INFO] Downloading Caltech101 via torchvision...")
        Caltech101(root=root, download=True)
    else:
        raise NotImplementedError(f"No automatic download method defined for dataset: {name}")

def ensure_data_available(entry, name):
    root = Path(entry["path"])
    if not root.exists():
        print(f"[INFO] Dataset path '{root}' not found for '{name}'.")
        
        try:
            automatic_download_dataset(name, root)
        except NotImplementedError as e:
            raise FileNotFoundError(
                f"[ERROR] Dataset '{name}' not found and automatic download not supported.\n{e}"
            )
        
def get_downstream_dataset(catalog, name, is_train, transform):
    entry = catalog[name]
    root = entry['path']
    ensure_data_available(entry, name)
    if entry['type'] == 'imagefolder':
        dataset = t_datasets.ImageFolder(os.path.join(root, entry['train'] if is_train else entry['test']),
            transform=transform)
    elif entry['type'] == 'special':
        if name == 'cifar10':
            dataset = t_datasets.CIFAR10(root, train=is_train,
                transform=transform, download=True)
        elif name == 'cifar100':
            dataset = t_datasets.CIFAR100(root, train=is_train,
                transform=transform, download=True)
        elif name == 'stl10':
            dataset = t_datasets.STL10(root, split='train' if is_train else 'test',
                transform=transform, download=True)
        elif name == 'mnist':
            dataset = t_datasets.MNIST(root, train=is_train,
                transform=transform, download=True)
    elif entry['type'] == 'filelist':
        path = entry['train'] if is_train else entry['test']
        val_images = os.path.join(root, path + '_images.npy')
        val_labels = os.path.join(root, path + '_labels.npy')
        if name == 'clevr_counts':
            target_transform = lambda x: ['count_10', 'count_3', 'count_4', 'count_5', 'count_6', 'count_7', 'count_8', 'count_9'].index(x)
        else:
            target_transform = None
        dataset = FileListDataset(val_images, val_labels, transform, target_transform)
    else:
        raise Exception('Unknown dataset')

    return dataset

class spacy_concepts_extractor():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        # Actions like "leaning against"
        self.act_matcher = Matcher(self.nlp.vocab)
        self.act_matcher.add("VERB_ADP", [[{"POS": "VERB"}, {"POS": "ADP"}]])

        # (Optional) NP extension rule — keep if you still want NP+PP like "a cat on the right"
        self.np_side = Matcher(self.nlp.vocab)
        self.np_side.add("NP_SIDE", [[
            {"POS":"DET","OP":"?"}, {"POS":{"IN":["ADJ","NOUN","PROPN"]},"OP":"+"},
            {"POS":"ADP"}, {"LOWER":"the","OP":"?"}, {"LOWER":{"IN":["left","right","top","bottom","front","back","middle","center"]}}, {"LOWER":"of","OP":"?"}
        ]])

        self.KEEP_POS = {"NOUN","PROPN","ADJ","VERB","ADP","ADV","NUM","SYM"}

        self.SPATIAL_HEADS = {
            "left","right","top","bottom","front","back",
            "inside","outside","above","below","under","over",
            "middle","center","centered","frontside","backside","near","nearby","far"
        }
        self.merge_pairs = (("VERB","ADP"), ("ADV","ADP"))
        self.keep_single_adp = False
        self.blacklist = ["the", "the image", "image"]

    def _spatial_phrases(self, doc, *, lemmatize=False):
        """Find ADP [DET]? SPATIAL_HEAD ['of']? and ADV+ADP ('next to')."""
        out = []
        i, n = 0, len(doc)

        def norm(tok):
            return (tok.lemma_ if lemmatize else tok.text).lower().strip()

        while i < n:
            t = doc[i]
            # 1) 'next to'
            if i+1 < n and t.pos_ == "ADV" and doc[i+1].pos_ == "ADP" and t.lower_ == "next" and doc[i+1].lower_ == "to":
                out.append(f"{norm(t)} {norm(doc[i+1])}")
                i += 2
                continue

            # 2) ADP [DET]? SPATIAL_HEAD ['of']?
            if t.pos_ == "ADP" and i+1 < n:
                j = i + 1
                if j < n and doc[j].pos_ == "DET":
                    j += 1
                if j < n and doc[j].lower_ in self.SPATIAL_HEADS:
                    j += 1
                    if j < n and doc[j].pos_ == "ADP" and doc[j].lower_ == "of":
                        j += 1
                    phrase = " ".join(norm(tok) for tok in doc[i:j])
                    out.append(phrase)
                    i = j
                    continue

            i += 1
        # dedupe preserving order
        seen, rels = set(), []
        for r in out:
            if r not in seen:
                seen.add(r); rels.append(r)
        return rels

    def concepts_minimal(self, text, *, lemmatize=False, include_np_side=True):
        doc = self.nlp(text) # doc is a Doc object

        # base NPs
        spans = list(doc.noun_chunks)

        # add extended NP spans for "... on the left/right (of ...)" if you still want NP+PP chunks
        if include_np_side:
            spans += [doc[s:e] for _, s, e in self.np_side(doc)] # doc[s:e] is a Span object

        # keep the longest overlapping spans
        noun_phrases = [s.text.strip() for s in filter_spans(spans)] # s is a Span object

        # actions like "leaning against"
        actions = [doc[s:e].text for _, s, e in self.act_matcher(doc)]

        # spatial relations via the same logic used in concept_words
        relations = self._spatial_phrases(doc, lemmatize=lemmatize) # relations is a list of strings

        # merge + dedupe preserving order
        seen, out = set(), []
        for x in noun_phrases + actions + relations:
            if x not in seen:
                seen.add(x)
                if x not in self.blacklist:
                    out.append(x)
        return out

    def concepts_words(
        self,
        text, *,
        unique=True,
        lemmatize=False,
    ):
        doc = self.nlp(text) # doc is a Doc object
        words, seen = [], set()
        i, n = 0, len(doc)

        def norm(tok):
            return (tok.lemma_ if lemmatize else tok.text).lower().strip()

        def push(s):
            if not unique or s not in seen:
                seen.add(s)
                if s not in self.blacklist:
                    words.append(s)

        while i < n:
            t = doc[i]
            if t.is_space or t.is_punct:
                i += 1; continue

            # --- 1) Merge spatial PP: ADP [DET]? SPATIAL_HEAD ['of']?
            if t.pos_ == "ADP" and i+1 < n:
                j = i + 1
                # optional DET (e.g., "the")
                if j < n and doc[j].pos_ == "DET":
                    j += 1
                # spatial head (adj/noun/adv/etc.)
                if j < n and doc[j].lower_ in self.SPATIAL_HEADS:
                    j += 1
                    # optional trailing 'of' (captures "to the right of")
                    if j < n and doc[j].lower_ == "of" and doc[j].pos_ == "ADP":
                        j += 1
                    phrase = " ".join(norm(tok) for tok in doc[i:j])
                    push(phrase)
                    i = j
                    continue

            # --- 2) Merge configured POS bigrams (e.g., VERB+ADP => "leaning against")
            if i+1 < n:
                t2 = doc[i+1]
                if any((t.pos_ == a and t2.pos_ == b) for (a,b) in self.merge_pairs): # (("VERB","ADP"), ("ADV","ADP"))
                    push(f"{norm(t)} {norm(t2)}")
                    i += 2
                    continue

            # --- 3) Single-token fallback (respect keep_pos and optional ADP drop)
            if t.pos_ not in self.KEEP_POS:
                i += 1; continue
            if (t.pos_ == "ADP") and not self.keep_single_adp: # False
                i += 1; continue

            w = norm(t)
            if w:
                push(w)
            i += 1

        return words

class sharegpt4v_train_dataset(torch.utils.data.Dataset):
    def __init__(self, metadata, root, transform, tokenizer, use_text_concepts, use_text_tokens, caption_type='original', caption_len=0, num_positive_samples=0, load_concepts=False, use_caption=True, use_multi_scale_caption=False):
        self.metadata = metadata
        self.root = root
        self.transform = transform
        self.total_len = 1000 #reserved for validation
        self.tokenizer = tokenizer
        self.use_text_concepts = use_text_concepts
        self.use_text_tokens = use_text_tokens
        self.caption_type = caption_type
        self.caption_len = caption_len
        self.num_positive_samples = num_positive_samples  # Number of positive sentences to sample
        self.load_concepts = load_concepts
        self.use_caption = use_caption
        self.use_multi_scale_caption = use_multi_scale_caption
        with open(self.metadata, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[self.total_len:]
        print('share4v_train_dataset loaded, total length:', len(self.json_data))
        print(f'num_positive_samples: {self.num_positive_samples}')

        self.spacy_concepts_extractor = spacy_concepts_extractor()

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        image_path = Path(self.root) / self.json_data[index]['image']
        image = pil_loader(image_path)
        image_tensor = self.transform(image)
        
        if not self.load_concepts:
            caption = deduplicate_sentence_fast(caption)
        
        sentences = [s.strip() for s in caption.split(". ") if s.strip()]
        num_sentences = len(sentences)
                
        if self.num_positive_samples > 0:
            # Sample num_positive_samples sentences (with replacement if needed)
            num_to_sample = min(self.num_positive_samples, num_sentences)
            if num_to_sample < self.num_positive_samples:
                sampled_sentences = sentences # if not enough to sample, use all sentences
            else:
                # Sample without replacement
                sampled_sentences = random.sample(sentences, num_to_sample)
            
            # Tokenize each sentence separately
            sentence_ids_list = []
            for sent in sampled_sentences:
                sentence_ids = self.tokenizer(sent)
                sentence_ids_list.append(sentence_ids)
            
            # Stack into [num_positive_samples, seq_len] tensor and make contiguous
            sentence_ids = torch.stack(sentence_ids_list, dim=0).contiguous().clone() # num_positive_samples, seq_len
            sampled_caption = '. '.join(sampled_sentences)
            if self.use_caption:
                caption_ids = self.tokenizer(sampled_caption)
                caption_ids = caption_ids.unsqueeze(0) # 1, seq_len
                caption_ids = torch.cat([caption_ids, sentence_ids], dim=0) # num_positive_samples+1, seq_len
            else:
                caption_ids = sentence_ids
            if self.use_text_concepts:
                if self.load_concepts:
                    concepts = self.json_data[index]['concepts_minimal']
                else:
                    concepts = self.spacy_concepts_extractor.concepts_minimal(sampled_caption)
                concepts_ids = self.tokenizer(concepts)
                return image_tensor, caption_ids, concepts_ids # [3, H, W], [num_positive_samples+1, seq_len], [K, seq_len]
            elif self.use_text_tokens:
                if self.load_concepts:
                    concepts = self.json_data[index]['concepts_words']
                else:
                    concepts = self.spacy_concepts_extractor.concepts_words(sampled_caption)
                concepts_ids = self.tokenizer(concepts)
                return image_tensor, caption_ids, concepts_ids
            elif self.use_multi_scale_caption:
                concepts = self.spacy_concepts_extractor.concepts_minimal(sampled_caption)
                concepts_ids = self.tokenizer(concepts)
                caption_ids = torch.cat([caption_ids, concepts_ids], dim=0) # num_positive_samples+1+K, seq_len
                return image_tensor, caption_ids  # [3, H, W], [num_positive_samples+1+K, seq_len]
            else:
                return image_tensor, caption_ids  # [3, H, W], [num_positive_samples+1, seq_len]
        
        else:
            if self.caption_len == 0:
                num_to_select = random.randint(1, num_sentences)
                caption = '. '.join(sentences[:num_to_select])
            elif self.caption_len == -1:
                caption = ". ".join(sentences)
            else:            
                caption = '. '.join(random.sample(sentences, min(self.caption_len, len(sentences))))
                
            if self.caption_type == 'concepts':            
                concepts = self.spacy_concepts_extractor.concepts_minimal(caption)
                caption_concepts = " ".join(concepts)
                caption_ids = self.tokenizer(caption_concepts)
            else:
                caption_ids = self.tokenizer(caption)
                        
            if self.use_text_concepts:
                if self.load_concepts:
                    concepts = self.json_data[index]['concepts_minimal']
                else:
                    concepts = self.spacy_concepts_extractor.concepts_minimal(caption)
                concepts_ids = self.tokenizer(concepts)
                return image_tensor, caption_ids, concepts_ids # [3, H, W], [L], [K, L]
            elif self.use_text_tokens:
                if self.load_concepts:
                    concepts = self.json_data[index]['concepts_words']
                else:
                    concepts = self.spacy_concepts_extractor.concepts_words(caption)
                concepts_ids = self.tokenizer(concepts)
                return image_tensor, caption_ids, concepts_ids
            else:
                return image_tensor, caption_ids

class sharegpt4v_val_dataset(torch.utils.data.Dataset): #need to get this later
    def __init__(self, metadata, root, transform, tokenizer, use_text_concepts, use_text_tokens, caption_type='original', caption_len=0, num_positive_samples=0):
        self.metadata = metadata
        self.root = root
        self.transform = transform
        self.total_len = 1000
        self.tokenizer = tokenizer
        self.use_text_concepts = use_text_concepts
        self.use_text_tokens = use_text_tokens
        self.caption_type = caption_type
        self.caption_len = caption_len
        self.num_positive_samples = num_positive_samples
        
        with open(self.metadata, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[:self.total_len]

        # Only initialize spacy when actually needed (avoid NFS contention in multi-node distributed mode)
        if self.use_text_concepts or self.use_text_tokens:
            self.spacy_concepts_extractor = spacy_concepts_extractor()
        else:
            self.spacy_concepts_extractor = None

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        # Fast text-level deduplication before tokenization
        caption = deduplicate_sentence_fast(caption)
        image_path = Path(self.root) / self.json_data[index]['image']
        image = pil_loader(image_path)
        image_tensor = self.transform(image)

        caption_ids = self.tokenizer(caption)
        return image_tensor, caption_ids

class urban1k_val_dataset(torch.utils.data.Dataset):
    def __init__(self, image_root, caption_root, transform, tokenizer, use_text_concepts, use_text_tokens, total_len=None):
        self.image_root = image_root
        self.caption_root = caption_root
        self.transform = transform
        self.tokenizer = tokenizer
        self.use_text_concepts = use_text_concepts
        self.use_text_tokens = use_text_tokens
        
        # Get list of caption files and corresponding image files
        self.caption_files = sorted(os.listdir(caption_root))
        self.image_files = []
        
        # Filter to only include caption files that have corresponding image files
        for caption_file in self.caption_files:
            image_name = caption_file[:-4] + '.jpg'  # Replace .txt with .jpg
            image_path = os.path.join(image_root, image_name)
            if os.path.exists(image_path):
                self.image_files.append(image_name)
            else:
                # If image doesn't exist, remove the caption file from consideration
                self.caption_files.remove(caption_file)
        
        # Limit dataset size if specified
        if total_len is not None:
            self.total_len = min(total_len, len(self.caption_files))
            self.caption_files = self.caption_files[:self.total_len]
            self.image_files = self.image_files[:self.total_len]
        else:
            self.total_len = len(self.caption_files)
            
        print('urban1k_val_dataset loaded, total length:', self.total_len)
        
        # Initialize concepts extractor if needed
        if self.use_text_concepts or self.use_text_tokens:
            self.spacy_concepts_extractor = spacy_concepts_extractor()

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        # Get caption file and corresponding image file
        caption_file = self.caption_files[index]
        image_file = self.image_files[index]
        
        # Load caption
        caption_path = os.path.join(self.caption_root, caption_file)
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.readlines()[0].strip()
        
        # Load image
        image_path = os.path.join(self.image_root, image_file)
        image = pil_loader(image_path)
        image_tensor = self.transform(image)
        
        caption_ids = self.tokenizer(caption)
        return image_tensor, caption_ids


class flickr30k_val_dataset(torch.utils.data.Dataset):
    def __init__(self, images_root, captions_file, transform, tokenizer, use_text_concepts, use_text_tokens, total_len=None):
        self.images_root = images_root
        self.transform = transform
        self.tokenizer = tokenizer
        self.use_text_concepts = use_text_concepts
        self.use_text_tokens = use_text_tokens
        
        # Load Flickr30K captions file
        self.samples = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    image_path = parts[0]
                    caption = parts[1]
                    # Check if image exists
                    full_image_path = os.path.join(images_root, image_path)
                    if os.path.exists(full_image_path):
                        self.samples.append((image_path, caption))
        
        # Limit dataset size if specified
        if total_len is not None:
            self.total_len = min(total_len, len(self.samples))
            self.samples = self.samples[:self.total_len]
        else:
            self.total_len = len(self.samples)
            
        print('flickr30k_val_dataset loaded, total length:', self.total_len)
        
        # Initialize concepts extractor if needed
        if self.use_text_concepts or self.use_text_tokens:
            self.spacy_concepts_extractor = spacy_concepts_extractor()

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        image_path, caption = self.samples[index]
        
        # Load image
        full_image_path = os.path.join(self.images_root, image_path)
        image = pil_loader(full_image_path)
        image_tensor = self.transform(image)
        
        # Process caption and concepts based on configuration
        if self.use_text_concepts:
            concepts = self.spacy_concepts_extractor.concepts_minimal(caption)
            concepts = self.tokenizer(concepts)
            caption = self.tokenizer(caption)
            return image_tensor, caption, concepts
        elif self.use_text_tokens:
            concepts = self.spacy_concepts_extractor.concepts_words(caption)
            concepts = self.tokenizer(concepts)
            caption = self.tokenizer(caption)
            return image_tensor, caption, concepts
        else:
            caption = self.tokenizer(caption)
            return image_tensor, caption

class flickr32k_val_dataset(torch.utils.data.Dataset):
    def __init__(self, images_root, annotations_file, transform, tokenizer, use_text_concepts, use_text_tokens, total_len=None, split="test"):
        self.images_root = images_root
        self.transform = transform
        self.tokenizer = tokenizer
        self.use_text_concepts = use_text_concepts
        self.use_text_tokens = use_text_tokens

        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.samples = []
        allowed_splits = {split} if isinstance(split, str) else set(split)
        images = data.get('images', [])
        for img_info in images:
            if allowed_splits and img_info.get('split') not in allowed_splits:
                continue
            filename = img_info.get('filename')
            if not filename:
                continue

            filepath = img_info.get('filepath', '')
            # Build candidate paths: prefer subdirectory if provided, fallback to root
            candidate_paths = []
            if filepath:
                candidate_paths.append(os.path.join(images_root, filepath, filename))
            candidate_paths.append(os.path.join(images_root, filename))

            full_image_path = None
            for candidate in candidate_paths:
                if os.path.exists(candidate):
                    full_image_path = candidate
                    break

            if full_image_path is None:
                continue

            sentences = img_info.get('sentences', [])
            captions = [s.get('raw') for s in sentences if isinstance(s, dict) and s.get('raw')]
            if not captions:
                continue

            self.samples.append((full_image_path, captions))

        if total_len is not None:
            self.total_len = min(total_len, len(self.samples))
            self.samples = self.samples[:self.total_len]
        else:
            self.total_len = len(self.samples)

        print(f'flickr32k_val_dataset loaded (split={split}), total length:', self.total_len)

        if self.use_text_concepts or self.use_text_tokens:
            self.spacy_concepts_extractor = spacy_concepts_extractor()

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        image_path, captions = self.samples[index]
        image = pil_loader(image_path)
        image_tensor = self.transform(image)

        caption = captions[0] if isinstance(captions, list) else captions

        if self.use_text_concepts:
            concepts = self.spacy_concepts_extractor.concepts_minimal(caption)
            concepts = self.tokenizer(concepts)
            caption = self.tokenizer(caption)
            return image_tensor, caption, concepts
        elif self.use_text_tokens:
            concepts = self.spacy_concepts_extractor.concepts_words(caption)
            concepts = self.tokenizer(concepts)
            caption = self.tokenizer(caption)
            return image_tensor, caption, concepts
        else:
            caption = self.tokenizer(caption)
            return image_tensor, caption

class coco_val_dataset(torch.utils.data.Dataset):
    def __init__(self, images_root, annotations_file, transform, tokenizer, use_text_concepts, use_text_tokens, total_len=None):
        self.images_root = images_root
        self.transform = transform
        self.tokenizer = tokenizer
        self.use_text_concepts = use_text_concepts
        self.use_text_tokens = use_text_tokens
        
        # Load COCO annotations
        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create image_id to filename mapping
        id_to_filename = {}
        for img in data['images']:
            id_to_filename[img['id']] = img['file_name']
        
        # Create samples from annotations - group by image_id to get all captions per image
        image_to_captions = {}
        for ann in data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            
            if image_id in id_to_filename:
                image_filename = id_to_filename[image_id]
                full_image_path = os.path.join(images_root, image_filename)
                if os.path.exists(full_image_path):
                    if image_id not in image_to_captions:
                        image_to_captions[image_id] = (image_filename, [])
                    image_to_captions[image_id][1].append(caption)
        
        # Create samples: each image with all its captions
        self.samples = []
        for image_id, (image_filename, captions) in image_to_captions.items():
            self.samples.append((image_filename, captions))
        
        # Limit dataset size if specified
        if total_len is not None:
            self.total_len = min(total_len, len(self.samples))
            self.samples = self.samples[:self.total_len]
        else:
            self.total_len = len(self.samples)
            
        print('coco_val_dataset loaded, total length:', self.total_len)
        
        # Initialize concepts extractor if needed
        if self.use_text_concepts or self.use_text_tokens:
            self.spacy_concepts_extractor = spacy_concepts_extractor()

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        image_filename, captions = self.samples[index]
        
        # Load image
        full_image_path = os.path.join(self.images_root, image_filename)
        image = pil_loader(full_image_path)
        image_tensor = self.transform(image)
        
        # For COCO, we need to handle multiple captions per image
        # For now, let's use the first caption for simplicity, but we'll handle multiple captions in evaluation
        caption = captions[0] if isinstance(captions, list) else captions
        
        # Process caption and concepts based on configuration
        if self.use_text_concepts:
            concepts = self.spacy_concepts_extractor.concepts_minimal(caption)
            concepts = self.tokenizer(concepts)
            caption = self.tokenizer(caption)
            return image_tensor, caption, concepts
        elif self.use_text_tokens:
            concepts = self.spacy_concepts_extractor.concepts_words(caption)
            concepts = self.tokenizer(concepts)
            caption = self.tokenizer(caption)
            return image_tensor, caption, concepts
        else:
            caption = self.tokenizer(caption)
            return image_tensor, caption

def get_dataset(train_transform, tokenizer, args):
    # Support mixture strings like "cc3m+datacomp@0.15"
    mix_names, mix_probs = _parse_mixture(args.dataset)
    print(f"Dataset mixture: {mix_names} (probs: {mix_probs})")
    if len(mix_names) > 1:
        # ---- 1. Map comma-separated roots / metadata to datasets -----------
        roots = [s.strip() for s in args.root.split(",")] if "," in args.root else [args.root] * len(mix_names)
        metas = [s.strip() for s in args.metadata.split(",")] if "," in args.metadata else [args.metadata] * len(mix_names)

        assert len(roots) == len(mix_names),     f"#roots ({len(roots)}) ≠ #datasets ({len(mix_names)})"
        assert len(metas) == len(mix_names),     f"#metadata ({len(metas)}) ≠ #datasets ({len(mix_names)})"

        # ---- 2. Build each constituent with its own args clone -------------
        constituents = []
        for name, root, meta in zip(mix_names, roots, metas):
            sub_args = argparse.Namespace(**vars(args))   # shallow copy of all flags
            sub_args.dataset   = name
            sub_args.root      = root
            sub_args.metadata  = meta
            constituents.append(get_dataset(train_transform, tokenizer, sub_args))

        # ---- 3. Default mixture probs = proportional to dataset length -----
        if mix_probs is None:
            lens = [len(d) for d in constituents]
            mix_probs = [l / sum(lens) for l in lens]

        return MixedDataset(constituents, mix_probs)

    if args.dataset == 'sharegpt4v':
        return sharegpt4v_train_dataset(args.metadata, args.root, train_transform, tokenizer, args.use_text_concepts, args.use_text_tokens, args.caption_type, args.caption_len, args.num_positive_samples, use_caption=args.use_caption, use_multi_scale_caption=args.use_multi_scale_caption)
    elif args.dataset == 'sharegpt4v_val':
        return sharegpt4v_val_dataset(args.metadata, args.root, train_transform, tokenizer, args.use_text_concepts, args.use_text_tokens, args.caption_type, args.caption_len, args.num_positive_samples, use_caption=args.use_caption)
    
    if args.model.startswith('CLIP'):
        return ImageCaptionDatasetCLIP(args.dataset, args.root, args.metadata, train_transform, tokenizer, args.use_text_concepts, args.use_text_tokens)