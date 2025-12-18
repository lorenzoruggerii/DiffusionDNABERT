import os
import argparse
import random
import pyBigWig

from multiprocessing import Pool, cpu_count
from functools import partial
from torch.utils import data
from tqdm import tqdm
from pyfaidx import Fasta
from datasets import Dataset

MODEL_LENGTH = 200  # it should be 12_288 but do not consider padding

def process_single_file(file_info, bigbed_folder, chrom_start, chrom_end, fasta_path, num_peaks):
    """Process a single bigbed file - designed to be called in parallel"""
    i, file = file_info
    dataset_entries = []
    
    # Load fasta in each worker process
    fasta = Fasta(fasta_path)
    
    bb = os.path.join(bigbed_folder, file)
    bb_file = pyBigWig.open(bb)
    
    for chrom in range(chrom_start, chrom_end):
        
        chrom = f"chr{chrom}"
        # Define entries
        if chrom not in bb_file.chroms():
            continue
            
        out_coords = bb_file.chroms()[chrom]
        entries = bb_file.entries(chrom, 0, out_coords, withString=False)
        entries = entries[:num_peaks]
        
        if entries is None:
            continue
        
        for (peak_start, peak_end) in entries:
            
            # Expand to model size
            peak_summit = (peak_start + peak_end) // 2
            cand_new_start = peak_summit - MODEL_LENGTH // 2
            new_start = cand_new_start + random.randint(-MODEL_LENGTH // 2, MODEL_LENGTH // 2)
            
            if new_start < 0 or new_start + MODEL_LENGTH > out_coords:
                continue
            
            new_end = new_start + MODEL_LENGTH
            ref_sequence = str(fasta[chrom][new_start:new_end])
            
            dataset_entries.append(
                {
                    "sequence": ref_sequence.upper(),
                    "label": i,
                    "chrom": chrom,
                    "peak_start": peak_start,
                    "peak_end": peak_end,
                    "start": new_start,
                    "end": new_end
                }
            )
    
    bb_file.close()
    return dataset_entries


class DatasetCreator:

    def __init__(self, bigbed_folder: str, chrom_start: int, chrom_end: int, fasta: str, outpath: str, n_workers: int = None, num_peaks: int = None):
        self.bigbed_folder = bigbed_folder
        self.chrom_start = chrom_start
        self.chrom_end = chrom_end
        self.outpath = outpath
        self.fasta_path = fasta
        self.n_workers = n_workers if n_workers is not None else 1
        self.num_peaks = num_peaks

        self.dataset = []

        # Load fasta file (main process only needs it for validation)
        self.fasta = Fasta(fasta)

    def create_dataset(self):
        
        # Create index
        bigbed_files = sorted([f for f in os.listdir(self.bigbed_folder) if f.endswith("bigBed")])
        self.index = {i: f for i, f in enumerate(bigbed_files)}
        
        print(f"Processing {len(self.index)} bigbed files using {self.n_workers} workers...")
        
        # Prepare file info for parallel processing
        file_infos = list(self.index.items())
        
        # Create partial function with fixed arguments
        process_func = partial(
            process_single_file,
            bigbed_folder=self.bigbed_folder,
            chrom_start=self.chrom_start,
            chrom_end=self.chrom_end,
            fasta_path=self.fasta_path,
            num_peaks=self.num_peaks
        )
        
        # Process files in parallel with progress bar
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, file_infos),
                total=len(file_infos),
                desc="Processing bigbed files"
            ))
        
        # Flatten results into single dataset
        print("Consolidating results...")
        for entries in tqdm(results, desc="Merging datasets"):
            self.dataset.extend(entries)
        
        print(f"Total entries created: {len(self.dataset)}")
        
        self._save_dataset()
        self._save_index()

    def _save_index(self):
        
        print(f"Saving index file to {self.outpath}...")
        
        os.makedirs(self.outpath, exist_ok=True)
        with open(os.path.join(self.outpath, "index.csv"), "w") as fout:
            for i, bb in self.index.items():
                fout.write(f"{i},{bb}\n")

    def _save_dataset(self):
        
        print(f"Saving dataset file to {self.outpath}...")
        os.makedirs(self.outpath, exist_ok=True)
        hf_dataset = Dataset.from_list(self.dataset)
        hf_dataset.save_to_disk(self.outpath)

class DiffusionNTDataset(data.Dataset):
    """Wrapper of torch.Dataset for training. To use after tokenization."""
    
    def __init__(self, tokenized_dataset, colname: str):
        self.sequences = tokenized_dataset[colname]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):

        return {
            "sequence": self.sequences[idx],
        }

def main():
    
    p = argparse.ArgumentParser()
    p.add_argument("--bb_folder", type=str, required=True, help="Folder with bigbed files.")
    p.add_argument("--chroms", type=str, required=True, help="Chrom ranges (e.g. 1-2).")
    p.add_argument("--fasta", type=str, required=True, help="Where reference FASTA is stored.")
    p.add_argument("--outpath", type=str, required=True, help="Where to store the dataset.")
    p.add_argument("--n_workers", type=int, default=None, help="Number of parallel workers (default: CPU count - 1).")
    p.add_argument("--num_peaks", type=int, required=True, help="How Many peaks per chromosome")
    
    args = p.parse_args()
    
    chrom_start, chrom_end = map(int, args.chroms.split("-"))
    
    dataset = DatasetCreator(
        args.bb_folder,
        chrom_start,
        chrom_end,
        args.fasta,
        args.outpath,
        args.n_workers,
        args.num_peaks
    )
    dataset.create_dataset()


if __name__ == "__main__":
    main()