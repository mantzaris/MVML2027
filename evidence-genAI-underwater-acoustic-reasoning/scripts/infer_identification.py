"""Fresh clean and stressed inference from the three original frozen anchors."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from underwater_reasoning.io import file_hash
from underwater_reasoning.runtime import budget_check
from train_classifier import features, AudioImages, new_model, infer

ROOT = Path(__file__).resolve().parents[1]


def main():
    frozen = json.loads((ROOT/'protocol/identification_freeze.json').read_text())
    for path, expected in frozen['files'].items():
        if file_hash(ROOT/path) != expected:
            raise ValueError('Freeze mismatch: '+path)
    rows = json.loads((ROOT/'data/private/identification_evaluation_manifest.json').read_text())['records']
    path = features(rows, cache_name='identification')
    training = json.loads((ROOT/'results/manifests/classifier_training.json').read_text())
    loader = DataLoader(AudioImages(path, rows, list(range(len(rows)))), batch_size=64, num_workers=4, pin_memory=True)
    for trained in training['seeds']:
        budget_check(reserve_seconds=600)
        seed = trained['seed']
        destination = ROOT/f'data/derived/identification_seed{seed}.npz'
        if destination.exists():
            existing = np.load(destination)
            assert list(existing['ids']) == [r['id'] for r in rows]
            continue
        model = new_model()
        model.load_state_dict(torch.load(ROOT/trained['checkpoint'], weights_only=False)['model'])
        logits, _, indices, embeddings = infer(model, loader, embedding=True)
        assert np.array_equal(indices, np.arange(len(rows)))
        original = np.load(ROOT/f'data/derived/classifier_seed{seed}.npz')
        pca = (embeddings-original['pca_mean'])@original['pca_components'].T
        np.savez_compressed(destination, ids=np.array([r['id'] for r in rows]), logits=logits,
                            embeddings=embeddings, pca=pca, temperature=original['temperature'])
        del model
        torch.cuda.empty_cache()
        print('identification inference complete', seed, len(rows), flush=True)


if __name__ == '__main__':
    main()
