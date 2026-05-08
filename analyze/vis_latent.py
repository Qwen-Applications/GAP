import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class LatentRecord:
    filepath: str
    filename: str
    dataset: str
    sample_id: str
    rank_id: str
    latent: np.ndarray


class LatentTensorAnalyzer:
    def __init__(self, latent_dir: str, output_dir: Optional[str] = None):
        self.latent_dir = latent_dir
        self.output_dir = output_dir or os.path.join(latent_dir, "analysis")
        os.makedirs(self.output_dir, exist_ok=True)

    def _parse_filename(self, filepath: str) -> Dict[str, str]:
        filename = os.path.basename(filepath)

        patterns = [
            r"^(?P<dataset>.+?)_sample(?P<sample_id>.+?)_rank(?P<rank_id>.+?)\.npy$",
            r"^(?P<dataset>.+?)_(?P<sample_id>.+?)_(?P<rank_id>.+?)\.npy$",
        ]

        for pattern in patterns:
            m = re.match(pattern, filename)
            if m:
                return m.groupdict()

        return {
            "dataset": "unknown",
            "sample_id": "unknown",
            "rank_id": "unknown",
        }

    def load_latents(self) -> List[LatentRecord]:
        records = []
        npy_files = sorted(glob.glob(os.path.join(self.latent_dir, "*.npy")))

        for fp in npy_files:
            try:
                latent = np.load(fp)
                meta = self._parse_filename(fp)
                records.append(
                    LatentRecord(
                        filepath=fp,
                        filename=os.path.basename(fp),
                        dataset=meta["dataset"],
                        sample_id=str(meta["sample_id"]),
                        rank_id=str(meta["rank_id"]),
                        latent=latent,
                    )
                )
            except Exception as e:
                print(f"[Warning] Failed to load {fp}: {e}")

        print(f"[Info] Loaded {len(records)} latent files from {self.latent_dir}")
        return records

    def _to_sample_embedding(self, latent: np.ndarray, pool: str = "mean") -> np.ndarray:
        arr = np.asarray(latent)

        if arr.ndim == 1:
            return arr

        if arr.ndim >= 2:
            last_dim = arr.shape[-1]
            arr = arr.reshape(-1, last_dim)

        if pool == "mean":
            return arr.mean(axis=0)
        elif pool == "max":
            return arr.max(axis=0)
        elif pool == "flatten":
            return arr.reshape(-1)
        else:
            raise ValueError(f"Unsupported pool method: {pool}")

    def build_feature_matrix(self, records: List[LatentRecord], pool: str = "mean"):
        features = []
        labels = []
        metas = []

        for rec in records:
            feat = self._to_sample_embedding(rec.latent, pool=pool)
            features.append(feat)
            labels.append(rec.dataset)
            metas.append(
                {
                    "filename": rec.filename,
                    "dataset": rec.dataset,
                    "sample_id": rec.sample_id,
                    "rank_id": rec.rank_id,
                }
            )

        features = np.stack(features, axis=0)
        return features, labels, metas

    def run_pca(
        self,
        features: np.ndarray,
        n_components: int = 2,
        random_state: int = 42,
    ) -> np.ndarray:
        pca = PCA(n_components=n_components, random_state=random_state)
        coords = pca.fit_transform(features)
        print(f"[Info] PCA explained variance ratio: {pca.explained_variance_ratio_}")
        return coords

    def run_tsne(
        self,
        features: np.ndarray,
        n_components: int = 2,
        perplexity: int = 30,
        random_state: int = 42,
    ) -> np.ndarray:
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, max(2, len(features) - 1)),
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(features)
        return coords

    def plot_2d(
        self,
        coords: np.ndarray,
        labels: List[str],
        metas: Optional[List[Dict[str, Any]]] = None,
        title: str = "Latent Visualization",
        save_name: str = "latent_vis.png",
        annotate: bool = False,
    ):
        plt.figure(figsize=(10, 8))

        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap("tab20", len(unique_labels))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}

        for i, label in enumerate(labels):
            color = cmap(label_to_idx[label])
            plt.scatter(coords[i, 0], coords[i, 1], color=color, s=40, alpha=0.8)

            if annotate and metas is not None:
                text = f'{metas[i]["dataset"]}:{metas[i]["sample_id"]}'
                plt.annotate(text, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.8)

        handles = []
        for label in unique_labels:
            color = cmap(label_to_idx[label])
            handles.append(
                plt.Line2D([0], [0], marker="o", color="w", label=label,
                           markerfacecolor=color, markersize=8)
            )

        plt.legend(handles=handles, loc="best")
        plt.title(title)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"[Info] Saved figure to {save_path}")

    def save_projection(
        self,
        coords: np.ndarray,
        metas: List[Dict[str, Any]],
        save_name: str = "projection.npy",
    ):
        save_path = os.path.join(self.output_dir, save_name)
        payload = {
            "coords": coords,
            "meta": metas,
        }
        np.save(save_path, payload, allow_pickle=True)
        print(f"[Info] Saved projection to {save_path}")

    def analyze(
        self,
        pool: str = "mean",
        run_pca: bool = True,
        run_tsne: bool = True,
        perplexity: int = 30,
        annotate: bool = False,
    ):
        records = self.load_latents()
        if not records:
            print("[Warning] No latent tensors found.")
            return

        features, labels, metas = self.build_feature_matrix(records, pool=pool)

        if run_pca:
            pca_coords = self.run_pca(features)
            self.plot_2d(
                pca_coords,
                labels,
                metas=metas,
                title=f"PCA of Latent Tensors ({pool})",
                save_name=f"latent_pca_{pool}.png",
                annotate=annotate,
            )
            self.save_projection(
                pca_coords,
                metas,
                save_name=f"latent_pca_{pool}.npy",
            )

        if run_tsne:
            tsne_coords = self.run_tsne(features, perplexity=perplexity)
            self.plot_2d(
                tsne_coords,
                labels,
                metas=metas,
                title=f"t-SNE of Latent Tensors ({pool})",
                save_name=f"latent_tsne_{pool}.png",
                annotate=annotate,
            )
            self.save_projection(
                tsne_coords,
                metas,
                save_name=f"latent_tsne_{pool}.npy",
            )