import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

# Résolution du chemin absolu pour les imports du projet
root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

from src.dataset.normalization import TARGET_FREQ



# Constantes temporelles
MAX_TEMPS = 144
MAX_SIGNAL_LENGTH = MAX_TEMPS * TARGET_FREQ + 10


class TurboDataset(IterableDataset):
    """
    Dataset itérable optimisé.
    
    Cette classe implémente une stratégie de lecture hybride type Mega-Batch Streaming.
    Plutôt que d'effectuer des requêtes I/O aléatoires (qui saturent les disques réseaux) ou de 
    charger des fichiers entiers en mémoire vive, l'itérateur extrait des blocs de données contigus.
    Ces blocs sont ensuite mélangés en mémoire vive avant d'être scindés en mini-batchs.
    
    Cette approche garantit un débit de lecture disque maximal tout en maintenant une empreinte 
    mémoire (RAM) strictement bornée et un mélange stochastique de haute qualité.

    Arguments :
        data_path (str): Chemin vers le répertoire contenant les shards (.npy et .csv).
        batch_size (int): Nombre d'échantillons par mini-batch envoyé au GPU.
        mega_batch_size (int): Nombre d'échantillons lus simultanément depuis le disque. 
            Détermine l'empreinte RAM maximale par worker.
        use_static_padding (bool): Si True, contraint la dimension temporelle à `max_signal_length`. 
            Si False, la dimension temporelle s'ajuste dynamiquement au signal le plus long du mini-batch.
        max_signal_length (int): Dimension temporelle maximale autorisée.
    """

    def __init__(self, data_path, batch_size=64, mega_batch_size=1024, 
                 use_static_padding=False, max_signal_length=MAX_SIGNAL_LENGTH):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.mega_batch_size = mega_batch_size
        self.use_static_padding = use_static_padding
        self.max_signal_length = max_signal_length

        # Identification des fichiers de signaux
        self.shard_files = sorted(glob.glob(os.path.join(data_path, "*_signals.npy")))
        if not self.shard_files:
            raise FileNotFoundError(f"Aucun fichier shard détecté dans {data_path}")

        # Calcul de la volumétrie totale via les métadonnées (mmap_mode='r' pour éviter la charge RAM)
        self.total_samples = 0
        for f in self.shard_files:
            lab_f = f.replace("_signals.npy", "_labels.npy")
            self.total_samples += np.load(lab_f, mmap_mode='r').shape[0]

    def __iter__(self):
        """
        Générateur de flux de données. 
        Gère la répartition multiprocessing, la lecture contiguë et le mélange intra-bloc.
        """
        worker_info = get_worker_info()
        indices_shards = list(range(len(self.shard_files)))

        # Répartition des shards entre les workers du DataLoader
        if worker_info is not None:
            per_worker = int(np.ceil(len(indices_shards) / float(worker_info.num_workers)))
            indices_shards = indices_shards[worker_info.id * per_worker : (worker_info.id + 1) * per_worker]

        # Mélange global de l'ordre de lecture des shards pour la diversité stochastique
        np.random.shuffle(indices_shards)

        for shard_idx in indices_shards:
            sig_f = self.shard_files[shard_idx]
            lab_f = sig_f.replace("_signals.npy", "_labels.npy")
            met_f = sig_f.replace("_signals.npy", "_meta.csv")

            # Ouverture des descripteurs de fichiers sans chargement en mémoire vive
            sig_mmap = np.load(sig_f, mmap_mode='r')
            lab_all = np.load(lab_f)

            # Utilisation du moteur C de Pandas pour une lecture optimisée des métadonnées
            len_all = pd.read_csv(met_f, usecols=['length'], engine='c')['length'].values

            shard_len = len(sig_mmap)

            # Découpage logique du shard en indices de blocs contigus (Mega-Batches)
            mb_starts = list(range(0, shard_len, self.mega_batch_size))
            np.random.shuffle(mb_starts)

            for start in mb_starts:
                end = min(start + self.mega_batch_size, shard_len)
                
                # Lecture contiguë forcée par np.array()
                mb_signals = np.array(sig_mmap[start:end]) 
                mb_labels = lab_all[start:end]
                mb_lengths = len_all[start:end]

                mb_size = len(mb_signals)

                # Mélange aléatoire des échantillons au sein de la mémoire vive allouée
                perm = np.random.permutation(mb_size)
                mb_signals = mb_signals[perm]
                mb_labels = mb_labels[perm]
                mb_lengths = mb_lengths[perm]

                # Construction et itération sur les mini-batchs finaux
                for j in range(0, mb_size, self.batch_size):
                    batch_idx = list(range(j, min(j + self.batch_size, mb_size)))
                    cur_bs = len(batch_idx)

                    # Stratégie de padding
                    if self.use_static_padding:
                        target_t = self.max_signal_length
                    else:
                        target_t = int(mb_lengths[batch_idx].max())

                    # Allocation du tenseur PyTorch pré-rempli de zéros
                    batch_x = torch.zeros((cur_bs, 12, target_t), dtype=torch.float32)

                    # Référence mémoire du mini-batch courant
                    raw_x = mb_signals[batch_idx]

                    # Remplissage par copie mémoire restreinte à la longueur utile
                    for k in range(cur_bs):
                        read_len = min(mb_lengths[batch_idx[k]], target_t)
                        batch_x[k, :, :read_len] = torch.from_numpy(raw_x[k, :, :read_len])

                    batch_y = torch.from_numpy(mb_labels[batch_idx])
                    batch_lens = torch.from_numpy(mb_lengths[batch_idx]).long()

                    yield batch_x, batch_y, batch_lens

    def __len__(self):
        """
        Évalue le nombre total d'itérations prévues sur l'ensemble du dataset.
        
        Returns:
            int: Le nombre estimé de mini-batchs.
        """
        return int(np.ceil(self.total_samples / self.batch_size))
