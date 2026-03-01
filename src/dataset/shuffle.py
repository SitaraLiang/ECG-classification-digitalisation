import os
import argparse
import time
import glob
import json
import numpy as np
import pandas as pd
import h5py
import multiprocessing
from tqdm import tqdm



# Fixation de la graine aléatoire pour garantir la reproductibilité des splits et des mélanges
np.random.seed(42)


def normalize_id(val):
    """
    Normalise les identifiants extraits des sources hétérogènes.
    
    Convertit les types bytes (fréquents lors de la lecture HDF5) en chaînes de caractères UTF-8 
    et supprime les espaces résiduels pour garantir l'intégrité des jointures relationnelles.

    Args:
        val (Union[bytes, str, int]): L'identifiant brut.

    Returns:
        str: L'identifiant normalisé.
    """
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode('utf-8')
    return str(val).strip()


def scan_sources(input_dir):
    """
    Effectue un inventaire exhaustif et réalise l'alignement entre les métadonnées (CSV) 
    et les index physiques (HDF5).

    Args:
        input_dir (str): Répertoire contenant les paires de fichiers .csv et .hdf5.

    Returns:
        pd.DataFrame: DataFrame contenant la jointure stricte (Inner Join) des métadonnées 
        et de l'adressage physique HDF5.
    """
    print("\n[1/5] Scan des sources et vérification de l'alignement structurel...")
    csv_files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    h5_files = sorted(glob.glob(os.path.join(input_dir, '*.hdf5')))

    # 1. Agrégation des métadonnées et vecteurs de labels
    df_list = []
    for f in tqdm(csv_files, desc="Chargement des métadonnées (CSV)"):
        df_list.append(pd.read_csv(f))
    full_csv = pd.concat(df_list, ignore_index=True)
    full_csv['exam_id'] = full_csv['exam_id'].apply(normalize_id)

    # Nettoyage préventif des variables de traçabilité potentiellement obsolètes
    for col in ['trace_file', 'h5_path_src', 'h5_idx_src']:
        if col in full_csv.columns:
            full_csv = full_csv.drop(columns=[col])

    # 2. Cartographie des adresses physiques des signaux
    map_records = []
    for h5_path in tqdm(h5_files, desc="Indexation spatiale (HDF5)"):
        with h5py.File(h5_path, 'r') as f:
            ids = f['exam_id'][:]
            for idx, val in enumerate(ids):
                map_records.append({
                    'exam_id': normalize_id(val),
                    'h5_path_src': h5_path,
                    'h5_idx_src': idx  # Index séquentiel du signal dans le dataset 'tracings'
                })

    df_h5 = pd.DataFrame(map_records)

    # 3. Jointure relationnelle garantissant l'alignement signal-label
    inventory = pd.merge(full_csv, df_h5, on='exam_id', how='inner')
    print(f"Validation terminée : {len(inventory)} correspondances exactes établies.")
    return inventory


def write_npy_shard(task):
    """
    Processus de travail dédié à l'extraction, l'alignement et la sérialisation des tenseurs.
    
    Optimisations I/O appliquées :
    - Tri préalable des requêtes par index physique pour forcer une lecture séquentielle du disque.
    - Allocation préventive du tenseur de destination pour un zero-padding implicite.

    Args:
        task (dict): Dictionnaire contenant les paramètres de la partition (DataFrame, chemins, classes).

    Returns:
        str: Code de statut de l'opération (SUCCESS ou ERROR) incluant des métriques.
    """
    output_base = task['output_base']
    df = task['df'].copy()
    classes = task['classes']
    shard_id = task['shard_id']
    split = task['split']

    # Extraction ordonnée des matrices de labels
    labels = df[classes].values.astype(np.float32)

    # Verrouillage de la topologie de destination
    df = df.reset_index(drop=True)
    df['d_idx'] = df.index 

    n_samples = len(df)
    max_t = int(df['length'].max())

    # Pré-allocation du tenseur cible (Zéro-padding structurel)
    signals = np.zeros((n_samples, 12, max_t), dtype=np.float32)

    # Groupement par ressource physique pour minimiser les descripteurs de fichiers ouverts
    sources = df.groupby('h5_path_src')

    try:
        for h5_path, group in sources:
            # Tri par index source (h5_idx_src) imposant une lecture séquentielle matérielle
            group = group.sort_values('h5_idx_src')
            
            with h5py.File(h5_path, 'r') as f_in:
                ds = f_in['tracings']
                for _, row in group.iterrows():
                    start = int(row['start_offset'])
                    length = int(row['length'])
                    s_idx = int(row['h5_idx_src'])
                    d_idx = int(row['d_idx']) 

                    # Copie mémoire vectorisée vers le segment pré-alloué
                    signals[d_idx, :, :length] = ds[s_idx, :, start:start+length]

        # Sérialisation binaire au format NumPy natif
        np.save(f"{output_base}_signals.npy", signals)
        np.save(f"{output_base}_labels.npy", labels)
        
        # Sauvegarde des métadonnées de traçabilité
        df.drop(columns=['d_idx']).to_csv(f"{output_base}_meta.csv", index=False)
        
        return f"SUCCESS|{split}|{shard_id}|{n_samples}|{max_t}"
    except Exception as e:
        return f"ERROR|{split}|{shard_id}|{str(e)}"



def run(args):
    """
    Fonction d'orchestration de la préparation des données.
    
    Implémente la stratégie de "Bucketing" assortie d'un "Chunk Shuffling" pour 
    optimiser le padding dynamique lors de l'entraînement, tout en prévenant 
    l'oubli catastrophique (Catastrophic Forgetting) par une distribution stochastique globale.
    """
    start_total = time.time()
    os.makedirs(args.output, exist_ok=True)

    # 1. Chargement des classes cibles
    print(f"\n[0/5] Initialisation du mappage des classes : {args.class_map}")
    with open(args.class_map, 'r') as f:
        target_classes = json.load(f)

    # 2. Établissement de l'inventaire physique
    inventory = scan_sources(args.input)

    # 3. Traitement et normalisation des vecteurs d'annotation
    print("\n[2/5] Normalisation des vecteurs de classification...")
    for c in tqdm(target_classes, desc="Conformation des labels"):
        if c not in inventory.columns:
            inventory[c] = 0.0
        else:
            inventory[c] = inventory[c].fillna(0).astype(int).astype(np.float32)

    # 4. Partitionnement indépendant des patients (Patient-Aware Split)
    print("\n[3/5] Partitionnement des sous-ensembles (Train/Val/Test)...")
    unique_patients = inventory['patient_id'].unique()
    np.random.shuffle(unique_patients)

    n_train = int(len(unique_patients) * args.train_prct)
    n_val = int(len(unique_patients) * args.val_prct)

    train_pats = set(unique_patients[:n_train])
    val_pats = set(unique_patients[n_train:n_train+n_val])

    inventory['split'] = inventory['patient_id'].apply(
        lambda x: 'train' if x in train_pats else ('val' if x in val_pats else 'test')
    )

    # 5. Algorithme de Bucketing et Chunk Shuffling
    print("\n[4/5] Structuration stochastique (Bucketing & Chunk Shuffling)...")
    tasks = []

    # Dimensionnement des blocs pour la stochasticité locale.
    # TODO Cette valeur doit être synchronisée avec 'mega_batch_size' dans le DataLoader.
    mega_batch_size = 1024 

    for split in ['train', 'val', 'test']:
        split_df = inventory[inventory['split'] == split].copy()
        if split_df.empty: 
            continue

        # Tri ascendant par dimension temporelle pour générer les clusters
        split_df = split_df.sort_values(by='length').reset_index(drop=True)

        # Segmentation en macro-blocs homogènes
        sub_chunks = [split_df.iloc[i : i + mega_batch_size] for i in range(0, len(split_df), mega_batch_size)]

        # Permutation globale des macro-blocs
        np.random.shuffle(sub_chunks)

        # Ré-assemblage du DataFrame
        split_df = pd.concat(sub_chunks).reset_index(drop=True)
        
        out_dir = os.path.join(args.output, split)
        os.makedirs(out_dir, exist_ok=True)

        # Découpage final en partitions de stockage (Shards)
        n_shards = int(np.ceil(len(split_df) / args.shard_size))
        chunks = np.array_split(split_df, n_shards)

        for i, chunk in enumerate(chunks):
            shard_name = f"{split}_shard_{i:04d}"
            tasks.append({
                'output_base': os.path.join(out_dir, shard_name),
                'df': chunk,
                'classes': target_classes,
                'shard_id': i,
                'split': split
            })

    # 6. Parallélisation du traitement I/O
    print(f"\n[5/5] Sérialisation multiprocessus ({args.workers} processus affectés)...")
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(args.workers) as pool:
        pbar = tqdm(pool.imap_unordered(write_npy_shard, tasks), total=len(tasks), desc="Écriture binaire")
        for r in pbar:
            if "SUCCESS" in r:
                _, split, s_id, n, t = r.split('|')
                pbar.set_postfix_str(f"Shard {s_id} ({split}) maxT={t}")
            else:
                print(f"\nÉchec d'écriture détecté : {r}")

    print(f"\nOpération de préparation terminée en {int(time.time() - start_total)}s.")
    print(f"Répertoire de sortie : {os.path.abspath(args.output)}")



def main():
    parser = argparse.ArgumentParser(
        description="Module de partitionnement offline et d'optimisation I/O pour signaux physiologiques."
    )

    parser.add_argument('-i', '--input', type=str, default='../../../output/normalize_data/',
                        help='Répertoire source contenant les fichiers normalisés (.hdf5, .csv)')
    parser.add_argument('-o', '--output', type=str, default='../../../output/final_data/',
                        help='Répertoire cible pour les tenseurs compilés (.npy)')
    parser.add_argument('--class_map', type=str, default='../../ressources/final_class.json',
                        help="Chemin vers le dictionnaire JSON de classification")
    parser.add_argument('-s', '--shard_size', type=int, default=8192,
                        help="Taille nominale des partitions de stockage (Défaut: 8192 échantillons)")

    parser.add_argument('--train_prct', type=float, default=0.80, help='Proportion allouée à l\'entraînement')
    parser.add_argument('--val_prct', type=float, default=0.10, help='Proportion allouée à la validation')

    parser.add_argument('-w', '--workers', type=int, default=os.cpu_count()-1,
                        help='Nombre de threads affectés à la sérialisation')

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
