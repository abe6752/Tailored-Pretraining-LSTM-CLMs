import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from src.chem_utils import read_smiles_list_tsv

def murcko_scaffold_smiles(smi: str):
    if not isinstance(smi, str) or smi.strip() == '':
        return None

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return None
    
    return Chem.MolToSmiles(scaf)

def build_scaffold_maps(smi_list: list[str]):
    
    smi2scaf = {}
    scaf_set = set()
    for smi in smi_list:
        scaf = murcko_scaffold_smiles(smi)
        smi2scaf[smi] = scaf
        if scaf is not None:
            scaf_set.add(scaf)
    return smi2scaf, scaf_set


def calc_rediscovered_set(gen_smi: list[str], test_smi: list[str]) -> set[str]:
    return set(test_smi) & set(gen_smi)

def prepare_novel_scaffold_context(train_smi: list[str], test_smi: list[str]):

    _, train_scaf_set = build_scaffold_maps(train_smi)
    test_smi2scaf, _  = build_scaffold_maps(test_smi)

    test_novel_scaf_set = set()
    test_novel_smi_set  = set()

    for s in test_smi:
        sc = test_smi2scaf.get(s)
        if sc is None:
            continue
        if sc not in train_scaf_set:
            test_novel_scaf_set.add(sc)
            test_novel_smi_set.add(s)

    return {
        'train_scaf_set': train_scaf_set,
        'test_smi2scaf': test_smi2scaf,
        'test_novel_scaf_set': test_novel_scaf_set,
        'test_novel_smi_set': test_novel_smi_set
    }

def calc_novel_scaff_redisc_metrics(redisc_smi_set: set[str], ctx: dict):

    test_smi2scaf = ctx['test_smi2scaf']
    train_scaf_set = ctx['train_scaf_set']

    test_novel_scaf_set = ctx['test_novel_scaf_set']
    test_novel_smi_set  = ctx['test_novel_smi_set']

    redisc_novel_scaf_set = set()
    redisc_novel_smi_set  = set()

    for s in redisc_smi_set:
        sc = test_smi2scaf.get(s)
        if sc is None:
            continue
        if sc not in train_scaf_set:
            redisc_novel_scaf_set.add(sc)
            redisc_novel_smi_set.add(s)

    # scaffold-level
    num_scaf   = len(redisc_novel_scaf_set)
    denom_scaf = len(test_novel_scaf_set)
    scaffold_level = (num_scaf / denom_scaf) if denom_scaf > 0 else np.nan

    # molecule-level
    num_mol   = len(redisc_novel_smi_set)
    denom_mol = len(test_novel_smi_set)
    molecule_level = (num_mol / denom_mol) if denom_mol > 0 else np.nan

    return{
        'scaffold_level': scaffold_level,
        'molecule_level': molecule_level,
        'num_scaf': num_scaf,
        'denom_scaf': denom_scaf,
        'num_mol': num_mol,
        'denom_mol': denom_mol
    }

# Epoch-wise calculation
def load_and_calculate_epochwise(
    gen_dir: str,
    test_smi: list[str],
    ctx: dict,
    n_epochs: int=100
):
    
    out = {}
    test_smi_set = set(test_smi)

    for epoch in range(1, n_epochs + 1):
        tsv = os.path.join(gen_dir, 'sampling', f'epoch{epoch}_10000samples.tsv')

        gen_smi = read_smiles_list_tsv(tsv, col='canonical_smiles', dropna=True, dedup=True)

        redisc_smi_set = (set(gen_smi) & test_smi_set)

        metrics = calc_novel_scaff_redisc_metrics(redisc_smi_set, ctx)
        out[f'epoch{epoch}'] = metrics
    
    return out

def dict_to_epoch_df(epoch_metrics_dict: dict, key: str) -> pd.Series:
    items = []
    for ep, md in epoch_metrics_dict.items():
        ep_i = int(ep.replace('epoch', ''))
        items.append((ep_i, md.get(key, np.nan)))
    items.sort(key=lambda x: x[0])
    idx = [f'epoch{i}' for i, _ in items]
    vals = [v for _, v in items]
    return pd.Series(vals, index=idx)

# pooled calculation
def calc_pooled_redisc_novel_scaf(
    alluniq_path: str,
    test_smi: list[str],
    ctx: dict,
    col: str='canonical_smiles'
):
    gen_all = read_smiles_list_tsv(alluniq_path, col=col, dropna=True, dedup=True)
    redisc_smi_set = calc_rediscovered_set(gen_all, test_smi)
    return calc_novel_scaff_redisc_metrics(redisc_smi_set, ctx)


def run_novel_scaff_redisc(
        FINETUNE_DATA: str,
        FINETUNE_RESULTS: str,
        FINETUNE_OUT: str,
        FINETUNE_FILTER: str,
        f_data_list: list[str],
        dataset_list: list[str],
        n_epochs: int=100,
):
    save_dir = os.path.join(FINETUNE_OUT, 'novel_scaff_redisc')
    os.makedirs(save_dir, exist_ok=True)

    for f_data in f_data_list:
        test_path = os.path.join(FINETUNE_DATA, f'{FINETUNE_FILTER}-{f_data}_test.tsv')
        train_path = os.path.join(FINETUNE_DATA, f'{FINETUNE_FILTER}-{f_data}_train.tsv')

        test_smi = read_smiles_list_tsv(test_path, col='rdkit_smiles', dropna=True, dedup=True)
        train_smi = read_smiles_list_tsv(train_path, col='rdkit_smiles', dropna=True, dedup=True)

        ctx = prepare_novel_scaffold_context(train_smi, test_smi)

        scaffold_level_df = pd.DataFrame(index=[f'epoch{i}' for i in range(1, n_epochs + 1)])
        molecule_level_df = pd.DataFrame(index=[f'epoch{i}' for i in range(1, n_epochs + 1)])

        num_scaf_df = pd.DataFrame(index=[f'epoch{i}' for i in range(1, n_epochs + 1)])
        num_mol_df = pd.DataFrame(index=[f'epoch{i}' for i in range(1, n_epochs + 1)])

        for dataset in dataset_list:
            gen_dir = os.path.join(FINETUNE_RESULTS, f'{dataset}_results', f'{f_data}_finetune')
            epoch_metrics = load_and_calculate_epochwise(gen_dir, test_smi, ctx, n_epochs=n_epochs)

            scaffold_level_df[dataset] = dict_to_epoch_df(epoch_metrics, 'scaffold_level').values
            molecule_level_df[dataset] = dict_to_epoch_df(epoch_metrics, 'molecule_level').values
            num_scaf_df[dataset]       = dict_to_epoch_df(epoch_metrics, 'num_scaf').values
            num_mol_df[dataset]        = dict_to_epoch_df(epoch_metrics, 'num_mol').values

        scaffold_level_df.to_csv(os.path.join(save_dir, f'{f_data}_scaffold_level.tsv'), sep='\t')
        molecule_level_df.to_csv(os.path.join(save_dir, f'{f_data}_molecule_level.tsv'), sep='\t')
        num_scaf_df.to_csv(os.path.join(save_dir, f'{f_data}_num_novel_scaf_rediscovered.tsv'), sep='\t')
        num_mol_df.to_csv(os.path.join(save_dir, f'{f_data}_num_novel_mol_rediscovered.tsv'), sep='\t')

        denom_info = {
            'denom_uniq_novel_scaf_in_test': len(ctx['test_novel_scaf_set']),
            'denom_novel_mol_in_test': len(ctx['test_novel_smi_set'])
        }
        pd.Series(denom_info).to_csv(os.path.join(save_dir, f'{f_data}_denominantors.tsv'), sep='\t')

    # pooled
    rows_scaf = []
    rows_mol  = []

    for f_data in f_data_list:
        test_path = os.path.join(FINETUNE_DATA, f'{FINETUNE_FILTER}-{f_data}_test.tsv')
        train_path = os.path.join(FINETUNE_DATA, f'{FINETUNE_FILTER}-{f_data}_train.tsv')

        test_smi = read_smiles_list_tsv(test_path, col='rdkit_smiles', dropna=True, dedup=True)
        train_smi = read_smiles_list_tsv(train_path, col='rdkit_smiles', dropna=True, dedup=True)
        ctx = prepare_novel_scaffold_context(train_smi, test_smi)

        row_scaf = {'finetune_data': f_data}
        row_mol  = {'finetune_data': f_data}

        for dataset in dataset_list:
            alluniq_path = os.path.join(
                FINETUNE_RESULTS, 'all_unique_smiles', f_data, f'{f_data}_{dataset}_all_uniq_smiles.tsv'
            )
            m = calc_pooled_redisc_novel_scaf(alluniq_path, test_smi, ctx, col='canonical_smiles')

            row_scaf[dataset] = m['scaffold_level']
            row_mol[dataset]  = m['molecule_level']

        rows_scaf.append(row_scaf)
        rows_mol.append(row_mol)

    overall_scaf_df = pd.DataFrame(rows_scaf).set_index('finetune_data').round(3)
    overall_mol_df  = pd.DataFrame(rows_mol).set_index('finetune_data').round(3)

    overall_scaf_df.to_csv(os.path.join(save_dir, 'overall_scaffold_level.tsv'), sep='\t')
    overall_mol_df.to_csv(os.path.join(save_dir, 'overall_molecule_level.tsv'), sep='\t')

    return overall_scaf_df, overall_mol_df


# plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_metric_with_peaks(ax, df, dataset_list, colors, title, is_last, ylabel):
    for dataset, color in zip(dataset_list, colors):
        values = df[dataset].values
        epochs = range(1, len(values) + 1)

        ax.plot(epochs, values, color=color, alpha=0.95)

        if np.all(np.isnan(values)):
            continue
        max_idx = np.nanargmax(values)
        max_epoch = max_idx + 1
        max_value = values[max_idx]

        ax.axvline(x=max_epoch, color=color, linestyle=':', alpha=0.7, linewidth=1)
        ax.plot(max_epoch, max_value, 'o', color=color, markersize=4)

    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=9)
    if is_last:
        ax.set_xlabel('Epoch', fontsize=13)


def plot_novel_scaff_redisc_with_peaks(
    FINETUNE_OUT: str,
    FIGURES_DIR: str,
    FINETUNE_FILTER: str,
    f_data_list: list[str],
    dataset_list: list[str],
    dataset_names: list[str],
    dataset_colors: list[str],
    metric: str = 'scaffold_level',   # 'scaffold_level' or 'molecule_level'
):
    assert metric in ['scaffold_level', 'molecule_level']

    save_dir = os.path.join(FINETUNE_OUT, 'novel_scaff_redisc')
    if metric == 'scaffold_level':
        suffix = 'scaffold_level'
        ylabel = 'Novel-scaffold rediscovery'
    else:
        suffix = 'molecule_level'
        ylabel = 'Novel-scaffold rediscovery\nmolecule-level'

    fig, axs = plt.subplots(len(f_data_list), 1, figsize=(8.27, 11.69))

    if len(f_data_list) == 1:
        axs = [axs]

    legend_handles = [
        Line2D([0], [0], color=c, linewidth=2, label=n)
        for c, n in zip(dataset_colors, dataset_names)
    ]

    for i, (ax, f_data) in enumerate(zip(axs, f_data_list)):
        df = pd.read_table(os.path.join(save_dir, f'{f_data}_{suffix}.tsv'), index_col=0)
        plot_metric_with_peaks(
            ax=ax,
            df=df,
            dataset_list=dataset_list,
            colors=dataset_colors,
            title=f_data,
            is_last=(i == len(f_data_list) - 1),
            ylabel=ylabel,
        )

    fig.legend(handles=legend_handles, loc='upper center', ncol=3,
               bbox_to_anchor=(0.505, 0.98), fontsize=10)
    fig.subplots_adjust(hspace=0.4, top=0.89)

    fig_dir = os.path.join(FIGURES_DIR, 'finetune', FINETUNE_FILTER)
    os.makedirs(fig_dir, exist_ok=True)

    outpath = os.path.join(fig_dir, f'novel_scaff_redisc_{metric}_with_peaks.png')
    plt.savefig(outpath, dpi=400, bbox_inches='tight')
    plt.close(fig)

    return outpath