"""Summarize the matched diagnostic runs without changing their raw artifacts."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent / 'diagnostics' / 'context'
SEEDS = (41100, 41200, 41300, 41400, 41500)
METRICS = {
    'base_clean': ('base', 'clean_base28'),
    'base_noise20': ('base', 'noise0.2_base28'),
    'base_noise40': ('base', 'noise0.4_base28'),
    'old_after_extension_28_candidates': ('post', 'clean_base28'),
    'old_after_extension_36_candidates': ('post', 'clean_base36'),
    'new_after_extension': ('post', 'clean_extension'),
}

def summarize():
    summary = {'seeds': SEEDS, 'dispersion': 'sample standard deviation across seeds',
               'conditions': {}}
    for beta, folder in [(0.03, ROOT), (0.003, ROOT/'core_beta_0.003'),
                         (0.0, ROOT/'core_beta_0.0')]:
        runs = [json.loads((folder/f'analysis_{s}.json').read_text()) for s in SEEDS]
        for readout in runs[0]['readouts']:
            condition = {'metrics': {}, 'terminal_SA_ceiling': {}, 'validation': {}}
            for label, (stage, metric) in METRICS.items():
                values = [r['readouts'][readout][stage][metric]['accuracy'] for r in runs]
                strict = [r['readouts'][readout][stage][metric]['strict_accuracy'] for r in runs]
                condition['metrics'][label] = {
                    'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)),
                    'by_seed': dict(zip(SEEDS, values)),
                    'strict_mean': float(np.mean(strict)),
                }
            ceilings = [r['geometry']['SA']['base']['exact_pattern_ceiling'] for r in runs]
            condition['terminal_SA_ceiling'] = {'mean': float(np.mean(ceilings)),
                                              'by_seed': dict(zip(SEEDS, ceilings))}
            condition['validation'] = {str(s): r['validation'] for s, r in zip(SEEDS, runs)}
            summary['conditions'][f'beta={beta}/{readout}'] = condition
            if readout in ('full_online', 'sa_online'):
                print(beta, readout, ' | '.join(
                    f'{label}: {100*m["mean"]:.1f} ± {100*m["sd"]:.1f}'
                    for label, m in condition['metrics'].items()))
    stress = ROOT/'physical_four_area_and_exposure.json'
    if stress.exists():
        summary['physical_validation_and_exposure'] = json.loads(stress.read_text())
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary

if __name__ == '__main__':
    summarize()
