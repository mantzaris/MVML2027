"""Explain saved category accuracy and reporting trade-offs in concrete units.

This is a presentation-only view of the frozen study, with no new inference.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from underwater_reasoning.io import atomic_json, file_hash

LABELS = {
    'background': 'Background',
    'cargo': 'Cargo ship',
    'passenger': 'Passenger vessel',
    'tanker': 'Tanker',
    'tug_tow': 'Tug / tow',
}
BLUE = '#0072B2'
ORANGE = '#C65316'
INK = '#173247'
MUTED = '#526373'
GREY = '#E5EAF0'


def main():
    source = ROOT / 'results/tables/main_results.json'
    result = json.loads(source.read_text())
    anchor = result['classifier_all_test']['17']
    matrix = np.asarray(anchor['confusion_matrix'], dtype=int)
    totals = matrix.sum(axis=1)
    correct = matrix.diagonal()
    recalls = correct / totals
    assert matrix.sum() == anchor['n'] == 2240
    assert np.allclose(recalls, anchor['per_class_recall'])
    assert np.isclose(correct.sum() / totals.sum(), anchor['accuracy'])
    categories = []
    for index, label in enumerate(result['labels']):
        errors = matrix[index].copy(); errors[index] = -1
        other = int(errors.argmax())
        categories.append({
            'category': label, 'label': LABELS[label], 'recordings': int(totals[index]),
            'correct': int(correct[index]), 'recall': float(recalls[index]),
            'most_common_wrong_label': LABELS[result['labels'][other]],
            'most_common_error_count': int(matrix[index, other]),
            'most_common_error_fraction_of_category': float(matrix[index, other] / totals[index]),
        })

    comparator = result['locked_comparator']
    assert comparator == 'B9_entropy'
    methods = [comparator, 'P', 'B8_deterministic']
    names = {comparator: 'Heuristic + AI', 'P': 'Learned selector + AI',
             'B8_deterministic': 'Fixed template'}
    clips = result['assessment_clips']
    slots = clips * 4
    reporting = []
    for method in methods:
        values = result['methods'][method]
        assert values['total_reports'] == clips == 100
        assert values['schema_validity_all'] == 1
        supported = values['supported_assertions']
        wrong = values['wrong_or_unsupported_assertions']
        assert supported + wrong == values['assertions']
        assert np.isclose(supported / slots, values['whole_test_useful_coverage'])
        calls = values['mean_tools_all'] * clips
        assert np.isclose(calls, round(calls))
        reporting.append({
            'method': method, 'label': names[method], 'clips': clips,
            'requested_physical_details': slots, 'supported_details': supported,
            'wrong_or_unsupported_details': wrong, 'reports_returned': values['retained_reports'],
            'uncertain_details_in_returned_reports': 4 * values['retained_reports'] - values['assertions'],
            'withheld_details': 4 * (clips - values['retained_reports']),
            'tool_calls': round(calls),
            'supported_details_ci95_scaled_to_400_requests': [
                slots * value for value in values['whole_test_useful_coverage_ci95']],
        })
    base, learned, template = reporting
    deltas = {key: learned[key] - base[key] for key in
              ('tool_calls', 'supported_details', 'wrong_or_unsupported_details')}
    template_deltas = {key: template[key] - base[key] for key in
                       ('supported_details', 'wrong_or_unsupported_details')}
    contrast = result['primary_contrasts'][comparator]['whole_test_useful_difference']
    assert np.isclose(deltas['supported_details'] / slots, contrast['difference'])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig = plt.figure(figsize=(14.4, 8.8), facecolor='white')
    fig.text(.05, .952, 'Vessel categories and useful acoustic details',
             fontsize=23, weight='bold', color=INK)
    fig.text(.05, .908,
             'An operator reading guide to the civilian-recording benchmark  |  Five-second hydrophone clips',
             fontsize=12, color=MUTED)
    fig.text(.155, .842, '1  Which categories are recognized?',
             fontsize=14, weight='bold', color=INK)
    fig.text(.155, .808,
             f"{anchor['accuracy']:.1%} correct overall  ·  {int(correct.sum()):,} / {anchor['n']:,} recordings",
             fontsize=11, color=MUTED)
    fig.text(.675, .842, '2  What details reach the reader?',
             fontsize=14, weight='bold', color=INK)
    fig.text(.675, .808, f'The same {clips} assessment clips  ·  {slots} details requested',
             fontsize=11, color=MUTED)

    left = fig.add_axes([.155, .325, .335, .425])
    y = np.arange(len(categories)) * 1.25
    left.barh(y, np.full(5, 100), height=.42, color=GREY, zorder=2)
    left.barh(y, 100 * recalls, height=.42, color=BLUE, zorder=3)
    for point, row in zip(y, categories):
        left.text(50 * row['recall'], point, f"{100 * row['recall']:.1f}%",
                  ha='center', va='center', color='white', fontsize=11, weight='bold')
        left.text(0, point + .43,
                  f"Most often confused with {row['most_common_wrong_label'].lower()}: "
                  f"{100 * row['most_common_error_fraction_of_category']:.1f}%",
                  va='center', fontsize=8.6, color=MUTED)
    left.set(yticks=y, yticklabels=[f"{v['label']}\n{v['correct']} correct / {v['recordings']} clips"
                                  for v in categories],
             xlim=(0, 100), ylim=(5.75, -.65), xticks=[0, 25, 50, 75, 100],
             xlabel='Correct labels within each dataset category (%)')
    left.tick_params(axis='y', length=0, pad=10, labelsize=10)
    left.tick_params(axis='x', labelsize=10)
    left.set_axisbelow(True); left.grid(axis='x', color=GREY, linewidth=.8)
    left.spines['left'].set_visible(False)
    left.legend(handles=[Patch(facecolor=BLUE, label='Correct'),
                         Patch(facecolor=GREY, label='Other label')],
                loc='lower left', bbox_to_anchor=(0, 1.00), ncol=2,
                frameon=False, fontsize=10, borderaxespad=0)

    right = fig.add_axes([.675, .325, .275, .425])
    y = np.array([.3, 2.25, 4.2])
    for point, row in zip(y, reporting):
        value = row['supported_details']
        lo, hi = row['supported_details_ci95_scaled_to_400_requests']
        right.barh(point - .20, value, height=.46, color=BLUE, zorder=3)
        right.errorbar(value, point - .20, xerr=[[value - lo], [hi - value]],
                       fmt='none', ecolor=INK, capsize=4, lw=1.2, zorder=4)
        right.text(value / 2, point - .20, str(value), color='white', weight='bold',
                   ha='center', va='center', fontsize=12, zorder=5)
        wrong = row['wrong_or_unsupported_details']
        right.barh(point + .27, wrong, height=.23, color=ORANGE, zorder=3)
        right.text(wrong + 2.5, point + .27, str(wrong), va='center', color=ORANGE,
                   fontsize=11, weight='bold')
    right.set(yticks=y, yticklabels=[f"{v['label']}\n{v['reports_returned']} / 100 reports returned"
                                   for v in reporting],
              xlim=(0, 180), ylim=(5.75, -.65), xticks=[0, 50, 100, 150],
              xlabel='Number of details (400 requested in total)')
    right.tick_params(axis='y', length=0, pad=10, labelsize=10)
    right.tick_params(axis='x', labelsize=10)
    right.spines['left'].set_visible(False)
    right.set_axisbelow(True); right.grid(axis='x', color=GREY, linewidth=.8)
    right.text(0, 5.03,
               f"Template vs heuristic: +{template_deltas['supported_details']} supported details\n"
               f"and +{template_deltas['wrong_or_unsupported_details']} wrong / unsupported details.",
               fontsize=8.3, color=MUTED, va='center', linespacing=1.4)
    right.legend(handles=[Patch(facecolor=BLUE, label='Supported'),
                          Patch(facecolor=ORANGE, label='Wrong / unsupported')],
                 loc='lower left', bbox_to_anchor=(0, 1.00), ncol=2,
                 frameon=False, fontsize=9.5, borderaxespad=0, columnspacing=1.0)

    fig.add_artist(Line2D([.05, .95], [.245, .245], transform=fig.transFigure,
                          color=GREY, lw=1.5))
    fig.text(.05, .209, 'Did learned selection help?  Changes versus the heuristic, across the same 100 clips:',
             fontsize=12, weight='bold', color=INK)
    cards = [
        (.05, 'tool_calls', 'tool calls', f"{base['tool_calls']} → {learned['tool_calls']} calls", BLUE),
        (.365, 'supported_details', 'supported details',
         f"{base['supported_details']} → {learned['supported_details']} useful details", ORANGE),
        (.68, 'wrong_or_unsupported_details', 'wrong / unsupported details',
         f"{base['wrong_or_unsupported_details']} → {learned['wrong_or_unsupported_details']} problematic details", BLUE),
    ]
    for x, key, label, note, color in cards:
        fig.text(x, .155, f'{deltas[key]:+d}'.replace('-', '−'), fontsize=24, weight='bold', color=color)
        fig.text(x + .071, .159, label, fontsize=11, color=INK)
        fig.text(x, .119, note, fontsize=10, color=MUTED)
    fig.text(.05, .066,
             'A detail is an energy band, persistent tone, modulation rate, or received-energy trend. '
             'Uncertain slots and withheld reports add no supported detail.', fontsize=9, color=MUTED)
    fig.text(.05, .039,
             'Whiskers: 95% source-bootstrap intervals. Category percentages are recall for the shared classifier (seed 17). '
             'Detail support uses automated proxies.', fontsize=9, color=MUTED)
    fig.text(.05, .012,
             'Fewer tool calls do not establish GPU savings. This civilian-data study did not evaluate sonar operators or operational deployment.',
             fontsize=9, color=MUTED)

    out = ROOT / 'results/figures'; out.mkdir(exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'operator_context.{suffix}', dpi=300, facecolor='white',
                    metadata={'Title': 'Vessel categories and useful acoustic details'})
    plt.close(fig)
    table = {
        'source': str(source.relative_to(ROOT)), 'source_sha256': file_hash(source),
        'presentation_only': True, 'classifier_seed': 17,
        'classifier_overall_correct': int(correct.sum()), 'classifier_recordings': int(totals.sum()),
        'classifier_accuracy': anchor['accuracy'], 'categories': categories,
        'reporting': reporting, 'learned_minus_locked_heuristic_counts': deltas,
        'template_minus_locked_heuristic_counts': template_deltas,
        'whole_test_useful_difference': contrast,
        'definitions': {
            'category_percentage': 'Recall: correct labels / all recordings in that dataset category.',
            'confusion_percentage': 'Most frequent wrong label / all recordings in that dataset category, not only errors.',
            'supported_detail': 'An asserted physical slot passing the frozen numerical-reference and citation-support checks.',
            'wrong_or_unsupported_detail': 'An asserted physical slot that is incorrect or lacks required support.',
            'intervals': 'Previously computed source-bootstrap whole-test useful-coverage intervals, multiplied by 400 for display.',
            'comparison': 'Calibrated systems on the same 100 requested clips, with different retained subsets and tool choices; not an isolated generator effect.',
        },
    }
    atomic_json(ROOT / 'results/tables/operator_context.json', table)
    caption = (
        'Operator-oriented reading guide to the saved UniqueShip civilian-recording study. '
        f"Left: the shared classifier correctly labels {int(correct.sum()):,} of {int(totals.sum()):,} "
        f"test clips ({100 * anchor['accuracy']:.1f}%). Category bars show recall, with exact counts and the "
        'most frequent wrong label; confusion percentages use all clips in the relevant category. '
        'Right: three calibrated reporting systems evaluated on the same 100 clips, requesting four physical '
        'details per clip. Supported and wrong/unsupported assertions are counts across all 400 requested slots. '
        'Uncertain slots and withheld reports contribute no supported assertion. Whiskers rescale the existing '
        'source-bootstrap 95% intervals to 400 requests. Retained subsets and tool choices differ between systems. '
        f"Learned selection uses {base['tool_calls'] - learned['tool_calls']} fewer tool calls, returns "
        f"{base['supported_details'] - learned['supported_details']} fewer supported details, and makes "
        f"{base['wrong_or_unsupported_details'] - learned['wrong_or_unsupported_details']} fewer wrong/unsupported assertions "
        'than the locked heuristic. This is a completeness/cost/error trade-off, not an improvement in category accuracy. '
        f"The fixed template supplies {template_deltas['supported_details']} more supported details and "
        f"{template_deltas['wrong_or_unsupported_details']} more wrong/unsupported details than the heuristic "
        'at its own calibrated operating point. Support is an automated proxy; no human/operator study or operational '
        'sonar validation was performed. No new experiments or test tuning were used for this figure.\n'
    )
    (out / 'operator_context_caption.txt').write_text(caption)
    print(f'Created operator_context.png/.pdf; category accuracy {anchor["accuracy"]:.4%}; '
          f'learned-minus-heuristic counts {deltas}')


if __name__ == '__main__':
    main()
