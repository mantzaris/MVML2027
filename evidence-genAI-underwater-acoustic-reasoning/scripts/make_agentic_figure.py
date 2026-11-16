"""Show implemented agent decisions and their measured reporting tradeoffs.

Presentation only: the two frozen cohorts remain separate. No new model calls.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

from underwater_reasoning.io import atomic_json, file_hash

OUT = ROOT / 'results/figures'
INK, MUTED = '#173247', '#526373'
TEAL, RED, GREY = '#00857b', '#c65242', '#dce3e8'
BLUE, PALE = '#2c5a88', '#eef4f7'


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    old = read('results/tables/main_results.json')
    fresh = read('results/tables/identification_results.json')
    assert old['assessment_clips'] == 100 and fresh['fresh_clean_clips'] == 235
    methods = [('B8_deterministic', 'Fixed template'),
               ('B9_entropy', 'Heuristic + Omni'),
               ('B6_planner', 'LLM planner + Omni'),
               ('P', 'Learned agent + Omni')]
    report_rows = []
    for key, name in methods:
        result = old['methods'][key]
        calls = result['mean_tools_all'] * result['total_reports']
        assert np.isclose(calls, round(calls)) and result['total_reports'] == 100
        report_rows.append({'method': key, 'label': name, 'tool_calls': round(calls),
                            'supported_details': result['supported_assertions'],
                            'wrong_or_unsupported_details': result['wrong_or_unsupported_assertions'],
                            'reports_returned': result['retained_reports']})
    heuristic, agent = report_rows[1], report_rows[-1]
    reduction = 100 * (1 - agent['tool_calls'] / heuristic['tool_calls'])
    fewer_wrong = heuristic['wrong_or_unsupported_details'] - agent['wrong_or_unsupported_details']
    fewer_supported = heuristic['supported_details'] - agent['supported_details']
    generation_rows = []
    for key, name in [('raw', 'Audio only'), ('acoustic_prior', 'Audio + acoustic prior'),
                      ('fused_evidence', 'Audio + fused evidence')]:
        result = fresh['omni'][key]
        counts = [result['correct_issued'], result['wrong_issued'], result['referred']]
        assert result['complete'] and result['valid'] == result['saved'] == sum(counts) == 235
        generation_rows.append({'method': key, 'label': name, 'correct': counts[0],
                                'wrong': counts[1], 'withheld': counts[2], 'macro_f1': result['macro_f1']})
    prior, grounded = generation_rows[1:]
    assert prior['correct'] == grounded['correct']
    error_change = prior['wrong'] - grounded['wrong']
    withholding_change = grounded['withheld'] - prior['withheld']
    assert error_change == withholding_change

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(7.1, 5.45), facecolor='white')
    fig.text(.04, .972, 'What the agent decides, and what evidence changes',
             color=INK, fontsize=12, weight='bold')
    fig.text(.04, .925, 'A  Adaptive tool use in the earlier report-support experiment',
             color=INK, fontsize=9.5, weight='bold')
    flow = fig.add_axes([.04, .645, .92, .26])
    flow.set(xlim=(0, 1), ylim=(0, 1)); flow.axis('off')

    def box(x, y, w, h, title, detail, face=PALE, edge=BLUE):
        flow.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.025',
                                     linewidth=1, edgecolor=edge, facecolor=face))
        flow.text(x+w/2, y+h*.71, title, ha='center', va='center', fontsize=8.5, color=INK, weight='bold')
        flow.text(x+w/2, y+h*.32, detail, ha='center', va='center', fontsize=8, color=INK, linespacing=1.2)

    def arrow(points, color=BLUE):
        for start, end in zip(points[:-2], points[1:-1]):
            flow.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=1.15)
        flow.annotate('', xy=points[-1], xytext=points[-2],
                      arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': 1.15, 'mutation_scale': 9})

    box(.015, .54, .23, .39, 'Observed state', 'Acoustic prior\n+ acquired evidence')
    box(.33, .54, .26, .39, 'Learned controller', 'Expected report benefit\nminus query cost', '#e4f3ef', TEAL)
    box(.745, .54, .24, .39, 'Omni report', 'Evidence citations\nor uncertainty')
    box(.33, .02, .655, .30, 'Query one available tool',
        'A Spectrum   B Modulation   C Time / quality   D Similar recordings')
    arrow([(.245, .735), (.33, .735)])
    arrow([(.59, .735), (.745, .735)])
    flow.text(.667, .81, 'STOP', ha='center', color=TEAL, weight='bold', fontsize=8)
    arrow([(.46, .54), (.46, .32)], TEAL)
    flow.text(.48, .415, 'QUERY', va='center', color=TEAL, weight='bold', fontsize=8)
    arrow([(.33, .17), (.13, .17), (.13, .54)], TEAL)
    flow.text(.23, .36, 'Update\nstate', ha='center', va='center', color=TEAL, fontsize=8)
    fig.text(.04, .625, 'A learned neural policy chooses at most two queries; only acquired cards enter the next decision.',
             fontsize=7.8, color=MUTED)

    fig.text(.04, .585, 'B  Acquisition and report content: same 100 clips, 400 physical details requested',
             fontsize=9, weight='bold', color=INK)
    tabax = fig.add_axes([.04, .405, .92, .16]); tabax.axis('off')
    cells = [[row['label'], row['tool_calls'], row['supported_details'],
              row['wrong_or_unsupported_details'], row['reports_returned']] for row in report_rows]
    table = tabax.table(cellText=cells,
                        colLabels=['System', 'Logical\nqueries', 'Supported\ndetails', 'Wrong /\nunsupported', 'Reports\nreturned'],
                        colWidths=[.32, .14, .17, .20, .17], cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(8)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#c9d5de'); cell.set_linewidth(.45)
        cell.set_text_props(color=INK)
        if column == 0: cell.set_text_props(ha='left')
        if row == 0:
            cell.set_facecolor(PALE); cell.set_text_props(weight='bold', fontsize=7.8); cell.set_height(.30)
        else:
            cell.set_height(.17)
        if row == len(report_rows):
            cell.set_facecolor('#e4f3ef'); cell.set_text_props(weight='bold')
    fig.text(.04, .379, f'Agent vs heuristic: {reduction:.1f}% fewer queries; {fewer_wrong} fewer wrong details; '
             f'{fewer_supported} fewer supported details.', color=INK, fontsize=8.1)
    fig.text(.04, .351, 'Cached replay, automated support checks, different retention thresholds; no measured runtime saving.',
             color=MUTED, fontsize=7.8)

    fig.text(.04, .305, 'C  Generative identification: separate 235 clips, fixed evidence inputs',
             fontsize=9.5, weight='bold', color=INK)
    bars = fig.add_axes([.30, .15, .66, .132])
    offset = np.zeros(3)
    for key, color in [('correct', TEAL), ('wrong', RED), ('withheld', GREY)]:
        values = np.array([row[key] for row in generation_rows])
        bars.barh(np.arange(3), values, left=offset, color=color, height=.65)
        for y, value, start in zip(range(3), values, offset):
            if value:
                bars.text(start+value/2, y, str(value), ha='center', va='center', fontsize=9,
                          color=INK if key == 'withheld' else 'white', weight='bold')
        offset += values
    bars.set(xlim=(0, 235), yticks=range(3), yticklabels=[row['label'] for row in generation_rows], xticks=[])
    bars.invert_yaxis(); bars.tick_params(axis='y', length=0, labelsize=8)
    for spine in bars.spines.values(): spine.set_visible(False)
    handles = [Patch(facecolor=TEAL, label='Correct category'), Patch(facecolor=RED, label='Wrong category'),
               Patch(facecolor=GREY, label='Withheld')]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.57, .119), ncol=3, frameon=False,
               fontsize=8, handlelength=1, columnspacing=1.2)
    fig.text(.04, .076, f'Fused vs acoustic prior: {error_change} fewer wrong labels; {withholding_change} more withheld; '
             f'correct total stays {grounded["correct"]}.', fontsize=8.1, color=INK)
    fig.text(.04, .041, 'No net identification gain is established. The follow-up uses fixed fusion, not the adaptive loop above.',
             fontsize=7.8, color=MUTED)
    for suffix in ['png', 'pdf']:
        options = {'dpi': 300, 'pil_kwargs': {'optimize': True}} if suffix == 'png' else {
            'metadata': {'Author': 'Anonymous', 'Title': 'Agentic acquisition and evidence-conditioned generation'}}
        fig.savefig(OUT / f'agentic_reasoning_contribution.{suffix}', facecolor='white', **options)
    plt.close(fig)

    caption = ('Agentic decisions and observed effects. A–B depict the earlier frozen report-support study: '
               'a learned controller queries acoustic tools using only acquired evidence, or stops for Omni generation; '
               'four methods are shown on the same 100 requested clips. Details use automated support proxies and '
               'report-retention thresholds differ. Queries count logical acquisitions in cached replay, not measured '
               'end-to-end runtime or billed GPU time. C shows three fixed-input Omni conditions on a separate cohort '
               'of 235 recordings. Evidence changes both class probabilities and measured inputs; the bars do not '
               'isolate a physical-feature effect or demonstrate an autonomous reasoning process. Fewer tool calls '
               'and fewer errors accompany losses of supported information or increased withholding. '
               'No human benefit or primary identification improvement is claimed. All counts use saved frozen outcomes.\n')
    (OUT / 'agentic_reasoning_contribution_caption.txt').write_text(caption)
    atomic_json(ROOT / 'results/tables/agentic_reasoning_contribution.json', {
        'presentation_only': True, 'new_model_calls': 0,
        'cohorts_combined_for_scoring': False,
        'earlier_report_study': {'clips': 100, 'physical_details_requested': 400, 'methods': report_rows,
                                 'agent_vs_heuristic': {'tool_call_reduction_percent': reduction,
                                                      'fewer_wrong_details': fewer_wrong,
                                                      'fewer_supported_details': fewer_supported}},
        'fresh_identification_study': {'clips': 235, 'methods': generation_rows,
                                       'fused_vs_acoustic_prior': {'fewer_wrong': error_change,
                                                                 'more_withheld': withholding_change,
                                                                 'additional_correct': grounded['correct']-prior['correct']}},
        'source_sha256': {path: file_hash(ROOT/path) for path in [
            'results/tables/main_results.json', 'results/tables/identification_results.json',
            'src/underwater_reasoning/controller.py', 'src/underwater_reasoning/acquisition.py',
            'src/underwater_reasoning/dsp.py']},
        'interpretation': caption.strip(),
    })
    print('Agentic figure, caption and independently checked aggregate values written; no model calls')


if __name__ == '__main__':
    main()
