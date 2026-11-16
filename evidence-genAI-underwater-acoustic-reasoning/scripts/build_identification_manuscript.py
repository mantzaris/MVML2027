"""Build the identification-focused anonymous paper in the official template."""
from copy import deepcopy
from pathlib import Path
import json
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT/'results/tables/identification_results.json').read_text())
TRAIN = json.loads((ROOT/'results/manifests/identification_training.json').read_text())
COHORT = json.loads((ROOT/'results/manifests/identification_cohort.json').read_text())
OLD = json.loads((ROOT/'results/tables/main_results.json').read_text())
CLEAN = R['conditions']['clean']['methods']
BASE = R['primary_comparator']
P, B = CLEAN['fusion'], CLEAN[BASE]
D = R['conditions']['clean']['primary_contrast']['macro_f1']
pct = lambda x: 'NA' if x is None else f'{100*x:.1f}'
num = lambda x: 'NA' if x is None else f'{x:.3f}'
NAMES = {'anchor':'Single acoustic model', 'ensemble':'Three-model ensemble', 'acoustic_refit':'Acoustic-only refit',
         'fusion':'Evidence fusion (proposed)', 'fusion_maxprob':'Fusion: confidence only', 'physical_only':'Measured features only',
         'no_physical':'Without physical features', 'no_prototypes':'Without prototypes', 'prototype_vote':'Prototype voting'}
LABELS = ['Background', 'Cargo', 'Passenger', 'Tanker', 'Tug/tow']
DOC=Document(ROOT/'manuscript/template/MVML27_Paper_Template.docx')
proto={'title':deepcopy(DOC.paragraphs[1]._p),'author':deepcopy(DOC.paragraphs[3]._p),'body':deepcopy(DOC.paragraphs[17]._p),'heading':deepcopy(DOC.paragraphs[16]._p)}
body=DOC._element.body
for node in list(body):
 if not node.tag.endswith('}sectPr'):body.remove(node)
PROSE=[]
def add(text,kind='body'):
 node=deepcopy(proto[kind]);body.insert(len(body)-1,node);p=Paragraph(node,DOC._body);p.text=text
 for run in p.runs:
  run.font.name='Arial' if kind in ['title','heading'] else 'Times New Roman';run.font.size=Pt({'title':16,'heading':12,'author':12,'body':11}[kind]);run.font.bold=kind in ['title','heading','author']
 if kind in ['title','author']:p.alignment=WD_ALIGN_PARAGRAPH.CENTER
 if kind=='heading':p.paragraph_format.keep_with_next=True
 if text.startswith('Abstract'):
  for run in p.runs:run.font.size=Pt(10)
 PROSE.append(text);return p

def table(caption,headers,rows):
 p=add(caption)
 for run in p.runs:run.font.size=Pt(10)
 p.paragraph_format.keep_with_next=True
 tab=DOC.add_table(rows=1,cols=len(headers));tab.style='Table Grid'
 for cell,text in zip(tab.rows[0].cells,headers):cell.text=str(text)
 for values in rows:
  for cell,value in zip(tab.add_row().cells,values):cell.text=str(value)
 if headers[0] in ('Method','Comparator','Intervention'):
  tab.autofit=False;widths=[2.1]+[(7.3-2.1)/(len(headers)-1)]*(len(headers)-1)
  for column,width in zip(tab.columns,widths):column.width=Inches(width)
  for row in tab.rows:
   for cell,width in zip(row.cells,widths):cell.width=Inches(width)
 repeat=OxmlElement('w:tblHeader');repeat.set(qn('w:val'),'true');tab.rows[0]._tr.get_or_add_trPr().append(repeat)
 for row_index,row in enumerate(tab.rows):
  row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
  for column_index,cell in enumerate(row.cells):
   for p in cell.paragraphs:
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if column_index==0 and headers[0]!='Seed' else WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent=Inches(0);p.paragraph_format.left_indent=Inches(0);p.paragraph_format.right_indent=Inches(0)
    p.paragraph_format.keep_with_next=row_index==0
    p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2)
    for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(10)
 for cell in tab.rows[0].cells:
  for p in cell.paragraphs:
   for run in p.runs:run.font.bold=True
 PROSE.append(' | '.join(headers)+'\n'+'\n'.join(' | '.join(map(str,r)) for r in rows))

def figure(name,caption,width=7.0):
 p=DOC.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
 p.add_run().add_picture(str(ROOT/f'results/figures/{name}.png'),width=Inches(width));p=add(caption)
 for run in p.runs:run.font.size=Pt(10)

add('Evidence-Grounded Generative AI for Underwater Acoustic Reasoning','title')
add('Anonymous author','author')
omni = R['omni']
assert all(v['complete'] for v in omni.values()), 'Manuscript requires every requested generative comparison'
success = 'The prospective identification success criterion was met.' if R['primary_success'] else 'The prospective identification success criterion was not met.'
add(f"Abstract — Acoustic assistance must identify the sound-source category before its interpretation can support a review decision. We investigate whether measured acoustic features and source-balanced training analogies improve vessel-category identification, and whether a generative model preserves those identifications. A frozen acoustic representation is combined with 62 spectral, modulation and quality features and 15 class-conditional prototype features. A disagreement score prioritizes uncertain recordings for review. Models and thresholds were frozen before evaluating 235 previously unused recordings from 218 recorded source groups, covering background, cargo, passenger, tanker and tug/tow. The proposed fusion attained {pct(P['accuracy'])}% accuracy and {num(P['macro_f1'])} macro-F1; the development-selected three-model ensemble attained {pct(B['accuracy'])}% and {num(B['macro_f1'])}. The macro-F1 difference was {pct(D['difference'])} percentage points (paired source-bootstrap 95% interval {pct(D['ci95'][0])} to {pct(D['ci95'][1])}). {success} Across 705 generative classifications, audio-only prompting withheld every category, while fused-evidence prompting correctly identified {pct(omni['fused_evidence']['accuracy'])}% of all requested recordings. Six waveform perturbations and a supplementary untrained-category challenge test reliability boundaries. Results quantify correct identifications, wrong labels and review referrals, using dataset category labels and no human reviewers. This same-collection study does not establish individual vessel recognition, continuous detection or improved human performance.")
add('Keywords: underwater acoustics; vessel identification; evidence fusion; selective classification; generative audio; operator support')

add('1. Introduction and Related Work','heading')
add('For an acoustic operator, identifying the source category is an immediate practical question: does a recording resemble background, a cargo vessel, a passenger vessel, a tanker or a tug? A useful automated answer must also indicate when that category is unresolved. Spectrograms, detections, classification and review are established parts of passive-acoustic software workflows [1]. A readable explanation adds little if it changes a correct category into a wrong one, or if apparent reliability is obtained by withholding nearly every useful answer. We therefore evaluate identification and referral outcomes directly, rather than treating fluent acoustic prose as the principal result.')
add('Identification assigns a civilian dataset category to a five-second recording; background does not prove vessel absence. The proposed method fuses a frozen neural representation, source-balanced prototypes and measured acoustic features, with a disagreement score for referral. Controls include a matched acoustic-only refit, a three-model ensemble, feature ablations and generative predictions. The contribution is this controlled identification/referral assessment and explicit accounting of generated overrides. Individual hull identity, contact counts and location are outside its scope.')
add('Underwater feature fusion is established: Wang et al. combine multiple acoustic feature families with a modified neural classifier [2]. SelectiveNet jointly learns classification and rejection [3]; our post-hoc confidence/disagreement thresholds are a different, simpler construction and carry no imported risk guarantee. UniqueShip emphasizes source separation in ship-classification evaluation [4]. These works motivate both strong classification baselines and explicit accounting for recordings withheld from automatic reporting.')
add('Audio-agent systems already acquire evidence and use analytical tools. AuTAgent learns tool selection [5], while Audio-Mind, SEAR and ACE-Cap investigate auditable or adaptive acoustic evidence [6–8]. From Signals to Evidence Parts I and II address physics-aware underwater interpretation using simulated data [9,10]. The latter reports and [6–8] are preprints in our verified source record. Our real-recording comparison asks whether the evidence actually improves the category decision and whether subsequent generation preserves it. Citation support, factual correctness and completeness remain different properties [11].')

add('2. Motivation, Data and the Fresh Evaluation','heading')
old_p=OLD['methods']['P'];old_b=OLD['methods'][OLD['locked_comparator']];old_d=OLD['methods']['B8_deterministic']
add('An earlier frozen report-support experiment motivated this follow-up. A learned neural controller selects spectral, modulation, temporal/quality or retrieval evidence, updates its observed state, and stops after at most two queries for Omni to generate a cited report. Unqueried tool results are hidden. Evaluation replays cached tool cards and terminal generations; query counts measure logical acquisition, not online runtime or billed savings. On 100 clips, it makes 101 queries versus the heuristic’s 200, returning 37 supported details versus 53; wrong/unsupported details fall from nine to one. Figure 3A–B includes LLM-planner and deterministic controls. The original success criterion failed; these automated proxies and different reporting thresholds motivated a category-identification endpoint.')
add('We use UniqueShip v1.0, fold 1 of the five-hour-per-class release, retaining background, cargo, passenger, tanker and tug/tow. The original acquired recordings are mono PCM at 32 kHz and five seconds long. Five duration mismatches and 42 filename/metadata date conflicts were excluded. Original partitions separate recorded vessels, background UTC-day groups and entire UTC days across all classes/hydrophones, with a global 60-second gap between clips in different partitions. The acoustic backbone and prototype bank use acoustic-fit only; the follow-up heads use policy-fit; development selects settings and calibration fixes final reporting thresholds.')
table('Table 1. Original fitting partitions and fresh category-identification cohort. Source groups are recorded vessel IDs or background days.',
      ['Partition','Clips','Source groups','Role'],[
          ['Acoustic-fit','2,459','658','Backbone, PCA, prototype bank'],
          ['Policy-fit','1,827','411','Identification refits'],
          ['Development','384','91','Settings and temperature'],
          ['Calibration','508','103','Reporting thresholds'],
          ['Original test','2,240','479','Prior study only'],
          ['Fresh identification test','235','218','Locked follow-up evaluation']])
add('The fresh cohort is selected from release members that were excluded from the original retained partitions. Every new recording ID and source group is absent from all 7,418 original records. Its UTC day is absent from acoustic-fit, policy-fit, development and calibration, and its start time is at least 65 seconds from every non-test recording start. Deterministic source-first sampling requests 50 clips per category, at most three per source and at least ten minutes apart. Passenger capacity is only 35 eligible clips, so this shortfall is retained. The resulting category counts are 50, 50, 35, 50 and 50, with 50, 50, 25, 50 and 43 source groups, respectively.')
add('This is a custom follow-up cohort, not the official fold test set: 216 of its unused members originated in the release training metadata and 19 in validation metadata. None entered the fitted models. All 185 fresh-cohort UTC days overlap days from the earlier study’s test partition, whose aggregate results informed the new research direction. Thus the evaluation concerns previously unused sources within the same collection and acoustic environments. It is not independent deployment validation. Parent-recording and deployment identifiers remain unavailable, and missing background hydrophone fields prevent a stronger provenance claim.')
add('Each new WAV member passes archive size and CRC checks; decoded waveform hashes show no duplicates within the cohort or against the original data. Source identities, dates, hydrophone fields and true labels are excluded from prediction inputs. Source IDs are used only for splitting, weighting, prototype construction and uncertainty estimation. Dataset category labels are the identification targets; they are not substituted with AI judgments. No human acoustic reviewers or measured hypothetical-human decisions are introduced. The deliberately source-respecting category mix is not an estimate of operational traffic prevalence.')

add('3. Evidence-Based Category Identification','heading')
add('The fixed seed-17 Swin V2 Tiny [12] receives anti-aliased 20 kHz audio as 128 log-mel bands over 20–8,000 Hz (1,024-sample window/FFT, 256-sample hop). A 90 dB relative range is resized to 256×256, replicated across three channels and ImageNet-normalized. Original AdamW training uses batch 32, head rate 0.001, weight decay 0.01, up to 30 epochs and patience five. Development selects backbone rate 10⁻⁵ from {10⁻⁶,10⁻⁵,3×10⁻⁵}; independent seeds 29 and 43 complete the ensemble. Fresh recordings never fine-tune a backbone.')
add('For the acoustic-only refit, the input contains 32 PCA coordinates of the anchor embedding and five log class probabilities. PCA is fitted on acoustic-fit embeddings only. The reference ensemble averages the separately temperature-calibrated probability vectors from the three original seeds. It requires three backbone evaluations; the proposed fusion uses the single seed-17 representation plus numerical measurements. This architectural difference is reported without converting cached batch timings into a claimed online speedup.')
add('Numerical evidence comprises 62 values. A native-rate Welch spectrum with an 8,192-sample window and 4,096-sample overlap supplies 32 logarithmically spaced relative band powers over 20–8,000 Hz, four broad-band energy fractions, normalized centroid and spread, spectral entropy, flatness, top-five-bin concentration and waveform crest factor. Four quality/temporal values describe first/last-second relative energy change, quarter-second RMS variability, exact-zero fraction and digital full-scale fraction. None is calibrated sound pressure. Sixteen modulation features are relative envelope-spectrum energies in eight intervals between 2 and 80 Hz for each of two carrier bands.')
add('Modulation extraction downsamples to 8 kHz with anti-aliasing, filters 100–1,000 and 1,000–3,500 Hz with fourth-order filters, removes 0.25 s Hilbert-envelope edges and resamples envelopes to 400 Hz. These are received modulation features, not shaft speed or blade count. Relative energies remove ordinary gain dependence; clipping and zero fractions retain digital-quality information. Synthetic checks verify gain invariance and known-band concentration.')
add('The exemplar bank contains one normalized mean embedding per acoustic-fit source. For each of five classes, we compute maximum cosine similarity and mean similarity to its three closest distinct-source prototypes. The maxima and means are centered across classes; a softmax of the five means with temperature 0.1 supplies five further values, giving 15 exemplar features. A repeated long recording therefore cannot contribute many separate prototype votes. Query-source prototypes are excluded. These are statistical class analogies, not a claim that the query shares the exemplars’ physical mechanism or identity.')
add('The full model concatenates 37 acoustic, 62 physical and 15 exemplar values into a 114-dimensional vector. Weighted standardization and multinomial logistic regression are fitted on policy-fit only. The sample weight is proportional to 1/(Gᶜ nᵍ), where Gᶜ is the number of recorded sources in class c and nᵍ the number of policy-fit clips from source g; weights are normalized to mean one. This gives equal total fitting mass to each class and each source within class. All refits share C ∈ {0.001, 0.01, 0.1, 1, 10}, a maximum of 2,000 optimizer iterations and development macro-F1 selection. A scalar probability temperature minimizes development negative log-likelihood after model selection.')
add(f"The full model selects C={TRAIN['search']['fusion']['selected_C']:g}. Baselines include the single anchor, three-model ensemble, acoustic-only refit, physical-only refit and prototype voting. Ablations remove the physical features or prototypes while retaining the same fit labels, standardization, weighting and C search. A pre-freeze diagnostic also tested fixed mixtures of each refit with the ensemble at weights 0, 0.25, 0.5, 0.75 and 1; none improved the ensemble’s development macro-F1. Those mixtures were not adopted. The full fusion remains the proposed method even when a baseline or ablation is stronger.")

add('4. Review Decisions, Generative Comparisons and Statistics','heading')
add('Let p be the fused category distribution, pᴹ the physical-only classifier distribution and pᴿ the prototype distribution. The review score is s = max(p) − γ[JSD(p,pᴹ) + JSD(p,pᴿ)]/2, with Jensen–Shannon divergence normalized to the interval [0,1]. Development chooses γ from {0,0.25,0.5,1} by accuracy among its top 70% ranked recordings; γ=0.25 is selected. A confidence-only ablation uses the same fused predictions with γ=0, isolating the ranking contribution from category classification. Disagreement is a heuristic indicator, not a calibrated probability that an error has occurred.')
add('For each method, the calibration threshold is the score of the ceil(0.70×508)=356th highest calibration recording, including ties. This fixes a deployable rule before evaluating fresh recordings; it does not guarantee 70% reporting on a shifted cohort. We count correct issued labels, wrong issued labels and referrals over every requested recording. A referral never counts as a correct identification. Separately, exact test-ranked reporting curves compare equal numbers of issued labels; these curves are descriptive and do not supply deployment thresholds. Vessel/background classification sums the four vessel probabilities and uses a development-selected binary threshold, without introducing event detection or false alerts per hour.')
add('The frozen Qwen2.5-Omni-7B Thinker [13] supplies three supplementary identification conditions: original audio alone; audio with the anchor probabilities; and audio with fused probabilities, broad-band relative energies, received-energy change, clipping fraction and prototype class analogies. Every condition uses the same category-only prompt, greedy decoding, a 128-token limit, one call and a JSON category plus one short reason. The native processor receives anti-aliased 16 kHz audio. A development-only format pilot returns 20 valid records from 20 requests. No query labels or provenance identifiers enter these prompts. No human-like skill is imputed to the model.')
add('All three generative conditions evaluate every fresh clean recording, yielding 705 requested category answers. One deterministic JSON extraction is permitted, with no second generation or content repair. Truncated, malformed and uncertain outputs remain in the denominator as withheld/failed identifications. For prior-assisted conditions, we count previously correct numeric classifications changed to an incorrect or withheld answer, and previously wrong classifications corrected by generation. The proposed operator record keeps the numeric classifier’s category authoritative; a generated interpretation is supplementary and cannot be credited with improving identification merely by repeating that category.')
add(f"A timestamped internal freeze at {R['freeze_utc']} records the cohort, fitting outputs, model hashes, prompts, analysis code and decision rules before any fresh test inference. This is prospective follow-up documentation, not an external preregistration, and the prior study’s results were already known. The ensemble is locked as the primary comparator because it has the highest development macro-F1 among the anchor, ensemble and matched acoustic-only refit. The primary endpoint is five-class macro-F1 across all fresh clips before rejection. Success requires a gain of at least 0.03 and a positive paired source-bootstrap 95% lower bound.")
add('Uncertainty uses 2,000 paired bootstrap resamples of recorded source groups, retaining all clips from each sampled group and keeping fitted parameters and thresholds fixed. A UTC-day-cluster sensitivity addresses shared-day dependence. We report accuracy, class recalls and confusion counts alongside macro-F1. Per-class, ablation, generative and degradation findings are exploratory; they do not replace the single primary criterion. Six fixed transformations—Gaussian noise at 20, 10 and 0 dB SNR, a 2 kHz low-pass, gain 0.25 and RMS-scaled digital clipping—are applied to all 235 fresh recordings. Every condition receives new acoustic features and classifier predictions without retraining. Nominal category labels remain unchanged, but recognizability after severe distortion is not independently annotated.')

add('5. Identification Results and Review Tradeoffs','heading')
add(f"The full fusion correctly identifies {round(P['accuracy']*235)}/235 recordings ({pct(P['accuracy'])}%), compared with {round(CLEAN['anchor']['accuracy']*235)}/235 ({pct(CLEAN['anchor']['accuracy'])}%) for the single anchor, {round(CLEAN['acoustic_refit']['accuracy']*235)}/235 ({pct(CLEAN['acoustic_refit']['accuracy'])}%) for the matched acoustic-only refit and {round(B['accuracy']*235)}/235 ({pct(B['accuracy'])}%) for the locked ensemble. Table 2 separates full-cohort identification from automatically issued labels. The primary macro-F1 difference is {pct(D['difference'])} points, with interval {pct(D['ci95'][0])} to {pct(D['ci95'][1])}; the UTC-day interval is {pct(R['utc_day_sensitivity']['macro_f1']['ci95'][0])} to {pct(R['utc_day_sensitivity']['macro_f1']['ci95'][1])}. {success} The interval does not establish either superiority or equivalence to the ensemble.")
table('Table 2. All 235 fresh recordings. Accuracy and macro-F1 precede rejection; final columns count outcomes at separately frozen reporting thresholds.',
      ['Method','Accuracy %','Macro-F1','Correct issued','Wrong issued','Referred'],
      [[NAMES[m],pct(CLEAN[m]['accuracy']),num(CLEAN[m]['macro_f1']),CLEAN[m]['correct_issued'],CLEAN[m]['wrong_issued'],CLEAN[m]['referred']]
       for m in ['anchor','ensemble','acoustic_refit','physical_only','prototype_vote','no_physical','no_prototypes','fusion','fusion_maxprob']])
add('The class-level tradeoff is substantial (Fig. 1). Fusion identifies 31/50 tankers, compared with 9/50 for the ensemble, but only 25/50 cargo vessels versus 40/50. The paired recall changes are +44.0 points for tanker (pointwise 95% interval +28.0 to +60.0) and −30.0 for cargo (−48.0 to −12.0). The matched acoustic-only refit also identifies 31 tankers, so that tanker improvement cannot be attributed specifically to the added measurements. Relative to that matched refit, fusion gains five correct background labels and six tug/tow labels, loses one cargo label, and leaves passenger/tanker totals unchanged. These are exploratory category comparisons, not simultaneous confidence claims.')
diag=json.loads((ROOT/'results/tables/identification_diagnostics.json').read_text())
physical_delta=diag['additional_paired_contrasts']['no_physical']['macro_f1']
matched_delta=diag['additional_paired_contrasts']['acoustic_refit']['macro_f1']
add(f"Removing physical features reduces macro-F1 from {num(P['macro_f1'])} to {num(CLEAN['no_physical']['macro_f1'])}; the exploratory paired improvement is {pct(physical_delta['difference'])} points (pointwise interval {pct(physical_delta['ci95'][0])} to {pct(physical_delta['ci95'][1])}). Relative to the acoustic-only refit, the improvement is {pct(matched_delta['difference'])} points ({pct(matched_delta['ci95'][0])} to {pct(matched_delta['ci95'][1])}). Removing prototypes gives {num(CLEAN['no_prototypes']['macro_f1'])}, almost the same as full fusion. Thus measured features contribute within the prototype-augmented model, while clear incremental value from prototypes over acoustic-plus-physical features is not established.")
figure('identification_class_accuracy','Fig. 1. Category recall before rejection, with the full cohort denominator for each category. The matched acoustic refit shows which changes already arise without physical or exemplar evidence.',width=6.8)
add(f"At the frozen reporting thresholds, fusion issues {P['issued']} labels: {P['correct_issued']} correct and {P['wrong_issued']} wrong, referring {P['referred']} recordings. The ensemble issues {B['issued']}: {B['correct_issued']} correct and {B['wrong_issued']} wrong, referring {B['referred']}. Accuracy among issued labels is {pct(P['conditional_accuracy'])}% versus {pct(B['conditional_accuracy'])}%, but fusion returns fewer labels. At the same descriptive count of 165 issued labels, fusion gives 115 correct and 50 wrong versus the ensemble’s 110 and 55. The confidence-only fusion ablation also gives 115 and 50 at that count. This run therefore provides no additional benefit for the disagreement term at that matched reporting point. These rankings are not frozen deployment thresholds.")
add(f"Broad vessel/background balanced accuracy is {pct(P['vessel_background_balanced_accuracy'])}% for fusion and {pct(B['vessel_background_balanced_accuracy'])}% for the ensemble on the five-category cohort. This easier distinction must not be presented as five-class object-identification accuracy. A prototype-channel permutation changes 101 category predictions, converting 55 correct labels to errors and correcting 28 errors; accuracy falls to {pct(R['prototype_intervention']['metrics']['accuracy'])}%. That intervention demonstrates reliance on the fitted evidence channels, while the no-prototype ablation shows that reliance is not itself proof of an incremental benefit.")

add('6. Degraded Recordings and Untrained Categories','heading')
add('Added noise substantially reduces identification accuracy. At 20 dB SNR, fusion reaches 48.5% versus 44.7% for the ensemble; at 10 dB, 35.7% versus 30.6%; at 0 dB, 22.6% versus 22.1%. These are exploratory stress outcomes. Gain 0.25 leaves the category predictions unchanged, consistent with relative-energy normalization. Low-pass filtering reduces fusion to 49.8%, below the ensemble’s 54.9%; the physical-only classifier falls to 22.1%. The explicit acoustic features are therefore not uniformly robust to loss of the upper frequency range. Digital clipping changes accuracy only slightly in this particular transformation and cohort; this does not establish tolerance to arbitrary clipping.')
add('The review mechanism fails to warn reliably under severe noise. Fusion’s automatic reporting rate rises from 56.2% on clean recordings to 87.2% at 0 dB SNR. It then issues 205 labels, of which 50 are correct and 155 wrong: only 24.4% accuracy among issued labels. A clean-calibrated disagreement penalty can therefore return more answers while the evidence becomes less informative. This is directly relevant to an identification assistant: a high score should not be treated as assurance that unfamiliar recording conditions are safe for automatic interpretation.')
figure('identification_robustness','Fig. 2. Percentage of all 235 recordings assigned the correct nominal category in each condition. Every cell includes every recording, with no rejection, tuning or omission of difficult cases.',width=6.8)
untrained=json.loads((ROOT/'results/tables/untrained_categories_results.json').read_text())
unknown_p=untrained['methods']['fusion']['overall'];unknown_b=untrained['methods']['ensemble']['overall']
add('After the primary results were available, we added a supplementary challenge using 72 previously unused recordings from 65 additional sources: pilot vessel (3 clips/1 source), rescue, sailing, pleasure craft and fishing (15 clips/15 sources each), and dredger (9 clips/4 sources). These explicit release categories were never fitted; the ambiguous catch-all category was excluded. Source/date/time exclusions and decoded-hash checks were repeated against both earlier cohorts. A separate pre-inference freeze records this exploratory cohort, while all five-class models and thresholds remain unchanged. The small pilot-vessel and dredger groups sharply limit category-specific inference.')
table('Table 3. Supplementary untrained-category challenge. Issued trained labels are out-of-vocabulary assignments under the release taxonomy; broad vessel recognition is a separate outcome.',
      ['Method','Assigned trained label','Referred','Recognized broadly as vessel %'],
      [[NAMES[m],untrained['methods'][m]['overall']['trained_category_issued'],untrained['methods'][m]['overall']['referred'],pct(untrained['methods'][m]['overall']['broad_vessel_recognition'])]
       for m in ['anchor','ensemble','acoustic_refit','fusion','fusion_maxprob']])
add(f"Fusion assigns trained-category labels to {unknown_p['trained_category_issued']}/72 untrained-category clips ({pct(unknown_p['trained_category_issue_rate'])}%), including {unknown_p['issued_background_labels']} background labels, and refers {unknown_p['referred']}. The ensemble assigns {unknown_b['trained_category_issued']} and refers {unknown_b['referred']}. Fusion recognizes only {pct(unknown_p['broad_vessel_recognition'])}% of these clips broadly as vessel. Referral of uncertain known classes is therefore not an adequate substitute for explicit unknown-category modeling. These taxonomy-based outcomes do not imply that the excluded categories have acoustically unique signatures or establish open-world detection performance.")

add('7. Agentic Decisions and Evidence-Conditioned Generation','heading')
add('Figure 3 separates the implemented adaptive acquisition loop from the fresh identification experiment, which supplies fixed evidence in one generator call. Against the earlier LLM planner, the learned controller preserves 37 supported details with 99 fewer queries and 20 fewer wrong/unsupported details; report retention differs. Against the locked heuristic it also loses 16 supported details. These are descriptive system tradeoffs, not isolated proof of superior reasoning or an identification gain.')
add('On the fresh cohort, all 705 generated outputs parse. Audio-only prompting withholds all 235 categories; this describes the fixed prompt/decoder, not general model competence. Acoustic-prior and fused-evidence prompting each identify 99/235 correctly (42.1%), while wrong labels fall from 68 to 44 and withheld labels rise from 68 to 92. Macro-F1 is 0, 0.411 and 0.456 for the three conditions, counting withheld labels as missed identifications. Both the supplied posterior and measured evidence change, so the comparison does not isolate the measurements’ effect.')
figure('agentic_reasoning_contribution','Fig. 3. Agentic acquisition and generative outcomes. A–B: earlier 100-clip cohort, logical queries in cached replay, automated support proxies and separately calibrated retention. C: separate 235-clip cohort with fixed inputs. Fewer queries and errors accompany reduced supported information or increased withholding; human benefit is unmeasured.',width=6.8)
overrides=diag['generator_overrides']['fused_evidence']
add(f"A descriptive decomposition separates changed errors from cautious withholding. With fused evidence, the generator changes {overrides['numeric_correct_to_wrong_label']} correct numeric identifications into wrong labels, withholds {overrides['numeric_correct_to_withheld']} additional correct identifications, and corrects {overrides['numeric_wrong_corrected']} prior errors. On the same 143 records that it labels, the numeric fusion classifier is correct on 103 versus the generator’s 99. Ranking numeric fusion to issue exactly 143 labels also gives 103 correct, a descriptive comparison with equal output counts. With the acoustic prior, generation changes three correct labels to errors, withholds 32 correct labels and corrects four prior errors. Thus withholding explains most of the reduction in full-cohort yield; direct harmful overrides are a distinct, smaller failure mode.")
add('The operator-facing design keeps the numeric category, probabilities, referral state and measured evidence separate. Generated prose cannot replace the category. This preserves numeric decisions by construction, without establishing a learned accuracy gain, interface usability or human time savings. The artifact bundle also retains two post-hoc, hash-ordered spectrogram examples with opposing fusion/ensemble outcomes; they are illustrations, not representative performance estimates.')

add('8. Discussion, Reproducibility and Conclusion','heading')
add('The physical-feature ablation motivates further study but does not replace the failed primary comparison. The matched refit controls extra labels and fitting procedure; source weighting and representation effects are not fully isolated. Prototype corruption shows dependence without proving benefit. Future tests should address unfamiliar-category training, recording-quality shift and longer context on new deployments, without tuning these observed test outcomes.')
add('The fresh cohorts have new sources but share the earlier test’s collection and recording days; fitting/selection/calibration dates remain disjoint. Missing parent/deployment provenance prevents complete physical-independence claims. Dataset labels are not expert judgments for every audible event. Passenger and some untrained-category source counts are small. Five-second clips omit contact history and overlapping events, and artificial noise does not establish perceptual recognizability. The sampled category mix does not represent operational traffic.')
add('Identification uses dataset labels and no human reviewers. The earlier report-support experiment used numerical proxies and two correlated AI reviewer roles; those roles are not identification ground truth here. New generated reasons lack independent validation, and record parsing is not a factuality score. This civilian, single-region experiment does not support propulsion-parameter or submarine-specific recognition claims.')
add('The reproducibility bundle retains all freezes, prompts, fitting searches, model/threshold hashes, confusion matrices, intervals and degradation results. Follow-up inference covers 235 clean and 1,410 transformed clips through three backbones, 705 generative classifications, 20 format checks and 72 untrained-category clips. Audio, models and detailed outputs have local copies outside Git. Portable checks cover weighting, prototypes, accounting, threshold ties and known-signal behavior. Stage timings are retained for the 32 GB RTX PRO 4500 Blackwell worker.')
add('The study makes agentic acquisition and evidence-conditioned generation testable through observable actions and outcomes. Fusion improves over its no-physical-feature ablation but fails the primary ensemble comparison. Clean calibration fails under severe noise and untrained categories. Fewer queries or more cautious AI answers therefore remain distinct from improved identification; measured human benefit remains an open question.')

add('References','heading')
records={r['key']:r['fields'] for path in ['sources/bibliography.json','sources/identification_bibliography.json'] for r in json.loads((ROOT/path).read_text())}
keys=['pamguard_workflow','underwater_feature_fusion','selectivenet','uniqueship','autagent','audio_mind','sear','ace_cap','signals1','signals2','alce','swin_v2','qwen_omni']
def reference_author(name):
 if ',' in name:
  surname,given=name.split(',',1)
 else:
  *given_parts,surname=name.split();given=' '.join(given_parts)
 return ' '.join(part[0]+'.' for part in given.strip().split())+' '+surname.strip()
for i,key in enumerate(keys,1):
 f=records[key];authors=f['author'].split(' and ');author=', '.join(reference_author(name) for name in authors)
 venue=f.get('booktitle',f.get('journal','arXiv:'+f.get('eprint','')))
 if key=='pamguard_workflow':
  text=f"[{i}] PAMGuard, “{f['title']},” official software documentation, accessed Nov. 16, 2026. {f['url']}"
 else:
  details=(' '+f['volume'] if 'volume' in f else '')+(', '+('art. ' if key=='underwater_feature_fusion' else 'pp. ')+f['pages'].replace('--','–') if 'pages' in f else '')
  text=f"[{i}] {author}, “{f['title']},” {venue}{details}, {f['year']}. {f['url']}"
 p=add(text)
 for run in p.runs:run.font.size=Pt(10)
for section in DOC.sections:
 for header in (section.header,section.first_page_header,section.even_page_header):
  for element in list(header._element):header._element.remove(element)
  p=header.add_paragraph('MVML 2027 — manuscript for author review');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
  for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(9)
 for footer in (section.footer,section.first_page_footer,section.even_page_footer):
  for element in list(footer._element):footer._element.remove(element)
  p=footer.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run('Manuscript — ')
  page=OxmlElement('w:fldSimple');page.set(qn('w:instr'),'PAGE');p._p.append(page)
DOC.core_properties.title='Evidence-Grounded Generative AI for Underwater Acoustic Reasoning'
DOC.core_properties.author='Anonymous';DOC.core_properties.last_modified_by=''
DOC.core_properties.subject='Vessel-category identification, review decisions and generative overrides on real recordings'
DOC.save(ROOT/'manuscript/manuscript.docx')
(ROOT/'manuscript/manuscript.txt').write_text('\n\n'.join(PROSE)+'\n')
print('Built identification manuscript; verify exactly eight PDF pages including references')
