# Evaluation

The pipeline uses a model in three places where it used to rely on rules or on the generator alone. This folder measures each of them on labeled data, through the same code paths the application runs.

| Question | Baseline | Model |
|----------|----------|-------|
| Is a symptom mention a current finding of the patient? | NegEx-style rules in `TextPreprocessor` | Jev |
| Should the red-flag gate escalate this text? | Keyword rules in `SafetyController` | Rules + Jev emergency signs |
| Do the retrieved guidelines support the diagnosis? | Token matching; Gemini judging its own output | OpenAI judge |

It also measures the rule-based diagnosis engine, which answers when the generator is unavailable, on a public benchmark (section 5).

## Running it

```bash
python -m evaluation.run_eval context --split heldout
python -m evaluation.run_eval gate --split heldout3
python -m evaluation.run_eval negex --split test
python -m evaluation.judge_eval --split heldout
python -m evaluation.fallback_eval --split test
```

- **Results** are written to `results/*.json`, with every item and each system's answer.
- **Responses are cached** in `.cache/` (not in git), so a re-run sends nothing. Without the cache a run needs `TYPESAFE_API_KEY` (context, gate, negex) or `OPENAI_API_KEY` (judge). The fallback evaluation calls no API; it downloads its benchmark on first use.
- **Cost so far**: everything in the caches came to about $0.14 for Jev (about 3,400 requests, earlier wordings of the questions included) and $0.02 for OpenAI (150 requests).
- **The Gemini judge is not in the default run.** The free tier allowed 20 requests per day for `gemini-2.5-flash` when this was written, shared with the application. Add it with `--systems tokens,gemini,openai`; `--cached-only gemini` scores what is already cached and sends nothing.

## Method

- **Development and held-out sets.** The `*_dev` sets were used to write the rules, the questions put to Jev, the thresholds and the judge instructions. Every `*_heldout` set was written, with its labels, before any system was run on it.
- **"Blind" is stated per number.** After a held-out run its errors were read, and twice the gate was changed because of them. A later run of the changed code on that same set is not blind and is marked so below; a new set was written for the next blind run.
- **Ambiguous items are not scored.** Items where a careful reader could go either way are labeled `ambiguous`, left out of the metrics and listed with each system's answer.
- **One annotator.** All labels except the public NegEx set were written by the author of the rules and of the model instructions.
- **Small sets, one run.** A model's answer is taken from a single call. For 43 correct out of 43, the 95% confidence interval for the true accuracy starts at 92%.

## Results

### 1. Mention context

`context_heldout.jsonl`: 101 patient-style texts, 133 symptom mentions. A mention counts as positive when it is a current finding of the patient, as opposed to denied, past, someone else's or only asked about. Blind.

| System | Accuracy | Sensitivity | Specificity | Missed | Wrongly kept |
|--------|----------|-------------|-------------|--------|--------------|
| Rules | 68.4% | 86.3% | 57.3% | 7 | 35 |
| Jev | 99.2% | 100% | 98.8% | 0 | 1 |

The rules have no notion of past history or of a question about a condition, which is most of the gap: on those mentions they score 17% and 0%. On the kinds they do handle they reach 86% for negated and for present mentions and 82% for family history, where Jev is at 100%. Jev's one error is "The patient in the next bed had a seizure". On the development set both systems score 100%, as expected of a set the rules were written against.

The rules row is a re-run with the current symptom vocabulary, which was extended afterwards for section 5 without looking at this set. On the vocabulary the set was first run with, the rules scored 67.7% (one more mention wrongly kept).

### 2. Red-flag gate

A text is positive when it describes a possible emergency in the patient now. The gate combines the keyword scan with Jev's emergency signs, and lets Jev drop a keyword flag only when it is close to certain the mention is history. Sensitivity and specificity, clear cases only:

| Set | Clear cases | Blind for the code that ran | Keyword rules | Rules + Jev |
|-----|-------------|-----------------------------|---------------|-------------|
| `gate_heldout`, round 1 | 106 | Yes | 27.3% / 64.7% | 98.2% / 86.3% |
| `gate_heldout2`, round 2 | 42 | Yes | 40.0% / 54.5% | 100% / 95.5% |
| `gate_heldout3`, round 3 | 24 | Yes | 30.0% / 71.4% | 100% / 71.4% |
| `gate_heldout`, current code | 106 | No | 27.3% / 64.7% | 100% / 94.1% |

- **Round 1** (`results/gate_heldout_round1.json`) missed one emergency: "My mother had a stroke last year and now she can't move her right arm and her speech is slurred". Without the clearing step, specificity stayed at the rules' 64.7%: every keyword false alarm remained.
- **After round 1** the rule for dropping a keyword flag was rewritten: at most 0.05 probability that the condition is current and at least 0.9 that it is past, someone else's or denied.
- **After round 2** the stroke question was split into two literal questions, "are these deficits reported" and "does the text call them long-standing and unchanged", which catches the round 1 miss. `results/gate_heldout2.json` is a re-run with this code; the scores are the same as in the blind run.
- **Round 3** is a stress set of stroke histories with new or unchanged deficits. No emergency is missed, and specificity is no better than the keyword rules: four of fourteen non-emergencies are flagged, three of them long-standing deficits after an old stroke.

Jev's signs alone miss one emergency on the round 3 set that the keyword scan catches, which is why the two are combined.

### 3. Negation on a public benchmark

The NegEx annotation set (Chapman et al.), reports 21 to 120: 1,869 annotated concepts located in their sentence. Neither system was built on it; reports 1 to 20 were used while wording Jev's questions.

| System | Accuracy | F1 | Missed negations | False negations |
|--------|----------|----|------------------|-----------------|
| Rules | 97.8% | 0.945 | 29 | 13 |
| Jev | 96.4% | 0.919 | 12 | 55 |

The rules are better here. Most of Jev's false negations are concepts that are themselves a negative or normal finding ("afebrile", "lungs are clear"): the benchmark marks them affirmed because the phrase is stated, while Jev answers that the patient does not have the finding. Without those 380 concepts Jev scores 99.1% (12 missed, 2 false) and the rules 97.2% (29 missed, 13 false). That second comparison is post hoc: the filter was written after reading the errors.

### 4. Guideline judge

Each item is a diagnosis label and the three passages `GuidelineValidator` retrieves for it. The label says whether those passages support the diagnosis: named conditions, synonyms and lay names, classifier labels, conditions the library does not cover, different conditions that share a word, status messages, and labels whose stated values contradict the guideline's own criteria. A diagnosis counts as supported at a score of 0.4 or more.

`judge_heldout.jsonl`: 49 pairs, 43 scored. Blind for the OpenAI judge and its instructions.

| System | Accuracy | Supported diagnoses sent to review | Unsupported diagnoses passed |
|--------|----------|------------------------------------|------------------------------|
| Token matching as first measured | 72.1% | 7 of 22 | 5 of 21 |
| Token matching, rewritten (the fallback; not blind) | 93.0% | 3 of 22 | 0 of 21 |
| GPT-6 Luna with the Gemini judge's prompt | 95.3% | 0 | 2 |
| GPT-6 Luna with the judge instructions (`OpenAIJudge`) | 100% | 0 | 0 |

- **The instructions matter on status messages.** With the earlier prompt the model passed "Insufficient information to determine a diagnosis" and "Further evaluation needed" as partly supported.
- **Token matching erred in both directions and was rewritten.** It matched substrings and added up passages, so it passed "Heart block" with a score of 1.0 ("block" is inside "beta-blocker") and "Pulmonary hypertension" from the passages on systemic hypertension and pulmonary edema, while rejecting "NSTEMI" and every classifier label. It now matches whole words, needs one passage to hold more than half of the diagnosis's words, and never passes a label that states a measurement.
- **The rewritten score is not a blind result.** The rewrite was made after reading these errors. On the development set it scores 88.5% (3 of 15 supported diagnoses sent to review, none passed wrongly). What it still rejects are synonyms it cannot know: "Heart attack", "Bladder infection", "Scarring of the lungs".
- **The comparison with the Gemini judge is incomplete.** Gemini's daily free allowance ran out after 23 of the 26 development pairs and before the held-out set. On those 23 pairs Gemini, GPT-6 Luna with the same prompt and `OpenAIJudge` all agree with every label. There is no measured accuracy difference between the two judge models. What is measured is that the OpenAI judge is correct on the held-out set, that on the development pairs it answered in a median 2.7 s against Gemini's 4.2 s, and that it does not use the Gemini allowance.
- **The Gemini judge keeps its original prompt** for that reason: the new instructions have not been run on Gemini.
- **Retrieval limits the judge.** For "CHF" and "Congestive heart failure exacerbation" the three retrieved passages do not include the heart failure section, so a correct diagnosis is sent to review. Both are among the six ambiguous items.

On the held-out set the OpenAI judge took a median 2.6 s (95th percentile 3.9 s) and about 1,070 input and 120 output tokens per diagnosis, which is $0.00017.

### 5. Rule-based diagnosis engine

`ClinicalFallbackEngine` names a disease from the extracted symptoms when the LLM generator is unavailable. A text result is shown as approved only when its confidence reaches 0.70, so two things are measured: how often the first diagnosis is right, and what passes that gate.

The benchmark is [gretelai/symptom_to_diagnosis](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) (Apache-2.0): 1,065 patient-style descriptions labeled with one of 22 diagnoses, not written for this project. Its train file is the development split and its test file the blind split. The mapping from its labels to the engine's diseases was fixed before the engine was run on it:

- **Known** (9 labels): conditions the engine has an entry for. The three skin conditions count as correct when the engine answers with its one skin entry.
- **Unknown** (10 labels): conditions it has no entry for, such as malaria or arthritis. Whatever it names is wrong, and showing it is the failure to catch.
- **Not scored** (3 labels): rash-dominated conditions for which the skin entry would be a defensible answer.

Blind split, 212 texts, 184 scored (88 known, 96 unknown). The committed engine is the code before this work, run from a copy of that commit:

| | Committed engine | Current engine |
|---|---|---|
| First diagnosis is the labeled one, known conditions | 27.3% or 33.0%, depending on the hash seed | 55.7% (49 of 88) |
| Diagnoses passing the gate | 0 | 16, of which 15 correct (93.8%) |
| Known conditions shown with the right diagnosis | 0% | 17.0% (15 of 88) |
| Unknown conditions shown with a diagnosis | 0 of 96 | 1 of 96 |

On the development split (853 texts, 733 scored) the same figures are 63.8%, 92 shown of which 89 correct (96.7%), 25.1%, and 3 of 379; the committed engine scored 32.8% or 37.0% and showed nothing.

What changed, in the order it was found:

- **Symptom extraction read clinical words only.** "Coughing", "dizzy", "tired", "phlegm" or "it burns when I pee" were not recognized: half of the urinary tract infection descriptions and two thirds of the diabetes descriptions of the development split yielded no symptom at all. A layer of everyday wording now maps such phrases to the clinical terms.
- **Hallmark symptoms carried no weight.** The engine could output a urinary tract infection or diabetes, but painful or frequent urination and thirst were not in its table. The table was completed from the symptom lists of the project's own guideline library, not from the benchmark's labels.
- **Equal scores were ranked by chance.** The leading disease among equal scores followed the iteration order of a set, so the same text could get a different diagnosis after a restart. That is the reason for the two figures of the committed engine. The order is now fixed.
- **Confidence followed the size of the score.** It now follows the margin of the leading disease over the runner-up, which is what predicted a correct answer on the development split: 97% correct at a margin of 0.7 or more with at least two supporting symptoms, about half between 0.6 and 0.7. Accuracy by confidence on the blind split: 4% at 0.1 to 0.2 (79 texts), 20% at 0.2 to 0.3 (46), 63% at 0.4 to 0.5 (8), 50% at 0.6 to 0.7 (34), 94% at 0.7 and above (16); one text fell between 0.5 and 0.6 and was wrong.

What the benchmark is and is not good for:

- **Its symptom profiles are not clinical truth.** Its "bronchial asthma" descriptions are fever with a productive cough, and its "hypertension" descriptions headache, chest pain and loss of balance. On the blind split the engine answers pneumonia or COPD for the first and acute coronary syndrome or stroke for the second, which counts as wrong here: 0 of 10 on each label. The engine was not tuned towards these profiles.
- **The engine reads chest pain cautiously.** Six of the ten pneumonia descriptions of the blind split add chest pain, sweating and a racing heart to the cough, and the engine answers acute coronary syndrome. That is scored as wrong, and it is the answer the red-flag gate sends to a doctor either way.
- **It covers 9 of the engine's 22 diseases.** The confidence curve is fitted on those. Its cap at 0.75 and the rule that one symptom is never a confident diagnosis are there for the rest.
- **Most correct diagnoses are still held back.** A picture shared by several diseases, such as cough, fever and sputum, leaves pneumonia ahead of tuberculosis by a margin of about 0.6 and is sent to review. On this benchmark that is the right call about half the time.

## What these numbers do not show

- How the full pipeline performs on real patient text. The texts were written for the evaluation, in English, and are short.
- Agreement between annotators, since there was one.
- How stable a model's answer is across repeated calls.
- How the judge behaves when the generator, not the retrieval, is wrong about the patient: the judge sees the diagnosis label and the guidelines, not the patient's text.
