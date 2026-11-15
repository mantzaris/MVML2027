# Development-only corrective decoding pass — 15 November 2026

The unconstrained development pilot was stopped after 176 reports. Although
175 were JSON-parseable, only 151 (85.8%) obeyed the nested slot contract. Errors
included measurement objects in citation arrays, numerical tone values in the
categorical field, and inconsistent uncertainty fields. Ninety reports violated
the requested interpretation length. No test terminal reports were generated.
All failed/free-greedy outputs are preserved under `free_greedy_pilot` and are
excluded from the corrected comparison cache.

The common corrective pass uses schema-constrained greedy decoding with
`xgrammar==0.2.9` (`apache-tvm-ffi==0.1.16`), following its official Transformers integration:
https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/contrib/hf.py .
The initial LM Format Enforcer 0.11.3 benchmark was terminated after 373 seconds
without producing its first batch; its failed job record is retained. The grammar fixes object shape,
status/value types, empty uncertainty fields, citation-array types, and the
60–100-word interpretation contract. It contains no query measurements,
reference values, correct classes, selected-action restriction or source IDs.
Arbitrary citation strings remain possible and still require semantic checking.
The same grammar applies to every generative terminal comparator and stress
report. The model checkpoint, precision, evidence budgets and scoring remain
shared. This is an explicit change from unrestricted greedy generation; format
constraints can affect content and are a limitation of the resulting study.

The common terminal output ceiling becomes 1,024 tokens to accommodate format
constraints; actual token counts and truncation remain reported. There is still
one terminal generation call and no generative repair or repeated selection of
preferred answers. Existing independent AI reviews remain unconstrained at
their original 768-token limit. The corrected development pilot must pass the
same schema/feasibility gate before expansion, and all test choices remain locked.
