Day 22 Lab Evidence
===================

Generated:
- `02_ab_routing_log.txt`: deterministic A/B routing log for 50 questions.
- `03_ragas_report.json`: copied RAGAS report with V1 and V2 scores.
- `04_pii_demo_log.txt`: custom Guardrails PII validator demo.
- `04_json_demo_log.txt`: custom Guardrails JSON repair validator demo.

Remaining screenshots to add manually:
- `01_langsmith_traces.png`
- `02_prompt_hub.png`
- `03_ragas_scores.png`

RAGAS summary:
- V1 faithfulness: 0.8921
- V2 faithfulness: 0.7401
- Target met: yes

V1 scored higher on faithfulness and answer relevancy because the concise prompt stayed closer to the retrieved context. V2 was more structured, but the extra explanatory style appears to have introduced more unsupported or less directly grounded wording.
