# How the demo documents were produced

All demo documents are **synthetic**. The company ("Meridian Maritime"), policies, suppliers, part numbers and prices are fictional.
They were written by an LLM (Claude) in this repository's development session from the prompts below, then edited by hand
for internal consistency (no contradictory numbers between documents). The text sources live in `*/sources/`; `build_demo_docs.py`
turns them into real PDFs (pure-python PDF writer) in `*/docs/`, which is what the factory reads.

## Prompt 1 - HR policy set (3 documents)
> Write three short HR policy documents for a fictional shipping company's shore staff: (1) a leave policy, (2) a working hours and
> overtime policy, (3) a conduct, grievance and business travel policy. Each 300-400 words, numbered sections, and dense with concrete
> checkable facts: day counts, notice periods, multipliers, deadlines, USD limits. Do NOT cover health insurance or pensions - the
> brief for this demo deliberately asks for them so that the groundability guard has real gaps to detect. No contradictions between documents.

## Prompt 2 - Ship-spares set (CSV + 2 documents)
> Create (a) a catalogue CSV of 36 ship spare parts with columns part_no, description, category, criticality_class (A/B/C),
> supplier_code (HLM/NWE/PVP), unit_price_usd, lead_time_days, stock_on_hand, min_order_qty; (b) a supplier terms-of-trade document for
> the three suppliers (payment terms, volume discount, freight, returns/restocking, emergency surcharge, warranty); (c) a fleet procurement
> procedure with approval limits, quotation rules, critical-spares rule, emergency purchase rule, delivery and documentation rules.
> The suppliers' terms must differ from each other so questions can combine two facts. Leave fuel prices and charter decisions out of scope.

The *briefs* (`brief.txt`) and the optional requester answers (`answers.json`) are part of each demo.
