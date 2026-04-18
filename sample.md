# SFT And DPO Samples

这份文档给当前新主线各放两个真实样本。

- SFT 数据来自 `/gpfs/projects/p32908/data/nlp/hf_cot_sft`
- DPO 数据来自 `/gpfs/projects/p32908/data/nlp/hf_cot_dpo`

所有样本都来自 `train` split，内容为便于阅读做了适度截断，但保留了真实输入输出结构。

## SFT Sample 1

元信息：

- `ticker`: `AAPL`
- `report_date`: `2021-01-28`
- `factor`: `tech_infra_platforms.capital_allocation_signal`
- `evidence_count`: `1`

输入：

```text
Task: Auditor (1.3)
Ticker: AAPL | Filing: 10-Q | Date: 2021-01-28
Factor: tech_infra_platforms.capital_allocation_signal

[Current Evidence]
  q_key : Q5::How does management justify capital deployment across M&A, buybacks, dividends, and internal R&D investment?
  quote : The Company uses the net proceeds from the commercial paper program for general corporate purposes, including dividends and share repurchases. As of December 26, 2020, the Company was authorized to purchase up to $225 billion of the Company’s common stock under a share repurchase program, of which $192.6 billion had been utilized. During the three months ended December 26, 2020, the Company repurchased 200 million shares of its common stock for $24.0 billion. The Company’s quarterly cash dividend was $0.205 per share. The Company intends to increase its dividend on an annual basis, subject to declaration by the Board of Directors.
  conf  : 0.85
[Historical Context]
  [2020-10-30 10-K] In April 2020, the Company announced an increase to its current share repurchase program authorization from $175 billion to $225 billion and raised its quarterly dividend from $0.1925 to $0.205 per share beginning in May 2020. During 2020, the Company repurchased $72.5 billion of its common stock and paid dividends and dividend equivalents of $14.1 billion.  (conf: 0.92)
  [2019-10-31 10-K] On April 30, 2019, the Company also announced the Board of Directors raised the Company’s quarterly cash dividend from $0.73 to $0.77 per share, beginning with the dividend paid during the third quarter of 2019. The Company intends to increase its dividend on an annual basis, subject to declaration by the Board of Directors.  (conf: 0.95)
  [2019-10-31 10-K] On April 30, 2019, the Company announced the Board of Directors increased the current share repurchase program authorization from $100 billion to $175 billion of the Company’s common stock, of which $96.1 billion had been utilized as of September 28, 2019.  (conf: 0.95)

Output reasoning_chain, sentiment_label, and confidence_score for this factor.
```

输出：

```json
{
  "reasoning_chain": "Summary of evidence: Current 10-Q states commercial paper proceeds used for general corporate purposes including dividends and share repurchases; $225bn repurchase authorization with $192.6bn utilized as of Dec 26, 2020; $24.0bn repurchased in the quarter; quarterly dividend $0.205 with an intention to increase annually. Historical filings (10-Ks 2019–2020) show consistent, increasing repurchase authorizations and material repurchases ($72.5bn in 2020) and material dividend payments ($14.1bn in 2020). Evidence quality and consistency: multiple filings across periods with high relevance confidence (0.85–0.95) provide a consistent, high-quality signal that management prioritizes shareholder returns via buybacks and dividends. Missing/weak evidence: the current excerpt contains no detailed justification or quantification of capital allocated specifically to M&A or internal R&D; historical context likewise focuses on returns rather than explicit platform investment amounts. Direction and magnitude of expected impact: the disclosures indicate a strong, large-magnitude tilt toward share repurchases and dividends—this materially improves near-term shareholder return metrics (EPS accretion, yield) but provides limited direct signal that incremental capital is being directed to tech infrastructure or platform expansion. Net factor implication: while shareholder-return policy is clearly favorable, its effect on the tech_infra_platforms.capital_allocation_signal is ambiguous because the filings do not document tradeoffs or explicit investment commitments to M&A/R&D. Confidence calibration: high confidence (0.8–0.9) that returns are prioritized; moderate confidence that platform-directed capital is unclear due to absence of evidence. Overall sentiment reflects this mixed/ambiguous signal.",
  "sentiment_label": "neutral",
  "confidence_score": 0.75
}
```

## SFT Sample 2

元信息：

- `ticker`: `AAPL`
- `report_date`: `2019-01-30`
- `factor`: `ai_infrastructure_cycle.ai_datacenter_buildout`
- `evidence_count`: `1`

输入：

```text
Task: Auditor (1.3)
Ticker: AAPL | Filing: 10-Q | Date: 2019-01-30
Factor: ai_infrastructure_cycle.ai_datacenter_buildout

[Current Evidence]
  q_key : Q6::How does the filing describe exposure to AI-driven datacenter buildouts and associated capital expenditure cycles?
  quote : The Company’s capital expenditures were $1.5 billion during the first three months of 2019. The Company anticipates utilizing approximately $12.0 billion for capital expenditures during 2019, which includes product tooling and manufacturing process equipment; data centers; corporate facilities and infrastructure, including information systems hardware, software and enhancements; and retail store facilities.
  conf  : 0.6

Output reasoning_chain, sentiment_label, and confidence_score for this factor.
```

输出：

```json
{
  "reasoning_chain": "The filing explicitly reports capital expenditures of $1.5B in the first three months of 2019 and guidance of approximately $12.0B for 2019, and it enumerates \"data centers\" among broad capex categories. Evidence quality: single, high-level disclosure that lists data centers but provides no breakdown of spending by category, no temporal phasing, and no explicit link to AI-driven or incremental datacenter buildouts. Direction: mildly positive — data centers are included in a sizable capex plan, implying capacity to invest in infrastructure that could support AI workloads. Magnitude: indeterminate — $12B is material, but the share attributable to data centers (let alone AI-specific buildout) is unspecified, so any impact magnitude is uncertain and could range from negligible to meaningful. Confidence calibration: low-to-moderate confidence due to single, non-specific evidence and moderate relevance (relevance_confidence 0.6); cannot reliably infer AI-driven datacenter exposure from this filing alone.",
  "sentiment_label": "neutral",
  "confidence_score": 0.35
}
```

## DPO Sample 1

元信息：

- `ticker`: `AAPL`
- `report_date`: `2021-01-28`
- `factor`: `tech_infra_platforms.capital_allocation_signal`
- `return_label`: `very_negative`
- `preference_source`: `return_aligned_over_conservative`

输入：

```text
System: You are a Senior Equity Strategist. Audit the current evidence by comparing it against the historical context. Analyze the deviation from the baseline and sector context. Score the sentiment as one of: very_negative, negative, neutral, positive, very_positive. Provide a reasoning_chain, sentiment_label, and confidence_score. Output only valid JSON.

User:
Task: Auditor (1.3)
Ticker: AAPL | Filing: 10-Q | Date: 2021-01-28
Factor: tech_infra_platforms.capital_allocation_signal

[Current Evidence]
  q_key : Q5::How does management justify capital deployment across M&A, buybacks, dividends, and internal R&D investment?
  quote : The Company uses the net proceeds from the commercial paper program for general corporate purposes, including dividends and share repurchases. As of December 26, 2020, the Company was authorized to purchase up to $225 billion of the Company’s common stock under a share repurchase program, of which $192.6 billion had been utilized. During the three months ended December 26, 2020, the Company repurchased 200 million shares of its common stock for $24.0 billion. The Company’s quarterly cash dividend was $0.205 per share. The Company intends to increase its dividend on an annual basis, subject to declaration by the Board of Directors.
  conf  : 0.85
[Historical Context]
  [2020-10-30 10-K] In April 2020, the Company announced an increase to its current share repurchase program authorization from $175 billion to $225 billion and raised its quarterly dividend from $0.1925 to $0.205 per share beginning in May 2020. During 2020, the Company repurchased $72.5 billion of its common stock and paid dividends and dividend equivalents of $14.1 billion.  (conf: 0.92)
  [2019-10-31 10-K] On April 30, 2019, the Company also announced the Board of Directors raised the Company’s quarterly cash dividend from $0.73 to $0.77 per share, beginning with the dividend paid during the third quarter of 2019. The Company intends to increase its dividend on an annual basis, subject to declaration by the Board of Directors.  (conf: 0.95)
  [2019-10-31 10-K] On April 30, 2019, the Company announced the Board of Directors increased the current share repurchase program authorization from $100 billion to $175 billion of the Company’s common stock, of which $96.1 billion had been utilized as of September 28, 2019.  (conf: 0.95)

Output reasoning_chain, sentiment_label, and confidence_score for this factor.
```

输出 `chosen`：

```json
{
  "reasoning_chain": "For tech_infra_platforms.capital_allocation_signal, the prompt provides 1 current evidence items and 3 historical references. Relative to the prior baseline, the evidence points to a strongly deteriorating setup. The conclusion should stay grounded in the filing details rather than unsupported macro assumptions.",
  "sentiment_label": "very_negative",
  "confidence_score": 0.64
}
```

输出 `rejected`：

```json
{
  "reasoning_chain": "For tech_infra_platforms.capital_allocation_signal, the prompt provides 1 current evidence items and 3 historical references. Relative to the prior baseline, the evidence looks broadly balanced versus prior context. The conclusion should stay grounded in the filing details rather than unsupported macro assumptions.",
  "sentiment_label": "neutral",
  "confidence_score": 0.55
}
```

## DPO Sample 2

元信息：

- `ticker`: `AAPL`
- `report_date`: `2017-02-01`
- `factor`: `structural_risks_tech.supply_chain_concentration`
- `return_label`: `positive`
- `preference_source`: `teacher_full_over_conservative`

输入：

```text
System: You are a Senior Equity Strategist. Audit the current evidence by comparing it against the historical context. Analyze the deviation from the baseline and sector context. Score the sentiment as one of: very_negative, negative, neutral, positive, very_positive. Provide a reasoning_chain, sentiment_label, and confidence_score. Output only valid JSON.

User:
Task: Auditor (1.3)
Ticker: AAPL | Filing: 10-Q | Date: 2017-02-01
Factor: structural_risks_tech.supply_chain_concentration

[Current Evidence]
  q_key : Q33::What can you say about geographic concentration of suppliers and mitigation strategies for supply risk?
  quote : The Company utilizes several outsourcing partners to manufacture sub-assemblies for the Company’s products and to perform final assembly and testing of finished products. These outsourcing partners acquire components and build product based on demand information supplied by the Company, which typically covers periods up to 150 days. The Company also obtains individual components for its products from a wide variety of individual suppliers.
  conf  : 0.75
[Historical Context]
  [2016-10-26 10-K] The Company utilizes several outsourcing partners to manufacture sub-assemblies for the Company’s products and to perform final assembly and testing of finished products. These outsourcing partners acquire components and build product based on demand information supplied by the Company, which typically covers periods up to 150 days. The Company also obtains individual components for its products from a wide variety of individual suppliers.  (conf: 0.75)
  [2016-07-27 10-Q] The Company utilizes several outsourcing partners to manufacture sub-assemblies for the Company’s products and to perform final assembly and testing of finished products. These outsourcing partners acquire components and build product based on demand information supplied by the Company, which typically covers periods up to 150 days. The Company also obtains individual components for its products from a wide variety of individual suppliers.  (conf: 0.85)
  [2015-10-28 10-K] The Company acquires components through a combination of purchase orders, supplier contracts, and open orders based on projected demand information. The Company also obtains individual components for its products from a wide variety of individual suppliers.  (conf: 0.85)

Output reasoning_chain, sentiment_label, and confidence_score for this factor.
```

输出 `chosen`：

```json
{
  "reasoning_chain": "Evidence quality & consistency: The current 10-Q quote and multiple historical filings (2015–2016) repeat nearly identical language, indicating high consistency but limited detail. They state use of several outsourcing partners, outsourcing of sub-assembly and final assembly, partner build based on demand windows up to ~150 days, and procurement from a “wide variety” of individual suppliers.",
  "sentiment_label": "neutral",
  "confidence_score": 0.6
}
```

输出 `rejected`：

```json
{
  "reasoning_chain": "For structural_risks_tech.supply_chain_concentration, the prompt provides 1 current evidence items and 3 historical references. Relative to the prior baseline, the evidence looks broadly balanced versus prior context. The conclusion should stay grounded in the filing details rather than unsupported macro assumptions.",
  "sentiment_label": "neutral",
  "confidence_score": 0.51
}
```
