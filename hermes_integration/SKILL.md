---
name: cwt-prediction
description: Executes a single forecasting and position sizing cycle for crypto markets.
version: 1.0.0
metadata:
  hermes:
    tags: [crypto, prediction, trading, finance]
    category: finance
    requires_toolsets: [terminal]
---

# CWT Prediction Skill

This skill allows the Hermes Agent to run the crypto short-term prediction pipeline on a 5-minute schedule.

## How to Install the Skill in Hermes

1. Copy this file into your local Hermes skills directory:
   - On Windows: `%USERPROFILE%\.hermes\skills\cwt-prediction\SKILL.md`
   - On Linux/macOS: `~/.hermes/skills/cwt-prediction/SKILL.md`

2. When the user asks `/cwt-prediction` or "run a prediction cycle", the agent will run the pipeline.

## Execution Procedure

When this skill is triggered, the agent should run the pipeline entrypoint using its terminal execution tool:

```bash
# In the workspace directory "d:\CWT prediction"
python -m cwt_prediction.main
```

Then, the agent reads the cycle summary printed to standard output and summarizes the results for the user:
- The markets analyzed.
- The forecast direction and probability for each market.
- The recommended Kelly position size.
- The resolution of past expired markets.
