# Scheduling CWT Prediction with Hermes Agent

You can schedule the CWT Prediction pipeline to run automatically every 5 minutes (or any chosen interval) using Hermes' built-in natural language cron scheduling.

## Prerequisites

1. Ensure the Python environment has all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have created and configured the `.env` file in the project root.

## Scheduling the Cron Job

Start the Hermes CLI or talk to your agent via Slack/Telegram, and run the following command:

```text
/cronjob create name="cwt-prediction" schedule="*/5 * * * *" prompt="Execute the Python script at d:\CWT prediction\cwt_prediction\main.py and output the cycle results summary"
```

### Alternative Natural Language Command

You can also prompt Hermes in plain English:

```text
"Create a cron job named 'cwt-prediction' that runs every 5 minutes. Every time it runs, execute 'python -m cwt_prediction.main' in the directory 'd:\CWT prediction', log the JSON summary, and notify me of any active trades with positive Kelly fraction."
```

## Managing the Cron Job

Once scheduled, you can manage it from the Hermes CLI:

- **List active cron jobs:**
  ```text
  /cronjob list
  ```

- **Run the job immediately:**
  ```text
  /cronjob run name="cwt-prediction"
  ```

- **Pause the job:**
  ```text
  /cronjob pause name="cwt-prediction"
  ```

- **Delete the job:**
  ```text
  /cronjob remove name="cwt-prediction"
  ```
