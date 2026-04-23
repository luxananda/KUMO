# KumoRFM × BigQuery (Streamlit)

A Streamlit frontend that lists and previews BigQuery tables, and initializes the [KumoRFM SDK](https://kumo.ai/docs/rfm/sdk-getting-started) so you can run predictive queries on them.

## Why two pieces?

Kumo ships two products:

| | Kumo SaaS (`kumoai`) | **KumoRFM** (`kumoai.experimental.rfm`) |
|---|---|---|
| API key | `<customer_id>:<secret>` | JWT (`eyJ...`) — free-trial & new product |
| URL | `https://<cid>.kumoai.cloud/api` | none |
| Native BigQuery connector | Yes | **No** |

The free-trial JWT key you get from kumorfm.ai is for **KumoRFM**, which does **not** have a native BigQuery connector. It takes pandas DataFrames. So this app pulls BQ data via `google-cloud-bigquery` and hands it to RFM.

## Setup

```bash
cd /Users/lakshmivenkatesh/KUMO
python3.11 -m venv .venv          # 3.10+ required by kumoai
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501.

## Sidebar inputs

1. **Kumo API Key (JWT)** — your KumoRFM key (`eyJ...`).
2. **GCP Project ID** — e.g. `onyx-smoke-486103-d1` (from your JDBC string).
3. **BigQuery Dataset ID** — pick any dataset inside the project.
4. **Service Account JSON** — open the file at `KeyFilePath` from your JDBC string and paste the full JSON.

Click **💾 Save** to persist these in `.kumo_config.json` (gitignored). Click **Connect & list tables**.

## What v1 does

- Init KumoRFM SDK with your JWT
- Connect to BigQuery with service-account creds
- List tables in the dataset
- Show schema + row count for each table
- Preview 50 rows on click

## Next step (not wired yet)

Load selected BQ tables into pandas, then:

```python
import kumoai.experimental.rfm as rfm

graph = rfm.Graph.from_data({"users": users_df, "orders": orders_df})
model = rfm.KumoRFM(graph)
result = model.predict("PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR users.user_id=1")
```
