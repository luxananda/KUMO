import json
from pathlib import Path

import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

import kumoai.experimental.rfm as rfm

CONFIG_PATH = Path(__file__).parent / ".kumo_config.json"
FIELDS = [
    "kumo_api_key",
    "project_id",
    "dataset_id",
    "sa_json",
]


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_config(values: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(values, indent=2))


def read_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


st.set_page_config(page_title="KumoRFM × BigQuery", layout="wide")
st.title("KumoRFM × BigQuery")
st.caption("Browse BigQuery tables, upload CSV/Excel, and feed them to KumoRFM.")

# Seed session_state from saved config on first load.
if "_loaded" not in st.session_state:
    saved = load_config()
    for f in FIELDS:
        st.session_state.setdefault(f, saved.get(f, ""))
    st.session_state._loaded = True

for k, default in [
    ("bq_client", None),
    ("bq_tables", []),
    ("uploaded_tables", {}),  # name -> DataFrame (CSV/Excel + pulled BQ)
    ("rfm_ready", False),
    ("preview_df", None),
    ("preview_table", None),
    ("graph", None),
    ("graph_tables", []),  # names in the built graph
    ("predict_result", None),
    ("predict_query", "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR users.user_id=1"),
]:
    if k not in st.session_state:
        st.session_state[k] = default

with st.sidebar:
    st.header("1. KumoRFM")
    st.text_input(
        "Kumo API Key (JWT)",
        key="kumo_api_key",
        type="password",
        placeholder="eyJhbGciOi...",
        help="Your KumoRFM free-trial / API key (JWT format).",
    )

    st.divider()
    st.header("2. BigQuery")
    st.text_input("GCP Project ID", key="project_id", placeholder="onyx-smoke-486103-d1")
    st.text_input("BigQuery Dataset ID", key="dataset_id", placeholder="my_dataset")

    sa_upload = st.file_uploader(
        "Service Account JSON file (recommended)",
        type=["json"],
        help="Upload the .json key file downloaded from GCP IAM → Service Accounts → Keys. "
             "Avoids copy-paste errors that mangle the PEM private key.",
    )
    if sa_upload is not None:
        try:
            st.session_state.sa_json = sa_upload.getvalue().decode("utf-8")
            st.caption(f"✅ Loaded {sa_upload.name}")
        except Exception as e:
            st.error(f"Couldn't read {sa_upload.name}: {e}")

    with st.expander("…or paste JSON manually", expanded=not st.session_state.get("sa_json")):
        st.text_area(
            "Service Account JSON",
            key="sa_json",
            height=180,
            placeholder="Paste the full service-account key JSON here",
            label_visibility="collapsed",
        )

    col_save, col_clear = st.columns(2)
    with col_save:
        if st.button("💾 Save", use_container_width=True):
            save_config({f: st.session_state[f] for f in FIELDS})
            st.toast(f"Saved to {CONFIG_PATH.name}", icon="✅")
    with col_clear:
        if st.button("🗑 Clear saved", use_container_width=True):
            if CONFIG_PATH.exists():
                CONFIG_PATH.unlink()
            for f in FIELDS:
                st.session_state[f] = ""
            st.toast("Cleared.", icon="🧹")
            st.rerun()

    st.caption(
        f"Values persist to `{CONFIG_PATH.name}` (gitignored)."
        if CONFIG_PATH.exists()
        else "Click Save to persist these values between reloads."
    )

    connect = st.button("Connect & list tables", type="primary", use_container_width=True)

    st.divider()
    st.header("3. Upload files")
    st.caption("CSV, TSV, or Excel. Each file becomes a table named after the file.")
    uploads = st.file_uploader(
        "Upload CSV / Excel",
        type=["csv", "tsv", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploads:
        for f in uploads:
            tname = Path(f.name).stem
            if tname in st.session_state.uploaded_tables:
                continue
            try:
                st.session_state.uploaded_tables[tname] = read_upload(f)
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")

    if st.session_state.uploaded_tables:
        st.caption(f"Loaded {len(st.session_state.uploaded_tables)} uploaded table(s).")
        if st.button("Clear uploaded", use_container_width=True):
            st.session_state.uploaded_tables = {}
            st.session_state.preview_df = None
            st.session_state.preview_table = None
            st.rerun()

kumo_api_key = st.session_state.kumo_api_key
project_id = st.session_state.project_id
dataset_id = st.session_state.dataset_id
sa_json = st.session_state.sa_json

if connect:
    if not kumo_api_key:
        st.error("Kumo API key is required.")
        st.stop()

    save_config({f: st.session_state[f] for f in FIELDS})

    with st.spinner("Initializing KumoRFM…"):
        try:
            rfm.init(api_key=kumo_api_key)
            st.session_state.rfm_ready = True
        except Exception as e:
            msg = str(e).lower()
            if "already been initialized" in msg or "already initialized" in msg:
                # RFM is a process-level singleton; Streamlit reruns the script
                # on every interaction so a second init() call raises. Treat as ok.
                st.session_state.rfm_ready = True
                st.info(
                    "KumoRFM was already initialized in this process — keeping the "
                    "existing session. To switch API keys, restart the Streamlit server."
                )
            else:
                st.session_state.rfm_ready = False
                st.error(f"rfm.init failed: {e}")
                st.stop()

    # BigQuery is optional — only connect if all BQ fields are provided.
    if project_id and dataset_id and sa_json:
        try:
            sa_info = json.loads(sa_json)
        except json.JSONDecodeError as e:
            st.error(f"Service Account JSON is not valid JSON: {e}")
            st.stop()

        pk = sa_info.get("private_key", "")
        if "\\n" in pk and "\n" not in pk:
            sa_info["private_key"] = pk.replace("\\n", "\n")
            pk = sa_info["private_key"]

        if not (pk.startswith("-----BEGIN") and "-----END" in pk and "..." not in pk):
            st.error(
                "The `private_key` field in your service-account JSON looks corrupted "
                "(truncated, redacted, or had newlines stripped). "
                "**Tip:** use the file uploader above instead of pasting — it avoids this. "
                "Or re-download the key JSON from GCP IAM and re-paste it raw."
            )
            st.stop()

        with st.spinner("Connecting to BigQuery…"):
            try:
                credentials = service_account.Credentials.from_service_account_info(sa_info)
                client = bigquery.Client(project=project_id, credentials=credentials)
            except Exception as e:
                st.error(f"BigQuery client failed: {e}")
                st.stop()

        with st.spinner(f"Listing tables in {project_id}.{dataset_id}…"):
            try:
                dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
                tables = sorted(t.table_id for t in client.list_tables(dataset_ref))
            except Exception as e:
                st.error(f"list_tables failed: {e}")
                st.stop()

        st.session_state.bq_client = client
        st.session_state.bq_tables = tables
        st.success(f"Connected. Found {len(tables)} BQ table(s). KumoRFM initialized.")
    else:
        st.success("KumoRFM initialized. (BigQuery skipped — fill those fields to list BQ tables.)")

client = st.session_state.bq_client
bq_tables = st.session_state.bq_tables
uploaded_tables = st.session_state.uploaded_tables

if not st.session_state.rfm_ready and not uploaded_tables:
    st.info("Enter your KumoRFM API key in the sidebar (BigQuery optional), then click **Connect**. Or upload a CSV/Excel.")
    st.stop()

# Build a combined table list with type tags.
combined: list[tuple[str, str]] = []  # (source, name) where source is "BQ" or "Upload"
for t in bq_tables:
    combined.append(("BQ", t))
for t in sorted(uploaded_tables.keys()):
    combined.append(("Upload", t))

left, right = st.columns([1, 2])

with left:
    st.subheader("Tables")
    st.caption(
        f"BigQuery: {len(bq_tables)} · Uploaded: {len(uploaded_tables)}"
    )
    if not combined:
        st.write("_No tables. Connect to BigQuery or upload a file._")
        selected_source = selected_name = None
    else:
        labels = [f"[{src}] {name}" for src, name in combined]
        idx = st.radio(
            "Select a table",
            options=list(range(len(combined))),
            format_func=lambda i: labels[i],
            index=0,
            label_visibility="collapsed",
        )
        selected_source, selected_name = combined[idx]

    if selected_source == "BQ" and selected_name:
        if st.button("Preview 50 rows", use_container_width=True):
            with st.spinner(f"Loading {selected_name}…"):
                try:
                    table_ref = bigquery.TableReference.from_string(
                        f"{project_id}.{dataset_id}.{selected_name}"
                    )
                    rows = client.list_rows(table_ref, max_results=50)
                    st.session_state.preview_df = rows.to_dataframe()
                    st.session_state.preview_table = ("BQ", selected_name)
                except Exception as e:
                    st.error(f"Preview failed: {e}")

with right:
    if selected_name is None:
        st.subheader("—")
    else:
        st.subheader(f"[{selected_source}] {selected_name}")

        if selected_source == "BQ":
            try:
                table_ref = bigquery.TableReference.from_string(
                    f"{project_id}.{dataset_id}.{selected_name}"
                )
                tbl = client.get_table(table_ref)

                st.markdown(
                    f"**Rows:** {tbl.num_rows:,} · **Size:** {tbl.num_bytes / 1e6:.2f} MB · "
                    f"**Created:** {tbl.created:%Y-%m-%d}"
                )
                schema_rows = [
                    {"column": f.name, "type": f.field_type, "mode": f.mode, "description": f.description or ""}
                    for f in tbl.schema
                ]
                st.markdown("**Schema**")
                st.dataframe(schema_rows, use_container_width=True, hide_index=True)

                if (
                    st.session_state.preview_df is not None
                    and st.session_state.preview_table == ("BQ", selected_name)
                ):
                    st.markdown("**Preview (first 50 rows)**")
                    st.dataframe(st.session_state.preview_df, use_container_width=True)

                already_loaded = selected_name in uploaded_tables
                load_label = (
                    f"✅ Loaded for RFM ({len(uploaded_tables[selected_name]):,} rows)"
                    if already_loaded
                    else f"Load full table for RFM ({tbl.num_rows:,} rows)"
                )
                if st.button(load_label, use_container_width=True, disabled=already_loaded):
                    with st.spinner(f"Pulling {selected_name} into pandas…"):
                        try:
                            df = client.list_rows(table_ref).to_dataframe()
                            st.session_state.uploaded_tables[selected_name] = df
                            st.success(f"Loaded {len(df):,} rows into RFM set.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Load failed: {e}")
            except Exception as e:
                st.error(f"Failed to load table '{selected_name}': {e}")

        else:  # Upload
            df = uploaded_tables[selected_name]
            st.markdown(f"**Rows:** {len(df):,} · **Columns:** {len(df.columns)}")
            schema_rows = [
                {"column": c, "type": str(df[c].dtype), "nulls": int(df[c].isna().sum())}
                for c in df.columns
            ]
            st.markdown("**Schema**")
            st.dataframe(schema_rows, use_container_width=True, hide_index=True)

            st.markdown(f"**Preview (first {min(50, len(df))} rows)**")
            st.dataframe(df.head(50), use_container_width=True)


st.divider()
st.header("🔮 Build Graph & Predict")

if not st.session_state.rfm_ready:
    st.warning("KumoRFM not initialized yet. Paste your JWT and click **Connect** in the sidebar.")
    st.stop()

available = sorted(st.session_state.uploaded_tables.keys())

if not available:
    st.info(
        "No tables loaded into the RFM set yet. Upload CSV/Excel in the sidebar, "
        "or select a BigQuery table on the left and click **Load full table for RFM**."
    )
    st.stop()

cols = st.columns([2, 1])
with cols[0]:
    chosen = st.multiselect(
        "Tables to include in the graph",
        options=available,
        default=available,
        help="All DataFrames from CSV/Excel uploads and BQ 'Load for RFM' pulls.",
    )
with cols[1]:
    st.write("")
    st.write("")
    build_clicked = st.button("🔨 Build graph", use_container_width=True, disabled=not chosen)

if build_clicked:
    with st.spinner("Building graph…"):
        try:
            tables_dict = {n: st.session_state.uploaded_tables[n] for n in chosen}
            graph = rfm.Graph.from_data(tables_dict)
            try:
                graph.validate()
            except Exception as ve:
                st.warning(f"Graph validation warning: {ve}")
            st.session_state.graph = graph
            st.session_state.graph_tables = chosen
            st.session_state.predict_result = None
            st.success(f"Graph built with {len(chosen)} table(s): {', '.join(chosen)}")
        except Exception as e:
            st.session_state.graph = None
            st.error(f"Graph build failed: {e}")

if st.session_state.graph is not None:
    st.markdown(f"**Graph ready** · tables: `{', '.join(st.session_state.graph_tables)}`")

    with st.expander("Graph details", expanded=False):
        try:
            st.code(repr(st.session_state.graph))
        except Exception:
            pass

    st.markdown("#### PREDICT query (PQL)")
    st.text_area(
        "Predictive query",
        key="predict_query",
        height=100,
        label_visibility="collapsed",
        help="KumoRFM Predictive Query Language. "
             "Example: PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR users.user_id=1",
    )

    if st.button("▶ Run PREDICT", type="primary"):
        with st.spinner("Running prediction…"):
            try:
                model = rfm.KumoRFM(st.session_state.graph)
                result = model.predict(st.session_state.predict_query)
                st.session_state.predict_result = result
            except Exception as e:
                st.session_state.predict_result = None
                st.error(f"Predict failed: {e}")

    result = st.session_state.predict_result
    if result is not None:
        st.markdown("#### Result")
        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
        else:
            try:
                st.dataframe(pd.DataFrame(result), use_container_width=True)
            except Exception:
                st.code(repr(result))
