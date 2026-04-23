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
    ("graph_pks", {}),  # table_name -> primary_key_column
    ("predict_result", None),
    ("predict_query", "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR users.user_id=1"),
]:
    if k not in st.session_state:
        st.session_state[k] = default

KUMO_FIELDS = ["kumo_api_key"]
BQ_FIELDS = ["project_id", "dataset_id", "sa_json"]


def connect_kumo(api_key: str) -> tuple[bool, str]:
    """Returns (ok, message)."""
    try:
        rfm.init(api_key=api_key)
        return True, "KumoRFM initialized."
    except Exception as e:
        msg = str(e).lower()
        if "already been initialized" in msg or "already initialized" in msg:
            return True, (
                "KumoRFM was already initialized in this process. "
                "Restart Streamlit to switch API keys."
            )
        return False, f"rfm.init failed: {e}"


def connect_bigquery(project_id: str, dataset_id: str, sa_json: str):
    """Returns (client, tables, error_message). client/tables are None on failure."""
    try:
        sa_info = json.loads(sa_json)
    except json.JSONDecodeError as e:
        return None, None, f"Service Account JSON is not valid JSON: {e}"

    pk = sa_info.get("private_key", "")
    if "\\n" in pk and "\n" not in pk:
        sa_info["private_key"] = pk.replace("\\n", "\n")
        pk = sa_info["private_key"]

    if not (pk.startswith("-----BEGIN") and "-----END" in pk and "..." not in pk):
        return None, None, (
            "The `private_key` field in your service-account JSON looks corrupted "
            "(truncated, redacted, or had newlines stripped). "
            "Use the file uploader instead of pasting."
        )

    try:
        credentials = service_account.Credentials.from_service_account_info(sa_info)
        client = bigquery.Client(project=project_id, credentials=credentials)
    except Exception as e:
        return None, None, f"BigQuery client failed: {e}"

    try:
        dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
        tables = sorted(t.table_id for t in client.list_tables(dataset_ref))
    except Exception as e:
        return None, None, f"list_tables failed: {e}"

    return client, tables, None


with st.sidebar:
    # ──────────────────────────── KumoRFM ────────────────────────────
    st.header("1. KumoRFM")
    st.text_input(
        "Kumo API Key (JWT)",
        key="kumo_api_key",
        type="password",
        placeholder="eyJhbGciOi...",
        help="Your KumoRFM free-trial / API key (JWT format).",
    )

    k_save, k_connect = st.columns(2)
    with k_save:
        if st.button("💾 Save Kumo", use_container_width=True):
            save_config({f: st.session_state[f] for f in FIELDS})
            st.toast("Kumo key saved.", icon="✅")
    with k_connect:
        if st.button("🔌 Connect Kumo", type="primary", use_container_width=True):
            if not st.session_state.kumo_api_key:
                st.error("Enter the Kumo API key first.")
            else:
                ok, msg = connect_kumo(st.session_state.kumo_api_key)
                st.session_state.rfm_ready = ok
                (st.success if ok else st.error)(msg)

    if st.session_state.rfm_ready:
        st.caption("✅ KumoRFM connected")
    else:
        st.caption("⚪ Not connected to Kumo")

    st.divider()

    # ──────────────────────────── BigQuery ────────────────────────────
    st.header("2. BigQuery (optional)")
    st.text_input("GCP Project ID", key="project_id", placeholder="onyx-smoke-486103-d1")
    st.text_input("BigQuery Dataset ID", key="dataset_id", placeholder="my_dataset")

    sa_upload = st.file_uploader(
        "Service Account JSON file (recommended)",
        type=["json"],
        help="Upload the .json key file downloaded from GCP IAM → Service Accounts → Keys.",
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

    b_save, b_connect = st.columns(2)
    with b_save:
        if st.button("💾 Save BQ", use_container_width=True):
            save_config({f: st.session_state[f] for f in FIELDS})
            st.toast("BigQuery config saved.", icon="✅")
    with b_connect:
        if st.button("🔌 Connect BQ", type="primary", use_container_width=True):
            if not (st.session_state.project_id and st.session_state.dataset_id and st.session_state.sa_json):
                st.error("Fill Project, Dataset, and SA JSON first.")
            else:
                with st.spinner("Connecting to BigQuery…"):
                    client, tbls, err = connect_bigquery(
                        st.session_state.project_id,
                        st.session_state.dataset_id,
                        st.session_state.sa_json,
                    )
                if err:
                    st.error(err)
                else:
                    st.session_state.bq_client = client
                    st.session_state.bq_tables = tbls
                    st.success(f"Connected. {len(tbls)} table(s) found.")

    if st.session_state.bq_client is not None:
        st.caption(f"✅ BigQuery connected · {len(st.session_state.bq_tables)} tables")
    else:
        st.caption("⚪ Not connected to BigQuery")

    st.divider()

    # ──────────────────────────── Upload files ────────────────────────────
    st.header("3. Upload CSV / Excel")
    st.caption("Drop files to use them directly — no connection needed.")
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
        st.caption(f"✅ {len(st.session_state.uploaded_tables)} uploaded table(s)")
        if st.button("Clear uploaded", use_container_width=True):
            st.session_state.uploaded_tables = {}
            st.session_state.preview_df = None
            st.session_state.preview_table = None
            st.rerun()

    st.divider()

    # ──────────────────────────── Saved config ────────────────────────────
    if CONFIG_PATH.exists():
        st.caption(f"Saved values live in `{CONFIG_PATH.name}` (gitignored).")
    if st.button("🗑 Clear all saved", use_container_width=True):
        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
        for f in FIELDS:
            st.session_state[f] = ""
        st.toast("Cleared all saved values.", icon="🧹")
        st.rerun()

kumo_api_key = st.session_state.kumo_api_key
project_id = st.session_state.project_id
dataset_id = st.session_state.dataset_id
sa_json = st.session_state.sa_json

client = st.session_state.bq_client
bq_tables = st.session_state.bq_tables
uploaded_tables = st.session_state.uploaded_tables

if not bq_tables and not uploaded_tables:
    st.info(
        "Nothing loaded yet. In the sidebar:\n\n"
        "- **1. KumoRFM** — paste your JWT, then click **🔌 Connect Kumo** (needed to run PREDICT).\n"
        "- **2. BigQuery** — (optional) connect to browse tables.\n"
        "- **3. Upload CSV / Excel** — drop files to use them directly.\n\n"
        "You need tables loaded before the Build Graph & Predict panel appears."
    )
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

def guess_primary_key(df: pd.DataFrame, table_name: str) -> str:
    """Heuristic PK pick: prefer an id-like column, unique, otherwise first column."""
    cols = list(df.columns)
    lower = [c.lower() for c in cols]
    # 1. Exact '<table>id' match, e.g. churn -> customerid isn't this, but users -> userid is
    for c, lc in zip(cols, lower):
        if lc == f"{table_name.lower()}id" or lc == f"{table_name.lower()}_id":
            if df[c].is_unique:
                return c
    # 2. Anything ending with 'id' that's unique
    for c, lc in zip(cols, lower):
        if lc.endswith("id") and df[c].is_unique:
            return c
    # 3. Column literally named 'id'
    for c, lc in zip(cols, lower):
        if lc == "id" and df[c].is_unique:
            return c
    # 4. First unique column
    for c in cols:
        if df[c].is_unique:
            return c
    # 5. Fallback: first column
    return cols[0]


if build_clicked:
    with st.spinner("Building graph…"):
        try:
            tables_dict = {n: st.session_state.uploaded_tables[n] for n in chosen}
            graph = rfm.Graph.from_data(tables_dict)

            # Auto-assign a primary key per table — required for FOR clauses.
            pk_guesses = {}
            for name in chosen:
                pk = guess_primary_key(tables_dict[name], name)
                try:
                    graph.table(name).primary_key = pk
                    pk_guesses[name] = pk
                except Exception as e:
                    st.warning(f"Couldn't set PK on {name}: {e}")

            try:
                graph.validate()
            except Exception as ve:
                st.warning(f"Graph validation warning: {ve}")

            st.session_state.graph = graph
            st.session_state.graph_tables = chosen
            st.session_state.graph_pks = pk_guesses
            st.session_state.predict_result = None
            st.success(
                f"Graph built with {len(chosen)} table(s): {', '.join(chosen)} · "
                f"PKs: {', '.join(f'{t}.{pk_guesses[t]}' for t in chosen)}"
            )
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

    # Show each table's columns + primary key. PQL FOR clause requires a PK.
    with st.expander("Tables & primary keys", expanded=True):
        st.caption(
            "The `FOR` clause must reference a **primary key** column. "
            "Override the auto-picked PK below if needed, then click **Apply PKs**."
        )
        pk_override = {}
        for t in st.session_state.graph_tables:
            df = st.session_state.uploaded_tables[t]
            current_pk = st.session_state.graph_pks.get(t, df.columns[0])
            cols = list(df.columns)
            idx = cols.index(current_pk) if current_pk in cols else 0
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**{t}** ({len(df):,} rows)")
            with c2:
                pk_override[t] = st.selectbox(
                    f"Primary key for `{t}`",
                    options=cols,
                    index=idx,
                    key=f"pk_select_{t}",
                    label_visibility="collapsed",
                )
            st.caption("Columns: " + ", ".join(f"`{t}.{c}`" for c in df.columns))

        if st.button("Apply PKs", use_container_width=True):
            for name, pk in pk_override.items():
                try:
                    st.session_state.graph.table(name).primary_key = pk
                except Exception as e:
                    st.error(f"Failed to set PK on {name}: {e}")
            st.session_state.graph_pks = pk_override
            try:
                st.session_state.graph.validate()
                st.success(f"PKs applied: {', '.join(f'{t}.{pk}' for t, pk in pk_override.items())}")
            except Exception as ve:
                st.warning(f"Graph validation warning: {ve}")

    st.markdown("#### PREDICT query (PQL)")

    st.info(
        "**Syntax:** `PREDICT <agg>(<target_table>.*, <start>, <end>, <unit>) [op value] "
        "FOR <entity_table>.<pk_column>=<value>`\n\n"
        "Every column in the `FOR` clause **must** be qualified with its table name "
        "(e.g. `customers.customerid=123`, not `customerid=123`)."
    )

    # Quick query builder
    with st.expander("🛠 Build a query from the tables", expanded=False):
        qb_cols = st.columns(4)
        with qb_cols[0]:
            target_tbl = st.selectbox(
                "Target table",
                options=st.session_state.graph_tables,
                key="qb_target_tbl",
                help="The table being aggregated (e.g. transactions, balance).",
            )
        with qb_cols[1]:
            entity_tbl = st.selectbox(
                "Entity table",
                options=st.session_state.graph_tables,
                key="qb_entity_tbl",
                help="The table whose rows you're predicting for.",
            )
        # Entity column must be the primary key of the entity table.
        pk_for_entity = st.session_state.graph_pks.get(entity_tbl)
        entity_cols = [pk_for_entity] if pk_for_entity else list(
            st.session_state.uploaded_tables[entity_tbl].columns
        )
        with qb_cols[2]:
            entity_col = st.selectbox(
                "Entity column (PK)",
                options=entity_cols,
                key="qb_entity_col",
                help="Must be the primary key of the entity table. "
                     "Change the PK in the 'Tables & primary keys' panel above.",
                disabled=len(entity_cols) == 1,
            )
        with qb_cols[3]:
            entity_val = st.text_input("Entity value", key="qb_entity_val")

        qb_cols2 = st.columns(4)
        with qb_cols2[0]:
            agg = st.selectbox("Aggregation", options=["COUNT", "SUM", "AVG", "MIN", "MAX"], key="qb_agg")
        with qb_cols2[1]:
            start = st.number_input("Window start", value=0, step=1, key="qb_start")
        with qb_cols2[2]:
            end = st.number_input("Window end", value=30, step=1, key="qb_end")
        with qb_cols2[3]:
            unit = st.selectbox("Unit", options=["days", "hours", "weeks", "months"], key="qb_unit")

        if agg == "COUNT":
            agg_expr = f"COUNT({target_tbl}.*, {start}, {end}, {unit})"
        else:
            target_cols = list(st.session_state.uploaded_tables[target_tbl].columns)
            target_col = st.selectbox(
                f"{agg} column (from {target_tbl})",
                options=target_cols,
                key="qb_target_col",
            )
            agg_expr = f"{agg}({target_tbl}.{target_col}, {start}, {end}, {unit})"

        threshold = st.text_input(
            "Optional threshold (e.g. `> 0`, `>= 100`)",
            key="qb_threshold",
            value="> 0",
        )
        threshold_part = f" {threshold}" if threshold.strip() else ""

        built = f"PREDICT {agg_expr}{threshold_part} FOR {entity_tbl}.{entity_col}={entity_val or '<value>'}"
        st.code(built, language="sql")
        if st.button("Use this query", use_container_width=True):
            st.session_state.predict_query = built
            st.rerun()

    st.text_area(
        "Predictive query",
        key="predict_query",
        height=100,
        label_visibility="collapsed",
        help="KumoRFM Predictive Query Language. "
             "Example: PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR users.user_id=1",
    )

    if st.button("▶ Run PREDICT", type="primary"):
        # Clean common whitespace issues around `.` and `=` that break the parser.
        import re
        raw_query = st.session_state.predict_query
        cleaned = re.sub(r"\s*\.\s*", ".", raw_query)   # no spaces around '.'
        cleaned = re.sub(r"\s*=\s*", "=", cleaned)       # no spaces around '='
        cleaned = cleaned.strip()

        if cleaned != raw_query:
            st.caption(f"Auto-cleaned query: `{cleaned}`")

        with st.spinner("Running prediction…"):
            try:
                model = rfm.KumoRFM(st.session_state.graph)
                result = model.predict(cleaned)
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
