"""Streamlit dashboard for AI/ML network traffic analysis.

Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (CLASSES, DEFAULT_WEIGHTS, FEATURES, LABEL_COL, METRICS_FILE,  # noqa: E402
                    OUT_DIR)
from detector import TrafficDetector  # noqa: E402
from generate_data import generate  # noqa: E402

st.set_page_config(page_title="Network Traffic Analysis (AI/ML)", page_icon="🛰️",
                   layout="wide")

COLORS = {"Normal": "#2f855a", "DDoS": "#e53e3e", "PortScan": "#dd6b20",
          "BruteForce": "#805ad5", "Botnet": "#2b6cb0"}


@st.cache_resource
def get_detector():
    return TrafficDetector()


def show_results(res: pd.DataFrame):
    total = len(res)
    attacks = int(res["is_attack"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows analysed", f"{total:,}")
    c2.metric("Attacks flagged", f"{attacks:,}", f"{attacks / max(total, 1):.1%} of traffic",
              delta_color="off")
    c3.metric("Anomalies (Isolation Forest)", f"{int(res['is_anomaly'].sum()):,}")
    if LABEL_COL in res.columns:
        c4.metric("Accuracy vs labels", f"{(res['prediction'] == res[LABEL_COL]).mean():.2%}")
    else:
        c4.metric("Avg confidence", f"{res['confidence'].mean():.2%}")

    left, right = st.columns(2)
    counts = res["prediction"].value_counts().reset_index()
    counts.columns = ["class", "flows"]
    left.plotly_chart(
        px.pie(counts, names="class", values="flows", hole=0.45, color="class",
               color_discrete_map=COLORS, title="Predicted traffic mix"),
        use_container_width=True)
    port_df = (res[res["is_attack"]].groupby(["dst_port", "prediction"]).size()
               .reset_index(name="flows").sort_values("flows", ascending=False).head(15))
    if len(port_df):
        right.plotly_chart(
            px.bar(port_df, x=port_df["dst_port"].astype(str), y="flows", color="prediction",
                   color_discrete_map=COLORS, title="Top targeted ports (attack flows)",
                   labels={"x": "destination port"}),
            use_container_width=True)
    else:
        right.info("No attack flows detected.")

    st.plotly_chart(
        px.scatter(res.sample(min(len(res), 4000), random_state=1), x="flow_pkts_per_s",
                   y="avg_packet_size", color="prediction", color_discrete_map=COLORS,
                   log_x=True, opacity=0.65, title="Packet rate vs packet size",
                   hover_data=["dst_port", "confidence"]),
        use_container_width=True)

    st.subheader("Alerts (highest confidence first)")
    cols = ["prediction", "confidence", "anomaly_score", "protocol", "dst_port", "duration",
            "fwd_packets", "bwd_packets", "flow_pkts_per_s"]
    alerts = res[res["is_attack"]].sort_values("confidence", ascending=False)
    st.dataframe(alerts[cols].head(200), use_container_width=True, hide_index=True)
    st.download_button("Download full results (CSV)", res.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")


st.title("🛰️ AI/ML Network Traffic Analysis")
st.caption("Classifies network flows as Normal, DDoS, PortScan, BruteForce or Botnet and "
           "flags anomalies with an Isolation Forest.")

try:
    detector = get_detector()
except FileNotFoundError:
    st.error("Models not found. Open a terminal in the project folder and run: "
             "`python src/train.py` — then refresh this page.")
    st.stop()

tab_sim, tab_upload, tab_manual, tab_perf = st.tabs(
    ["🎲 Simulate traffic", "📁 Upload CSV", "🔬 Inspect one flow", "📊 Model performance"])

# ---------------------------------------------------------------- simulate
with tab_sim:
    st.write("Generate a fresh batch of mixed traffic the model has never seen, then run detection.")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Number of flows", 200, 20000, 3000, step=100)
    attack_pct = c2.slider("Attack share (%)", 0, 90, 35)
    seed = c3.number_input("Random seed", 0, 99999, 7)
    if st.button("Run simulation", type="primary"):
        attack_total = sum(v for k, v in DEFAULT_WEIGHTS.items() if k != "Normal")
        w = {"Normal": 1 - attack_pct / 100}
        for k, v in DEFAULT_WEIGHTS.items():
            if k != "Normal":
                w[k] = (attack_pct / 100) * v / attack_total
        sim = generate(n, weights=w, seed=int(seed))
        st.session_state["sim_result"] = detector.predict(sim)
    if "sim_result" in st.session_state:
        show_results(st.session_state["sim_result"])

# ---------------------------------------------------------------- upload
with tab_upload:
    st.write("Upload a CSV of flows. Required columns:")
    st.code(", ".join(FEATURES), language="text")
    f = st.file_uploader("CSV file", type="csv")
    if f is not None:
        try:
            res = detector.predict(pd.read_csv(f))
            show_results(res)
        except ValueError as e:
            st.error(str(e))

# ---------------------------------------------------------------- manual
with tab_manual:
    st.write("Craft a single flow and see how the model classifies it.")
    preset = st.selectbox("Start from a preset", ["Custom"] + CLASSES)
    if preset == "Custom":
        base = generate(1, weights={"Normal": 1}, seed=1).iloc[0]
    else:
        base = generate(1, weights={preset: 1}, seed=11).iloc[0]
    a, b, c = st.columns(3)
    vals = {}
    vals["protocol"] = a.selectbox("Protocol", [6, 17, 1],
                                   index=[6, 17, 1].index(int(base["protocol"])),
                                   format_func={6: "TCP (6)", 17: "UDP (17)", 1: "ICMP (1)"}.get)
    vals["dst_port"] = a.number_input("Destination port", 0, 65535, int(base["dst_port"]))
    vals["duration"] = a.number_input("Duration (s)", 0.0001, 5000.0, float(round(base["duration"], 4)),
                                      format="%.4f")
    vals["fwd_packets"] = b.number_input("Forward packets", 1, 100000, int(base["fwd_packets"]))
    vals["bwd_packets"] = b.number_input("Backward packets", 0, 100000, int(base["bwd_packets"]))
    vals["fwd_bytes"] = b.number_input("Forward bytes", 0, 50_000_000, int(base["fwd_bytes"]))
    vals["bwd_bytes"] = b.number_input("Backward bytes", 0, 50_000_000, int(base["bwd_bytes"]))
    vals["syn_count"] = c.number_input("SYN flags", 0, 100000, int(base["syn_count"]))
    vals["ack_count"] = c.number_input("ACK flags", 0, 100000, int(base["ack_count"]))
    vals["rst_count"] = c.number_input("RST flags", 0, 100000, int(base["rst_count"]))
    vals["fin_count"] = c.number_input("FIN flags", 0, 100000, int(base["fin_count"]))
    vals["psh_count"] = c.number_input("PSH flags", 0, 100000, int(base["psh_count"]))
    iat_cv = st.slider("Timing irregularity (std / mean of inter-arrival)", 0.0, 3.0,
                       float(min(3.0, round(base["iat_std"] / max(base["iat_mean"], 1e-9), 2))))

    total_pk = vals["fwd_packets"] + vals["bwd_packets"]
    total_b = max(vals["fwd_bytes"] + vals["bwd_bytes"], 1)
    iat_mean = vals["duration"] / max(total_pk - 1, 1)
    vals.update({
        "avg_packet_size": total_b / total_pk,
        "flow_bytes_per_s": total_b / vals["duration"],
        "flow_pkts_per_s": total_pk / vals["duration"],
        "iat_mean": iat_mean,
        "iat_std": iat_mean * iat_cv,
    })
    one = detector.predict(pd.DataFrame([vals]))
    row = one.iloc[0]
    st.subheader(f"Prediction: {row['prediction']}  ({row['confidence']:.1%} confidence)")
    proba = detector.pipeline.predict_proba(pd.DataFrame([vals])[FEATURES])[0]
    st.plotly_chart(
        px.bar(x=detector.classes, y=proba, labels={"x": "class", "y": "probability"},
               color=detector.classes, color_discrete_map=COLORS),
        use_container_width=True)
    st.write("Isolation Forest verdict:",
             "🚨 anomalous" if row["is_anomaly"] else "✅ looks like normal traffic",
             f"(score {row['anomaly_score']:.3f})")

# ---------------------------------------------------------------- performance
with tab_perf:
    if not METRICS_FILE.exists():
        st.warning("No metrics yet. Run `python src/train.py`.")
    else:
        m = json.loads(METRICS_FILE.read_text())
        st.write(f"Best model: **{m['best_model']}**  |  train flows: {m['n_train']:,}  |  "
                 f"test flows: {m['n_test']:,}")
        comp = pd.DataFrame(m["models"]).T.round(4)
        st.dataframe(comp, use_container_width=True)
        rep = pd.DataFrame(m["per_class_report"]).T.round(4)
        st.write("Per-class report (test set)")
        st.dataframe(rep, use_container_width=True)
        an = m["anomaly_detector"]
        st.write(f"Isolation Forest (attack vs normal) — precision {an['precision']:.3f}, "
                 f"recall {an['recall']:.3f}, F1 {an['f1']:.3f}")
        imgs = ["confusion_matrix.png", "feature_importance.png", "model_comparison.png",
                "class_distribution.png"]
        cols = st.columns(2)
        for i, name in enumerate(imgs):
            p = OUT_DIR / name
            if p.exists():
                cols[i % 2].image(str(p), caption=name.replace("_", " ").replace(".png", "").title())
