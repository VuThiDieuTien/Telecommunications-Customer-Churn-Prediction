import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .dashboard-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .dashboard-header p { margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem; }

    .kpi-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid; height: 100%; transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-5px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
    .kpi-value { font-size: 2rem; font-weight: 700; margin: 0.5rem 0; }
    .kpi-label { font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-delta { font-size: 0.8rem; margin-top: 0.3rem; }

    .chart-container {
        background: white; border-radius: 10px; padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); height: 100%;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #f8fafc; padding: 0.5rem; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; border-radius: 8px; padding: 0 2rem; font-weight: 600; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""",
    unsafe_allow_html=True
)

def create_kpi_card(label, value, delta=None, color="#667eea"):
    delta_html = ""
    if delta is not None and str(delta).strip() != "":
        delta_str = str(delta)
        delta_color = "#64748b"
        if "↑" in delta_str or "+" in delta_str:
            delta_color = "#ef4444"
        elif "↓" in delta_str or "-" in delta_str:
            delta_color = "#10b981"
        delta_html = f'<div class="kpi-delta" style="color: {delta_color};">{delta_str}</div>'

    return f"""
    <div class="kpi-card" style="border-left-color: {color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {color};">{value}</div>
        {delta_html}
    </div>
    """

def filter_churn_stayed(df: pd.DataFrame) -> pd.DataFrame:
    if "customer_status" not in df.columns:
        return df
    return df[df["customer_status"].isin(["Churned", "Stayed"])].copy()

def safe_yes_rate(df: pd.DataFrame, col: str):
    """% Yes cho cột Yes/No hoặc 1/0. Nếu không tồn tại -> None"""
    if col not in df.columns:
        return None
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return float((s == 1).mean() * 100)
    return float((s.astype(str).str.strip().str.lower() == "yes").mean() * 100)

def require_files(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        st.error("Thiếu file output từ pipeline:")
        for p in missing:
            st.write(f"- {p}")
        st.info("Hãy chạy pipeline để tạo đủ file trong data/dashboard/")
        st.stop()

DATA_DIR = "data/dashboard"
MASTER_CSV = os.path.join(DATA_DIR, "dashboard_master_data.csv")
METRICS_JSON = os.path.join(DATA_DIR, "model_metrics.json")
FEATURE_IMP_CSV = os.path.join(DATA_DIR, "feature_importance.csv")

MODEL_XGB = os.path.join(DATA_DIR, "churn_model_xgb.pkl")
MODEL_KMEANS = os.path.join(DATA_DIR, "kmeans_model.pkl")
MODEL_FEATURES = os.path.join(DATA_DIR, "model_features.pkl")

require_files(MASTER_CSV, METRICS_JSON)

@st.cache_data
def load_dashboard_data():
    df = pd.read_csv(MASTER_CSV)
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    feature_imp = pd.DataFrame(columns=["feature", "importance"])
    if os.path.exists(FEATURE_IMP_CSV):
        feature_imp = pd.read_csv(FEATURE_IMP_CSV)

    return df, metrics, feature_imp

@st.cache_resource
def load_models():
    xgb_model = joblib.load(MODEL_XGB) if os.path.exists(MODEL_XGB) else None
    kmeans_model = joblib.load(MODEL_KMEANS) if os.path.exists(MODEL_KMEANS) else None
    model_features = joblib.load(MODEL_FEATURES) if os.path.exists(MODEL_FEATURES) else None
    return xgb_model, kmeans_model, model_features

df, metrics, feature_imp = load_dashboard_data()
xgb_model, kmeans_model, model_features = load_models()

if "customer_status" in df.columns:
    df["customer_status"] = df["customer_status"].astype(str)

df_churn = filter_churn_stayed(df)

min_cols = ["customer_status", "tenure_in_months", "monthly_charge"]
missing_min = [c for c in min_cols if c not in df_churn.columns]
if missing_min:
    st.error(f"Master data thiếu các cột tối thiểu: {missing_min}")
    st.stop()


def render_overview_page():
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>📊 CUSTOMER CHURN DASHBOARD</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_customers = len(df_churn)
    churned = int((df_churn["customer_status"] == "Churned").sum())
    churn_rate = (churned / total_customers * 100) if total_customers > 0 else 0.0

    active_df = df_churn[df_churn["customer_status"] == "Stayed"].copy()
    monthly_active = float(active_df["monthly_charge"].sum()) if len(active_df) else 0.0
    yearly_active = monthly_active * 12
    arpu_active = float(active_df["monthly_charge"].mean()) if len(active_df) else 0.0

    kpis = []
    kpis.append(("CHURNED CUSTOMERS", f"{churned:,}", f"↑ {churn_rate:.1f}%", "#ef4444"))

    if "Tech Tickets" in df_churn.columns:
        kpis.append(("# TECH TICKETS", f"{int(df_churn['Tech Tickets'].sum()):,}", None, "#f59e0b"))
    if "Admin Tickets" in df_churn.columns:
        kpis.append(("# ADMIN TICKETS", f"{int(df_churn['Admin Tickets'].sum()):,}", None, "#8b5cf6"))

    if "monthly_charge" in df_churn.columns:
        kpis.append(("YEARLY ACTIVE CHARGES", f"${yearly_active/1e6:.2f}M", None, "#10b981"))
        kpis.append(("MONTHLY ACTIVE CHARGES", f"${monthly_active/1e3:.1f}K", f"ARPU: ${arpu_active:.2f}", "#3b82f6"))

    cols = st.columns(min(5, len(kpis)))
    for i, (label, value, delta, color) in enumerate(kpis[:5]):
        with cols[i]:
            st.markdown(create_kpi_card(label, value, delta, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    row1_col1, row1_col2 = st.columns([1, 1])

    with row1_col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 👥 DEMOGRAPHICS")

        has_gender = "Gender" in df_churn.columns
        fig1 = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=("Churned by Gender" if has_gender else "Churned vs Stayed", "Tenure Distribution (%)")
        )

        if has_gender:
            churned_gender = df_churn[df_churn["customer_status"] == "Churned"]["Gender"].value_counts()
            fig1.add_trace(go.Pie(labels=churned_gender.index, values=churned_gender.values, hole=0.5, textinfo="percent"), 1, 1)
        else:
            vs = df_churn["customer_status"].value_counts()
            fig1.add_trace(go.Pie(labels=vs.index, values=vs.values, hole=0.5, textinfo="percent"), 1, 1)

        tenure_bins = pd.cut(df_churn["tenure_in_months"], bins=[-0.01, 12, 24, 36, 48, 60, 72, 10**9])
        tenure_dist = tenure_bins.value_counts().sort_index()
        tenure_pct = tenure_dist.values / tenure_dist.sum() * 100
        tenure_labels = ["<1Y", "1-2Y", "2-3Y", "3-4Y", "4-5Y", "5-6Y", ">6Y"][:len(tenure_pct)]

        fig1.add_trace(go.Bar(x=tenure_labels, y=tenure_pct, text=[f"{p:.1f}%" for p in tenure_pct], textposition="outside"), 1, 2)
        fig1.update_layout(height=320, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("**Demographics Stats:**")
        c1, c2, c3 = st.columns(3)

        if "Senior Citizen" in df_churn.columns:
            c1.metric("Senior Citizen", f"{safe_yes_rate(df_churn,'Senior Citizen'):.0f}%")
        if "Married" in df_churn.columns:
            c2.metric("Married", f"{safe_yes_rate(df_churn,'Married'):.0f}%")
        if "number_of_dependents" in df_churn.columns:
            dep_pct = float((df_churn["number_of_dependents"] > 0).mean() * 100)
            c3.metric("Has Dependents", f"{dep_pct:.0f}%")

        st.markdown("</div>", unsafe_allow_html=True)

    with row1_col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 📱 SUBSCRIBED SERVICES")

        candidate_service_cols = [
            "Phone Service", "Online Security", "Online Backup", "Device Protection Plan",
            "Premium Tech Support", "Streaming TV", "Streaming Movies", "Paperless Billing", "Multiple Lines"
        ]
        existing = [c for c in candidate_service_cols if c in df_churn.columns]

        if existing:
            cols3 = st.columns(3)
            for i, col in enumerate(existing[:9]):
                rate = safe_yes_rate(df_churn, col)
                cols3[i % 3].metric(col, f"{rate:.0f}%")
            st.markdown("---")

        if "internet_type" in df_churn.columns:
            st.markdown("**INTERNET TYPE USERS**")
            internet_dist = df_churn["internet_type"].value_counts()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=internet_dist.index,
                x=internet_dist.values,
                orientation="h",
                text=[f"{v/internet_dist.sum()*100:.1f}%" for v in internet_dist.values],
                textposition="outside"
            ))
            fig.update_layout(height=230, margin=dict(l=20, r=20, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)

    with b1:
        if "payment_method" in df_churn.columns:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 💳 PAYMENT METHOD")
            payment_dist = df_churn["payment_method"].value_counts()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=payment_dist.index,
                x=payment_dist.values,
                orientation="h",
                text=[f"{v/payment_dist.sum()*100:.1f}%" for v in payment_dist.values],
                textposition="outside"
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with b2:
        if "Paperless Billing" in df_churn.columns:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 🧾 PAPERLESS BILLING")
            paper = df_churn["Paperless Billing"].value_counts()
            fig = go.Figure()
            fig.add_trace(go.Pie(labels=paper.index, values=paper.values, hole=0.5, textinfo="percent"))
            fig.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.metric("ACTIVE ARPU", f"${arpu_active:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

    with b3:
        if "contract" in df_churn.columns:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 📄 CONTRACT TYPE")
            contract_dist = df_churn["contract"].value_counts()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=contract_dist.index,
                y=contract_dist.values,
                text=[f"{v/contract_dist.sum()*100:.1f}%" for v in contract_dist.values],
                textposition="outside"
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


def render_customer_360():
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>👥 CHÂN DUNG KHÁCH HÀNG (CUSTOMER 360)</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "Cluster_Label" not in df_churn.columns:
        st.warning("Không có Cluster_Label trong master data => bỏ toàn bộ persona/cluster section.")
        return

    churn_by_cluster = (
        df_churn.groupby("Cluster_Label")["customer_status"]
        .apply(lambda x: float((x == "Churned").mean()))
        .sort_values(ascending=False)
    )
    ordered_clusters = churn_by_cluster.index.tolist()
    labels_rank = ["🔴 Rủi Ro Cao", "🟡 Rủi Ro TB", "🟢 Ổn Định", "🔵 Trung Thành"]
    name_rank = {int(cl): (labels_rank[i] if i < len(labels_rank) else f"Cluster {cl}") for i, cl in enumerate(ordered_clusters)}
    palette = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6"]
    color_rank = {int(cl): (palette[i] if i < len(palette) else "#64748b") for i, cl in enumerate(ordered_clusters)}

    st.markdown("### 📊 Personas (Cluster Distribution)")
    col1, col2 = st.columns([1, 2])

    with col1:
        dist = df_churn["Cluster_Label"].value_counts().sort_index()
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=[name_rank.get(int(i), f"Cluster {i}") for i in dist.index],
            values=dist.values,
            marker=dict(colors=[color_rank.get(int(i), "#64748b") for i in dist.index]),
            textinfo="label+percent",
            hole=0.4
        ))
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Persona stats")
        agg = {"tenure_in_months": "mean", "monthly_charge": "mean"}
        if "number_of_referrals" in df_churn.columns:
            agg["number_of_referrals"] = "mean"
        if "number_of_dependents" in df_churn.columns:
            agg["number_of_dependents"] = "mean"

        stats = df_churn.groupby("Cluster_Label").agg(agg)
        stats["churn_rate_%"] = df_churn.groupby("Cluster_Label")["customer_status"].apply(lambda x: (x == "Churned").mean() * 100)
        stats = stats.round(2)
        stats.index = [name_rank.get(int(i), f"Cluster {i}") for i in stats.index]
        st.dataframe(stats, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💰 Churned vs Stayed")

    churned_df = df_churn[df_churn["customer_status"] == "Churned"]
    stayed_df = df_churn[df_churn["customer_status"] == "Stayed"]

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Box(y=churned_df["monthly_charge"], name="Churned"))
        fig.add_trace(go.Box(y=stayed_df["monthly_charge"], name="Stayed"))
        fig.update_layout(title="Monthly Charge", height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=churned_df["tenure_in_months"], name="Churned", opacity=0.7, nbinsx=20))
        fig.add_trace(go.Histogram(x=stayed_df["tenure_in_months"], name="Stayed", opacity=0.7, nbinsx=20))
        fig.update_layout(title="Tenure (Months)", height=350, barmode="overlay", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        churned_ltv = float((churned_df["monthly_charge"] * churned_df["tenure_in_months"]).mean()) if len(churned_df) else 0.0
        stayed_ltv = float((stayed_df["monthly_charge"] * stayed_df["tenure_in_months"]).mean()) if len(stayed_df) else 0.0
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Churned", "Stayed"], y=[churned_ltv, stayed_ltv],
                             text=[f"${churned_ltv:.0f}", f"${stayed_ltv:.0f}"], textposition="outside"))
        fig.update_layout(title="Avg LTV (proxy)", height=350, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_root_cause_analysis():
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>🔍 NGUYÊN NHÂN CỐT LÕI</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    if feature_imp is not None and not feature_imp.empty and set(["feature", "importance"]).issubset(feature_imp.columns):
        st.markdown("### 🎯 Feature Importance (XGBoost)")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        top = feature_imp.sort_values("importance", ascending=False).head(15)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=top["feature"], x=top["importance"], orientation="h",
                             text=top["importance"].round(3), textposition="outside"))
        fig.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Churn rate theo nhóm")

    cols = st.columns(3)

    def churn_rate_bar(group_col: str, title: str, col_container):
        if group_col not in df_churn.columns:
            return
        tmp = df_churn.groupby(group_col).agg(
            churned=("customer_status", lambda x: (x == "Churned").sum()),
            total=("customer_status", "count")
        ).reset_index()
        tmp["churn_rate"] = tmp["churned"] / tmp["total"] * 100

        col_container.markdown('<div class="chart-container">', unsafe_allow_html=True)
        col_container.markdown(f"#### {title}")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=tmp[group_col].astype(str), y=tmp["churn_rate"],
                             text=[f"{v:.1f}%" for v in tmp["churn_rate"]], textposition="outside"))
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Churn Rate (%)", xaxis_title="")
        col_container.plotly_chart(fig, use_container_width=True)
        col_container.markdown("</div>", unsafe_allow_html=True)

    churn_rate_bar("contract", "Contract Type", cols[0])
    churn_rate_bar("internet_type", "Internet Type", cols[1])
    churn_rate_bar("payment_method", "Payment Method", cols[2])

    if "tenure_in_months" in df_churn.columns and "monthly_charge" in df_churn.columns:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔥 Heatmap: Tenure vs Monthly Charge (Churn Rate)")

        df_heat = df_churn.copy()
        df_heat["tenure_group"] = pd.cut(df_heat["tenure_in_months"], bins=[-0.01, 12, 24, 36, 48, 72, 10**9],
                                         labels=["0-1yr", "1-2yr", "2-3yr", "3-4yr", "4-6yr", ">6yr"])
        df_heat["charge_group"] = pd.cut(df_heat["monthly_charge"], bins=[0, 30, 60, 90, 10**9],
                                         labels=["<$30", "$30-60", "$60-90", ">$90"])

        hm = df_heat.groupby(["tenure_group", "charge_group"]).agg(
            churned=("customer_status", lambda x: (x == "Churned").sum()),
            total=("customer_status", "count")
        ).reset_index()
        hm["churn_rate"] = hm["churned"] / hm["total"] * 100
        pivot = hm.pivot(index="tenure_group", columns="charge_group", values="churn_rate")

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            text=np.round(pivot.values, 1),
            texttemplate="%{text}%",
            colorbar=dict(title="Churn Rate %")
        ))
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    keys = ["accuracy", "auc", "precision", "recall"]
    if all(k in metrics for k in keys):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Model Performance (from pipeline)")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(create_kpi_card("ACCURACY", f"{metrics['accuracy']*100:.2f}%", color="#10b981"), unsafe_allow_html=True)
        c2.markdown(create_kpi_card("AUC-ROC", f"{metrics['auc']:.4f}", color="#3b82f6"), unsafe_allow_html=True)
        c3.markdown(create_kpi_card("PRECISION", f"{metrics['precision']*100:.2f}%", color="#f59e0b"), unsafe_allow_html=True)
        c4.markdown(create_kpi_card("RECALL", f"{metrics['recall']*100:.2f}%", color="#8b5cf6"), unsafe_allow_html=True)


def render_prediction_page():
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>🔮 DỰ BÁO & HÀNH ĐỘNG</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    if xgb_model is None or model_features is None:
        st.warning("Không có model hoặc model_features => bỏ tab Prediction.")
        return

    st.markdown("### 📝 Nhập Thông Tin Khách Hàng")
    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 👤 Personal")
        gender = None
        age = None
        senior = None
        married = None
        dependents = None

        if "Gender" in df_churn.columns:
            gender = st.selectbox("Gender", options=sorted(df_churn["Gender"].dropna().unique().tolist()))
        if "Age" in df_churn.columns:
            age = st.number_input("Age", min_value=18, max_value=100, value=int(df_churn["Age"].dropna().median()) if df_churn["Age"].notna().any() else 35)
        if "Senior Citizen" in df_churn.columns:
            senior = st.selectbox("Senior Citizen", options=["No", "Yes"])
        if "Married" in df_churn.columns:
            married = st.selectbox("Married", options=["No", "Yes"])
        if "number_of_dependents" in df_churn.columns:
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)

        st.markdown("</div>", unsafe_allow_html=True)

    with f2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 📞 Service")
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)

        phone_service = st.selectbox("Phone Service", ["No", "Yes"]) if "Phone Service" in df_churn.columns else None
        online_security = st.selectbox("Online Security", ["No", "Yes"]) if "Online Security" in df_churn.columns else None
        online_backup = st.selectbox("Online Backup", ["No", "Yes"]) if "Online Backup" in df_churn.columns else None
        device_protection = st.selectbox("Device Protection Plan", ["No", "Yes"]) if "Device Protection Plan" in df_churn.columns else None
        tech_support = st.selectbox("Premium Tech Support", ["No", "Yes"]) if "Premium Tech Support" in df_churn.columns else None

        internet_type = None
        if "internet_type" in df_churn.columns:
            internet_type = st.selectbox("Internet Type", options=sorted(df_churn["internet_type"].dropna().unique().tolist()))

        st.markdown("</div>", unsafe_allow_html=True)

    with f3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 💰 Financial")

        contract = None
        payment = None
        paperless = None

        if "contract" in df_churn.columns:
            contract = st.selectbox("Contract", options=sorted(df_churn["contract"].dropna().unique().tolist()))
        if "payment_method" in df_churn.columns:
            payment = st.selectbox("Payment Method", options=sorted(df_churn["payment_method"].dropna().unique().tolist()))
        if "Paperless Billing" in df_churn.columns:
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

        monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=100000.0, value=float(monthly_charge * max(tenure, 1)), step=10.0)

        referrals = None
        if "number_of_referrals" in df_churn.columns:
            referrals = st.number_input("Number of Referrals", min_value=0, max_value=50, value=0)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("🔮 DỰ BÁO NGAY", use_container_width=True, type="primary")
    if not btn:
        return


    input_row = {}

    if gender is not None: input_row["Gender"] = gender
    if age is not None: input_row["Age"] = age
    if senior is not None: input_row["Senior Citizen"] = 1 if senior == "Yes" else 0
    if married is not None: input_row["Married"] = 1 if married == "Yes" else 0
    if dependents is not None: input_row["Number of Dependents"] = dependents

    input_row["Tenure in Months"] = tenure
    input_row["Monthly Charge"] = monthly_charge
    input_row["Total Charges"] = total_charges

    if phone_service is not None: input_row["Phone Service"] = phone_service
    if online_security is not None: input_row["Online Security"] = online_security
    if online_backup is not None: input_row["Online Backup"] = online_backup
    if device_protection is not None: input_row["Device Protection Plan"] = device_protection
    if tech_support is not None: input_row["Premium Tech Support"] = tech_support
    if referrals is not None: input_row["Number of Referrals"] = referrals

    if internet_type is not None: input_row["Internet Type"] = internet_type
    if contract is not None: input_row["Contract"] = contract
    if payment is not None: input_row["Payment Method"] = payment
    if paperless is not None: input_row["Paperless Billing"] = paperless

    input_df = pd.DataFrame([input_row])

    input_encoded = pd.get_dummies(input_df, drop_first=True)
    for col in model_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_features]

    try:
        prob = float(xgb_model.predict_proba(input_encoded)[0][1])
        pred = int(xgb_model.predict(input_encoded)[0])

        st.markdown("### 🎯 KẾT QUẢ DỰ BÁO")
        left, right = st.columns([1, 1])

        with left:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "XÁC SUẤT CHURN (%)"},
                number={"suffix": "%", "font": {"size": 60}},
                gauge={"axis": {"range": [0, 100]}}
            ))
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"Prediction (0=Stayed, 1=Churned): **{pred}**")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 💡 Gợi ý hành động (rule-based)")
            recs = []
            if contract is not None and str(contract).lower().startswith("month"):
                recs.append("Đề xuất ưu đãi chuyển sang hợp đồng dài hạn (1–2 năm).")
            if monthly_charge > 70:
                recs.append("Tư vấn gói phù hợp / ưu đãi add-on để giảm churn.")
            if payment is not None and "electronic" in str(payment).lower():
                recs.append("Khuyến khích chuyển sang auto-payment (bank/credit automatic).")
            if tenure < 12:
                recs.append("Chăm sóc khách mới (tenure < 12): gọi follow-up + ưu đãi nhỏ.")
            if referrals is not None and referrals == 0:
                recs.append("Kích hoạt chương trình giới thiệu bạn bè (referral).")

            if recs:
                for r in recs:
                    st.write(f"- {r}")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("Lỗi khi dự báo.")
        st.exception(e)


def main():
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 BỨC TRANH TOÀN CẢNH",
        "👥 CHÂN DUNG KHÁCH HÀNG",
        "🔍 NGUYÊN NHÂN CỐT LÕI",
        "🔮 DỰ BÁO & HÀNH ĐỘNG"
    ])

    with tab1:
        render_overview_page()
    with tab2:
        render_customer_360()
    with tab3:
        render_root_cause_analysis()
    with tab4:
        render_prediction_page()

    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align:center; color:#64748b; font-size:0.85rem;">
            <p>Data Last Updated: {datetime.now().strftime("%Y-%m-%d")} | AUC: {metrics.get('auc', 0):.4f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
