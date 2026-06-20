import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import mysql.connector

st.set_page_config(
    layout="wide",
    page_title="Retail Churn Dashboard",
    page_icon="🛍️"
)

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "Capsule@1603",
    "database": "churn_analytics"
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def run_query(query):
    conn = get_db_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r'data\processed\customer_churn_predictions.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        model = joblib.load(r'models\best_churn_model.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_feature_importances(model, feature_columns):
    if hasattr(model, 'feature_importances_'):
        return pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
    elif hasattr(model, 'coef_'):
        return pd.DataFrame({
            'Feature': feature_columns,
            'Importance': np.abs(model.coef_[0])
        }).sort_values(by='Importance', ascending=False)
    return pd.DataFrame()

df_features = load_data()
model = load_model()

if df_features.empty or model is None:
    st.stop()

X_for_importance = df_features.drop(
    ['Customer ID', 'is_churned', 'churn_probability', 'predicted_churn', 'PrimaryCountry_Grouped_Original'],
    axis=1, errors='ignore'
)
feature_importance_df = get_feature_importances(model, X_for_importance.columns)

st.title("🛍️ Retail Customer Churn Analysis Dashboard")
st.markdown("End-to-end churn prediction platform — ML model + SQL analytics layer + business insights.")

st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)

total_customers = df_features['Customer ID'].nunique()
actual_churn_rate = df_features['is_churned'].mean() * 100
predicted_churn_rate = df_features['predicted_churn'].mean() * 100
predicted_churners = int(df_features['predicted_churn'].sum())
revenue_at_risk = df_features[df_features['churn_probability'] >= 0.7]['Monetary'].sum()

with col1:
    st.metric("Total Customers", f"{total_customers:,}")
with col2:
    st.metric("Actual Churn Rate", f"{actual_churn_rate:.1f}%")
with col3:
    st.metric("Predicted Churn Rate", f"{predicted_churn_rate:.1f}%")
with col4:
    st.metric("Predicted Churners", f"{predicted_churners:,}")
with col5:
    st.metric("Revenue at Risk (High Risk)", f"£{revenue_at_risk:,.0f}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Churn Drivers",
    "🌍 Country Analysis",
    "🗄️ SQL Insights",
    "🔍 Customer Lookup"
])

with tab1:
    st.subheader("Top Features Influencing Churn")
    col_a, col_b = st.columns(2)

    with col_a:
        if not feature_importance_df.empty:
            fig = px.bar(
                feature_importance_df.head(10),
                x='Importance', y='Feature',
                orientation='h',
                title='Top 10 Churn Drivers',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        churn_counts = df_features['is_churned'].value_counts().reset_index()
        churn_counts.columns = ['Status', 'Count']
        churn_counts['Status'] = churn_counts['Status'].map({0: 'Retained', 1: 'Churned'})
        fig2 = px.pie(
            churn_counts, values='Count', names='Status',
            title='Churned vs Retained Customers',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("RFM Distribution — Churned vs Retained")
    col_c, col_d, col_e = st.columns(3)

    with col_c:
        fig3 = px.box(df_features, x='is_churned', y='Recency',
                      title='Recency by Churn Status',
                      labels={'is_churned': 'Churned', 'Recency': 'Recency (Days)'},
                      color='is_churned',
                      color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        fig4 = px.box(df_features, x='is_churned', y='Frequency',
                      title='Frequency by Churn Status',
                      labels={'is_churned': 'Churned', 'Frequency': 'No. of Orders'},
                      color='is_churned',
                      color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig4, use_container_width=True)

    with col_e:
        fig5 = px.box(df_features, x='is_churned', y='Monetary',
                      title='Monetary by Churn Status',
                      labels={'is_churned': 'Churned', 'Monetary': 'Revenue (£)'},
                      color='is_churned',
                      color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("🎯 At-Risk Customers")
    churn_threshold = st.slider(
        "Churn Probability Threshold:",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )
    at_risk = df_features[df_features['churn_probability'] >= churn_threshold].sort_values(
        by='churn_probability', ascending=False
    )
    st.info(f"{len(at_risk):,} customers with churn probability ≥ {churn_threshold:.2f}")

    if not at_risk.empty:
        display_cols = ['Customer ID', 'churn_probability', 'Recency',
                        'Frequency', 'Monetary', 'Tenure', 'PrimaryCountry_Grouped_Original']
        st.dataframe(
            at_risk[display_cols].head(500),
            hide_index=True,
            column_config={
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn Probability", format="%.2f", min_value=0, max_value=1),
                "Monetary": st.column_config.NumberColumn("Revenue (£)", format="£%.2f"),
                "Recency": st.column_config.NumberColumn("Recency (Days)"),
                "Frequency": st.column_config.NumberColumn("Orders"),
                "Tenure": st.column_config.NumberColumn("Tenure (Days)"),
            }
        )

with tab2:
    st.subheader("Churn Rate by Country")
    col_f, col_g = st.columns(2)

    with col_f:
        churn_by_country = df_features.groupby('PrimaryCountry_Grouped_Original')['is_churned'].mean().mul(100).reset_index()
        churn_by_country.columns = ['Country', 'Churn Rate (%)']
        fig6 = px.bar(
            churn_by_country.sort_values('Churn Rate (%)', ascending=False),
            x='Country', y='Churn Rate (%)',
            title='Churn Rate by Country',
            color='Churn Rate (%)',
            color_continuous_scale='Plasma'
        )
        st.plotly_chart(fig6, use_container_width=True)

    with col_g:
        revenue_by_country = df_features.groupby('PrimaryCountry_Grouped_Original')['Monetary'].sum().reset_index()
        revenue_by_country.columns = ['Country', 'Total Revenue (£)']
        fig7 = px.pie(
            revenue_by_country,
            values='Total Revenue (£)', names='Country',
            title='Revenue Share by Country'
        )
        st.plotly_chart(fig7, use_container_width=True)

with tab3:
    st.subheader("🗄️ SQL Analytics Layer — Live MySQL Queries")
    st.markdown("All results below are fetched **live from MySQL database** using analytical SQL queries.")

    sql_option = st.selectbox("Select Analysis:", [
        "Q1 — Overall Churn Rate",
        "Q2 — Revenue: Churned vs Retained",
        "Q3 — Monthly Revenue Trend",
        "Q4 — Top 10 Countries by Revenue",
        "Q5 — Top 10 Best Selling Products",
        "Q6 — Customer RFM Segments",
        "Q7 — Churn Rate by Segment",
        "Q8 — Top 20 High Risk Customers",
        "Q9 — Revenue at Risk",
        "Q10 — Churn by Tenure Bucket",
        "Q11 — Customer Lifetime Value (CLV)",
        "Q13 — Average Order Value Trend (LAG)",
        "Q14 — Churn Risk by Country",
    ])

    queries = {
        "Q1 — Overall Churn Rate": """
            SELECT COUNT(*) AS total_customers,
                   SUM(is_churned) AS churned_customers,
                   ROUND(SUM(is_churned) * 100.0 / COUNT(*), 2) AS churn_rate_percent
            FROM customer_features
        """,
        "Q2 — Revenue: Churned vs Retained": """
            SELECT is_churned,
                   COUNT(*) AS customer_count,
                   ROUND(AVG(Monetary), 2) AS avg_revenue,
                   ROUND(SUM(Monetary), 2) AS total_revenue,
                   ROUND(AVG(Frequency), 2) AS avg_orders
            FROM customer_features
            GROUP BY is_churned
        """,
        "Q3 — Monthly Revenue Trend": """
            SELECT DATE_FORMAT(STR_TO_DATE(InvoiceDate, '%Y-%m-%d %H:%i:%s'), '%Y-%m') AS month,
                   COUNT(DISTINCT `Customer ID`) AS unique_customers,
                   ROUND(SUM(Revenue), 2) AS monthly_revenue,
                   COUNT(DISTINCT Invoice) AS total_orders
            FROM transactions
            GROUP BY month
            ORDER BY month
        """,
        "Q4 — Top 10 Countries by Revenue": """
            SELECT Country,
                   COUNT(DISTINCT `Customer ID`) AS customers,
                   ROUND(SUM(Revenue), 2) AS total_revenue,
                   COUNT(DISTINCT Invoice) AS total_orders
            FROM transactions
            GROUP BY Country
            ORDER BY total_revenue DESC
            LIMIT 10
        """,
        "Q5 — Top 10 Best Selling Products": """
            SELECT StockCode, Description,
                   SUM(Quantity) AS total_quantity_sold,
                   ROUND(SUM(Revenue), 2) AS total_revenue
            FROM transactions
            WHERE Quantity > 0
            GROUP BY StockCode, Description
            ORDER BY total_revenue DESC
            LIMIT 10
        """,
        "Q6 — Customer RFM Segments": """
            SELECT `Customer ID`, Recency, Frequency, ROUND(Monetary, 2) AS Monetary,
                   CASE
                       WHEN Recency <= 30  AND Frequency >= 10 AND Monetary >= 1000 THEN 'High Value'
                       WHEN Recency <= 60  AND Frequency >= 5                        THEN 'Loyal'
                       WHEN Recency <= 90  AND Frequency >= 3                        THEN 'At Risk'
                       WHEN Recency > 90   AND Frequency <= 2                        THEN 'Lost'
                       ELSE 'New / Occasional'
                   END AS customer_segment
            FROM customer_features
            ORDER BY Monetary DESC
            LIMIT 100
        """,
        "Q7 — Churn Rate by Segment": """
            WITH segs AS (
                SELECT is_churned,
                       CASE
                           WHEN Recency <= 30  AND Frequency >= 10 AND Monetary >= 1000 THEN 'High Value'
                           WHEN Recency <= 60  AND Frequency >= 5                        THEN 'Loyal'
                           WHEN Recency <= 90  AND Frequency >= 3                        THEN 'At Risk'
                           WHEN Recency > 90   AND Frequency <= 2                        THEN 'Lost'
                           ELSE 'New / Occasional'
                       END AS customer_segment
                FROM customer_features
            )
            SELECT customer_segment,
                   COUNT(*) AS total_customers,
                   SUM(is_churned) AS churned,
                   ROUND(SUM(is_churned) * 100.0 / COUNT(*), 2) AS churn_rate_percent
            FROM segs
            GROUP BY customer_segment
            ORDER BY churn_rate_percent DESC
        """,
        "Q8 — Top 20 High Risk Customers": """
            SELECT `Customer ID`,
                   ROUND(churn_probability, 4) AS churn_probability,
                   RANK() OVER (ORDER BY churn_probability DESC) AS risk_rank,
                   ROUND(Monetary, 2) AS total_spent,
                   Recency AS days_since_last_purchase,
                   PrimaryCountry_Grouped_Original AS country
            FROM customer_predictions
            ORDER BY churn_probability DESC
            LIMIT 20
        """,
        "Q9 — Revenue at Risk": """
            WITH high_risk AS (
                SELECT cp.`Customer ID`, cp.churn_probability, cf.Monetary
                FROM customer_predictions cp
                JOIN customer_features cf ON cp.`Customer ID` = cf.`Customer ID`
                WHERE cp.churn_probability >= 0.7
            )
            SELECT COUNT(*) AS high_risk_customers,
                   ROUND(SUM(Monetary), 2) AS revenue_at_risk,
                   ROUND(AVG(churn_probability) * 100, 2) AS avg_churn_prob_percent
            FROM high_risk
        """,
        "Q10 — Churn by Tenure Bucket": """
            SELECT CASE
                       WHEN Tenure <= 90  THEN '0-3 months'
                       WHEN Tenure <= 180 THEN '3-6 months'
                       WHEN Tenure <= 365 THEN '6-12 months'
                       WHEN Tenure <= 730 THEN '1-2 years'
                       ELSE '2+ years'
                   END AS tenure_bucket,
                   COUNT(*) AS customers,
                   SUM(is_churned) AS churned,
                   ROUND(SUM(is_churned) * 100.0 / COUNT(*), 2) AS churn_rate_percent,
                   ROUND(AVG(Monetary), 2) AS avg_revenue
            FROM customer_features
            GROUP BY tenure_bucket
            ORDER BY churn_rate_percent DESC
        """,
        "Q11 — Customer Lifetime Value (CLV)": """
            SELECT `Customer ID`,
                   ROUND(Monetary, 2) AS total_revenue,
                   Frequency AS total_orders,
                   ROUND(Monetary / NULLIF(Frequency, 0), 2) AS avg_order_value,
                   Tenure AS days_active,
                   ROUND((Monetary / NULLIF(Tenure, 0)) * 365, 2) AS estimated_annual_clv
            FROM customer_features
            ORDER BY estimated_annual_clv DESC
            LIMIT 20
        """,
        "Q13 — Average Order Value Trend (LAG)": """
            WITH monthly_aov AS (
                SELECT DATE_FORMAT(STR_TO_DATE(InvoiceDate, '%Y-%m-%d %H:%i:%s'), '%Y-%m') AS month,
                       ROUND(SUM(Revenue) / COUNT(DISTINCT Invoice), 2) AS avg_order_value
                FROM transactions
                WHERE Revenue > 0
                GROUP BY month
            )
            SELECT month, avg_order_value,
                   LAG(avg_order_value) OVER (ORDER BY month) AS prev_month_aov,
                   ROUND(avg_order_value - LAG(avg_order_value) OVER (ORDER BY month), 2) AS mom_change
            FROM monthly_aov
            ORDER BY month
        """,
        "Q14 — Churn Risk by Country": """
            SELECT PrimaryCountry_Grouped_Original AS country,
                   COUNT(*) AS total_customers,
                   SUM(CASE WHEN churn_probability >= 0.7 THEN 1 ELSE 0 END) AS high_risk,
                   SUM(CASE WHEN churn_probability BETWEEN 0.4 AND 0.7 THEN 1 ELSE 0 END) AS medium_risk,
                   SUM(CASE WHEN churn_probability < 0.4 THEN 1 ELSE 0 END) AS low_risk,
                   ROUND(AVG(churn_probability) * 100, 2) AS avg_churn_prob_percent
            FROM customer_predictions
            GROUP BY country
            ORDER BY avg_churn_prob_percent DESC
        """,
    }

    if st.button("▶ Run Query"):
        try:
            result = run_query(queries[sql_option])
            st.success(f"✅ {len(result)} rows returned from MySQL")
            st.dataframe(result, hide_index=True, use_container_width=True)

            if sql_option == "Q3 — Monthly Revenue Trend":
                fig = px.line(result, x='month', y='monthly_revenue',
                              title='Monthly Revenue Trend', markers=True)
                st.plotly_chart(fig, use_container_width=True)

            elif sql_option == "Q7 — Churn Rate by Segment":
                fig = px.bar(result, x='customer_segment', y='churn_rate_percent',
                             title='Churn Rate by RFM Segment',
                             color='churn_rate_percent',
                             color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)

            elif sql_option == "Q10 — Churn by Tenure Bucket":
                fig = px.bar(result, x='tenure_bucket', y='churn_rate_percent',
                             title='Churn Rate by Customer Tenure',
                             color='churn_rate_percent',
                             color_continuous_scale='Oranges')
                st.plotly_chart(fig, use_container_width=True)

            elif sql_option == "Q13 — Average Order Value Trend (LAG)":
                fig = px.line(result, x='month', y=['avg_order_value', 'prev_month_aov'],
                              title='AOV Trend with Month-over-Month Comparison',
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)

            elif sql_option == "Q14 — Churn Risk by Country":
                fig = px.bar(result, x='country',
                             y=['high_risk', 'medium_risk', 'low_risk'],
                             title='Risk Distribution by Country',
                             barmode='stack')
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Query error: {e}")

with tab4:
    st.subheader("🔍 Customer Risk Profile Lookup")
    st.markdown("Enter a Customer ID to get their complete churn risk profile from MySQL.")

    customer_id_input = st.number_input(
        "Enter Customer ID:", min_value=0, step=1, value=12347
    )

    if st.button("🔍 Get Risk Profile"):
        try:
            query = f"""
                SELECT cp.`Customer ID`,
                       cp.PrimaryCountry_Grouped_Original AS country,
                       ROUND(cp.churn_probability * 100, 2) AS churn_probability_percent,
                       CASE
                           WHEN cp.churn_probability >= 0.7 THEN 'HIGH RISK'
                           WHEN cp.churn_probability >= 0.4 THEN 'MEDIUM RISK'
                           ELSE 'LOW RISK'
                       END AS risk_level,
                       ROUND(cf.Monetary, 2) AS total_spent,
                       cf.Frequency AS total_orders,
                       cf.Recency AS days_since_last_purchase,
                       cf.Tenure AS days_as_customer,
                       cf.is_churned AS actually_churned
                FROM customer_predictions cp
                JOIN customer_features cf ON cp.`Customer ID` = cf.`Customer ID`
                WHERE cp.`Customer ID` = {customer_id_input}
            """
            result = run_query(query)

            if result.empty:
                st.warning(f"No customer found with ID {customer_id_input}")
            else:
                row = result.iloc[0]

                risk_color = {
                    "HIGH RISK": "🔴",
                    "MEDIUM RISK": "🟡",
                    "LOW RISK": "🟢"
                }
                risk = row['risk_level']
                st.markdown(f"### {risk_color[risk]} Customer {int(row['Customer ID'])} — {risk}")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Churn Probability", f"{row['churn_probability_percent']}%")
                c2.metric("Total Spent", f"£{row['total_spent']:,.2f}")
                c3.metric("Total Orders", int(row['total_orders']))
                c4.metric("Days Since Last Purchase", int(row['days_since_last_purchase']))

                c5, c6, c7 = st.columns(3)
                c5.metric("Country", row['country'])
                c6.metric("Days as Customer", int(row['days_as_customer']))
                c7.metric("Actually Churned", "Yes" if row['actually_churned'] else "No")

        except Exception as e:
            st.error(f"Lookup error: {e}")

st.markdown("---")
st.markdown("Built with Python · Scikit-learn · MySQL · Streamlit | Taswi Shahpar")