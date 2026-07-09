import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Akıllı Toplu Ulaşım",
    page_icon="🚌",
    layout="wide"
)

st.title("🚌 Akıllı Toplu Ulaşım Karar Destek Sistemi")

st.markdown("""
Bu uygulama İstanbul Mart 2021 toplu ulaşım verileri kullanılarak
geliştirilen veri analizi ve yapay zekâ destekli karar destek sistemidir.
""")

st.divider()

# --------------------------------------------------
# Verileri Yükle
# --------------------------------------------------

df = pd.read_csv("hourly_data.csv")

top_lines = pd.read_csv("top10_lines.csv")

model = joblib.load("ibb_passenger_model.pkl")

features = joblib.load("features.pkl")   # <-- Bu satır mutlaka olmalı

test_result = pd.read_csv("test_result.csv")

importance = pd.read_csv("feature_importance.csv")

tab1, tab2, tab3 = st.tabs([

    "📊 Veri Analizi",

    "🤖 Model Performansı",

    "🚍 Karar Destek"

])

with tab1:

    st.header("📊 Veri Analizi Dashboard")

    # ===================================================
    # KPI KARTLARI
    # ===================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "📅 Gün Sayısı",
        df["transition_date"].nunique()
    )

    c2.metric(
        "🕒 Saat",
        len(df)
    )

    c3.metric(
        "👥 Toplam Yolcu",
        f"{df['number_of_passenger'].sum():,}"
    )

    c4.metric(
        "📈 Saatlik Ortalama",
        f"{df['number_of_passenger'].mean():,.0f}"
    )

    st.divider()

    # ===================================================
    # GRAFİKLER
    # ===================================================

    col1, col2 = st.columns(2)

    # Saatlik Yoğunluk

    hourly = (
        df.groupby("transition_hour")["number_of_passenger"]
        .mean()
        .reset_index()
    )

    fig1 = px.line(
        hourly,
        x="transition_hour",
        y="number_of_passenger",
        markers=True,
        title="Saatlik Ortalama Yolcu Yoğunluğu"
    )

    fig1.update_layout(
        xaxis_title="Saat",
        yaxis_title="Yolcu"
    )

    col1.plotly_chart(
        fig1,
        use_container_width=True
    )

    # Hafta içi / Hafta sonu

    weekend = (
        df.groupby("is_weekend")["number_of_passenger"]
        .mean()
        .reset_index()
    )

    weekend["is_weekend"] = weekend["is_weekend"].replace({
        0:"Hafta İçi",
        1:"Hafta Sonu"
    })

    fig2 = px.pie(
        weekend,
        names="is_weekend",
        values="number_of_passenger",
        title="Hafta İçi / Hafta Sonu"
    )

    col2.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    # ===================================================
    # En Yoğun Saatler
    # ===================================================

    col3, col4 = st.columns(2)

    top_lines = (
        top_lines
        .sort_values("number_of_passenger", ascending=False)
        .head(10)
    )

    fig3 = px.bar(
        top_lines.sort_values("number_of_passenger"),
        x="number_of_passenger",
        y="line_name",
        orientation="h",
        text="number_of_passenger",
        title="🚌 En Yoğun 10 Hat"
    )

    fig3.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    fig3.update_layout(
        xaxis_title="Toplam Yolcu",
        yaxis_title="Hat"
    )

    col3.plotly_chart(
        fig3,
        use_container_width=True,
        key="top10_lines"
    )

    # ===================================================
    # Ortalama Yolcu Göstergesi
    # ===================================================

    avg = df["number_of_passenger"].mean()
    maxx = df["number_of_passenger"].max()

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg,
        title={"text": "Ortalama Saatlik Yolcu"},
        gauge={
            "axis": {"range": [0, maxx]},
            "bar": {"color": "darkblue"}
        }
    ))

    col4.plotly_chart(
        gauge,
        use_container_width=True
    )


    

    st.divider()

    # ===================================================
    # TABLO
    # ===================================================

    with st.expander("📋 Saatlik Veri Tablosu"):

        st.dataframe(
            df,
            use_container_width=True
        )
with tab2:

    st.header("🤖 Model Performansı")

    # =======================================
    # KPI
    # =======================================

    c1, c2, c3 = st.columns(3)

    c1.metric("📉 MAE","5.476")

    c2.metric("📊 RMSE","8.987")

    c3.metric("🎯 Doğruluk","%99.60")

    st.divider()

    # =======================================
    # Feature Importance
    # =======================================

    col1, col2 = st.columns(2)

    fig_imp = px.bar(

        importance.sort_values(
            "Importance",
            ascending=True
        ),

        x="Importance",

        y="Feature",

        orientation="h",

        title="Feature Importance"

    )
    fig_imp.update_layout(
        title="📊 Özellik Önem Dereceleri",
        xaxis_title="Önem Skoru",
        yaxis_title=""
    )

    col1.plotly_chart(
        fig_imp,
        use_container_width=True
    )

    # =======================================
    # Scatter
    # =======================================

    max_val = max(
        test_result["number_of_passenger"].max(),
        test_result["prediction"].max()
    )

    fig_scatter = px.scatter(
        test_result,
        x="number_of_passenger",
        y="prediction",
        opacity=0.7,
        title="Gerçek vs Model Tahmini"
    )

    fig_scatter.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_val,
        y1=max_val,
        line=dict(
            color="red",
            width=3,
            dash="dash"
        )
    )

    fig_scatter.update_layout(
        xaxis_title="Gerçek Yolcu",
        yaxis_title="Tahmin",
        coloraxis_colorbar_title="Hata"
    )
    col2.plotly_chart(
        fig_scatter,
        use_container_width=True,
        key="scatter"
    )
    st.divider()
    # =======================================
    # Saat Bazında Gerçek vs Tahmin
    # =======================================
    fig_compare = go.Figure()

    fig_compare.add_trace(

        go.Scatter(

            x=test_result["datetime"],

            y=test_result["number_of_passenger"],

            mode="lines",

            name="Gerçek",
            line=dict(
                color = "#00E5FF",
                width=3
            )

        )

    )

    fig_compare.add_trace(

        go.Scatter(

            x=test_result["datetime"],

            y=test_result["prediction"],

            mode="lines",

            name="Tahmin",
            line=dict(
                color="#FFA500",
                width=3,
                dash="dash"
            )

        )

    )

    fig_compare.update_layout(

        title="📈 Saat Bazında Gerçek ve Tahmin",

        xaxis_title="Tarih",

        yaxis_title="Yolcu"

    )

    st.plotly_chart(

        fig_compare,

        use_container_width=True,

        key="compare"

    )
    # =======================================
    # Residual
    # =======================================

    fig_res = px.line(

        test_result,

        x="datetime",

        y="error",

        title="Residual Analizi"

    )

    fig_res.add_hline(

        y=0,

        line_dash="dash",

        line_color="red"

    )

    st.plotly_chart(

        fig_res,

        use_container_width=True

    )

    st.divider()

    # =======================================
    # En Büyük Hatalar
    # =======================================

    st.subheader("🚨 Modelin En Çok Zorlandığı 10 Saat")
    worst = (

        test_result

        .sort_values(

            "abs_error",

            ascending=False

        )

        [[

            "datetime",

            "number_of_passenger",

            "prediction",

            "abs_error"

        ]]

        .head(10)

    )

    st.dataframe(

        worst,

        use_container_width=True

    )
    accuracy = (1 - test_result["abs_error"].mean() /
            test_result["number_of_passenger"].mean()) * 100

    st.success(f"""
    ### 🤖 Model Özeti

    ✅ Model başarıyla eğitildi.

    📈 Ortalama Doğruluk : **%{accuracy:.2f}**

    📉 Ortalama Hata : **{test_result['abs_error'].mean():,.0f} yolcu**

    🎯 Model gerçek verilere oldukça yakın tahminler üretmektedir.
    """)
with tab3:

    st.header("🚍 Belediye Karar Destek Sistemi")

    st.markdown("### 📅 Analiz Seçimi")

    selected_date = st.selectbox(
        "Tarih",
        sorted(df["transition_date"].unique())
    )

    selected_hour = st.selectbox(
        "Saat",
        sorted(df["transition_hour"].unique())
    )

    selected = df[
        (df["transition_date"] == selected_date) &
        (df["transition_hour"] == selected_hour)
    ]

    if not selected.empty:

        # ==================================================
        # Gerçek Veri
        # ==================================================

        real_passenger = int(selected["number_of_passenger"].iloc[0])

        # ==================================================
        # Model Tahmini
        # ==================================================

        feature_input = selected[features].copy()

        prediction = model.predict(feature_input)[0]
        prediction = int(round(prediction))

        # ==================================================
        # Hata Hesabı
        # ==================================================

        error = abs(real_passenger - prediction)
        error_percent = (error / real_passenger) * 100

        st.divider()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🤖 Model Tahmini",
            f"{prediction:,}"
        )

        col2.metric(
            "📊 Gerçek Yolcu",
            f"{real_passenger:,}"
        )

        col3.metric(
            "📉 Tahmin Hatası",
            f"{error:,}",
            f"%{error_percent:.2f}"
        )

        st.divider()

        # ==================================================
        # Akıllı Risk Analizi
        # ==================================================

        hour = int(selected["transition_hour"].iloc[0])
        is_weekend = int(selected["is_weekend"].iloc[0])
        is_peak = int(selected["is_peak"].iloc[0])

        advice = []

        # Risk Seviyesi
        if prediction >= 350000:
            density = "🔴 Çok Yüksek"
        elif prediction >= 250000:
            density = "🟠 Yüksek"
        elif prediction >= 100000:
            density = "🟡 Orta"
        else:
            density = "🟢 Düşük"

        # Kritik öneriler
        if prediction > 350000:
            advice.append("🔴 KRİTİK | Pik saatler için ek otobüs planlanmalı.")
            advice.append("🔴 KRİTİK | Metro sefer sıklığı artırılmalı.")
            advice.append("🔴 KRİTİK | Aktarma istasyonlarına ek personel atanmalı.")

        elif prediction > 200000:
            advice.append("🟠 ORTA | Sefer sıklığı gözden geçirilmeli.")
            advice.append("🟠 ORTA | Yoğun duraklar canlı olarak izlenmeli.")

        # Pik saat kontrolü
        if is_peak:
            advice.append("🟡 BİLGİ | Pik saat yoğunluğu bekleniyor.")

        # Hafta sonu / hafta içi
        if is_weekend:
            advice.append("🟢 BİLGİ | Hafta sonu olduğu için etkinlik bölgeleri takip edilmeli.")
        else:
            advice.append("🟢 BİLGİ | İşe gidiş-geliş yoğunluğu dikkate alınmalı.")

        # Saat bazlı öneriler
        if hour >= 22:
            advice.append("🌙 Gece tarifesi değerlendirilebilir.")

        if hour <= 5:
            advice.append("🛠️ Bakım ve altyapı çalışmaları için uygun zaman.")

        # Düşük yoğunluk
        if prediction < 80000:
            advice.append("✅ Mevcut sefer planı yeterli görünüyor.")

        st.metric(
            "🚦 Tahmini Yoğunluk Seviyesi",
            density
        )
        st.subheader("🧠 Tahmini Etkileyen Faktörler")

        factors = []

        if is_peak:
            factors.append(("★★★★★", "Pik Saat"))

        if prediction > 250000:
            factors.append(("★★★★☆", "Önceki Saatte Yoğunluk"))

        if selected["lag_24"].iloc[0] > 200000:
            factors.append(("★★★★☆", "Bir Önceki Gün Aynı Saat"))

        if not is_weekend:
            factors.append(("★★★☆☆", "Hafta İçi"))

        factors.append(("★★☆☆☆", "Günün Saati"))

        for stars, text in factors:
            st.write(f"{stars}  {text}")
        st.subheader("💡 Yapay Zekâ Destekli Belediye Önerileri")

        for item in advice:
            st.success(item)

        st.divider()

        st.info(f"""
        📅 **Tarih:** {selected_date}

        🕒 **Saat:** {selected_hour}:00

        🤖 **Model Tahmini:** {prediction:,}

        📊 **Gerçek Yolcu:** {real_passenger:,}

        📉 **Tahmin Hatası:** %{error_percent:.2f}
        """)
        st.divider()

        st.subheader("📋 Yönetici Özeti")

        summary = []

        # Yoğunluk
        if prediction >= 350000:
            summary.append("🔴 Beklenen yolcu yoğunluğu çok yüksektir.")
        elif prediction >= 250000:
            summary.append("🟠 Beklenen yolcu yoğunluğu yüksektir.")
        elif prediction >= 100000:
            summary.append("🟡 Beklenen yolcu yoğunluğu orta seviyededir.")
        else:
            summary.append("🟢 Beklenen yolcu yoğunluğu düşüktür.")

        # Pik saat
        if is_peak:
            summary.append("⏰ Seçilen saat pik saat aralığındadır.")

        # Hafta içi / hafta sonu
        if is_weekend:
            summary.append("📅 Hafta sonu olması nedeniyle yolcu profili farklılık gösterebilir.")
        else:
            summary.append("🏢 Hafta içi işe gidiş-geliş yoğunluğu beklenmektedir.")

        # Gece
        if selected_hour >= 22:
            summary.append("🌙 Gece saatleri nedeniyle düşük talep beklenmektedir.")

        # Genel değerlendirme
        summary.append("🤖 Tahmin, XGBoost makine öğrenmesi modeli kullanılarak oluşturulmuştur.")

        for item in summary:
            st.write("• " + item)