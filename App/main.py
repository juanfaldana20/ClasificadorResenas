import os
import sys
import pathlib
import io

# --- Streamlit/UI ---
import streamlit as st
import pandas as pd

# --- gRPC ---
import grpc

# Asegura que Python encuentre los stubs generados en ml/
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "ml"))

# Intento de import de stubs gRPC; si falla, la UI sigue pero desactiva funciones que dependen de gRPC
try:
    import sentiment_pb2 as pb
    import sentiment_pb2_grpc as pb_grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False


# --------- Helpers de normalización ----------
def to_std(label: str) -> str:
    """
    Normaliza etiquetas del servidor a: 'positive'|'negative'|'neutral'.
    Acepta 'POS'/'NEG'/'NEU' o variantes en minúsculas/mayúsculas.
    """
    if not label:
        return "neutral"
    u = str(label).strip().upper()
    if u in ("POS", "POSITIVE"):
        return "positive"
    if u in ("NEG", "NEGATIVE"):
        return "negative"
    return "neutral"

def read_table(file) -> pd.DataFrame:
    """
    Lee CSV/XLSX con columna 'texto'. Soporta UTF-8, UTF-8-BOM, Latin-1, CP1252.
    """
    name = file.name.lower()
    if name.endswith(".csv"):
        # Probar varias codificaciones y autodetectar separador
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        last_err = None
        for enc in encodings:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, sep=None, engine="python")
                break
            except Exception as e:
                last_err = e
        else:
            raise ValueError(f"No se pudo leer el CSV. Último error: {last_err}")
    else:
        # XLSX no usa encoding de texto
        df = pd.read_excel(file)

    if "texto" not in df.columns:
        raise ValueError("El archivo debe tener una columna llamada 'texto'.")
    df = df.dropna(subset=["texto"]).copy()
    df["texto"] = df["texto"].astype(str)
    return df


# --------- Cliente gRPC ----------
def make_stub(addr: str) -> "pb_grpc.SentimentServiceStub":
    """
    Crea y devuelve el stub del servicio gRPC.
    """
    channel = grpc.insecure_channel(addr)
    return pb_grpc.SentimentServiceStub(channel)


def ping(stub: "pb_grpc.SentimentServiceStub") -> str:
    """
    Verifica salud del servicio. Devuelve el status.
    """
    return stub.Ping(pb.PingRequest()).status


def predict_text(stub: "pb_grpc.SentimentServiceStub", text: str) -> tuple[str, float]:
    """
    Envía un texto y obtiene (label, score).
    """
    resp = stub.Predict(pb.PredictRequest(text=text))
    return resp.label, resp.score


def predict_batch(
    stub: "pb_grpc.SentimentServiceStub", texts: list[str], chunk: int = 128
) -> list[tuple[str, float]]:
    """
    Predicción en lote con particionado para no exceder tamaño de mensaje.
    Devuelve lista de pares (label, score) alineada a 'texts'.
    """
    out: list[tuple[str, float]] = []
    for i in range(0, len(texts), chunk):
        part = texts[i : i + chunk]
        resp = stub.PredictBatch(pb.PredictBatchRequest(texts=part))
        out.extend(list(zip(resp.labels, resp.scores)))
    return out


# --------- Base de datos simulada de reseñas ----------
if "reviews_db" not in st.session_state:
    st.session_state.reviews_db = []


def save_review(
    text: str,
    recommend: bool,
    sentiment_label: str | None = None,
    sentiment_score: float | None = None,
):
    """
    Guarda una reseña en la base de datos simulada (session_state).
    """
    review = {
        "id": len(st.session_state.reviews_db) + 1,
        "texto": text,
        "recomienda": recommend,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "timestamp": pd.Timestamp.now(),
    }
    st.session_state.reviews_db.append(review)


# --------- CSS Personalizado ----------
def apply_custom_css():
    """
    Inyecta estilos CSS para personalizar la apariencia de la app.
    """
    st.markdown(
        """
    <style>
        .stApp { background-color: #C73D36; }
        .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 900px !important; }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
        .main-title { color: white !important; font-size: 28px !important; font-weight: 600 !important; margin-bottom: 1.5rem !important; text-align: center !important; padding: 1rem 0 !important; }
        .tab-subheader { color: white !important; font-size: 22px !important; font-weight: 500 !important; margin-bottom: 1rem !important; text-align: left !important; }
        .recommend-container { background-color: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0; border: 1px solid rgba(255,255,255,0.2); }
        .recommend-text { color: white !important; font-size: 16px !important; font-weight: 500 !important; margin: 0 !important; }
        .stTextArea > div > div > textarea { background-color: #3e4651 !important; color: white !important; border: 1px solid rgba(255,255,255,0.3) !important; border-radius: 8px !important; font-size: 16px !important; padding: 1rem !important; min-height: 120px !important; }
        .stTextArea > div > div > textarea::placeholder { color: rgba(255,255,255,0.6) !important; }
        .stButton > button { background-color: #4a90e2 !important; color: white !important; border: none !important; padding: 0.8rem 2rem !important; border-radius: 8px !important; font-size: 16px !important; font-weight: 600 !important; width: 100% !important; margin-top: 1rem !important; transition: all 0.3s ease !important; box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important; }
        .stButton > button:hover { background-color: #357abd !important; transform: translateY(-1px) !important; box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important; }
        .stSuccess { background-color: rgba(40,167,69,0.2) !important; color: white !important; border: 1px solid rgba(40,167,69,0.4) !important; }
        .stError { background-color: rgba(220,53,69,0.2) !important; color: white !important; border: 1px solid rgba(220,53,69,0.4) !important; }
        .stInfo { background-color: rgba(74,144,226,0.2) !important; color: white !important; border: 1px solid rgba(74,144,226,0.4) !important; }
        .stWarning { background-color: rgba(255,193,7,0.2) !important; color: white !important; border: 1px solid rgba(255,193,7,0.4) !important; }
        .stDataFrame { background-color: rgba(255,255,255,0.1) !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { background-color: rgba(255,255,255,0.1) !important; color: white !important; border-radius: 8px 8px 0 0 !important; }
        hr { border-color: rgba(255,255,255,0.3) !important; margin: 1.5rem 0 !important; }
        p, div, span, li, label { color: white !important; }
        .metric-container { background-color: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border: 1px solid rgba(255,255,255,0.2); }
        .footer-text { text-align: center; color: rgba(255,255,255,0.8) !important; font-size: 14px !important; margin-top: 1.5rem !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )


# --------- Lectura de archivos ----------
def read_table(file) -> pd.DataFrame:
    """
    Lee CSV o XLSX y retorna DataFrame con columna 'texto' obligatoria.
    Elimina filas vacías y fuerza a string.
    """
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    if "texto" not in df.columns:
        raise ValueError("El archivo debe tener una columna llamada 'texto'.")
    df = df.dropna(subset=["texto"]).copy()
    df["texto"] = df["texto"].astype(str)
    return df


# --------- UI: Escribir reseña ----------
def ui_write_review():
    """
    Interfaz para escribir reseñas con análisis de sentimientos y guardado local.
    """
    st.markdown('<h2 class="tab-subheader">📝 Escribir Reseña</h2>', unsafe_allow_html=True)

    # Sección de recomendación a amigos
    st.markdown('<div class="recommend-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([3.5, 1])
    with col1:
        st.markdown('<p class="recommend-text">¿Recomendarías a amigos?</p>', unsafe_allow_html=True)
    with col2:
        recommend_friends = st.toggle("", value=False, key="recommend_toggle")
    st.markdown("</div>", unsafe_allow_html=True)

    # Área de texto para la reseña
    review_text = st.text_area(
        "",
        placeholder="Escribe tu reseña aquí...",
        height=120,
        key="review_textarea",
        label_visibility="collapsed",
    )

    # Botón de envío
    if st.button("📤 Enviar Reseña", type="primary", use_container_width=True):
        if review_text.strip():
            # Análisis de sentimientos si gRPC está disponible
            sentiment_label = None
            sentiment_score = None

            if GRPC_AVAILABLE:
                try:
                    addr = os.getenv("APP_GRPC_ADDR", "localhost:50051")
                    stub = make_stub(addr)
                    raw_label, sentiment_score = predict_text(stub, review_text.strip())
                    sentiment_label = to_std(raw_label)
                except Exception as e:
                    st.warning(f"⚠ Análisis de sentimientos no disponible: {e}")

            # Guardar reseña
            save_review(review_text.strip(), recommend_friends, sentiment_label, sentiment_score)

            # Mostrar resultados
            st.success("✅ ¡Reseña enviada exitosamente!")
            st.markdown("---")
            st.markdown("📋 Resumen de tu reseña:")

            col1, col2, col3 = st.columns(3)
            with col1:
                recommend_text = "✅ Sí" if recommend_friends else "❌ No"
                st.info(f"*Recomendarías:* {recommend_text}")
            with col2:
                st.info(f"*Caracteres:* {len(review_text)}")
            with col3:
                if sentiment_label and sentiment_score is not None:
                    sentiment_emoji = (
                        "😊" if sentiment_label == "positive" else "😔" if sentiment_label == "negative" else "😐"
                    )
                    st.info(f"*Sentimiento:* {sentiment_emoji} {sentiment_label.upper()} ({sentiment_score:.2f})")
                else:
                    st.info("*Sentimiento:* No disponible")

            st.markdown("*Tu reseña:*")
            st.markdown(f'"{review_text}"')

            st.success("Gracias por tu feedback.")
        else:
            st.error("⚠ Por favor, escribe una reseña antes de enviar.")


# --------- UI: Análisis individual ----------
def ui_sentiment_analysis():
    """
    Interfaz para análisis individual de sentimientos contra el backend gRPC.
    """
    st.markdown('<h2 class="tab-subheader">🧠 Análisis de Sentimientos</h2>', unsafe_allow_html=True)

    if not GRPC_AVAILABLE:
        st.error("❌ Servicio de análisis de sentimientos no disponible. Instala las dependencias gRPC.")
        return

    txt = st.text_area("Texto para analizar:", height=160, placeholder="Pega aquí el texto a analizar...")

    if st.button("🔍 Analizar Sentimiento", type="primary", use_container_width=True, disabled=not txt.strip()):
        try:
            addr = os.getenv("APP_GRPC_ADDR", "localhost:50051")
            stub = make_stub(addr)
            raw_label, score = predict_text(stub, txt.strip())
            label = to_std(raw_label)

            # Mostrar resultado con colores
            sentiment_emoji = "😊" if label == "positive" else "😔" if label == "negative" else "😐"
            sentiment_color = "success" if label == "positive" else "error" if label == "negative" else "info"

            if sentiment_color == "success":
                st.success(f"{sentiment_emoji} *Sentimiento:* {label.upper()} | *Confianza:* {score:.3f}")
            elif sentiment_color == "error":
                st.error(f"{sentiment_emoji} *Sentimiento:* {label.upper()} | *Confianza:* {score:.3f}")
            else:
                st.info(f"{sentiment_emoji} *Sentimiento:* {label.upper()} | *Confianza:* {score:.3f}")

        except Exception as e:
            st.error(f"❌ Error conectando con el servicio: {e}")


# --------- UI: Base de datos local ----------
def ui_reviews_database():
    """
    Interfaz para ver todas las reseñas guardadas en session_state, con métricas simples.
    """
    st.markdown('<h2 class="tab-subheader">📊 Base de Datos de Reseñas</h2>', unsafe_allow_html=True)

    if not st.session_state.reviews_db:
        st.info("📝 No hay reseñas guardadas aún.")
        return

    # Convertir a DataFrame
    df = pd.DataFrame(st.session_state.reviews_db)

    # Normaliza etiquetas si existen
    if "sentiment_label" in df.columns:
        df["label_std"] = df["sentiment_label"].apply(to_std)
    else:
        df["label_std"] = "neutral"

    # Métricas generales
    st.markdown("📈 Estadísticas Generales:")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Reseñas", len(df))
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        recommend_count = int(df["recomienda"].sum())
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Recomiendan", recommend_count)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        positive_count = int((df["label_std"] == "positive").sum())
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Positivas", positive_count)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        avg_length = df["texto"].str.len().mean()
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Longitud Prom.", f"{avg_length:.0f} chars")
        st.markdown("</div>", unsafe_allow_html=True)

    # Mostrar tabla de reseñas
    st.markdown("---")
    st.markdown("📋 Todas las Reseñas:")

    display_df = df.copy()
    display_df["recomienda"] = display_df["recomienda"].map({True: "✅ Sí", False: "❌ No"})
    display_df["sentiment_label"] = display_df["label_std"].str.upper()
    if "sentiment_score" in display_df.columns:
        display_df["sentiment_score"] = display_df["sentiment_score"].fillna(0).round(3)

    st.dataframe(display_df.drop(columns=["label_std"]), use_container_width=True)

    # Botón para limpiar base de datos
    if st.button("🗑 Limpiar Base de Datos", type="secondary"):
        st.session_state.reviews_db = []
        st.success("✅ Base de datos limpiada.")
        st.experimental_rerun()


# --------- UI: Análisis por archivo ----------
def ui_sentiment_file():
    """
    Carga un CSV/XLSX con columna 'texto', ejecuta lote por gRPC y muestra resumen:
    conteos y porcentajes por clase. Permite descargar CSV con predicciones.
    """
    st.markdown('<h2 class="tab-subheader">📂 Análisis por Archivo</h2>', unsafe_allow_html=True)

    if not GRPC_AVAILABLE:
        st.error("❌ Servicio gRPC no disponible.")
        return

    up = st.file_uploader("Sube CSV o XLSX con columna 'texto'", type=["csv", "xlsx"])
    if not up:
        return

    try:
        df = read_table(up)
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Error leyendo archivo: {e}")
        return

    if st.button("▶ Procesar archivo", type="primary", use_container_width=True):
        try:
            addr = os.getenv("APP_GRPC_ADDR", "localhost:50051")
            stub = make_stub(addr)

            # Predicción por lotes (chunking interno)
            pairs = predict_batch(stub, df["texto"].tolist(), chunk=128)
            raw_labels = [p[0] for p in pairs]
            scores = [p[1] for p in pairs]

            # Normaliza etiquetas a positive/negative/neutral
            labels = [to_std(x) for x in raw_labels]

            # DataFrame de salida
            out = df.copy()
            out["label"] = labels
            out["score"] = scores

            # Resumen de conteos y porcentajes
            counts = pd.Series(labels).value_counts().reindex(
                ["positive", "negative", "neutral"], fill_value=0
            )
            total = int(counts.sum())
            pct = (counts / total * 100).round(1) if total > 0 else counts

            # Métricas
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total reseñas", total)
            with c2:
                st.metric("Positivas", f"{int(counts['positive'])} ({pct['positive']}%)")
            with c3:
                st.metric("Negativas", f"{int(counts['negative'])} ({pct['negative']}%)")
            with c4:
                st.metric("Neutras", f"{int(counts['neutral'])} ({pct['neutral']}%)")

            # Gráfico simple
            st.bar_chart(
                counts.rename({"positive": "Positivas", "negative": "Negativas", "neutral": "Neutras"})
            )

            # Descarga CSV
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button(
                "💾 Descargar CSV con predicciones",
                buf.getvalue().encode("utf-8"),
                file_name="predicciones.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.success("Archivo procesado.")
        except Exception as e:
            st.error(f"Error en predicción por lote: {e}")


# --------- Entry point ----------
def main():
    """
    Construye la UI con cuatro pestañas y prueba de salud al inicio.
    """
    st.set_page_config(page_title="Sistema de Reseñas con IA", page_icon="🍽", layout="wide")
    apply_custom_css()

    st.markdown('<h1 class="main-title">🍽 Sistema de Reseñas de Restaurante con IA</h1>', unsafe_allow_html=True)

    # Verificar conexión gRPC si está disponible
    if GRPC_AVAILABLE:
        try:
            addr = os.getenv("APP_GRPC_ADDR", "localhost:50051")
            stub = make_stub(addr)
            status = ping(stub)
            st.toast(f"🤖 IA conectada: {status}", icon="✅")
        except Exception as e:
            st.toast(f"⚠ IA no disponible: {str(e)[:80]}...", icon="⚠")

    # Pestañas
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Escribir Reseña", "🧠 Análisis IA", "📊 Base de Datos", "📂 Archivo"]
    )
    with tab1:
        ui_write_review()
    with tab2:
        ui_sentiment_analysis()
    with tab3:
        ui_reviews_database()
    with tab4:
        ui_sentiment_file()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class="footer-text">
            <p>🍽 Sistema de Reseñas con Análisis de Sentimientos | Powered by BETO + gRPC 🤖</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
