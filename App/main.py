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

try:
    import sentiment_pb2 as pb
    import sentiment_pb2_grpc as pb_grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

# --------- Cliente gRPC ----------
def make_stub(addr: str) -> 'pb_grpc.SentimentServiceStub':
    """
    Crea y devuelve el stub del servicio gRPC.
    """
    channel = grpc.insecure_channel(addr)
    return pb_grpc.SentimentServiceStub(channel)

def ping(stub: 'pb_grpc.SentimentServiceStub') -> str:
    """
    Verifica salud del servicio. Devuelve el status.
    """
    return stub.Ping(pb.PingRequest()).status

def predict_text(stub: 'pb_grpc.SentimentServiceStub', text: str) -> tuple[str, float]:
    """
    Envía un texto y obtiene (label, score).
    """
    resp = stub.Predict(pb.PredictRequest(text=text))
    return resp.label, resp.score

def predict_batch(
    stub: 'pb_grpc.SentimentServiceStub', texts: list[str], chunk: int = 128
) -> list[tuple[str, float]]:
    """
    Predicción en lote con particionado para no exceder tamaño de mensaje.
    """
    out: list[tuple[str, float]] = []
    for i in range(0, len(texts), chunk):
        part = texts[i : i + chunk]
        resp = stub.PredictBatch(pb.PredictBatchRequest(texts=part))
        out.extend(list(zip(resp.labels, resp.scores)))
    return out

def summarize_text(
    stub: 'pb_grpc.SentimentServiceStub', text: str, sentences: int = 3
) -> list[str]:
    """
    Solicita resumen extractivo LexRank. Devuelve oraciones.
    """
    resp = stub.Summarize(pb.SummarizeRequest(text=text, sentences=sentences))
    return list(resp.sentences)

# --------- Base de datos simulada de reseñas ----------
if 'reviews_db' not in st.session_state:
    st.session_state.reviews_db = []

def save_review(text: str, recommend: bool, sentiment_label: str = None, sentiment_score: float = None):
    """
    Guarda una reseña en la base de datos simulada.
    """
    review = {
        'id': len(st.session_state.reviews_db) + 1,
        'texto': text,
        'recomienda': recommend,
        'sentiment_label': sentiment_label,
        'sentiment_score': sentiment_score,
        'timestamp': pd.Timestamp.now()
    }
    st.session_state.reviews_db.append(review)

# --------- CSS Personalizado ----------
def apply_custom_css():
    st.markdown("""
    <style>
        /* Fondo beige para toda la aplicación */
        .stApp {
            background-color: #C73D36;
        }
        
        /* Espaciado y contenedor principal */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 700px !important;
        }
        
        /* Ocultar elementos de streamlit por defecto */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Título principal */
        .main-title {
            color: white !important;
            font-size: 28px !important;
            font-weight: 600 !important;
            margin-bottom: 1.5rem !important;
            text-align: center !important;
            padding: 1rem 0 !important;
        }
        
        /* Subtítulos de tabs */
        .tab-subheader {
            color: white !important;
            font-size: 22px !important;
            font-weight: 500 !important;
            margin-bottom: 1rem !important;
            text-align: left !important;
        }
        
        /* Sección de recomendación */
        .recommend-container {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .recommend-text {
            color: white !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            margin: 0 !important;
        }
        
        /* Área de texto personalizada */
        .stTextArea > div > div > textarea {
            background-color: #3e4651 !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            padding: 1rem !important;
            min-height: 120px !important;
        }
        
        .stTextArea > div > div > textarea::placeholder {
            color: rgba(255, 255, 255, 0.6) !important;
        }
        
        /* Botones */
        .stButton > button {
            background-color: #4a90e2 !important;
            color: white !important;
            border: none !important;
            padding: 0.8rem 2rem !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            width: 100% !important;
            margin-top: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        }
        
        .stButton > button:hover {
            background-color: #357abd !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* Mensajes de estado */
        .stSuccess {
            background-color: rgba(40, 167, 69, 0.2) !important;
            color: white !important;
            border: 1px solid rgba(40, 167, 69, 0.4) !important;
        }
        
        .stError {
            background-color: rgba(220, 53, 69, 0.2) !important;
            color: white !important;
            border: 1px solid rgba(220, 53, 69, 0.4) !important;
        }
        
        .stInfo {
            background-color: rgba(74, 144, 226, 0.2) !important;
            color: white !important;
            border: 1px solid rgba(74, 144, 226, 0.4) !important;
        }
        
        .stWarning {
            background-color: rgba(255, 193, 7, 0.2) !important;
            color: white !important;
            border: 1px solid rgba(255, 193, 7, 0.4) !important;
        }
        
        /* DataFrames */
        .stDataFrame {
            background-color: rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-radius: 8px 8px 0 0 !important;
        }
        
        /* Separadores */
        hr {
            border-color: rgba(255, 255, 255, 0.3) !important;
            margin: 1.5rem 0 !important;
        }
        
        /* Texto general */
        p, div, span, li, label {
            color: white !important;
        }
        
        /* Métricas */
        .metric-container {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Footer */
        .footer-text {
            text-align: center;
            color: rgba(255, 255, 255, 0.8) !important;
            font-size: 14px !important;
            margin-top: 1.5rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --------- UI Functions ----------
def ui_write_review():
    """
    Interfaz para escribir reseñas con análisis de sentimientos.
    """
    st.markdown('<h2 class="tab-subheader">📝 Escribir Reseña</h2>', unsafe_allow_html=True)
    
    # Sección de recomendación a amigos
    st.markdown('<div class="recommend-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([3.5, 1])
    
    with col1:
        st.markdown('<p class="recommend-text">¿Recomendarías a amigos?</p>', unsafe_allow_html=True)
    
    with col2:
        recommend_friends = st.toggle("", value=False, key="recommend_toggle")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Área de texto para la reseña
    review_text = st.text_area(
        "",
        placeholder="Escribe tu reseña aquí...",
        height=120,
        key="review_textarea",
        label_visibility="collapsed"
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
                    sentiment_label, sentiment_score = predict_text(stub, review_text.strip())
                except Exception as e:
                    st.warning(f"⚠ Análisis de sentimientos no disponible: {e}")
            
            # Guardar reseña
            save_review(review_text.strip(), recommend_friends, sentiment_label, sentiment_score)
            
            # Mostrar resultados
            st.success("✅ ¡Reseña enviada exitosamente!")
            
            # Mostrar resumen
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
                    sentiment_emoji = "😊" if sentiment_label.lower() == "positive" else "😔" if sentiment_label.lower() == "negative" else "😐"
                    st.info(f"*Sentimiento:* {sentiment_emoji} {sentiment_label} ({sentiment_score:.2f})")
                else:
                    st.info("*Sentimiento:* No disponible")
            
            st.markdown("*Tu reseña:*")
            st.markdown(f'"{review_text}"')
            
            # Mensaje de agradecimiento
            st.success("¡Gracias por tu feedback! 💙")
            
        else:
            st.error("⚠ Por favor, escribe una reseña antes de enviar.")

def ui_sentiment_analysis():
    """
    Interfaz para análisis individual de sentimientos.
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
            label, score = predict_text(stub, txt.strip())
            
            # Mostrar resultado con colores
            sentiment_emoji = "😊" if label.lower() == "positive" else "😔" if label.lower() == "negative" else "😐"
            sentiment_color = "success" if label.lower() == "positive" else "error" if label.lower() == "negative" else "info"
            
            if sentiment_color == "success":
                st.success(f"{sentiment_emoji} *Sentimiento:* {label} | *Confianza:* {score:.3f}")
            elif sentiment_color == "error":
                st.error(f"{sentiment_emoji} *Sentimiento:* {label} | *Confianza:* {score:.3f}")
            else:
                st.info(f"{sentiment_emoji} *Sentimiento:* {label} | *Confianza:* {score:.3f}")
                
        except Exception as e:
            st.error(f"❌ Error conectando con el servicio: {e}")

def ui_reviews_database():
    """
    Interfaz para ver todas las reseñas guardadas.
    """
    st.markdown('<h2 class="tab-subheader">📊 Base de Datos de Reseñas</h2>', unsafe_allow_html=True)
    
    if not st.session_state.reviews_db:
        st.info("📝 No hay reseñas guardadas aún.")
        return
    
    # Convertir a DataFrame
    df = pd.DataFrame(st.session_state.reviews_db)
    
    # Métricas generales
    st.markdown("📈 Estadísticas Generales:")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Reseñas", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        recommend_count = df['recomienda'].sum()
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Recomiendan", recommend_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if 'sentiment_label' in df.columns and df['sentiment_label'].notna().any():
            positive_count = (df['sentiment_label'] == 'positive').sum()
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Positivas", positive_count)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Sentimiento", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        avg_length = df['texto'].str.len().mean()
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Longitud Prom.", f"{avg_length:.0f} chars")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Mostrar tabla de reseñas
    st.markdown("---")
    st.markdown("📋 Todas las Reseñas:")
    
    # Preparar DataFrame para mostrar
    display_df = df.copy()
    display_df['recomienda'] = display_df['recomienda'].map({True: '✅ Sí', False: '❌ No'})
    if 'sentiment_label' in display_df.columns:
        display_df['sentiment_label'] = display_df['sentiment_label'].fillna('N/A')
        display_df['sentiment_score'] = display_df['sentiment_score'].fillna(0).round(3)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Botón para limpiar base de datos
    if st.button("🗑 Limpiar Base de Datos", type="secondary"):
        st.session_state.reviews_db = []
        st.success("✅ Base de datos limpiada.")
        st.experimental_rerun()

def main():
    """
    Función principal que construye la interfaz completa.
    """
    # Configuración de página
    st.set_page_config(
        page_title="Sistema de Reseñas con IA",
        page_icon="🍽",
        layout="wide"
    )
    
    # Aplicar CSS personalizado
    apply_custom_css()
    
    # Título principal
    st.markdown('<h1 class="main-title">🍽 Sistema de Reseñas de Restaurante con IA</h1>', unsafe_allow_html=True)
    
    # Verificar conexión gRPC si está disponible
    if GRPC_AVAILABLE:
        try:
            addr = os.getenv("APP_GRPC_ADDR", "localhost:50051")
            stub = make_stub(addr)
            status = ping(stub)
            st.toast(f"🤖 IA conectada: {status}", icon="✅")
        except Exception as e:
            st.toast(f"⚠ IA no disponible: {str(e)[:50]}...", icon="⚠")
    
    # Pestañas principales
    tab1, tab2, tab3 = st.tabs(["📝 Escribir Reseña", "🧠 Análisis IA", "📊 Base de Datos"])
    
    with tab1:
        ui_write_review()
    
    with tab2:
        ui_sentiment_analysis()
    
    with tab3:
        ui_reviews_database()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class="footer-text">
            <p>🍽 Sistema de Reseñas con Análisis de Sentimientos | Powered by BETO + gRPC 🤖</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()