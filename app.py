import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO

# ==================================================
# CONFIGURACIÓN GENERAL
# ==================================================
st.set_page_config(
    page_title="SVD-Gram",
    page_icon="📸",
    layout="wide"
)

MAX_MB = 10
MAX_BYTES = MAX_MB * 1024 * 1024
FORMATOS_ACEPTADOS = ["jpg", "jpeg", "png"]

# ==================================================
# ESTILOS PREMIUM PROFESIONAL
# ==================================================
st.markdown("""
<style>
/* Fondo general */
.stApp {
    background:
        radial-gradient(circle at top left, rgba(56,189,248,0.10), transparent 32%),
        radial-gradient(circle at top right, rgba(29,78,216,0.10), transparent 30%),
        linear-gradient(180deg, #020617 0%, #0B1120 45%, #0F172A 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
    border-right: 1px solid rgba(148,163,184,0.12);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F8FAFC;
}

/* Encabezado */
.hero {
    background:
        linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.92));
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 24px;
    padding: 26px 24px;
    margin-bottom: 28px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.30);
}

.hero-title {
    font-size: 48px;
    font-weight: 900;
    color: #F8FAFC;
    text-align: center;
    letter-spacing: -1px;
    margin-bottom: 6px;
}

.hero-subtitle {
    font-size: 18px;
    color: #CBD5E1;
    text-align: center;
    margin-bottom: 18px;
}

.hero-badges {
    text-align: center;
}

.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    margin: 4px;
    font-size: 12.5px;
    font-weight: 700;
    color: #BAE6FD;
    background: rgba(56,189,248,0.10);
    border: 1px solid rgba(56,189,248,0.28);
}

/* Títulos */
.section-title {
    font-size: 30px;
    font-weight: 900;
    color: #F8FAFC;
    margin-top: 22px;
    margin-bottom: 16px;
    letter-spacing: -0.4px;
}

/* Tarjetas métricas */
.metric-card {
    background:
        linear-gradient(180deg, rgba(15,23,42,0.98), rgba(17,24,39,0.95));
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 20px;
    padding: 20px;
    min-height: 120px;
    box-shadow: 0 14px 30px rgba(0,0,0,0.22);
    transition: transform 0.15s ease, border 0.15s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border: 1px solid rgba(56,189,248,0.35);
}

.metric-label {
    font-size: 14px;
    color: #94A3B8;
    font-weight: 700;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 31px;
    color: #F8FAFC;
    font-weight: 900;
    line-height: 1.1;
}

/* Tarjetas de imagen */
.image-card {
    background:
        linear-gradient(180deg, rgba(15,23,42,0.98), rgba(17,24,39,0.95));
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 22px;
    padding: 15px;
    box-shadow: 0 14px 30px rgba(0,0,0,0.22);
    margin-bottom: 14px;
}

.image-title {
    font-size: 18px;
    font-weight: 850;
    color: #F8FAFC;
    margin-bottom: 12px;
}

.image-caption {
    color: #94A3B8;
    font-size: 13px;
    margin-top: 10px;
    line-height: 1.5;
}

/* Mensajes de estado */
.success-box {
    background: linear-gradient(90deg, rgba(34,197,94,0.18), rgba(34,197,94,0.08));
    border: 1px solid rgba(34,197,94,0.34);
    color: #86EFAC;
    padding: 15px 18px;
    border-radius: 15px;
    font-weight: 800;
    margin-top: 16px;
    margin-bottom: 10px;
}

.warning-box {
    background: linear-gradient(90deg, rgba(245,158,11,0.16), rgba(245,158,11,0.08));
    border: 1px solid rgba(245,158,11,0.35);
    color: #FDE68A;
    padding: 15px 18px;
    border-radius: 15px;
    font-weight: 800;
    margin-top: 16px;
    margin-bottom: 10px;
}

.soft-note {
    color: #94A3B8;
    font-size: 14px;
    margin-top: 8px;
    margin-bottom: 20px;
}

/* Bienvenida */
.welcome-card {
    background:
        linear-gradient(180deg, rgba(15,23,42,0.98), rgba(17,24,39,0.95));
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 22px;
    padding: 24px;
    min-height: 165px;
    box-shadow: 0 14px 30px rgba(0,0,0,0.22);
}

.welcome-card h4 {
    color: #F8FAFC;
    font-size: 22px;
    margin-bottom: 10px;
}

.welcome-card p {
    color: #CBD5E1;
    font-size: 15px;
    line-height: 1.5;
}

/* Gráficos */
.graph-card {
    background:
        linear-gradient(180deg, rgba(15,23,42,0.98), rgba(17,24,39,0.95));
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 22px;
    padding: 16px;
    box-shadow: 0 14px 30px rgba(0,0,0,0.22);
}

/* Imágenes */
img {
    border-radius: 16px !important;
}

/* Botón descarga */
.stDownloadButton button {
    border-radius: 13px !important;
    font-weight: 800 !important;
    border: 1px solid rgba(56,189,248,0.35) !important;
    background: rgba(15,23,42,0.95) !important;
    color: #E0F2FE !important;
}

.stDownloadButton button:hover {
    border: 1px solid rgba(56,189,248,0.75) !important;
    color: #FFFFFF !important;
}

/* Uploader */
div[data-testid="stFileUploaderDropzone"] {
    border-radius: 16px !important;
    border: 1px dashed rgba(148,163,184,0.42) !important;
    background: rgba(2,6,23,0.45) !important;
}

/* Nota pequeña sidebar */
.sidebar-note {
    font-size: 13px;
    color: #94A3B8;
    line-height: 1.45;
    margin-top: -8px;
    margin-bottom: 18px;
}

/* Selects */
div[data-baseweb="select"] > div {
    background-color: rgba(2,6,23,0.65) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(148,163,184,0.20) !important;
}

div[data-baseweb="select"] > div:hover {
    border: 1px solid rgba(56,189,248,0.45) !important;
}

/* Sliders: celeste tecnológico */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #38BDF8 !important;
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 4px rgba(56,189,248,0.18) !important;
}

/* Radios */
.stRadio label {
    color: #E2E8F0 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(15,23,42,0.8) !important;
    border-radius: 12px !important;
    color: #F8FAFC !important;
}

/* Separación general */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# FUNCIONES
# ==================================================
def peso_kb(datos):
    if hasattr(datos, "getvalue"):
        datos = datos.getvalue()
    return len(datos) / 1024


def formatear_peso(kb):
    if kb >= 1024:
        return f"{kb / 1024:.2f} MB"
    return f"{kb:.1f} KB"


def leer_imagen_con_opencv(bytes_data):
    archivo_np = np.frombuffer(bytes_data, np.uint8)
    imagen_bgr = cv2.imdecode(archivo_np, cv2.IMREAD_COLOR)

    if imagen_bgr is None:
        return None

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    return imagen_rgb


def redimensionar(imagen_array, max_ancho=800):
    alto, ancho, _ = imagen_array.shape

    if ancho > max_ancho:
        escala = max_ancho / ancho
        nuevo_ancho = max_ancho
        nuevo_alto = int(alto * escala)

        imagen_array = cv2.resize(
            imagen_array,
            (nuevo_ancho, nuevo_alto),
            interpolation=cv2.INTER_AREA
        )

    return imagen_array


@st.cache_data(show_spinner=False)
def calcular_svd(imagen_array):
    imagen_float = imagen_array.astype(float)
    canales = []

    for i in range(3):
        canal = imagen_float[:, :, i]
        U, S, Vt = np.linalg.svd(canal, full_matrices=False)
        canales.append((U, S, Vt))

    return canales


def reconstruir_con_svd(canales_svd, k):
    canales_reconstruidos = []

    for U, S, Vt in canales_svd:
        canal_k = (U[:, :k] * S[:k]) @ Vt[:k, :]
        canales_reconstruidos.append(canal_k)

    imagen = np.stack(canales_reconstruidos, axis=2)
    imagen = np.clip(imagen, 0, 255)

    return imagen.astype(np.uint8)


def aplicar_sepia(imagen):
    matriz_sepia = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    imagen_sepia = imagen.astype(float) @ matriz_sepia.T
    imagen_sepia = np.clip(imagen_sepia, 0, 255)

    return imagen_sepia.astype(np.uint8)


def aplicar_grises(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    gris_rgb = cv2.cvtColor(gris, cv2.COLOR_GRAY2RGB)
    return gris_rgb


def aplicar_contraste_suave(imagen):
    imagen_float = imagen.astype(np.float32)
    imagen_float = imagen_float * 1.12
    imagen_float = np.clip(imagen_float, 0, 255)
    return imagen_float.astype(np.uint8)


def aplicar_vintage_calido(imagen):
    imagen_float = imagen.astype(np.float32)
    imagen_float[:, :, 0] *= 1.10
    imagen_float[:, :, 1] *= 1.03
    imagen_float[:, :, 2] *= 0.90
    imagen_float = np.clip(imagen_float, 0, 255)
    return imagen_float.astype(np.uint8)


def aplicar_filtro(imagen, filtro):
    if filtro == "Vintage Sepia":
        return aplicar_sepia(imagen)
    elif filtro == "Blanco y negro":
        return aplicar_grises(imagen)
    elif filtro == "Contraste suave":
        return aplicar_contraste_suave(imagen)
    elif filtro == "Vintage cálido":
        return aplicar_vintage_calido(imagen)
    else:
        return imagen


def calcular_calidad(canales_svd, k):
    porcentajes = []

    for _, S, _ in canales_svd:
        total = np.sum(S ** 2)
        parcial = np.sum(S[:k] ** 2)

        if total == 0:
            porcentajes.append(0)
        else:
            porcentajes.append((parcial / total) * 100)

    return np.mean(porcentajes)


def calcular_ahorro_teorico(alto, ancho, k):
    datos_originales = alto * ancho * 3
    datos_svd = k * (alto + ancho + 1) * 3

    ahorro = 100 * (1 - datos_svd / datos_originales)
    return ahorro


def crear_grafico_valores_singulares(canales_svd, k):
    S_rojo = canales_svd[0][1]
    S_verde = canales_svd[1][1]
    S_azul = canales_svd[2][1]

    longitud = min(len(S_rojo), len(S_verde), len(S_azul))
    promedio = (S_rojo[:longitud] + S_verde[:longitud] + S_azul[:longitud]) / 3

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, len(promedio) + 1)),
        y=promedio,
        mode="lines",
        name="Promedio RGB",
        line=dict(color="#38BDF8", width=3)
    ))

    fig.add_vline(
        x=k,
        line_dash="dash",
        line_color="rgba(255,255,255,0.55)",
        annotation_text=f"k = {k}",
        annotation_position="top"
    )

    fig.update_layout(
        title="Curva real de valores singulares",
        xaxis_title="Componentes SVD",
        yaxis_title="Valor singular",
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,0.65)",
        font=dict(color="white"),
        margin=dict(l=40, r=30, t=70, b=40)
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")

    return fig


def crear_grafico_pastel_compresion(ahorro_teorico):
    ahorro = max(0, min(100, ahorro_teorico))
    usado = 100 - ahorro

    fig = go.Figure(data=[
        go.Pie(
            labels=["Información usada", "Ahorro teórico SVD"],
            values=[usado, ahorro],
            hole=0.58,
            textinfo="label+percent",
            marker=dict(colors=["#7DD3FC", "#2563EB"]),
            pull=[0.02, 0.04]
        )
    ])

    fig.update_layout(
        title="Compresión matemática teórica",
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=70, b=20)
    )

    return fig


def imagen_a_descarga(imagen_array, formato, calidad):
    imagen_pil = Image.fromarray(imagen_array)
    buffer = BytesIO()

    if formato == "JPG":
        imagen_pil.save(
            buffer,
            format="JPEG",
            quality=calidad,
            optimize=True
        )
        mime = "image/jpeg"
        extension = "jpg"
    else:
        imagen_pil.save(
            buffer,
            format="PNG",
            optimize=True
        )
        mime = "image/png"
        extension = "png"

    buffer.seek(0)
    return buffer, mime, extension


def tarjeta_metrica(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# ENCABEZADO
# ==================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">📸 SVD-Gram</div>
    <div class="hero-subtitle">
        Compresión matemática de imágenes con SVD, filtros vintage y análisis visual interactivo
    </div>
    <div class="hero-badges">
        <span class="badge">Álgebra lineal aplicada</span>
        <span class="badge">Procesamiento digital de imágenes</span>
        <span class="badge">Python + Streamlit</span>
        <span class="badge">SVD + RGB</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# PANEL LATERAL
# ==================================================
st.sidebar.title("⚙️ Panel de control")

st.sidebar.markdown("### 1. Entrada de imagen")

entrada_tipo = st.sidebar.radio(
    "Selecciona una opción",
    ["Subir imagen", "Tomar foto con cámara"]
)

archivo = None

if entrada_tipo == "Subir imagen":
    archivo = st.sidebar.file_uploader(
        "Sube una imagen JPG, JPEG o PNG",
        type=FORMATOS_ACEPTADOS
    )
else:
    archivo = st.sidebar.camera_input("Tomar foto con cámara")

st.sidebar.markdown(
    f'<div class="sidebar-note">Formatos aceptados: JPG, JPEG y PNG · Máximo: {MAX_MB} MB</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown("### 2. Ajustes visuales")

filtro = st.sidebar.selectbox(
    "Filtro visual",
    [
        "Sin filtro",
        "Vintage Sepia",
        "Blanco y negro",
        "Contraste suave",
        "Vintage cálido"
    ]
)

formato_descarga = st.sidebar.selectbox(
    "Formato de descarga",
    ["JPG", "PNG"]
)

calidad_jpg = 85

if formato_descarga == "JPG":
    calidad_jpg = st.sidebar.slider(
        "Calidad JPG",
        min_value=50,
        max_value=100,
        value=85,
        step=5
    )

max_ancho = st.sidebar.slider(
    "Tamaño máximo de procesamiento",
    min_value=400,
    max_value=1000,
    value=800,
    step=100
)

# ==================================================
# PANTALLA INICIAL
# ==================================================
if archivo is None:
    st.markdown('<div class="section-title">👋 Bienvenido al panel de demostración</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div class="welcome-card">
            <h4>🧮 Compresión SVD</h4>
            <p>La imagen se reconstruye usando solo los componentes matemáticos más importantes.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="welcome-card">
            <h4>🎞️ Filtros visuales</h4>
            <p>Aplica efectos como Sepia, Blanco y negro, Contraste suave o Vintage cálido.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class="welcome-card">
            <h4>📊 Análisis interactivo</h4>
            <p>Observa métricas, gráfica de valores singulares y pastel de compresión.</p>
        </div>
        """, unsafe_allow_html=True)

# ==================================================
# PROCESAMIENTO PRINCIPAL
# ==================================================
else:
    bytes_originales = archivo.getvalue()

    if len(bytes_originales) > MAX_BYTES:
        st.error(f"La imagen supera el límite máximo de {MAX_MB} MB. Usa una imagen más liviana.")
        st.stop()

    imagen_original = leer_imagen_con_opencv(bytes_originales)

    if imagen_original is None:
        st.error("No se pudo leer la imagen. Usa un archivo JPG, JPEG o PNG válido.")
        st.stop()

    imagen_original = redimensionar(imagen_original, max_ancho)

    alto, ancho, _ = imagen_original.shape
    k_max = min(alto, ancho)

    st.sidebar.markdown("### 3. Compresión SVD")

    porcentaje = st.sidebar.slider(
        "Porcentaje de componentes SVD",
        min_value=5,
        max_value=100,
        value=45,
        step=5,
        help="Menor porcentaje = más compresión y menos nitidez. Mayor porcentaje = mejor calidad."
    )

    k = max(1, int((porcentaje / 100) * k_max))

    st.sidebar.caption(f"Componentes usados: k = {k} de {k_max}")
    st.sidebar.caption("k representa componentes SVD, no kilobytes.")

    with st.spinner("Procesando imagen con SVD..."):
        canales_svd = calcular_svd(imagen_original)
        imagen_comprimida = reconstruir_con_svd(canales_svd, k)
        imagen_final = aplicar_filtro(imagen_comprimida, filtro)

        imagen_gris_original = aplicar_grises(imagen_original)
        canales_svd_gris = calcular_svd(imagen_gris_original)
        imagen_gris_comprimida = reconstruir_con_svd(canales_svd_gris, k)

    calidad = calcular_calidad(canales_svd, k)
    calidad_gris = calcular_calidad(canales_svd_gris, k)
    ahorro_teorico = calcular_ahorro_teorico(alto, ancho, k)

    descarga, mime, extension = imagen_a_descarga(
        imagen_final,
        formato_descarga,
        calidad_jpg
    )

    peso_original = peso_kb(bytes_originales)
    peso_final = peso_kb(descarga)
    reduccion_real = 100 * (1 - peso_final / peso_original)

    # ==================================================
    # PANEL DE RESULTADOS
    # ==================================================
    st.markdown('<div class="section-title">📊 Panel de resultados</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        tarjeta_metrica("k usado", f"{k}")
    with c2:
        tarjeta_metrica("Calidad SVD", f"{calidad:.1f}%")
    with c3:
        tarjeta_metrica("Ahorro teórico", f"{ahorro_teorico:.1f}%")
    with c4:
        tarjeta_metrica("Peso original", formatear_peso(peso_original))
    with c5:
        tarjeta_metrica("Peso final", formatear_peso(peso_final))

    if reduccion_real >= 0:
        st.markdown(
            f'<div class="success-box">✅ Reducción real del archivo descargado: {reduccion_real:.1f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="warning-box">⚠️ El archivo final aumentó {abs(reduccion_real):.1f}%. Esto puede pasar con PNG o calidad JPG alta.</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="soft-note">El ahorro teórico corresponde a la compresión matemática por SVD. El peso final depende del formato de descarga y de la calidad seleccionada.</div>',
        unsafe_allow_html=True
    )

    # ==================================================
    # COMPARACIÓN VISUAL
    # ==================================================
    st.markdown('<div class="section-title">🖼️ Comparación visual</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="image-card"><div class="image-title">Imagen original</div>', unsafe_allow_html=True)
        st.image(imagen_original, use_container_width=True)
        st.markdown('<div class="image-caption">Imagen base tomada como referencia para la compresión.</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="image-card"><div class="image-title">Imagen comprimida / filtrada</div>', unsafe_allow_html=True)
        st.image(imagen_final, use_container_width=True)
        st.markdown('<div class="image-caption">Resultado generado con SVD y el filtro visual seleccionado.</div></div>', unsafe_allow_html=True)

        st.download_button(
            label=f"⬇️ Descargar imagen ({formatear_peso(peso_final)})",
            data=descarga,
            file_name=f"svd_gram_k{k}.{extension}",
            mime=mime
        )

    with col3:
        st.markdown('<div class="image-card"><div class="image-title">Escala de grises comprimida</div>', unsafe_allow_html=True)
        st.image(imagen_gris_comprimida, use_container_width=True)
        st.markdown(
            f'<div class="image-caption">Comparación sin color, trabajando solo con intensidad. Calidad SVD: {calidad_gris:.1f}%.</div></div>',
            unsafe_allow_html=True
        )

    # ==================================================
    # ANÁLISIS GRÁFICO
    # ==================================================
    st.markdown('<div class="section-title">📈 Análisis gráfico</div>', unsafe_allow_html=True)

    graf_col1, graf_col2 = st.columns(2)

    with graf_col1:
        fig_linea = crear_grafico_valores_singulares(canales_svd, k)
        st.plotly_chart(fig_linea, use_container_width=True)

    with graf_col2:
        fig_pastel = crear_grafico_pastel_compresion(ahorro_teorico)
        st.plotly_chart(fig_pastel, use_container_width=True)

    # ==================================================
    # EXPLICACIÓN
    # ==================================================
    with st.expander("📚 Explicación breve para la exposición"):
        st.write(
            "La aplicación toma una imagen, la divide en canales de color y aplica SVD para reconstruirla usando solo una parte de su información matemática."
        )
        st.write(
            "El valor k indica cuántos componentes se conservan. Si k es bajo, la imagen pierde nitidez pero aumenta la compresión matemática. Si k es alto, conserva más calidad."
        )
        st.write(
            "La gráfica de pastel muestra el ahorro teórico de la compresión SVD, mientras que la curva muestra cómo se distribuyen los valores singulares."
        )
        st.write(
            "La escala de grises comprimida permite comparar la imagen sin color, usando solo la intensidad visual."
        )
