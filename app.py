import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
from io import BytesIO
import plotly.graph_objects as go

# ==================================================
# CONFIGURACIÓN GENERAL
# ==================================================
st.set_page_config(
    page_title="SVD-Gram",
    page_icon="📸",
    layout="wide"
)

# ==================================================
# ESTILOS VISUALES
# ==================================================
st.markdown("""
<style>
.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
    margin-bottom: 0px;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cfcfcf;
    margin-bottom: 25px;
}
.card {
    background-color: #1f222b;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #343846;
    margin-bottom: 15px;
}
.help-box {
    background-color: #242936;
    padding: 15px;
    border-radius: 14px;
    border-left: 6px solid #f5c542;
}
.small-text {
    font-size: 14px;
    color: #cfcfcf;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# FUNCIONES
# ==================================================
def convertir_a_kb(bytes_data):
    return len(bytes_data) / 1024


def redimensionar_imagen(imagen, max_ancho):
    ancho, alto = imagen.size

    if ancho > max_ancho:
        nuevo_alto = int((max_ancho / ancho) * alto)
        imagen = imagen.resize((max_ancho, nuevo_alto))

    return imagen


@st.cache_data(show_spinner=False)
def calcular_svd(imagen_array):
    canales = []

    for i in range(3):
        canal = imagen_array[:, :, i]
        U, S, Vt = np.linalg.svd(canal, full_matrices=False)
        canales.append((U, S, Vt))

    return canales


def comprimir_imagen(canales_svd, k):
    canales_reconstruidos = []

    for U, S, Vt in canales_svd:
        canal_k = (U[:, :k] * S[:k]) @ Vt[:k, :]
        canales_reconstruidos.append(canal_k)

    imagen_comprimida = np.stack(canales_reconstruidos, axis=2)
    imagen_comprimida = np.clip(imagen_comprimida, 0, 255)

    return imagen_comprimida.astype(np.uint8)


def aplicar_sepia(imagen_array):
    matriz_sepia = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    imagen_sepia = imagen_array @ matriz_sepia.T
    imagen_sepia = np.clip(imagen_sepia, 0, 255)

    return imagen_sepia.astype(np.uint8)


def aplicar_grises(imagen_array):
    gris = np.mean(imagen_array, axis=2)
    imagen_gris = np.stack([gris, gris, gris], axis=2)

    return imagen_gris.astype(np.uint8)


def aplicar_contraste_suave(imagen_array):
    imagen = Image.fromarray(imagen_array)
    imagen = ImageEnhance.Contrast(imagen).enhance(1.25)
    imagen = ImageEnhance.Sharpness(imagen).enhance(1.15)

    return np.array(imagen)


def aplicar_vintage_calido(imagen_array):
    imagen = imagen_array.astype(np.float32)

    imagen[:, :, 0] *= 1.12
    imagen[:, :, 1] *= 1.03
    imagen[:, :, 2] *= 0.88

    imagen = np.clip(imagen, 0, 255)

    return imagen.astype(np.uint8)


def aplicar_filtro(imagen_array, filtro):
    if filtro == "Vintage Sepia":
        return aplicar_sepia(imagen_array)
    elif filtro == "Blanco y negro":
        return aplicar_grises(imagen_array)
    elif filtro == "Contraste suave":
        return aplicar_contraste_suave(imagen_array)
    elif filtro == "Vintage cálido":
        return aplicar_vintage_calido(imagen_array)
    else:
        return imagen_array


def guardar_para_descarga(imagen_array, formato, calidad_jpg):
    imagen = Image.fromarray(imagen_array)
    buffer = BytesIO()

    if formato == "PNG":
        imagen.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
        extension = "png"
    else:
        imagen.save(
            buffer,
            format="JPEG",
            quality=calidad_jpg,
            optimize=True
        )
        mime = "image/jpeg"
        extension = "jpg"

    buffer.seek(0)

    return buffer, mime, extension


def calcular_ahorro_teorico(alto, ancho, k):
    datos_originales = alto * ancho * 3
    datos_comprimidos = k * (alto + ancho + 1) * 3

    ahorro = 100 * (1 - datos_comprimidos / datos_originales)

    return ahorro


def calcular_calidad_svd(canales_svd, k):
    calidades = []

    for U, S, Vt in canales_svd:
        energia_total = np.sum(S ** 2)
        energia_k = np.sum(S[:k] ** 2)

        if energia_total == 0:
            calidad = 0
        else:
            calidad = (energia_k / energia_total) * 100

        calidades.append(calidad)

    return np.mean(calidades)


def obtener_valores_singulares(canales_svd, canal_grafico):
    S_rojo = canales_svd[0][1]
    S_verde = canales_svd[1][1]
    S_azul = canales_svd[2][1]

    if canal_grafico == "Rojo":
        return S_rojo
    elif canal_grafico == "Verde":
        return S_verde
    elif canal_grafico == "Azul":
        return S_azul
    else:
        return (S_rojo + S_verde + S_azul) / 3


def crear_grafico(canales_svd, k, canal_grafico, escala_log):
    S = obtener_valores_singulares(canales_svd, canal_grafico)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, len(S) + 1)),
        y=S,
        mode="lines",
        name=f"Canal: {canal_grafico}"
    ))

    fig.add_vline(
        x=k,
        line_dash="dash",
        annotation_text=f"k = {k}",
        annotation_position="top"
    )

    fig.update_layout(
        title="Curva real de valores singulares",
        xaxis_title="Componentes SVD",
        yaxis_title="Valor singular",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30)
    )

    if escala_log:
        fig.update_yaxes(type="log")

    return fig


def nombre_archivo_seguro(filtro):
    return filtro.lower().replace(" ", "_").replace("á", "a").replace("é", "e")


# ==================================================
# TÍTULO
# ==================================================
st.markdown(
    '<div class="main-title">📸 SVD-Gram: Compresión Matemática y Filtros Vintage</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Aplicación interactiva para comprimir imágenes usando SVD y aplicar filtros mediante transformaciones de color.</div>',
    unsafe_allow_html=True
)

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("⚙️ Controles")

modo_presentacion = st.sidebar.toggle(
    "Modo presentación",
    value=False,
    help="Muestra la app de forma más limpia para exponer en el stand."
)

st.sidebar.markdown("### 1. Entrada de imagen")

tipo_entrada = st.sidebar.radio(
    "Elige una opción",
    ["Subir imagen", "Tomar foto con cámara"]
)

entrada = None
bytes_originales = None

if tipo_entrada == "Subir imagen":
    entrada = st.sidebar.file_uploader(
        "Sube una imagen",
        type=["jpg", "jpeg", "png"]
    )

    if entrada is not None:
        bytes_originales = entrada.getvalue()

else:
    entrada = st.sidebar.camera_input("Tomar foto")

    if entrada is not None:
        bytes_originales = entrada.getvalue()

st.sidebar.markdown("### 2. Configuración")

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
        min_value=40,
        max_value=100,
        value=85,
        step=5,
        help="Menor calidad JPG reduce más el peso del archivo."
    )

max_ancho = st.sidebar.slider(
    "Tamaño de procesamiento",
    min_value=300,
    max_value=1000,
    value=600,
    step=100
)

canal_grafico = st.sidebar.selectbox(
    "Canal para el gráfico",
    ["Promedio RGB", "Rojo", "Verde", "Azul"]
)

escala_log = st.sidebar.checkbox(
    "Usar escala logarítmica en gráfico",
    value=False
)

# ==================================================
# PANTALLA INICIAL
# ==================================================
if entrada is None:
    st.markdown("""
    <div class="help-box">
        <h3>👋 Bienvenido a SVD-Gram</h3>
        <p>Sube una imagen o toma una foto con la cámara para comenzar.</p>
        <p>Después podrás mover el valor <b>k</b>, aplicar filtros, ver métricas y descargar la imagen procesada.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div class="card">
            <h4>🧮 Compresión SVD</h4>
            <p>Reduce información de la imagen usando componentes matemáticos importantes.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card">
            <h4>🎞️ Filtros vintage</h4>
            <p>Transforma los colores RGB para crear efectos visuales antiguos.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class="card">
            <h4>📊 Visualización</h4>
            <p>Muestra métricas, ahorro de espacio y curva de valores singulares.</p>
        </div>
        """, unsafe_allow_html=True)

# ==================================================
# PROCESAMIENTO
# ==================================================
else:
    imagen = Image.open(BytesIO(bytes_originales)).convert("RGB")
    imagen = redimensionar_imagen(imagen, max_ancho)

    imagen_array = np.array(imagen).astype(float)

    alto, ancho, _ = imagen_array.shape
    k_max = min(alto, ancho)
    k_recomendado = max(1, int(k_max * 0.15))

    st.sidebar.markdown("### 3. Compresión SVD")

    k = st.sidebar.slider(
        "Componentes SVD usados: k",
        min_value=1,
        max_value=k_max,
        value=k_recomendado,
        help="Menor k = más compresión y menos nitidez. Mayor k = más calidad y menos compresión."
    )

    st.sidebar.caption(
        f"Recomendado para demostrar: k ≈ {k_recomendado}. "
        "Recuerda: k no es KB, es cantidad de componentes SVD."
    )

    with st.spinner("Aplicando SVD y procesando imagen..."):
        canales_svd = calcular_svd(imagen_array)
        imagen_comprimida = comprimir_imagen(canales_svd, k)
        imagen_final = aplicar_filtro(imagen_comprimida, filtro)

    archivo_descarga, mime, extension = guardar_para_descarga(
        imagen_final,
        formato_descarga,
        calidad_jpg
    )

    peso_original_kb = convertir_a_kb(bytes_originales)
    peso_final_kb = convertir_a_kb(archivo_descarga.getvalue())

    reduccion_real = 100 * (1 - peso_final_kb / peso_original_kb)
    ahorro_teorico = calcular_ahorro_teorico(alto, ancho, k)
    calidad_svd = calcular_calidad_svd(canales_svd, k)

    # ==================================================
    # MÉTRICAS
    # ==================================================
    st.markdown("## 📊 Panel de resultados")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("k usado", k)
    col2.metric("Información conservada", f"{calidad_svd:.1f}%")
    col3.metric("Ahorro teórico SVD", f"{ahorro_teorico:.1f}%")
    col4.metric("Peso original", f"{peso_original_kb:.1f} KB")
    col5.metric("Peso final", f"{peso_final_kb:.1f} KB")

    if reduccion_real >= 0:
        st.success(f"Reducción real del archivo descargado: {reduccion_real:.1f}%")
    else:
        st.warning(
            f"El archivo final aumentó {abs(reduccion_real):.1f}%. "
            "Esto puede pasar con PNG o imágenes pequeñas."
        )

    st.progress(min(max(calidad_svd / 100, 0), 1))

    st.caption(
        "Nota: el valor k controla los componentes matemáticos usados para reconstruir la imagen. "
        "El peso final depende también del formato de descarga JPG o PNG."
    )

    st.write("")

    # ==================================================
    # IMÁGENES
    # ==================================================
    st.markdown("## 🖼️ Comparación visual")

    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown("### Imagen original")
        st.image(imagen, use_container_width=True)

    with img_col2:
        st.markdown("### Imagen procesada")
        st.image(imagen_final, use_container_width=True)

        st.download_button(
            label=f"⬇️ Descargar imagen procesada ({peso_final_kb:.1f} KB)",
            data=archivo_descarga,
            file_name=f"svd_gram_k{k}_{nombre_archivo_seguro(filtro)}.{extension}",
            mime=mime
        )

    st.write("")

    # ==================================================
    # GRÁFICO
    # ==================================================
    st.markdown("## 📈 Curva de valores singulares")

    fig = crear_grafico(
        canales_svd,
        k,
        canal_grafico,
        escala_log
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==================================================
    # EXPLICACIÓN PARA EL JURADO
    # ==================================================
    if not modo_presentacion:
        with st.expander("📚 Explicación para el jurado"):
            st.write(
                "El programa toma una imagen digital y la separa en tres canales: rojo, verde y azul. "
                "Cada canal se trata como una matriz numérica."
            )

            st.write(
                "Luego se aplica SVD, que permite descomponer cada matriz en componentes. "
                "El valor k indica cuántos componentes se conservan para reconstruir la imagen."
            )

            st.write(
                "Cuando k es bajo, la imagen ocupa menos información matemática, pero pierde nitidez. "
                "Cuando k es alto, la imagen se parece más a la original, pero se comprime menos."
            )

            st.write(
                "Los filtros vintage se aplican modificando matemáticamente los colores RGB. "
                "Por eso el proyecto no solo comprime imágenes, sino que también demuestra cómo una matriz puede transformar visualmente una fotografía."
            )

        with st.expander("🧪 Consejos para probar en la exposición"):
            st.write("1. Sube una imagen clara, con rostro u objetos visibles.")
            st.write("2. Baja k a valores pequeños como 5, 10 o 15 para mostrar pérdida de nitidez.")
            st.write("3. Sube k a 50 o más para mostrar cómo mejora la calidad.")
            st.write("4. Activa el filtro Vintage Sepia para explicar la transformación RGB.")
            st.write("5. Descarga la imagen para demostrar que el programa genera un resultado final.")