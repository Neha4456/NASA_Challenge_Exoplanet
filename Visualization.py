import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("TOI.csv")

df = load_data()

# Sidebar navigation
st.sidebar.title("🔭 Exoplanet Explorer")
page = st.sidebar.radio(
    "Navigate:",
    ["Exoplanet Main Page", "Plots", "Popular Exoplanets 3D Models", "AR Experience"]
)

# -------------------------------
# 1) MAIN PAGE
# -------------------------------
if page == "Exoplanet Main Page":
    st.title("🌍 What are Exoplanets?")
    st.write("""
    **Exoplanets** are planets that orbit stars outside our Solar System.  
    Since the first discovery in the 1990s, astronomers have confirmed thousands of them.  

    ### Why do we hunt for exoplanets?
    - To understand how common planetary systems are in the universe  
    - To study different types of worlds, from hot Jupiters to rocky Earth-like planets  
    - To search for planets in the **habitable zone**, where liquid water might exist  
    - To explore the possibility of **life beyond Earth**  

    TESS (Transiting Exoplanet Survey Satellite) has discovered thousands of candidate exoplanets.  
    Let's explore them through data and 3D models! 🚀
    """)

    # st.image(
    #     "https://science.nasa.gov/wp-content/uploads/2023/09/PIA21472.jpg",
    #     caption="Artist's concept of exoplanets (NASA)",
    #     use_container_width=True
    # )

# -------------------------------
# 2) PLOTS
# -------------------------------
elif page == "Plots":
    st.title("📊 Explore Exoplanet Data")

    # dataset_option = st.selectbox(
    #     "Choose a dataset:",
    #     ["TOI (TESS Objects of Interest)", "KOI (Kepler Objects of Interest)", "K2OI(K2 Objects of Interest)"]
    # )
    # file_map = {
    #     "TOI (TESS Objects of Interest)": "TOI.csv",
    #     "KOI": "koi.csv",
    #     "K2OI": "k2oi.csv"
    # }
    # df = load_data(file_map[dataset_option])

    option = st.selectbox(
        "Choose a visualization:",
        (
            "Orbital Period Distribution",
            "Planet Radius Distribution",
            "Radius vs Orbital Period",
            "Radius vs Star Temperature",
            "Insolation vs Radius (Habitability)"
        )
    )

    # Orbital Period
    if option == "Orbital Period Distribution":
        st.subheader("📊 Orbital Period Distribution")
        st.write("Most exoplanets found by TESS orbit their stars in just a few days to weeks.")
        fig = px.histogram(df, x="pl_orbper", nbins=50,
                           labels={"pl_orbper": "Orbital Period (days)"},
                           title="Orbital Periods of TESS Planets")
        st.plotly_chart(fig, use_container_width=True)

    # Planet Radius
    elif option == "Planet Radius Distribution":
        st.subheader("🪐 Planet Size Distribution")
        st.write("Most exoplanets are between Earth and Neptune in size. Gas giants are less common.")
        fig = px.histogram(df, x="pl_rade", nbins=50,
                           labels={"pl_rade": "Planet Radius (Earth radii)"},
                           title="Distribution of Planet Sizes")
        st.plotly_chart(fig, use_container_width=True)

    # Radius vs Period
    elif option == "Radius vs Orbital Period":
        st.subheader("📉 Planet Radius vs Orbital Period")
        st.write("This reveals clusters like 'Hot Jupiters' — large planets with very short orbits.")
        fig = px.scatter(df, x="pl_orbper", y="pl_rade",
                         hover_data=["toi", "tfopwg_disp"],
                         labels={"pl_orbper": "Orbital Period (days)", "pl_rade": "Planet Radius (Earth radii)"},
                         log_x=True, log_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # Radius vs Star Temperature
    elif option == "Radius vs Star Temperature":
        st.subheader("🌟 Planet Size vs Star Temperature")
        st.write("Shows how planet sizes vary with their host star's temperature.")
        fig = px.scatter(df, x="st_teff", y="pl_rade",
                         hover_data=["toi"],
                         labels={"st_teff": "Stellar Temperature (K)", "pl_rade": "Planet Radius (Earth radii)"},
                         title="Planet Radius vs Stellar Temperature")
        st.plotly_chart(fig, use_container_width=True)

    # Insolation vs Radius
    elif option == "Insolation vs Radius (Habitability)":
        st.subheader("🌍 Habitability: Insolation vs Planet Size")
        st.write("Planets in the habitable zone receive similar starlight to Earth — key for potential life.")
        fig = px.scatter(df, x="pl_insol", y="pl_rade",
                         hover_data=["toi"],
                         labels={"pl_insol": "Insolation (Earth flux)", "pl_rade": "Planet Radius (Earth radii)"},
                         log_x=True, log_y=True,
                         title="Insolation vs Planet Radius")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 3) 3D MODELS
# -------------------------------
elif page == "Popular Exoplanets 3D Models":
    st.title("🪐 Popular Exoplanets in 3D")
    st.write("Rotate and zoom in on these interactive models of famous exoplanets.")

    col1, col2 = st.columns(2)

    # Kepler-186f
    kepler186f_html = """
    <div class="sketchfab-embed-wrapper">
    <iframe title="Kepler-186f" frameborder="0" allowfullscreen 
    mozallowfullscreen="true" webkitallowfullscreen="true" 
    allow="autoplay; fullscreen; xr-spatial-tracking" 
    xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered 
    web-share src="https://sketchfab.com/models/c484b8b4aa9248b6998b6222d62f5a77/embed" 
    width="100%" height="400">
    </iframe>
    <p style="font-size: 13px; font-weight: normal; margin: 5px; color: #4A4A4A;">
        <a href="https://sketchfab.com/3d-models/kepler-186f-c484b8b4aa9248b6998b6222d62f5a77" target="_blank">Kepler-186f</a> 
        by <a href="https://sketchfab.com/uperesito" target="_blank">uperesito</a> on 
        <a href="https://sketchfab.com" target="_blank">Sketchfab</a>
    </p>
    </div>
    """


    kepler22b_html = """
    <div class="sketchfab-embed-wrapper"> 
    <iframe title="Kepler 22b" frameborder="0" allowfullscreen 
    mozallowfullscreen="true" webkitallowfullscreen="true" 
    allow="autoplay; fullscreen; xr-spatial-tracking" 
    xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered 
    web-share src="https://sketchfab.com/models/3589154676b7465c815a4aa1d8c4354a/embed"
    width="100%" height="400"> 
    </iframe> 
    <p style="font-size: 13px; font-weight: normal; margin: 5px; color: #4A4A4A;"> 
    <a href="https://sketchfab.com/3d-models/kepler-22b-3589154676b7465c815a4aa1d8c4354a?utm_medium=embed&utm_campaign=share-popup&utm_content=3589154676b7465c815a4aa1d8c4354a" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;"> Kepler 22b </a> 
    by <a href="https://sketchfab.com/per.rb1?utm_medium=embed&utm_campaign=share-popup&utm_content=3589154676b7465c815a4aa1d8c4354a" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;"> per.rb1 </a> on 
    <a href="https://sketchfab.com?utm_medium=embed&utm_campaign=share-popup&utm_content=3589154676b7465c815a4aa1d8c4354a" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;">Sketchfab</a>
    </p></div>
    """

    # Display in columns
    with col1:
        st.subheader("Kepler-186f")
        components.html(kepler186f_html, height=400)

    with col2:
        st.subheader("Kepler-22b")
        components.html(kepler22b_html, height=400)

    col3, col4 = st.columns(2)
    kepler452b_html = """
    <div class="sketchfab-embed-wrapper"> <iframe title="Kepler - 452b Planet" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/8310c4ebc8c642feaba50996911f80e9/embed" width="100%" height="400"> </iframe> <p style="font-size: 13px; font-weight: normal; margin: 5px; color: #4A4A4A;"> <a href="https://sketchfab.com/3d-models/kepler-452b-planet-8310c4ebc8c642feaba50996911f80e9?utm_medium=embed&utm_campaign=share-popup&utm_content=8310c4ebc8c642feaba50996911f80e9" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;"> Kepler - 452b Planet </a> by <a href="https://sketchfab.com/ahnaf.yasintx?utm_medium=embed&utm_campaign=share-popup&utm_content=8310c4ebc8c642feaba50996911f80e9" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;"> Ahnaf Yasin </a> on <a href="https://sketchfab.com?utm_medium=embed&utm_campaign=share-popup&utm_content=8310c4ebc8c642feaba50996911f80e9" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;">Sketchfab</a></p></div>
    """


    canceri_e_html = """
    <div class="sketchfab-embed-wrapper"> <iframe title="Pixel Planet 55 Cancri e" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/e4e4e7a6a77743b69e41d1c32ddc932e/embed" width="100%" height="400"> </iframe> <p style="font-size: 13px; font-weight: normal; margin: 5px; color: #4A4A4A;"> <a href="https://sketchfab.com/3d-models/pixel-planet-55-cancri-e-e4e4e7a6a77743b69e41d1c32ddc932e?utm_medium=embed&utm_campaign=share-popup&utm_content=e4e4e7a6a77743b69e41d1c32ddc932e" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;"> Pixel Planet 55 Cancri e </a> by <a href="https://sketchfab.com/AstroJar?utm_medium=embed&utm_campaign=share-popup&utm_content=e4e4e7a6a77743b69e41d1c32ddc932e" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;"> AstroJar </a> on <a href="https://sketchfab.com?utm_medium=embed&utm_campaign=share-popup&utm_content=e4e4e7a6a77743b69e41d1c32ddc932e" target="_blank" rel="nofollow" style="font-weight: bold; color: #1CAAD9;">Sketchfab</a></p></div>
    """

    # Display in columns
    with col1:
        st.subheader("Kepler-452b")
        components.html(kepler452b_html, height=400)

    with col2:
        st.subheader("55 Canceri e")
        components.html(canceri_e_html, height=400)

# -------------------------------
# 4) AR View
# -------------------------------
elif page == "AR Experience":
    st.title("🌌 AR Exoplanet Experience")
    st.write("👉 Point your camera at the exoplanet marker images below to see them in Augmented Reality!")

    st.image("Hiro-marker.png", caption="Exoplanet Marker")

    # Embed your AR.js app hosted on GitHub Pages
    st.markdown("""
        <a href="https://neha4456.github.io/NASA_Challenge_AR/" target="_blank">
            <button style="padding:10px 20px; font-size:16px;">Open Web AR app on Phone</button>
        </a>
    """, unsafe_allow_html=True)
