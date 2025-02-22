import streamlit as st
from crewai import Crew, Agent, Task
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
from fpdf import FPDF
from crewai.tools import tool

# ---------- 🔹 1. Définition des outils 🔹 ---------- #

@tool("Google Trends Analysis")
def google_trends_analysis(keyword: str, geo: str = "US", timeframe: str = "today 3-m") -> dict:
    """Analyse les tendances Google Trends pour un mot-clé donné."""
    pytrends = TrendReq()
    pytrends.build_payload([keyword], geo=geo, timeframe=timeframe)

    trends_data = pytrends.interest_over_time()
    if not trends_data.empty:
        return trends_data[keyword].to_dict()
    else:
        return {"error": "Aucune donnée trouvée"}

@tool("Data Visualization")
def visualize_trends(trends_data: dict) -> str:
    """Génère un graphique et le retourne encodé en base64."""
    if "error" in trends_data:
        return trends_data["error"]

    df = pd.DataFrame.from_dict(trends_data, orient='index', columns=['Interest'])
    df.index = pd.to_datetime(df.index)

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Interest"], marker='o', linestyle='-', color='b')
    plt.title("Tendances Google Trends")
    plt.xlabel("Date")
    plt.ylabel("Intérêt")
    plt.xticks(rotation=45)
    plt.grid()

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png")
    plt.close()
    
    return base64.b64encode(img_buffer.getvalue()).decode("utf-8")

@tool("PDF Report Generation")
def generate_pdf_report(trends_data: dict, graph_base64: str) -> str:
    """Crée un rapport PDF avec les tendances et l'image du graphique."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Rapport de Tendances Google Trends", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(200, 10, "Analyse des tendances récentes:", ln=True)
    
    for date, interest in trends_data.items():
        pdf.cell(200, 10, f"{date}: {interest}", ln=True)

    if graph_base64:
        image_data = base64.b64decode(graph_base64)
        image_path = "temp_graph.png"
        with open(image_path, "wb") as f:
            f.write(image_data)
        pdf.image(image_path, x=10, y=pdf.get_y(), w=180)

    pdf_path = "trend_report.pdf"
    pdf.output(pdf_path)
    return pdf_path

# ---------- 🔹 2. Création des Agents CrewAI 🔹 ---------- #

trend_research_agent = Agent(
    role="Analyste des tendances",
    goal="Extraire les tendances des recherches Google.",
    tools=[google_trends_analysis],
    memory=True, verbose=True, max_iter=3
)

data_analyst_agent = Agent(
    role="Analyste de données",
    goal="Créer des visualisations des tendances.",
    tools=[visualize_trends],
    memory=True, verbose=True, max_iter=2
)

report_writer_agent = Agent(
    role="Rédacteur de rapports",
    goal="Générer un rapport détaillé.",
    tools=[generate_pdf_report],
    memory=True, verbose=True, max_iter=2
)

marketing_strategist_agent = Agent(
    role="Stratégiste Marketing",
    goal="Déduire des recommandations commerciales.",
    memory=True, verbose=True, max_iter=2
)

# ---------- 🔹 3. Définition des tâches CrewAI 🔹 ---------- #

extract_trends_task = Task(
    description="Extraire les tendances Google Trends pour un mot-clé.",
    expected_output="Données des tendances.",
    agent=trend_research_agent,
    async_execution=True
)

generate_visuals_task = Task(
    description="Créer un graphique des tendances.",
    expected_output="Image graphique.",
    agent=data_analyst_agent,
    context=[extract_trends_task]
)

write_report_task = Task(
    description="Rédiger un rapport détaillé basé sur les tendances et analyses visuelles.",
    expected_output="Rapport PDF.",
    agent=report_writer_agent,
    context=[generate_visuals_task]
)

generate_strategy_task = Task(
    description="Proposer des recommandations marketing.",
    expected_output="Stratégie commerciale.",
    agent=marketing_strategist_agent,
    context=[write_report_task]
)

# ---------- 🔹 4. Création du CrewAI 🔹 ---------- #

trend_analysis_crew = Crew(
    agents=[trend_research_agent, data_analyst_agent, report_writer_agent, marketing_strategist_agent],
    tasks=[extract_trends_task, generate_visuals_task, write_report_task, generate_strategy_task],
    verbose=True
)

# ---------- 🔹 5. Interface Streamlit 🔹 ---------- #

st.title("🔍 Analyse des Tendances Google Trends")
keyword = st.text_input("Entrez un mot-clé pour analyser la tendance :")

if st.button("Lancer l'analyse"):
    if keyword:
        # Exécuter CrewAI
        result = trend_analysis_crew.kickoff()
        
        # Récupération des résultats
        trends_data = google_trends_analysis(keyword)
        graph_base64 = visualize_trends(trends_data)
        pdf_path = generate_pdf_report(trends_data, graph_base64)

        # Affichage des résultats
        st.subheader("📊 Graphique des tendances")
        if "error" not in graph_base64:
            img_data = base64.b64decode(graph_base64)
            st.image(img_data, caption="Tendances Google Trends", use_column_width=True)
        else:
            st.warning("Aucune donnée trouvée pour ce mot-clé.")

        st.subheader("📄 Rapport de tendances")
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        st.download_button(label="📥 Télécharger le rapport PDF", data=pdf_bytes, file_name="trend_report.pdf")

        st.subheader("🎯 Recommandations stratégiques")
        st.write(result)

    else:
        st.warning("Veuillez entrer un mot-clé.")