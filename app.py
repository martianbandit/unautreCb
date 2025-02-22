import os
import streamlit as st
from textwrap import dedent
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain.llms import Ollama

# Charger les variables d'environnement
load_dotenv()
MODEL = os.getenv('MODEL', 'mistral')

# Initialisation du modèle LLM
llm = Ollama(model=MODEL)

# Définition des outils
search_tool = SerperDevTool()

# Création des agents
def create_agents():
    return {
        "Lead Market Analyst": Agent(
            role="Lead Market Analyst",
            goal="Analyser les produits et concurrents pour fournir des insights marketing.",
            backstory="Expert en analyse de marché, vous scrutez les tendances et identifiez les opportunités.",
            tools=[search_tool],
            llm=llm,
            verbose=True
        ),
        "Chief Marketing Strategist": Agent(
            role="Chief Marketing Strategist",
            goal="Synthétiser les analyses pour formuler des stratégies marketing innovantes.",
            backstory="Stratège marketing reconnu, vous créez des campagnes percutantes.",
            tools=[search_tool],
            llm=llm,
            verbose=True
        ),
        "Creative Content Creator": Agent(
            role="Creative Content Creator",
            goal="Créer du contenu engageant pour les réseaux sociaux.",
            backstory="Spécialiste en storytelling digital, vous transformez les idées en visuels attractifs.",
            tools=[search_tool],
            llm=llm,
            verbose=True
        ),
        "Senior Photographer": Agent(
            role="Senior Photographer",
            goal="Capturer des images percutantes pour les publicités Instagram.",
            backstory="Photographe chevronné, vous donnez vie aux produits grâce à vos clichés.",
            tools=[search_tool],
            llm=llm,
            verbose=True
        ),
        "Chief Creative Director": Agent(
            role="Chief Creative Director",
            goal="Superviser et approuver le contenu créé par l'équipe marketing.",
            backstory="Directeur artistique expérimenté, vous assurez la cohérence et la qualité des productions.",
            tools=[search_tool],
            llm=llm,
            verbose=True
        ),
    }

# Création des tâches
def create_tasks(agents):
    return {
        "Competitor Analysis": Task(
            description="Réaliser une analyse approfondie des concurrents pour identifier les tendances et opportunités.",
            expected_output="Un rapport structuré des forces et faiblesses des concurrents.",
            tools=[search_tool],
            agent=agents["Lead Market Analyst"]
        ),
        "Strategy Planning": Task(
            description="Élaborer une stratégie marketing basée sur l'analyse concurrentielle.",
            expected_output="Un document avec des recommandations stratégiques détaillées.",
            tools=[search_tool],
            agent=agents["Chief Marketing Strategist"]
        ),
        "Content Creation": Task(
            description="Produire du contenu créatif et engageant pour les réseaux sociaux.",
            expected_output="Un ensemble de visuels et textes adaptés aux campagnes Instagram.",
            tools=[search_tool],
            agent=agents["Creative Content Creator"]
        ),
        "Photography": Task(
            description="Prendre des photos percutantes alignées avec la stratégie marketing.",
            expected_output="Une série d'images professionnelles prêtes à être utilisées dans les publicités.",
            tools=[search_tool],
            agent=agents["Senior Photographer"]
        ),
        "Review & Approval": Task(
            description="Vérifier et valider les productions de l'équipe avant publication.",
            expected_output="Un feedback détaillé et une version finale approuvée du contenu.",
            tools=[search_tool],
            agent=agents["Chief Creative Director"]
        ),
    }

# Interface Streamlit
st.title("🚀 Marketing CrewAI avec Streamlit")

if st.button("Lancer l'Analyse Marketing"):
    st.write("🔍 Initialisation des agents et des tâches...")

    # Instanciation des agents et des tâches
    agents = create_agents()
    tasks = create_tasks(agents)

    # Création du Crew
    crew = Crew(
        agents=list(agents.values()),
        tasks=list(tasks.values()),
        process=Process.sequential
    )

    st.write("🛠️ Exécution du workflow...")
    
    # Exécution du Crew avec un sujet générique
    result = crew.kickoff(inputs={'topic': 'Marketing Digital'})

    # Affichage du résultat
    st.subheader("📊 Résultats de l'analyse marketing :")
    st.write(result)

st.write("Appuyez sur le bouton ci-dessus pour lancer l'analyse marketing complète.")