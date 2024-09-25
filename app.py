import os
import gradio as gr
import requests
from crewai import Agent, Task, Crew, Process

from langchain_openai import ChatOpenAI

from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from crewai_tools import tool, SeleniumScrapingTool, ScrapeWebsiteTool
from duckduckgo_search import DDGS

from newspaper import Article

# S'assurer que les variables d'environnement essentielles sont définies
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY n'est pas définie dans les variables d'environnement")

def fetch_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print("ERREUR: " + str(e))
        return f"Erreur lors de la récupération du contenu : {e}"

# Définir l'outil de recherche DuckDuckGo
@tool('DuckDuckGoSearchResults')
def search_results(search_query: str) -> dict:
    """
    Effectue une recherche web pour rassembler et retourner une collection de résultats de recherche.
    Cet outil automatise la récupération d'informations web liées à une requête spécifiée.
    Args:
    - search_query (str): La chaîne de requête qui spécifie l'information à rechercher sur le web. Cela devrait être une expression claire et concise des besoins d'information de l'utilisateur.
    Retourne:
    - list: Une liste de dictionnaires, où chaque dictionnaire représente un résultat de recherche. Chaque dictionnaire inclut un 'extrait' de la page et le 'lien' avec l'url correspondante.
    """
    results = DDGS().text(search_query, max_results=5, timelimit='m')
    results_list = [{"title": result['title'], "snippet": result['body'], "link": result['href']} for result in results]
    return results_list

@tool('WebScrapper')
def web_scrapper(url: str, topic: str) -> str:
    """
    Un outil conçu pour extraire et lire le contenu d'un lien spécifié et générer un résumé sur un sujet spécifique.
    Il est capable de traiter divers types de pages web en faisant des requêtes HTTP et en analysant le contenu HTML reçu.
    Cet outil est particulièrement utile pour les tâches de web scraping, la collecte de données, ou l'extraction d'informations spécifiques de sites web.
    
    Args:
    - url (str): L'URL à partir de laquelle scraper le contenu.
    - topic (str): Le sujet spécifique sur lequel générer un résumé.
    Retourne:
    - summary (str): résumé de l'url sur le sujet
    """
    # Scraper le contenu de l'URL spécifiée
    content = fetch_content(url)
    
    # Préparer le prompt pour générer le résumé
    prompt = f"Générez un résumé du contenu suivant sur le sujet ## {topic} ### \n\nCONTENU:\n\n" + content
    
    # Générer le résumé en utilisant OpenAI (GPT-3.5-turbo pour cet exemple)
    openai_client = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4)
    response = openai_client.predict(prompt)

    summary_response = f"""###
    Résumé:
    {response}
    
    URL: {url}
    ###
    """
    
    return summary_response

def kickoff_crew(topic: str, model_choice: str) -> str:
    try:
        # Initialiser les modèles de langage en fonction du choix de l'utilisateur
        llm = ChatOpenAI(temperature=0, model_name=model_choice)
    
        # Définir les Agents avec le LLM OpenAI
        researcher = Agent(
            role='Chercheur',
            goal='Rechercher et collecter des informations détaillées sur le sujet ## {topic} ##',
            tools=[search_results, web_scrapper],
            llm=llm,
            backstory=(
                "Vous êtes un chercheur méticuleux, doué pour naviguer dans de vastes quantités d'informations afin d'extraire des insights essentiels sur n'importe quel sujet donné. "
                "Votre dévouement au détail assure la fiabilité et l'exhaustivité de vos découvertes. "
                "Avec une approche stratégique, vous analysez et documentez soigneusement les données, visant à fournir des résultats précis et dignes de confiance."
            ),
            allow_delegation=False,
            max_iter=15,
            max_rpm=20,
            memory=True,
            verbose=True
        )

        editor = Agent(
            role='Éditeur',
            goal='Compiler et affiner les informations en un rapport complet sur le sujet ## {topic} ##',
            llm=llm,
            backstory=(
                "En tant qu'éditeur expert, vous vous spécialisez dans la transformation de données brutes en rapports clairs et engageants. "
                "Votre forte maîtrise du langage et votre attention aux détails garantissent que chaque rapport non seulement transmet des insights essentiels "
                "mais est également facilement compréhensible et attrayant pour divers publics. "
            ),
            allow_delegation=False,
            max_iter=5,
            max_rpm=15,
            memory=True,
            verbose=True
        )
        
        # Définir les Tâches
        research_task = Task(
            description=(
                "Utilisez l'outil DuckDuckGoSearchResults pour collecter des extraits de recherche initiaux sur ## {topic} ##. "
                "Si des recherches plus détaillées sont nécessaires, générez et exécutez de nouvelles requêtes liées à ## {topic} ##. "
                "Ensuite, utilisez l'outil WebScrapper pour approfondir les URL importantes identifiées à partir des extraits, en extrayant plus d'informations et d'insights. "
                "Compilez ces découvertes en un brouillon préliminaire, en documentant toutes les sources pertinentes, les titres et les liens associés au sujet. "
                "Assurez une haute précision tout au long du processus et évitez toute fabrication ou représentation erronée de l'information."
            ),
            expected_output=(
                "Un rapport brouillon structuré sur le sujet, comportant une introduction, un corps principal détaillé organisé selon différents aspects du sujet, et une conclusion. "
                "Chaque section doit citer correctement les sources, fournissant un aperçu complet des informations recueillies."
            ),
            agent=researcher
        )

        
        edit_task = Task(
            description=(
                "Examinez et affinez le rapport brouillon initial de la tâche de recherche. Organisez le contenu de manière logique pour améliorer le flux d'information. "
                "Vérifiez l'exactitude de toutes les données, corrigez les divergences et mettez à jour les informations pour vous assurer qu'elles reflètent les connaissances actuelles et sont bien étayées par des sources. "
                "Améliorez la lisibilité du rapport en renforçant la clarté du langage, en ajustant les structures des phrases et en maintenant un ton cohérent. "
                "Incluez une section listant toutes les sources utilisées, formatée en points suivant ce modèle : "
                "- titre : url'."
            ),
            expected_output=(
                "Un rapport poli et complet sur le sujet ## {topic} ##, avec un récit clair et professionnel qui reflète fidèlement les résultats de la recherche. "
                "Le rapport doit inclure une introduction, une section de discussion approfondie, une conclusion concise et une liste de sources bien organisée. "
                "Assurez-vous que le document est grammaticalement correct et prêt pour la publication ou la présentation."
            ),
            agent=editor,
            context=[research_task]
        )
    
        # Former l'équipe
        crew = Crew(
            agents=[researcher, editor],
            tasks=[research_task, edit_task],
            process=Process.sequential,
        )
    
        # Lancer le processus de recherche
        result = crew.kickoff(inputs={'topic': topic})
        if not isinstance(result, str):
            result = str(result)
        return result
    except Exception as e:
        return f"Erreur : {str(e)}"


def main():
    """Configurer l'interface Gradio pour l'outil de recherche CrewAI."""
    with gr.Blocks() as demo:
        gr.Markdown("## Outil de Recherche CrewAI")
        topic_input = gr.Textbox(label="Entrez le sujet", placeholder="Tapez ici...")
        model_choice = gr.Radio(choices=["gpt-3.5-turbo", "gpt-4"], label="Choisissez le modèle")
        submit_button = gr.Button("Démarrer la recherche")
        output = gr.Markdown(label="Résultat")

        submit_button.click(
            fn=kickoff_crew,
            inputs=[topic_input, model_choice],
            outputs=output
        )

    demo.launch()

if __name__ == "__main__":
    main()
