from PIL import Image
import openai
import gradio as gr
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ID des assistants (Remplacez par vos vrais IDs)
assistant_ids = {
    "GPTBay": "asst_jkf2zv9jlzYj4prJ7dHECOWL",
    "specialiste_du_vrac": "asst_HSUZYtNegNGOfXRC3p7NqInX",
    "Scraping and Crawling Expert Code (RAG)": "asst_Lt1AWleH509FJYTVu0sy9AUE",
    "Hybrid Designer": "asst_5l3HvMS8AoKRahvrhph1Z5L0",
    "Math tuthor": "asst_9vmMiolEozJTNjQN4LBBM3AX",
    "SaaS starter": "asst_Lt1AWleH509FJYTVu0sy9AUE"
}

# Fonction pour interroger l'API OpenAI en utilisant l'ID de l'assistant
def chat_with_assistant(input_text, assistant, temp, file_upload, chat_history):

    if assistant not in assistant_ids:
        return [("Erreur", f"L'assistant '{assistant}' n'est pas valide.")]

    assistant_id = assistant_ids[assistant]  # Récupérer l'ID de l'assistant sélectionné
    
    # Préparer l'historique des messages pour l'API ChatCompletion
    messages = [{"role": "system", "content": f"Tu es un assistant nommé {assistant} spécialisé dans le domaine de la mécanique."}]
    
    # Ajouter l'historique des messages existant
    if chat_history is not None:
        for user_message, assistant_message in chat_history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_message})
    
    # Ajouter le message utilisateur actuel
    messages.append({"role": "user", "content": input_text})
    
    # Appel à l'API OpenAI
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Ou gpt-3.5-turbo si vous avez accès
            messages=messages,
            temperature=temp,
            max_tokens=1500,
        )
        response_text = response.choices[0].message["content"].strip()

        # Réponse personnalisée avec icône
        response_text = f"{'🤖' if assistant == 'OBD2 Diagnostic' else '👨‍💻'} {response_text}"
        
        if file_upload:
            response_text += f"\nVous avez téléchargé une image : {file_upload.name}."
        
        chat_history.append((input_text, response_text))
        return chat_history
    
    except Exception as e:
        return [("Erreur", f"Erreur avec l'API OpenAI: {str(e)}")]

# Liste des assistants disponibles
assistants = ["GPTBay", "specialiste_du_vrac", "Scraping and Crawling Expert Code (RAG)", "Hybrid Designer", "SaaS starter", "Math tuthor"]

# Interface utilisateur avec Gradio
with gr.Blocks() as interface:
    # Titre, sous-titre et image
    with gr.Row():
        gr.Markdown("# 🧰 Mes assistants Mécaniciens 🔧\n### Des assistants spécialisés dans plusieurs facettes du domaine de la Mécanique.")

    # Layout de la barre latérale et du chat
    with gr.Row():
        with gr.Column():
            contrast_mode = gr.Radio(label="Mode Contraste de l'application:", choices=["Clair", "Sombre"], value="Clair")
            assistant_choice = gr.Radio(label="Sélectionnez un Assistant:", choices=assistants, value=assistants[0])
            file_upload = gr.File(label="Télécharger une image ou fichier:", height=150)
            temperature = gr.Slider(label="Température du modèle", minimum=0, maximum=1, value=0.7, step=0.1)

        with gr.Column():
            chat_window = gr.Chatbot(label="Chat", height=400)
            user_input = gr.Textbox(label="Votre message ici", placeholder="Tapez votre question...")

            # Bouton d'envoi
            send_button = gr.Button("Envoyer")

            # Annonce sous le chat
            gr.Markdown("### Visitez [🤖GPTsIndex🤖](http://www.gpts-index.com) pour d’autres applications IA!!")

    # Action du bouton envoyer
    send_button.click(
        chat_with_assistant,
        inputs=[user_input, assistant_choice, temperature, file_upload, chat_window],
        outputs=chat_window
    )

# Lancer l'interface Gradio
interface.launch()


