import openai
import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()
# Votre clé API OpenAI (remplacez-la par la vôtre)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ID des assistants (Remplacez par vos vrais IDs)
assistant_ids = {
    "GPTBay": "asst_jkf2zv9jlzYj4prJ7dHECOWL",
    "specialiste_du_vrac": "asst_HSUZYtNegNGOfXRC3p7NqInX"
}

# Fonction pour interroger l'API OpenAI en utilisant l'ID de l'assistant
def chat_with_assistant(input_text, assistant, temp, contrast, image):
    assistant_id = assistant_ids[assistant]  # Récupérer l'ID de l'assistant sélectionné
    
    # Créer le prompt en fonction de l'assistant sélectionné
    prompt = f"Utilisateur: {input_text}\nAssistant ({assistant_id}):"
    
    # Appel à l'API OpenAI
    try:
        response = openai.Completion.create(
            engine="GPT-4o-mini",  # Ou GPT-4 si vous avez accès
            prompt=prompt,
            temperature=temp,
            max_tokens=1500,
            n=1,
            stop=["Utilisateur:", "Assistant:"]
        )
        response_text = response.choices[0].text.strip()

        # Réponse personnalisée avec icône
        response_text = f"{'🤖' if assistant == 'OBD2 Diagnostic' else '👨‍💻'} {response_text}"
        
        if image:
            response_text += f"\nVous avez téléchargé une image : {image.name}."
            
        return response_text
    
    except Exception as e:
        return f"Erreur avec l'API OpenAI: {str(e)}"

# Liste des assistants disponibles
assistants = ["Assistant SAAQ", "OBD2 Diagnostic", "Expert Mécanique", "Analyse d'Accident"]

# Interface utilisateur avec Gradio
with gr.Blocks() as interface:
    # Titre, sous-titre et image
    with gr.Row():
        gr.Markdown("# 🧰 Mes assistants Mécaniciens 🔧\n### Des assistants spécialisés dans plusieurs facettes du domaine de la Mécanique.")
        gr.Image("/workspaces/unautreCb/cti43y3h.png", elem_id="header_img", interactive=False)
    
    # Layout de la barre latérale et du chat
    with gr.Row():
        with gr.Column():
            assistant_choice = gr.Radio(label="Sélectionnez un Assistant:", choices=assistants, value=assistants[0])
            file_upload = gr.File(label="Télécharger une image ou fichier:")
            temperature = gr.Slider(label="Température du modèle", minimum=0, maximum=1, value=0.7, step=0.1)
            contrast_mode = gr.Radio(label="Mode Contraste de l'image:", choices=["Clair", "Sombre"], value="Clair")
        
        with gr.Column():
            chat_window = gr.Chatbot(label="Chat", height=400)
            user_input = gr.Textbox(label="Votre message ici", placeholder="Tapez votre question...")

            # Bouton d'envoi
            send_button = gr.Button("Envoyer")
            
            # Annonce sous le chat
            gr.Markdown("Visitez [GPTsIndex](http://www.gpts-index.com) pour d’autres applications IA!!")
            gr.image("/workspaces/unautreCb/cti43y3h.png")
    
    # Action du bouton envoyer
    send_button.click(chat_with_assistant, inputs=[user_input, assistant_choice, temperature, contrast_mode, file_upload], outputs=chat_window)

# Lancer l'interface Gradio
interface.launch()
