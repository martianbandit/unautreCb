import os
import time
import streamlit as st
import google.generativeai as genai


##############################################
# Configuration et fonctions utilitaires
##############################################

# Configuration de l'API Gemini
if "GEMINI_API_KEY" not in os.environ:
    st.error("Veuillez définir la variable d'environnement GEMINI_API_KEY avant de lancer l'application.")
    st.stop()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

def wait_for_files_active(files):
    """Wait until all files are ACTIVE."""
    st.write("En attente du traitement des fichiers…")
    for file in files:
        name = file.name
        f = genai.get_file(name)
        while f.state.name == "PROCESSING":
            time.sleep(5)
            f = genai.get_file(name)
        if f.state.name != "ACTIVE":
            raise Exception(f"Le fichier {f.name} n'a pas pu être activé.")
    st.write("Tous les fichiers sont actifs et prêts à l'emploi.")
    return files

# Configuration du modèle
generation_config = {
  "temperature": .45,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8160,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  tools = [
    genai.protos.Tool(
      function_declarations = [],
    ),
  ],
)

##############################################
# Interface Streamlit
##############################################

st.title("Application interactive avec Gemini")

st.write("Cette application permet de :")
st.write("- Uploader des fichiers vidéo vers l'API Gemini.")
st.write("- Attendre leur activation.")
st.write("- Interagir avec un modèle de langage afin de poser des questions sur le contenu vidéo.")
st.write("Le snippet original est répliqué avec une interface utilisateur plus conviviale.")

# Zone d’upload de fichiers
uploaded_files = st.file_uploader(
    "Chargez un ou plusieurs fichiers (vidéo) :",
    type=["webm", "mov", "mp4"],
    accept_multiple_files=True
)

if "uploaded_to_gemini" not in st.session_state:
    st.session_state["uploaded_to_gemini"] = False

if "gemini_files" not in st.session_state:
    st.session_state["gemini_files"] = []

if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = None

# Bouton pour envoyer les fichiers à Gemini
if st.button("Envoyer et traiter les fichiers sur Gemini") and uploaded_files:
    # Sauvegarde temporaire des fichiers
    local_paths = []
    for uf in uploaded_files:
        # Enregistrer le fichier localement
        with open(uf.name, "wb") as f:
            f.write(uf.read())
        local_paths.append(uf.name)
    
    # Upload sur Gemini
    gemini_files = []
    for path in local_paths:
        # Déduire le type MIME de base
        if path.endswith(".webm"):
            mime = "video/webm"
        elif path.endswith(".mov"):
            mime = "video/quicktime"
        elif path.endswith(".mp4"):
            mime = "video/mp4"
        else:
            mime = None
        f = upload_to_gemini(path, mime_type=mime)
        gemini_files.append(f)
        st.write(f"Fichier '{path}' uploadé : {f.uri}")

    # Attente de l'activation des fichiers
    active_files = wait_for_files_active(gemini_files)
    st.session_state["gemini_files"] = active_files
    st.session_state["uploaded_to_gemini"] = True

# Une fois les fichiers traités, on peut initialiser la session de chat
if st.session_state["uploaded_to_gemini"] and st.session_state["chat_session"] is None:
    # Créez une session de chat semblable à l'exemple
    # Ajoutons un historique par défaut basé sur l'exemple
    # Ici on suppose que le premier fichier est utilisé dans le contexte
    files = st.session_state["gemini_files"]
    
    # Historique par défaut comme dans le snippet
    history = [
        {
          "role": "user",
          "parts": [files[0]] if len(files) > 0 else [],
        },
        {
          "role": "user",
          "parts": [
              "Using this video, please answer the below questions:\n\n1. Who is the main subject of the video?\n2. List out how Ashley navigated her trip.\n3. At what timestamp did musicians appear playing instruments?",
          ],
        },
        {
          "role": "model",
          "parts": [
              "Here are the answers based on the video:\n\n1. The main subject of the video is Ashley, a woman going on a trip to Barcelona.\n\n\n2. Ashley navigated her trip using the Google Trips app.  This allowed her to:\n    * Access her itinerary offline, including flight and hotel information.\n    * View a map of her saved places.\n    * Create a personalized itinerary for each day, including travel time to each location.\n    * Find a restaurant recommended by a friend.\n\n\n3. Musicians playing instruments appear at timestamp 0:33 and again at 1:33.",
          ],
        },
    ]

    # Ajout du second fichier et des messages utilisateurs/modèle liés
    if len(files) > 1:
        history.extend([
            {
              "role": "user",
              "parts": [files[1]],
            },
            {
              "role": "user",
              "parts": [
                "Raconte se qui se passe dans le video",
              ],
            },
            {
              "role": "model",
              "parts": [
                "Bien sûr, voici une description du contenu de la vidéo :\n\nLa vidéo montre un gros plan d’une roue de camion sale et rouillée. La roue est blanche avec des taches de rouille. Le pneu est noir et semble usé. Le sol sous la roue est également sale et de couleur brun foncé.",
              ],
            },
            {
              "role": "user",
              "parts": [
                "Esce que quelque choses semble dangereux?"
              ],
            },
            {
              "role": "model",
              "parts": [
                "Oui, il y a quelques éléments dangereux potentiels dans la vidéo :\n\n* **Rouille :** ...\n* **Usure des pneus :** ...\n"
              ],
            },
        ])

    st.session_state["chat_session"] = model.start_chat(history=history)
    st.success("Session de chat initialisée avec l'historique!")

##############################################
# Interaction avec le modèle
##############################################

if st.session_state["chat_session"] is not None:
    user_input = st.text_input("Entrez votre message pour le modèle :")
    if st.button("Envoyer"):
        if user_input.strip():
            response = st.session_state["chat_session"].send_message(user_input)
            st.write("**Réponse du modèle :**")
            st.write(response.text)
        else:
            st.warning("Veuillez entrer une question ou un message.")