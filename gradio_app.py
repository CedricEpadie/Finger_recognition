import gradio as gr
import torch
import json, os
import uuid
import numpy as np

from model import SiameseNetwork
from PIL import Image
from utils import preprocess_image, compute_embedding, transform

MODEL_PATH = "siamese_model.pth"
DB_PATH = "embeddings/users.json"

# Charger le modèle
model = SiameseNetwork()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Créer la base si elle n'existe pas
if not os.path.exists(DB_PATH):
    with open(DB_PATH, 'w') as f:
        json.dump([], f)

def save_user(name, matricule, email, image_np):
    image_pil = Image.fromarray(image_np).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0)

    with torch.no_grad():
        embedding = model.forward_once(image_tensor).squeeze(0).numpy().tolist()

    user_data = {
        "name": name,
        "matricule": matricule,
        "email": email,
        "embedding": embedding
    }

    # Vérifier si le fichier existe et n'est pas vide
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
        with open(DB_PATH, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(user_data)

    with open(DB_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    return f"Utilisateur {name} enregistré avec succès."

def recognize(image_np):
    """
    Prend une image NumPy (chargée depuis Gradio), applique le prétraitement,
    extrait l'embedding et le compare aux embeddings en base.
    """
    # 1. Prétraitement avec transforms (Grayscale, Resize, ToTensor)
    image_pil = Image.fromarray(image_np).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0)  # shape: (1, 1, 100, 100)

    # 2. Calcul de l'embedding
    with torch.no_grad():
        input_emb = model.forward_once(image_tensor).squeeze(0).numpy()

    # 3. Chargement de la base d'utilisateurs
    with open(DB_PATH, 'r') as f:
        users = json.load(f)

    results = []
    for user in users:
        db_emb = np.array(user["embedding"])
        dist = np.linalg.norm(input_emb - db_emb, ord=2)
        results.append((user["name"], user["matricule"], user["email"], round(dist, 4)))

    if not results:
        return "Aucun utilisateur enregistré."

    results.sort(key=lambda x: x[3])  # trie par distance
    best = results[0]
    if best[3] < 0.5:
        return f"✅ Empreinte reconnue : {best[0]} (Matricule : {best[1]}, Email : {best[2]}) - Distance : {best[3]}"
    else:
        return f"❌ Aucune correspondance trouvée. Distance minimale : {best[3]}"


def list_users():
    with open(DB_PATH, 'r') as f:
        users = json.load(f)
    return "\n".join([f"{u['name']} | {u['matricule']} | {u['email']}" for u in users]) or "Aucun utilisateur"

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🔐 Reconnaissance d'Empreintes par Réseau Siamois")

    with gr.Tab("Ajouter un utilisateur"):
        name = gr.Text(label="Nom")
        matricule = gr.Text(label="Matricule")
        email = gr.Text(label="Email")
        image = gr.Image(label="Empreinte")
        submit = gr.Button("Enregistrer")
        output = gr.Textbox(label="Résultat")
        submit.click(fn=save_user, inputs=[name, matricule, email, image], outputs=output)

    with gr.Tab("Tester une empreinte"):
        test_image = gr.Image(label="Image test")
        recognize_btn = gr.Button("Reconnaître")
        recog_result = gr.Textbox(label="Résultat")
        recognize_btn.click(fn=recognize, inputs=[test_image], outputs=recog_result)

    with gr.Tab("Utilisateurs enregistrés"):
        user_list = gr.Textbox(label="Base d'utilisateurs", lines=10)
        list_btn = gr.Button("Afficher")
        list_btn.click(fn=list_users, outputs=user_list)

demo.launch()
