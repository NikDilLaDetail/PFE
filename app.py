from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Charger le modèle et le tokenizer fine-tuné
model_path = "./legal_bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Dictionnaire pour mapper les classes aux messages explicites
class_mapping = {
    0: "Le document est une assignation et il y a un vice de procédure.",
    1: "Le document est une assignation sans vice de procédure.",
    2: "Le document est une notification et il y a un vice de procédure.",
    3: "Le document est une notification sans vice de procédure."
}


# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(file_stream):
    reader = PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# Fonction pour effectuer la prédiction
def model_predict(text):
    if not text.strip():
        return "Le document est vide ou illisible."

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).item()

    return class_mapping.get(predictions, "Classe inconnue")


# Initialiser Flask
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    pdf_file = request.files.get('pdf_file')  # Récupérer le fichier PDF

    if pdf_file:
        # Extraire le texte directement depuis le fichier en mémoire
        text = extract_text_from_pdf(pdf_file.stream)
        prediction = model_predict(text)
        return render_template("index.html", prediction=prediction)
    else:
        return render_template("index.html", prediction="Aucun fichier PDF fourni.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
