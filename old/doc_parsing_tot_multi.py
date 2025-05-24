import os
import requests
import re
from PIL import Image
from io import BytesIO
from mistralai import Mistral
import datauri
from mistralai.models import OCRResponse

# Clé API Mistral (à sécuriser)
key_Mistral = "ThSJGAwoQibJjTiNJq1YjLdVyGvtmpeM"
client = Mistral(api_key=key_Mistral)

path = "C:/Users/menar/Documents/projet_certifIA_banque"
path_output = "C:/Users/menar/Documents/projet_certifIA_banque/output"
# Charger le fichier PDF

annees = ["2020", "2021", "2022", "2023", "2024", "2025"]

for annee in annees:
    input_file = f"RESF_{annee}.pdf"
    input_dir = path + '/RESF' 
    input_path = os.path.join(input_dir, input_file)

    uploaded_pdf = client.files.upload(
        file={
            "file_name": input_path,
            "content": open(input_path, "rb"),
        },
        purpose="ocr"
    )

    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

    # Effectuer l'OCR
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )

    ocr_text_with_pages = ""
    for page_number, page in enumerate(ocr_response.pages, start=1):
        if page.markdown:
            ocr_text_with_pages += f"[{page_number}]\n{page.markdown}\n"

    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = os.path.join(path_output, "textes_extraits")
    os.makedirs(output_dir, exist_ok=True)

    file_name = f"RESF_{annee}" + ".txt"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(ocr_text_with_pages)