import os
import time
import torch
import httpx
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from mistralai import Mistral
import datauri  
from mistralai.models import OCRResponse
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
from langchain.schema import Document as LangDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from docx import Document as WordDocument
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from credentials.key import key_Mistral, HF_TOKEN
from parametres.params import parent_dir, input_dir, output_dir, exemple_dir, notes_generees_dir, database_dir, RESF_dir, docs_dir, tables_dir, txt_dir

# ============================ Initialisation des clés API ============================
os.environ["HF_TOKEN"] = HF_TOKEN
login(os.environ["HF_TOKEN"])

# ============================ Initialisation des clients ============================
client = Mistral(api_key=key_Mistral)
try:
    model_chat = ChatMistralAI(model="mistral-large-latest", api_key=key_Mistral)
except Exception as e:
    print(f"Erreur lors de l'initialisation du modèle Mistral: {e}")
    exit()

# ============================ Modèle d'embeddings local ============================
model = AutoModel.from_pretrained("BAAI/bge-m3")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# ============================ Fonctions utilitaires ============================
def invoke_with_retry(model, prompt, max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            return model.invoke(prompt)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = 2 ** retries
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                raise e
    raise Exception("Max retries reached, request failed.")

# ============================ Prompt Template pour Q&A ============================
QA_PROMPT_TEMPLATE = """
En tant qu'expert en analyse de finances publiques, réponds à la question suivante en te basant uniquement sur le contexte fourni du Rapport Économique, Social et Financier (RESF) :

Question : {question}

Contexte du RESF :
{context}

Instructions :
1. Réponds de manière précise et factuelle
2. Cite les chiffres et données spécifiques du document
3. Utilise un langage professionnel mais accessible
4. Si la réponse n'est pas dans le contexte, indique-le clairement
5. Structure ta réponse de manière claire avec des paragraphes si nécessaire
6. Réponds en français

Réponse :
"""
qa_prompt_template = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)

# ============================ Définition des chemins ============================
path_input = os.path.join(parent_dir, input_dir)
path_resf = os.path.join(path_input, RESF_dir)
path_output = os.path.join(parent_dir, output_dir)
path_exemple = os.path.join(path_input, exemple_dir)
path_notes_generees = os.path.join(path_output, notes_generees_dir)
path_notes_generees_docs = os.path.join(path_notes_generees, docs_dir)
path_notes_generees_tables = os.path.join(path_notes_generees, tables_dir)
path_notes_generees_txt = os.path.join(path_notes_generees, txt_dir)
path_database = os.path.join(path_output, database_dir)

# Création des répertoires
os.makedirs(path_input, exist_ok=True)
os.makedirs(path_output, exist_ok=True)
os.makedirs(path_exemple, exist_ok=True)
os.makedirs(path_notes_generees, exist_ok=True)
os.makedirs(path_notes_generees_docs, exist_ok=True)
os.makedirs(path_notes_generees_tables, exist_ok=True)
os.makedirs(path_database, exist_ok=True)
os.makedirs(path_notes_generees_txt, exist_ok=True)
os.makedirs(path_resf, exist_ok=True)

# ============================ Fonction de traitement OCR ============================
def process_ocr(annee):
    input_file = f"RESF_{annee}.pdf"
    input_path = os.path.join(path_resf, input_file)
    
    if not os.path.exists(input_path):
        print(f"⚠️ Fichier {input_file} non trouvé dans {input_path}")
        return None

    print(f"📄 Traitement du fichier {input_file}...")
    
    try:
        uploaded_pdf = client.files.upload(
            file={
                "file_name": input_path,
                "content": open(input_path, "rb"),
            },
            purpose="ocr"
        )

        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

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

        file_name = f"RESF_{annee}.txt"
        file_path = os.path.join(path_notes_generees_txt, file_name)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(ocr_text_with_pages)
        
        print(f"✅ OCR terminé pour {input_file}")
        return file_path
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement OCR de {input_file}: {str(e)}")
        return None

# ============================ Fonction de préparation de la base de données ============================
def prepare_database(annee, input_path):
    """Prépare la base de données vectorielle pour une année donnée."""
    print(f"🔍 Préparation de la base de données pour RESF {annee}...")
    
    # Lecture du contenu du fichier
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    document = LangDocument(page_content=content, metadata={"source": input_path, "annee": annee})
    
    # Découpage du texte en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents([document]) 
    
    # Création de la base de données vectorielle
    chroma_path = os.path.join(path_database, f"db_RESF_{annee}")
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=key_Mistral)

    if not os.path.exists(chroma_path):
        db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_path)
        print(f"✅ Base de données créée pour RESF {annee}")
    else:
        db_chroma = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        print(f"✅ Base de données existante chargée pour RESF {annee}")
    
    return db_chroma

# ============================ Fonction de réponse aux questions ============================
def answer_question(db_chroma, question, annee):
    """Répond à une question en utilisant la base de données vectorielle."""
    print(f"🤔 Recherche de réponse pour : {question}")
    
    # Recherche des passages pertinents
    docs_chroma = db_chroma.similarity_search_with_score(question, k=5)
    context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

    # Génération de la réponse
    prompt = qa_prompt_template.format(question=question, context=context_text)
    response = invoke_with_retry(model_chat, prompt)
    
    return response.content

# ============================ Fonction d'affichage du menu ============================
def display_menu():
    """Affiche le menu principal."""
    print("\n" + "="*60)
    print("📊 SYSTÈME DE QUESTIONS-RÉPONSES SUR LES RAPPORTS RESF")
    print("="*60)
    print("1. Choisir une année et poser une question")
    print("2. Voir les années disponibles")
    print("3. Quitter")
    print("="*60)

def display_available_years():
    """Affiche les années disponibles avec leurs statuts."""
    print("\n📅 ANNÉES DISPONIBLES :")
    print("-" * 40)
    
    annees = ["2020", "2021", "2022", "2023", "2024", "2025"]
    available_years = []
    
    for annee in annees:
        # Vérifier si le fichier PDF existe
        pdf_path = os.path.join(path_resf, f"RESF_{annee}.pdf")
        pdf_exists = os.path.exists(pdf_path)
        
        # Vérifier si la base de données existe
        db_path = os.path.join(path_database, f"db_RESF_{annee}")
        db_exists = os.path.exists(db_path)
        
        status = ""
        if pdf_exists and db_exists:
            status = "✅ Prêt"
            available_years.append(annee)
        elif pdf_exists:
            status = "📄 PDF disponible (base à créer)"
            available_years.append(annee)
        else:
            status = "❌ Non disponible"
        
        print(f"RESF {annee} : {status}")
    
    return available_years

# ============================ Fonction de sélection d'année ============================
def select_year(available_years):
    """Permet à l'utilisateur de sélectionner une année."""
    if not available_years:
        print("❌ Aucune année disponible. Veuillez d'abord traiter les documents PDF.")
        return None
    
    print(f"\n📋 Années disponibles : {', '.join(available_years)}")
    
    while True:
        try:
            annee = input("\n🎯 Entrez l'année souhaitée (ex: 2024) : ").strip()
            if annee in available_years:
                return annee
            else:
                print(f"❌ L'année {annee} n'est pas disponible. Veuillez choisir parmi : {', '.join(available_years)}")
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir !")
            exit()
        except Exception as e:
            print(f"❌ Erreur : {e}")

# ============================ Fonction de traitement d'une année ============================
def process_year(annee):
    """Traite une année complète (OCR + base de données)."""
    print(f"\n🔄 Traitement de l'année {annee}...")
    
    # Étape 1: OCR
    input_path = process_ocr(annee)
    if input_path is None:
        return None
        
    # Étape 2: Préparation de la base de données
    db_chroma = prepare_database(annee, input_path)
    
    return db_chroma

# ============================ Fonction de session de questions ============================
def question_session(annee, db_chroma):
    """Gère une session de questions pour une année donnée."""
    print(f"\n💬 Session de questions pour RESF {annee}")
    print("💡 Tapez 'quit' pour revenir au menu principal")
    print("💡 Tapez 'help' pour des exemples de questions")
    print("-" * 50)
    
    example_questions = [
        "Quel est le déficit public prévu pour cette année ?",
        "Quelles sont les principales hypothèses de croissance ?",
        "Comment évolue la dette publique ?",
        "Quelles sont les principales mesures fiscales ?",
        "Quel est l'impact de la conjoncture économique sur les finances publiques ?"
    ]
    
    while True:
        try:
            question = input(f"\n❓ Question sur RESF {annee} : ").strip()
            
            if question.lower() == 'quit':
                print("👋 Retour au menu principal...")
                break
            elif question.lower() == 'help':
                print("\n📝 Exemples de questions que vous pouvez poser :")
                for i, example in enumerate(example_questions, 1):
                    print(f"{i}. {example}")
                continue
            elif not question:
                print("❌ Veuillez entrer une question.")
                continue
            
            print("\n🔄 Recherche en cours...")
            response = answer_question(db_chroma, question, annee)
            
            print(f"\n📋 Réponse :")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir !")
            exit()
        except Exception as e:
            print(f"❌ Erreur lors de la génération de la réponse : {e}")

# ============================ Programme principal ============================
def main():
    print("🚀 Démarrage du système de Q&A sur les Rapports RESF...")
    
    while True:
        try:
            display_menu()
            choice = input("\n🎯 Votre choix (1-3) : ").strip()
            
            if choice == "1":
                # Afficher les années disponibles
                available_years = display_available_years()
                
                # Sélectionner une année
                selected_year = select_year(available_years)
                if selected_year is None:
                    continue
                
                # Vérifier si la base de données existe
                db_path = os.path.join(path_database, f"db_RESF_{selected_year}")
                if not os.path.exists(db_path):
                    print(f"\n🔄 La base de données pour {selected_year} n'existe pas. Création en cours...")
                    db_chroma = process_year(selected_year)
                    if db_chroma is None:
                        continue
                else:
                    # Charger la base de données existante
                    input_path = os.path.join(path_notes_generees_txt, f"RESF_{selected_year}.txt")
                    if os.path.exists(input_path):
                        db_chroma = prepare_database(selected_year, input_path)
                    else:
                        print(f"❌ Fichier texte pour {selected_year} non trouvé. Traitement OCR nécessaire...")
                        db_chroma = process_year(selected_year)
                        if db_chroma is None:
                            continue
                
                # Démarrer la session de questions
                question_session(selected_year, db_chroma)
                
            elif choice == "2":
                display_available_years()
                
            elif choice == "3":
                print("👋 Au revoir !")
                break
                
            else:
                print("❌ Choix invalide. Veuillez entrer 1, 2 ou 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    main()