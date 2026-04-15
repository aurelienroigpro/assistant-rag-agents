# ===== IMPORTS =====
import streamlit as st
import datetime
import re
import requests

from functools import lru_cache
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
# from duckduckgo_search import DDGS
from ddgs import DDGS
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


# ===== TITRE de la page de navigateur =====
st.title("Agent RAG + Calcul")
if "history" not in st.session_state:
    st.session_state.history = []


# ===== Chargement des documents: le Pipeline RAG =====

@st.cache_resource
def load_pipeline():

    # Load PDFs
    loader1 = PyPDFLoader("PDF_cours_informatique/IUT.pdf")
    loader2 = PyPDFLoader("PDF_cours_informatique/Architecture-ordinateur.pdf")
    loader3 = PyPDFLoader("PDF_cours_informatique/assembleur.pdf")
    loader4 = PyPDFLoader("PDF_cours_informatique/constitution-ordinateur.pdf")

    pages = [
        *loader1.load(),
        *loader2.load(),
        *loader3.load(),
        *loader4.load()
    ]

    # Nettoyage des textes:

    def clean_text(text):
        text = text.replace("\xa0", " ").replace("\n", " ")
        return " ".join(text.split())

    for doc in pages:
        doc.page_content = clean_text(doc.page_content)


    # Split pour créer les chunks:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(pages)

    # Embeddings + DB des chunks, pour recherche ultérieure via le LLM:

    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(chunks, embedding)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})




    # Mise en place du LLM: gpt d'Open AI
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Prompt distribué au LLM:
    prompt = ChatPromptTemplate.from_template("""
    Réponds à la question en utilisant uniquement le contexte ci-dessous.

    Contexte:
    {context}

    Question:
    {question}
    """)


# Revoir ce passage là. 
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
    )

    return rag_chain, retriever, llm


rag_chain, retriever, llm = load_pipeline()



# ===== Les TOOLS utilisés par le LLM pour remplir sa tâche =====



# La calculatrice:
def calculator_tool(question: str):
    try:
        expression = re.findall(r"[0-9\+\-\*/\.\(\)]+", question)
        expression = "".join(expression)
        return str(eval(expression))
    except:
        return None
    
# ===== WEATHER TOOL =====

@lru_cache(maxsize=128)
def get_coordinates(city: str):
    """
    Recherche une ville et retourne ses coordonnées + infos utiles.
    Cache les résultats pour éviter de refaire la requête à chaque fois.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "count": 5,
        "language": "fr",
        "format": "json"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = data.get("results")
    if not results:
        return None

    # On prend le premier résultat pour rester simple en V1
    best = results[0]

    return {
        "name": best.get("name"),
        "country": best.get("country", ""),
        "admin1": best.get("admin1", ""),
        "latitude": best.get("latitude"),
        "longitude": best.get("longitude"),
        "timezone": best.get("timezone", "auto"),
    }


def extract_city(question: str):
    """
    Demande au LLM d'extraire uniquement la ville.
    """
    prompt = f"""
Tu extrais uniquement le nom de la ville mentionnée dans la question.

Règles :
- Réponds uniquement par le nom de la ville.
- Si aucune ville n'est mentionnée, réponds uniquement : inconnu
- N'ajoute aucune ponctuation ni explication.

Question : {question}
"""
    city = llm.invoke(prompt).content.strip()
    return city


def weather_code_to_text(code: int):
    mapping = {
        0: "ciel dégagé",
        1: "principalement dégagé",
        2: "partiellement nuageux",
        3: "couvert",
        45: "brouillard",
        48: "brouillard givrant",
        51: "bruine légère",
        53: "bruine modérée",
        55: "bruine dense",
        56: "bruine verglaçante légère",
        57: "bruine verglaçante dense",
        61: "pluie faible",
        63: "pluie modérée",
        65: "pluie forte",
        66: "pluie verglaçante légère",
        67: "pluie verglaçante forte",
        71: "neige faible",
        73: "neige modérée",
        75: "neige forte",
        77: "grains de neige",
        80: "averses faibles",
        81: "averses modérées",
        82: "averses violentes",
        85: "averses de neige faibles",
        86: "averses de neige fortes",
        95: "orage",
        96: "orage avec grêle légère",
        99: "orage avec forte grêle",
    }
    return mapping.get(code, f"code météo {code}")


def build_weather_summary(weather_data: dict, location: dict):
    current = weather_data.get("current", {})
    daily = weather_data.get("daily", {})

    weather_text = weather_code_to_text(current.get("weather_code"))
    temp = current.get("temperature_2m")
    apparent = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")
    precip = current.get("precipitation")

    temp_max = None
    temp_min = None
    rain_sum = None

    if daily:
        if daily.get("temperature_2m_max"):
            temp_max = daily["temperature_2m_max"][0]
        if daily.get("temperature_2m_min"):
            temp_min = daily["temperature_2m_min"][0]
        if daily.get("precipitation_sum"):
            rain_sum = daily["precipitation_sum"][0]

    lines = [
        f"Météo à {location['name']}, {location['country']}",
        f"- Temps actuel : {weather_text}",
        f"- Température : {temp}°C",
        f"- Ressenti : {apparent}°C",
        f"- Humidité : {humidity}%",
        f"- Vent : {wind} km/h",
        f"- Précipitations actuelles : {precip} mm",
    ]

    if temp_min is not None and temp_max is not None:
        lines.append(f"- Aujourd’hui : min {temp_min}°C / max {temp_max}°C")

    if rain_sum is not None:
        lines.append(f"- Pluie prévue sur la journée : {rain_sum} mm")

    return "\n".join(lines)


def fetch_weather(latitude: float, longitude: float, timezone: str = "auto"):
    """
    Récupère météo actuelle + résumé du jour.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "current": ",".join([
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "precipitation",
            "weather_code",
            "wind_speed_10m"
        ]),
        "daily": ",".join([
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum"
        ]),
        # Optionnel mais utile pour plus tard si vous voulez enrichir l’outil
        "forecast_days": 1
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def weather_tool(question: str):
    """
    Outil météo principal.
    """
    try:
        city = extract_city(question)

        if city.lower() == "inconnu":
            return "Je n’ai pas trouvé de ville dans ta question."

        location = get_coordinates(city)

        if not location:
            return f"Je n’ai pas trouvé la ville : {city}"

        weather_data = fetch_weather(
            latitude=location["latitude"],
            longitude=location["longitude"],
            timezone=location["timezone"]
        )

        return build_weather_summary(weather_data, location)

    except requests.Timeout:
        return "Le service météo met trop de temps à répondre."
    except requests.RequestException as e:
        return f"Erreur réseau météo : {e}"
    except Exception as e:
        return f"Erreur météo : {e}"

# Le début de calendrier (en chantier, ne répond pas aux questions sur le calendrier):
def date_tool():
    now = datetime.datetime.now()
    return now.strftime("Nous sommes le %d/%m/%Y, %A")


# Le navigateur web:
def web_search_tool(question: str):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(question, max_results=3)

            formatted = ""
            for r in results:
                formatted += f"{r['title']}\n{r['body']}\nSource: {r['href']}\n\n"

            return formatted if formatted else "Aucun résultat trouvé."

    except Exception as e:
        return f"Erreur recherche web : {e}"


# La recherche sur le RAG:
def rag_tool(question: str, history_text: str):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = f"""
Tu es un assistant spécialisé dans l'analyse de documents.

Historique récent de la conversation :
{history_text}

Contexte documentaire :
{context}

Question actuelle :
{question}

Consignes :
- Réponds de manière claire, naturelle et structurée.
- Utilise uniquement les informations présentes dans le contexte documentaire.
- Tiens compte de l'historique si la question est une relance.
- Si l'information n'est pas dans le contexte, dis simplement que tu ne sais pas.
- N'invente rien.
"""

    response = llm.invoke(rag_prompt)
    return response.content, docs

def chat_tool(question: str, history_text: str):
    chat_prompt = f"""
Tu es un assistant conversationnel sympathique et naturel.

Historique récent :
{history_text}

Question :
{question}

Consignes :
- Réponds comme un assistant généraliste, naturellement.
- Si l'utilisateur te parle simplement, réponds simplement.
- Ne fais pas de recherche web.
- N'utilise pas les documents.
- Ne donne pas la signification linguistique de la phrase sauf si l'utilisateur le demande explicitement.

Réponse :
"""
    response = llm.invoke(chat_prompt)
    return response.content

# ===== AGENT =====
def format_history(history, max_turns=5):
    recent_history = history[-max_turns:]
    formatted = []

    for msg in recent_history:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role} : {msg['content']}")

    return "\n".join(formatted)


def ask(question: str):
    history_text = format_history(st.session_state.history)

    decision_prompt = f"""
Tu dois choisir exactement un seul outil parmi :
- calculator
- date
- rag
- web
- weather
- chat

Historique récent :
{history_text}

Question actuelle : {question}

Règles :
- Choisis "calculator" pour un calcul.
- Choisis "date" pour une question sur la date du jour.
- Choisis "weather" pour la météo, la température, la pluie, le climat actuel d'une ville.
- Choisis "rag" si la question concerne un sujet académique, technique ou présent dans les documents (ex : architecture ordinateur, assembleur, cours).
- Choisis "web" uniquement si la question nécessite une information externe ou récente (actualité, personnes, événements, etc.).
- Choisis "chat" pour une conversation générale ou des questions non traitées par les autres outils.

Réponds uniquement par un seul mot parmi :
calculator
date
rag
web
weather
chat
"""

    decision = llm.invoke(decision_prompt).content.strip().lower()

    if "calculator" in decision:
        result = calculator_tool(question)
        return f"Résultat : {result}"

    elif "date" in decision:
        return date_tool()
    
    elif "chat" in decision:
        response = chat_tool(question, history_text)
        return response

    elif "weather" in decision:
        result = weather_tool(question)
        return f"Météo :\n{result}"

    elif "web" in decision:
        web_results = web_search_tool(question)

        web_prompt = f"""
    Tu es un assistant.

    Voici des résultats de recherche web :

    {web_results}

    Question :
    {question}

    Consignes :
    - Donne une réponse claire et synthétique
    - Utilise les informations pertinentes
    - Ne copie pas les résultats tels quels
    - Résume intelligemment

    Réponse :
    """

        response = llm.invoke(web_prompt).content

        return f"""
    Réponse :
    {response}

    Sources :
    {web_results}
    """

    else:
        answer, docs = rag_tool(question, history_text)
        sources = "\n".join(str(doc.metadata) for doc in docs)

        return f"""
Réponse :
{answer}

Sources :
{sources}
"""


# ===== INTERFACE =====
question = st.text_input("Pose ta question :")

if st.button("Envoyer"):
    if question:
        with st.spinner("Réflexion en cours..."):
            response = ask(question)

        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "content": response})

        st.write(response)

if st.session_state.history:
    st.subheader("Historique de conversation")
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**Toi :** {msg['content']}")
        else:
            st.markdown(f"**Assistant :** {msg['content']}")
        