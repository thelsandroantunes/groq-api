# src/app/chatbot.py
import os
from dotenv import load_dotenv
import requests
from requests.exceptions import Timeout, HTTPError, ConnectionError, RequestException

# Carregar variáveis do arquivo .env
load_dotenv()

def configure_llm():
    """
    Configura o LLM da Groq utilizando a chave de API.
    """
    api_key = os.getenv("API_KEY")
    api_url = os.getenv("API_URL")
    model = os.getenv("MODEL", "llama-3.3-70b-versatile")  # Modelo padrão

    if not api_key:
        raise ValueError("API_KEY não configurada no arquivo .env")

    if not api_url:
        raise ValueError("API_URL não configurada no arquivo .env")

    return GroqLLM(api_url, api_key, model)

class GroqLLM:
    def __init__(self, api_url, api_key, model):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def generate_response(self, messages, max_tokens=150, temperature=0.7):
        if not isinstance(messages, list):
            raise ValueError("Messages deve ser uma lista de mensagens.")

        for message in messages:
            if "role" not in message or "content" not in message:
                raise ValueError("Cada mensagem deve conter os campos 'role' e 'content'.")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
        except Timeout:
            raise Timeout("Tempo limite excedido ao conectar com a API da Groq.")
        except HTTPError as http_err:
            raise HTTPError(f"Erro HTTP: {http_err}")
        except ConnectionError:
            raise ConnectionError("Erro ao conectar com a API da Groq.")
        except RequestException as err:
            raise Exception(f"Erro na requisição: {err}")

        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Erro ao obter resposta da API.")
