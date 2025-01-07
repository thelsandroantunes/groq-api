# src/tests/test_groq_api.py
import pytest
import requests
import logging
from requests.exceptions import Timeout, HTTPError, ConnectionError
from dotenv import load_dotenv
from app.chatbot import configure_llm

# Configurar o logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Carregar o .env explicitamente no início dos testes
load_dotenv()

@pytest.fixture
def llm():
    return configure_llm()

def test_groq_api_success(llm, monkeypatch):
    logger.info("Iniciando teste: test_groq_api_success")
    monkeypatch.setenv("MODEL", "llama-3.3-70b-versatile")
    messages = [{"role": "user", "content": "Explique a importância de modelos de linguagem rápidos."}]
    response = llm.generate_response(messages)
    logger.info(f"Resposta recebida: {response}")
    assert response, "A resposta não deve ser vazia."

def test_groq_api_invalid_message_structure(llm):
    logger.info("Iniciando teste: test_groq_api_invalid_message_structure")
    with pytest.raises(ValueError, match="Messages deve ser uma lista de mensagens."):
        logger.info("Testando com mensagem inválida")
        llm.generate_response("Mensagem inválida")

    with pytest.raises(ValueError, match="Cada mensagem deve conter os campos 'role' e 'content'."):
        logger.info("Testando com mensagem sem o campo 'role'")
        llm.generate_response([{"content": "Sem role"}])

def test_groq_api_connection_error(llm, monkeypatch):
    logger.info("Iniciando teste: test_groq_api_connection_error")
    def mock_post(*args, **kwargs):
        raise ConnectionError("Erro ao conectar com a API da Groq.")
    monkeypatch.setattr(requests, "post", mock_post)

    with pytest.raises(ConnectionError, match="Erro ao conectar com a API da Groq."):
        logger.info("Simulando erro de conexão")
        llm.generate_response([{"role": "user", "content": "Teste"}])

def test_groq_api_timeout(llm, monkeypatch):
    logger.info("Iniciando teste: test_groq_api_timeout")
    def mock_post(*args, **kwargs):
        raise Timeout("Tempo limite excedido ao conectar com a API da Groq.")
    monkeypatch.setattr(requests, "post", mock_post)

    with pytest.raises(Timeout, match="Tempo limite excedido ao conectar com a API da Groq."):
        logger.info("Simulando timeout")
        llm.generate_response([{"role": "user", "content": "Teste de timeout"}])

def test_groq_api_http_error(llm, monkeypatch):
    logger.info("Iniciando teste: test_groq_api_http_error")
    def mock_post(*args, **kwargs):
        response = requests.Response()
        response.status_code = 400
        response._content = b'{"error": {"message": "Solicitacao invalida"}}'
        raise HTTPError(response=response)

    monkeypatch.setattr(requests, "post", mock_post)

    with pytest.raises(HTTPError):
        logger.info("Simulando erro HTTP")
        llm.generate_response([{"role": "user", "content": "Teste de erro HTTP"}])

def test_groq_api_max_tokens(llm):
    logger.info("Iniciando teste: test_groq_api_max_tokens")
    messages = [{"role": "user", "content": "Diga olá."}]

    # Cenário 1: Mínimo de tokens
    logger.info("Testando com max_tokens=1")
    response_min = llm.generate_response(messages, max_tokens=1)
    logger.info(f"Resposta com max_tokens=1: {response_min}")
    assert response_min, "A resposta com max_tokens=1 não deve ser vazia."

    # Cenário 2: Máximo de tokens
    logger.info("Testando com max_tokens=4096")
    response_max = llm.generate_response(messages, max_tokens=4096)
    logger.info(f"Resposta com max_tokens=4096: {response_max}")
    assert response_max, "A resposta com max_tokens=4096 não deve ser vazia."


def test_groq_api_temperature(llm):
    logger.info("Iniciando teste: test_groq_api_temperature")
    messages = [{"role": "user", "content": "Conte uma história curta."}]

    # Cenário 1: Temperature mínima
    logger.info("Testando com temperature=0.0")
    response_cold = llm.generate_response(messages, temperature=0.0)
    logger.info(f"Resposta com temperature=0.0: {response_cold}")
    assert response_cold, "A resposta com temperature=0.0 não deve ser vazia."

    # Cenário 2: Temperature máxima
    logger.info("Testando com temperature=1.0")
    response_hot = llm.generate_response(messages, temperature=1.0)
    logger.info(f"Resposta com temperature=1.0: {response_hot}")
    assert response_hot, "A resposta com temperature=1.0 não deve ser vazia."
