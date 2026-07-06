from pathlib import Path
from threading import Lock

import joblib
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


app = FastAPI(title="API com troca de modelo")

templates = Jinja2Templates(directory="templates")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

modelo_atual = {
    "nome": None,
    "objeto": None
}

lock = Lock()


class DadosEntrada(BaseModel):
    valores: list[float]


def listar_modelos():
    return [arquivo.name for arquivo in MODEL_DIR.glob("*.joblib")]


def carregar_modelo(nome_modelo: str):
    caminho = MODEL_DIR / nome_modelo

    if not caminho.exists():
        raise HTTPException(status_code=404, detail="Modelo não encontrado")

    modelo = joblib.load(caminho)

    with lock:
        modelo_atual["nome"] = nome_modelo
        modelo_atual["objeto"] = modelo

    return modelo


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "modelos": listar_modelos(),
            "modelo_atual": modelo_atual["nome"]
        }
    )


@app.post("/trocar-modelo")
def trocar_modelo(nome_modelo: str = Form(...)):
    carregar_modelo(nome_modelo)
    return RedirectResponse(url="/", status_code=303)


@app.post("/upload-modelo")
async def upload_modelo(arquivo: UploadFile = File(...)):
    if not arquivo.filename.endswith(".joblib"):
        raise HTTPException(
            status_code=400,
            detail="Envie apenas arquivos .joblib"
        )

    caminho = MODEL_DIR / arquivo.filename

    conteudo = await arquivo.read()

    with open(caminho, "wb") as f:
        f.write(conteudo)

    return RedirectResponse(url="/", status_code=303)


@app.post("/prever")
def prever(dados: DadosEntrada):
    if modelo_atual["objeto"] is None:
        raise HTTPException(
            status_code=400,
            detail="Nenhum modelo carregado"
        )

    entrada = [dados.valores]

    resultado = modelo_atual["objeto"].predict(entrada)

    return {
        "modelo_usado": modelo_atual["nome"],
        "entrada": dados.valores,
        "previsao": resultado.tolist()
    }


@app.get("/modelo-atual")
def ver_modelo_atual():
    return {
        "modelo_atual": modelo_atual["nome"],
        "modelos_disponiveis": listar_modelos()
    }