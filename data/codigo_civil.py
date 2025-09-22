import os
import re
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# URL do Código Civil compilado no Planalto
URL_CODIGO = "https://www.planalto.gov.br/ccivil_03/leis/2002/L10406.htm"

# Diretório de saída
OUTPUT_DIR = Path("codigo_civil_1314_1358")

# Faixa de artigos a extrair
ARTIGO_MIN = 1314
ARTIGO_MAX = 1358

# Regex para capitulos, seção, subseção etc.
capitulo_re = re.compile(r"^CAP[IÍ]TULO\s+([IVXLCDM]+)\s*(.*)$", re.IGNORECASE)
secao_re = re.compile(r"^Seç[aã]o\s+([IVXLCDM]+)\s*(.*)$", re.IGNORECASE)
subsecao_re = re.compile(r"^Subseç[aã]o\s+([IVXLCDM]+)\s*(.*)$", re.IGNORECASE)
artigo_re = re.compile(r"^Art\.\s*1\.(\d+)\b(.*)", re.IGNORECASE)


def sanitize(name: str) -> str:
    """Sanitize names for filesystem: remove / replace problematic chars."""
    # Remove acentos, pontuações, etc., simplificar
    # Aqui bem simples: substituir espaços, barras, dois pontos, etc.
    name = name.strip()
    name = re.sub(r'[\/:*?"<>|]', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name

def download_codigo() -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    resp = requests.get(URL_CODIGO, headers=headers, timeout=30)
    resp.raise_for_status()
    # O site está em ISO-8859-1
    resp.encoding = "latin-1"
    return resp.text

def extract_sections_parsed(text_html: str) -> list:
    """Retorna lista de linhas de interesse com marcação de capítulos, seções, subseções e artigos."""
    # Transformar HTML em texto limpo mantendo títulos
    soup = BeautifulSoup(text_html, "html.parser")

    # O plano: encontrar todos os elementos <p>, <hX>, etc., ou blocos que contenham títulos e artigos
    # No site do Planalto, o Código está em parágrafos com <p> e com <p CLASS="artigo">, ou sem class.
    # Para simplificar, vamos pegar o texto todo e dividir em linhas.
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # remover vazias
    return lines

def process_lines(lines: list):
    capitulo = None
    secao = None
    subsecao = None
    artigo_num = None
    artigo_buffer = []

    for line in lines:
        # Detecta capítulo
        m = capitulo_re.match(line)
        if m:
            capitulo = sanitize(line)
            secao = None
            subsecao = None
            continue

        # Detecta seção
        m = secao_re.match(line)
        if m:
            secao = sanitize(line)
            subsecao = None
            continue

        # Detecta subseção
        m = subsecao_re.match(line)
        if m:
            subsecao = sanitize(line)
            continue

        # Detecta artigo
        m = artigo_re.match(line)
        if m:
            num = int(m.group(1))
            if num < 1000:
                num += 1000
            if ARTIGO_MIN <= num <= ARTIGO_MAX:
                # salva artigo anterior, se ativo
                if artigo_buffer and artigo_num is not None:
                    salvar_artigo(capitulo, secao, subsecao, artigo_num, "\n".join(artigo_buffer))
                # iniciar novo artigo
                artigo_num = num
                artigo_buffer = [line]
            else:
                # Se for artigo fora da faixa, e estávamos acumulando, finalizar
                if artigo_buffer and artigo_num is not None:
                    salvar_artigo(capitulo, secao, subsecao, artigo_num, "\n".join(artigo_buffer))
                    artigo_buffer = []
                    artigo_num = None
            continue

        # Se estamos dentro de um artigo da faixa, acumula
        if artigo_num is not None:
            artigo_buffer.append(line)

    # salvar ultimo artigo se existir
    if artigo_buffer and artigo_num is not None:
        salvar_artigo(capitulo, secao, subsecao, artigo_num, "\n".join(artigo_buffer))


def salvar_artigo(capitulo, secao, subsecao, num, conteudo):
    # Define caminho
    path = OUTPUT_DIR
    if capitulo:
        path = path / capitulo
    if secao:
        path = path / secao
    if subsecao:
        path = path / subsecao

    path.mkdir(parents=True, exist_ok=True)

    # Nome do arquivo: Art_1.xxx.txt
    fname = f"Art_1.{num}.txt"
    arquivo = path / sanitize(fname)

    with open(arquivo, "w", encoding="utf-8") as f:
        f.write(conteudo + "\n")

    print(f"Salvo: {arquivo}")

def main():
    print("Baixando Código Civil compilado...")
    html = download_codigo()
    print("Convertendo e extraindo linhas...")
    lines = extract_sections_parsed(html)
    print(lines)
    print("Processando artigos entre", ARTIGO_MIN, "e", ARTIGO_MAX)
    process_lines(lines)
    print("Concluído.")

if __name__ == "__main__":
    main()
