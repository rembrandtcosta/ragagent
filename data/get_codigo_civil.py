import os
import re
import requests

# URL oficial do Código Civil no site do Planalto
URL = "https://www.planalto.gov.br/ccivil_03/leis/2002/L10406.htm"

# Pasta de saída
OUTPUT_DIR = "artigos"

# Intervalo desejado
ARTIGO_INICIAL = 1314
ARTIGO_FINAL = 1358

# Garante que a pasta de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_codigo() -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    resp = requests.get(URL, headers=headers, timeout=30)
    resp.raise_for_status()
    # O site está em ISO-8859-1
    resp.encoding = "latin-1"
    return resp.text


# Baixa o HTML do Planalto
html = download_codigo()

# Remove tags HTML para ficar só o texto
# O site usa <p> para artigos
texto_limpo = re.sub(r"<[^>]+>", "", html)  # remove tags
texto_limpo = re.sub(r"&nbsp;", " ", texto_limpo)  # substitui entidades
texto_limpo = re.sub(r"\s+\n", "\n", texto_limpo)  # limpa espaços extras

# Regex para capturar os artigos (caput + parágrafos + §§)
pattern = re.compile(r"(Art\.\s*1\.(\d+(?:-[A-Z])?)\.\s.*?)(?=(?:\n\s*Art\.\s*\d+\.|\Z))", re.DOTALL)
# artigo_re = re.compile(r"^Art\.\s*1\.(\d+)\b(.*)", re.IGNORECASE)

print("Extraindo artigos...")
for match in pattern.finditer(texto_limpo):
    artigo = match.group(1).strip()
    print(artigo)

    # Extrai o número do artigo
    num_match = re.search(r"Art\.\s*1\.(\d+)(?:-([A-Z]))?", artigo)
    num_artigo = None
    if num_match is None:
        continue

    num_artigo = int(num_match.group(1))
    num_artigo += 1000

    if num_artigo > ARTIGO_FINAL:
        break

    # Filtra apenas os artigos no intervalo 1314–1358
    if ARTIGO_INICIAL <= num_artigo <= ARTIGO_FINAL:
        letra = num_match.group(2)
        if letra:
            num_artigo = f"{num_artigo}-{letra}"

        filename = os.path.join(OUTPUT_DIR, f"Art_{num_artigo}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(artigo)
        print(f"Salvo: {filename}")

print("Concluído! ✅")
