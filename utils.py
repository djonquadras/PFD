import pypdf

def separarPagina(arquivo, pagina, destino):
    paginaSelecionada = arquivo.pages[pagina -1]
    escritor = pypdf.PdfWriter()
    escritor.add_page(paginaSelecionada)
    escritor.write(destino)
    

def separarRange(arquivo, paginaInicial, paginaFinal, destino):
    escritor = pypdf.PdfWriter()
    for pagina in arquivo.pages[(paginaInicial -1):paginaFinal]:
        escritor.add_page(pagina)
    escritor.write(destino)