import asyncio
import typer
import structlog
from pathlib import Path
from src.core.pdf_parser import VectorExtractor
from src.core.graph_builder import BipartiteGraphBuilder
from src.core.text_associator import TextAssociator
from src.llm.tools import GraphContext
from src.llm.agent import SchematicAgent, OllamaClient, MockClient, LLMClient

app = typer.Typer(help="CLI per l'agente LLM Schematic Extractor")
logger = structlog.get_logger("cli")

@app.callback()
def callback() -> None:
    pass

@app.command(name="query")
def query_cmd(
    question: str = typer.Argument(..., help="La domanda in linguaggio naturale"),
    pdf: Path = typer.Option(..., "--pdf", help="Percorso al file PDF dello schema"),
    mock: bool = typer.Option(False, "--mock", help="Usa il MockClient invece di Ollama"),
    model: str = typer.Option("llama3.1:8b-instruct-q4_K_M", "--model", help="Modello Ollama da utilizzare")
) -> None:
    """
    Interroga il grafo bipartito (Componenti <-> Nets) estratto da un PDF tramite un LLM.
    """
    if not pdf.exists():
        typer.secho(f"Errore: Il file {pdf} non esiste.", fg=typer.colors.RED)
        raise typer.Exit(1)
        
    typer.secho(f"Estrazione del grafo da {pdf}...", fg=typer.colors.CYAN)
    
    # 1. Parsing del PDF
    parser = VectorExtractor()
    try:
        pages = parser.extract(str(pdf))
    except Exception as e:
        typer.secho(f"Errore nel parsing PDF: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)
        
    if not pages:
        typer.secho("Errore: Nessuna pagina valida estratta.", fg=typer.colors.RED)
        raise typer.Exit(1)
        
    page = pages[0]
    
    # 2. Costruzione del grafo bipartito
    text_associator = TextAssociator()
    builder = BipartiteGraphBuilder(text_associator=text_associator)
    graph = builder.build_from_page(page)
    
    typer.secho(f"Grafo costruito: {graph.number_of_nodes()} nodi, {graph.number_of_edges()} archi.", fg=typer.colors.GREEN)
    
    # 3. Inizializzazione LLM
    graph_context = GraphContext(graph)
    client: LLMClient
    if mock:
        typer.secho("Uso MockClient (modalità mock).", fg=typer.colors.YELLOW)
        client = MockClient()
    else:
        typer.secho(f"Uso OllamaClient (modello: {model}).", fg=typer.colors.YELLOW)
        client = OllamaClient(model=model)
        
    agent = SchematicAgent(graph_context=graph_context, llm_client=client)
    
    # 4. Esecuzione query
    typer.secho("\nQuery: " + question, fg=typer.colors.MAGENTA)
    typer.secho("Agente in elaborazione...\n", fg=typer.colors.CYAN)
    
    response = asyncio.run(agent.query(question))
    
    typer.secho("Risposta:", fg=typer.colors.GREEN, bold=True)
    typer.echo(response)

if __name__ == "__main__":
    app()
