"""
SkySense: AI Co-Pilot for Flight Log Analysis
Entry point for the application
"""
import click
from pathlib import Path
from src.core.processor import FlightLogProcessor
from src.api.server import create_app
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@click.group()
def cli():
    """SkySense - AI Co-Pilot for Flight Log Analysis"""
    pass


@cli.command()
@click.argument('log_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='data/insights', help='Output directory for insights')
def analyze(log_file, output_dir):
    """Analyze a flight log and generate insights"""
    click.echo(f"Analyzing flight log: {log_file}")
    
    processor = FlightLogProcessor()
    insights = processor.process_log(log_file, output_dir)
    
    click.echo(f"Generated {len(insights)} insights")
    click.echo(f"Results saved to: {output_dir}")


@cli.command()
@click.option('--insights-dir', default='data/insights', help='Insights directory to clean')
@click.option('--dry-run', is_flag=True, help='Show what would be deleted without deleting')
def cleanup(insights_dir, dry_run):
    """Clean up duplicate analyses, keep only the latest for each log"""
    from cleanup_insights import cleanup_old_analyses
    
    if dry_run:
        click.echo("DRY RUN - No files will be deleted\n")
    
    cleanup_old_analyses(insights_dir, dry_run)


@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    """Start the SkySense API server"""
    click.echo(f"Starting SkySense server on {host}:{port}")
    
    app = create_app()
    uvicorn.run(app, host=host, port=port, reload=reload)


@cli.command()
@click.argument('log_file', type=click.Path(exists=True), required=False)
def copilot(log_file):
    """
    Launch interactive SkySense Copilot
    
    Examples:
        python main.py copilot ulogs/flight.ulg  # Analyze and chat
        python main.py copilot                    # Chat with knowledge base
    """
    from src.copilot.cli import SkySenseCopilot
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]🚁 SkySense Copilot[/bold blue]\n"
        "Your AI-powered flight log analysis assistant",
        border_style="blue"
    ))
    
    # Initialize RAG engine
    from src.copilot.rag_engine import SkySenseRAG
    rag_engine = SkySenseRAG()
    rag_engine.build_knowledge_base()
    
    copilot_instance = SkySenseCopilot(rag_engine)
    
    # Load flight if provided
    if log_file:
        if not copilot_instance.load_flight(log_file):
            return
    else:
        try:
            copilot_instance.load_flight()
        except:
            console.print("[yellow]No flight data loaded. You can still ask knowledge questions![/yellow]")
    
    console.print("\n[green]Type your questions or 'exit' to quit.[/green]")
    console.print("[dim]Examples: 'Why did battery sag?', 'Show critical events', 'How do I fix vibration?'[/dim]\n")
    
    while True:
        try:
            query = console.input("\n[bold cyan]You:[/bold cyan] ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break
            
            if not query.strip():
                continue
            
            # Process query
            with console.status("[yellow]Thinking...[/yellow]"):
                response = copilot_instance.process_query(query)
            
            # Display response
            console.print(Panel(
                Markdown(response),
                title="[bold green]Copilot[/bold green]",
                border_style="green"
            ))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
def setup_copilot():
    """Set up SkySense Copilot (build knowledge base and vector store)"""
    from src.copilot.knowledge_builder import KnowledgeBaseBuilder
    from src.copilot.rag_engine import SkySenseRAG
    from rich.console import Console
    
    console = Console()
    
    console.print("[bold blue]Setting up SkySense Copilot...[/bold blue]\n")
    
    # Build knowledge base
    console.print("1. Building knowledge base from legal sources...")
    builder = KnowledgeBaseBuilder()
    knowledge = builder.build_complete_knowledge_base(scrape_web=False)
    kb_path = builder.save()
    
    # Build vector store
    console.print("\n2. Creating vector embeddings (this may take a minute)...")
    rag = SkySenseRAG()
    rag.build_vector_store(force_rebuild=True)
    
    console.print("\n[bold green]✓ Copilot setup complete![/bold green]")
    console.print(f"\n  Knowledge base: {kb_path}")
    console.print(f"  Vector store: {rag.vector_store_path}")
    console.print("\n[cyan]You can now use:[/cyan]")
    console.print("  python main.py copilot <log_file>  - Analyze a flight")
    console.print("  python main.py copilot              - Knowledge Q&A")


if __name__ == '__main__':
    cli()