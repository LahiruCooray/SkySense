"""
SkySense: AI Co-Pilot for Flight Log Analysis
Entry point for the application
"""
import click
from pathlib import Path
from src.core.processor import FlightLogProcessor
from src.api.server import create_app
import uvicorn


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
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    """Start the SkySense API server"""
    click.echo(f"Starting SkySense server on {host}:{port}")
    
    app = create_app()
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == '__main__':
    cli()