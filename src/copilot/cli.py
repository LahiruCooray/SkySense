"""SkySense Copilot CLI - Interactive Flight Log Analysis Assistant"""

import os
import sys
import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.copilot.rag_engine import SkySenseRAG
from src.copilot.structured_query import InsightQueryEngine
from src.copilot.llm_router import LLMRouter, QueryType
from src.core.processor import FlightLogProcessor

console = Console()


class SkySenseCopilot:
    """
    Interactive copilot for flight log analysis.
    Intelligently routes queries to structured data or RAG.
    """
    
    def __init__(self, rag_engine: SkySenseRAG):
        """Initialize copilot with RAG engine"""
        self.rag_engine = rag_engine
        self.query_engine = InsightQueryEngine()
        self.current_flight_loaded = False
        
        # Initialize LLM router with Groq LLM
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            temperature=0.1,  # Low temp for classification
            model_name="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.router = LLMRouter(llm, self.query_engine, self.rag_engine)
        
    def initialize_rag(self):
        """Lazy load RAG (only when needed)"""
        if self.rag is None:
            console.print("[yellow]Initializing RAG engine (first time may take a moment)...[/yellow]")
            self.rag = SkySenseRAG()
            self.rag.initialize()
    
    def load_flight(self, log_path: str = None):
        """Load a flight log for analysis"""
        
        if log_path:
            # Process new log
            console.print(f"[blue]Processing flight log: {log_path}[/blue]")
            
            processor = FlightLogProcessor()
            insights = processor.process_log(log_path)
            
            # Load insights
            log_name = Path(log_path).stem
            self.query_engine.load_flight_insights(log_name=log_name)
            
        else:
            # Load latest
            try:
                self.query_engine.load_flight_insights()
            except FileNotFoundError:
                console.print("[red]No flight data found. Process a log first.[/red]")
                return False
        
        self.current_flight_loaded = True
        
        # Show summary
        summary = self.query_engine.generate_text_summary()
        console.print(Panel(summary, title="Flight Summary", border_style="green"))
        
        return True
    
    def classify_query(self, query: str) -> str:
        """
        Classify query type to route to appropriate handler.
        
        Returns:
            'structured' - Direct data query
            'rag' - Knowledge-based query
            'hybrid' - Needs both
        """
        
        query_lower = query.lower()
        
        # Flight-specific patterns (when flight is loaded)
        if self.current_flight_loaded:
            flight_specific = [
                'this flight', 'the flight', 'last flight', 'went wrong', 'happened', 
                'summary', 'overview', 'general idea', 'what did',
                'problems', 'issues', 'errors', 'warnings', 'about the', 'tell me about',
                'what can you say', 'analyze', 'report'
            ]
            if any(kw in query_lower for kw in flight_specific):
                return 'structured'  # Use structured for flight data
        
        # Structured query patterns
        structured_keywords = [
            'how many', 'count', 'show me', 'list', 'what was the',
            'duration', 'when did', 'at what time', 'statistics',
            'maximum', 'minimum', 'average', 'total', 'critical',
            'battery', 'motor', 'tracking', 'ekf', 'vibration'
        ]
        
        # RAG query patterns (general knowledge) - be more specific!
        rag_keywords = [
            'why does', 'why did', 'how do i', 'what causes', 
            'explain what is', 'what is ekf', 'what is px4',
            'how does', 'help me understand', 'recommend', 'suggest fix',
            'normal range for', 'should i', 'is this normal', 'troubleshoot',
            'what does', 'define', 'definition of'
        ]
        
        has_structured = any(kw in query_lower for kw in structured_keywords)
        has_rag = any(kw in query_lower for kw in rag_keywords)
        
        # If flight is loaded, prefer structured queries
        if self.current_flight_loaded and has_structured:
            return 'structured'
        
        if has_structured and not has_rag:
            return 'structured'
        elif has_rag and not has_structured:
            return 'rag'
        else:
            # If flight loaded and mixed query, use hybrid
            if self.current_flight_loaded:
                return 'hybrid'
            return 'rag'
    
    def handle_structured_query(self, query: str) -> str:
        """Handle queries about current flight data"""
        
        if not self.current_flight_loaded:
            return "No flight data loaded. Use 'analyze <log_file>' first."
        
        query_lower = query.lower()
        
        # General flight summary/overview - CHECK THIS FIRST!
        summary_keywords = [
            'summary', 'overview', 'general', 'went wrong', 'happened', 
            'problems', 'issues', 'last flight', 'about the', 'tell me about',
            'what can you say', 'what can u say', 'analyze', 'report',
            'how was', 'flight status', 'overall'
        ]
        if any(kw in query_lower for kw in summary_keywords):
            return self._generate_flight_summary()
        
        # Pattern matching for common queries
        if 'critical' in query_lower and 'event' in query_lower:
            critical = self.query_engine.get_critical_events()
            if not critical:
                return "✓ No critical events detected in this flight."
            
            response = f"Found {len(critical)} critical events:\n\n"
            for event in critical:
                response += f"• [{event['type']}] at {event['t_start']:.1f}s: {event['text']}\n"
            return response
        
        elif 'battery' in query_lower and 'sag' in query_lower:
            sag_events = self.query_engine.get_insights_by_type('battery_sag')
            if not sag_events:
                return "✓ No battery sag events detected."
            
            response = f"Found {len(sag_events)} battery sag events:\n\n"
            for event in sag_events:
                dv = event['metrics'].get('dv_min', 0)
                response += f"• At {event['t_start']:.1f}s: {dv:.1f}V drop (severity: {event['severity']})\n"
            return response
        
        elif 'motor' in query_lower and 'dropout' in query_lower:
            dropout_events = self.query_engine.get_insights_by_type('motor_dropout')
            if not dropout_events:
                return "✓ No motor dropout events detected."
            
            response = f"⚠️ Found {len(dropout_events)} motor dropout events:\n\n"
            for event in dropout_events:
                motor_idx = event.get('motor_index', '?')
                response += f"• Motor {motor_idx} at {event['t_start']:.1f}s: {event['text']}\n"
            return response
        
        elif 'tracking error' in query_lower or 'attitude error' in query_lower:
            tracking = self.query_engine.get_insights_by_type('tracking_error')
            if not tracking:
                return "✓ No significant tracking errors detected."
            
            stats = self.query_engine.get_detector_statistics('tracking_error')
            response = f"Tracking Error Summary:\n\n"
            response += f"• Count: {stats['count']} events\n"
            response += f"• Total duration: {stats['total_duration']:.1f}s\n"
            response += f"• Average RMS: {stats.get('rms_deg_mean', 0):.1f}°\n"
            response += f"• Maximum: {stats.get('max_deg_max', 0):.1f}°\n"
            return response
        
        elif 'vibration' in query_lower:
            vibe = self.query_engine.get_insights_by_type('vibration_peak')
            if not vibe:
                return "✓ No significant vibration peaks detected."
            
            response = f"Found {len(vibe)} vibration peaks:\n\n"
            for event in vibe:
                freq = event['metrics'].get('peak_hz', 0)
                db = event['metrics'].get('peak_db', 0)
                response += f"• {freq:.0f} Hz at {db:.0f} dB: {event['text']}\n"
            return response
        
        elif 'duration' in query_lower or 'how long' in query_lower:
            duration = self.query_engine.get_flight_duration()
            return f"Flight duration: {duration:.1f} seconds ({duration/60:.1f} minutes)"
        
        elif 'assessment' in query_lower or 'quality' in query_lower:
            assessment = self.query_engine.get_flight_assessment()
            return f"Overall flight assessment: {assessment}"
        
        elif 'summary' in query_lower:
            return self.query_engine.generate_text_summary()
        
        else:
            # Fallback: try text search
            results = self.query_engine.search_by_text(query)
            if results:
                response = f"Found {len(results)} matching insights:\n\n"
                for r in results[:5]:
                    response += f"• [{r['type']}] {r['text']}\n"
                return response
            
            return "I couldn't find specific data matching your query. Try rephrasing or ask for explanation instead."
    
    def _generate_flight_summary(self) -> str:
        """Generate a comprehensive summary of the current flight"""
        
        if not self.current_flight_loaded:
            return "No flight data loaded."
        
        # Get summary insight if available
        summary_insights = self.query_engine.get_insights_by_type('summary')
        if summary_insights:
            summary = summary_insights[0]
            # Clean up the summary text for better display
            text = summary['text']
            # Remove extra separators
            text = text.replace('============================================================', '')
            text = text.replace('\n\n\n', '\n\n')
            return text.strip()
        
        # Otherwise generate manual summary
        all_insights = self.query_engine.insights
        
        # Count by severity
        severity_counts = self.query_engine.count_by_severity()
        
        # Build summary
        response = "## Flight Analysis\n\n"
        
        # Overall status
        critical_count = severity_counts.get('critical', 0)
        warning_count = severity_counts.get('warning', 0)
        
        if critical_count > 0:
            response += f"🔴 **CRITICAL** - {critical_count} critical issue{'s' if critical_count > 1 else ''}\n\n"
        elif warning_count > 0:
            response += f"🟡 **WARNING** - {warning_count} warning{'s' if warning_count > 1 else ''}\n\n"
        else:
            response += f"🟢 **EXCELLENT** - No significant issues\n\n"
        
        # Issue breakdown
        if critical_count > 0 or warning_count > 0:
            response += "### Issues Found:\n"
            
            # Get critical events
            critical = self.query_engine.get_critical_events()
            for i, event in enumerate(critical[:3], 1):  # Limit to top 3
                response += f"{i}. **{event['type'].replace('_', ' ').title()}** (t={event['t_start']:.1f}s)\n"
                response += f"   {event['text'][:80]}...\n\n" if len(event['text']) > 80 else f"   {event['text']}\n\n"
            
            if len(critical) > 3:
                response += f"*...plus {len(critical) - 3} more issue{'s' if len(critical) - 3 > 1 else ''}*\n\n"
        else:
            response += "✅ Flight completed successfully with no critical issues.\n\n"
        
        response += "💡 *Try: 'Show critical events' or 'What is [term]?'*"
        
        return response
    
    def handle_rag_query(self, query: str) -> str:
        """Handle knowledge-based queries"""
        
        self.initialize_rag()
        
        result = self.rag.query(query, include_sources=False)
        return result['answer']
    
    def handle_hybrid_query(self, query: str) -> str:
        """Combine structured data with RAG explanation"""
        
        # Get structured data
        structured_response = self.handle_structured_query(query)
        
        # If structured response is informative enough, return it
        if len(structured_response) > 200 or "No " in structured_response:
            return structured_response
        
        # Otherwise add RAG explanation
        self.initialize_rag()
        explanation_query = f"Based on this flight data: {structured_response[:200]}\n\nQuestion: {query}\n\nProvide brief context (2-3 sentences max)."
        
        rag_result = self.rag.query(explanation_query, include_sources=False)
        
        # Combine cleanly
        return f"{structured_response}\n\n💡 **Quick Explanation:**\n{rag_result['answer']}"
    
    def process_query(self, query: str) -> str:
        """
        Main query processing using LLM-powered router.
        
        Flow:
        1. LLM classifies query → [FLIGHT_DATA, KNOWLEDGE, CONVERSATIONAL]
        2. Router gathers relevant context (insights + RAG docs)
        3. Router builds structured prompt with all context
        4. LLM generates final answer
        """
        
        query = query.strip()
        
        # Use LLM router for intelligent query handling
        result = self.router.answer_query(query, self.current_flight_loaded)
        
        # Format response with metadata
        answer = result['answer']
        classification = result['classification']
        confidence = result.get('confidence', 0.0)
        
        # Add debug info for low confidence
        if confidence < 0.5:
            debug_info = f"\n\n_Classification: {classification['type'].value}, Confidence: {confidence:.2f}_"
            answer += debug_info
        
        return answer
    
    def _process_query_fallback(self, query: str) -> str:
        """Fallback query processing (old method)"""
        
        query_lower = query.lower().strip()
        
        # Handle greetings and casual queries
        greetings = ['hi', 'hello', 'hey', 'help']
        if query_lower in greetings:
            if self.current_flight_loaded:
                return (
                    "👋 Hello! I've analyzed your flight and found some insights.\n\n"
                    "Here's what you can ask me:\n"
                    "- 'Give me a summary' - Overview of the flight\n"
                    "- 'Show critical events' - List all critical issues\n"
                    "- 'Why did battery sag?' - Explain specific issues\n"
                    "- 'What is EKF?' - Learn about drone systems\n\n"
                    "Type your question, or try: **'What went wrong in this flight?'**"
                )
            else:
                return (
                    "👋 Hello! I'm your SkySense flight analysis assistant.\n\n"
                    "I can help you:\n"
                    "- Understand drone flight logs\n"
                    "- Explain PX4 systems and terminology\n"
                    "- Troubleshoot flight issues\n\n"
                    "To analyze a flight, load a log file first.\n"
                    "Or ask me general questions like: **'What is EKF?'**"
                )
        
        query_type = self.classify_query(query)
        
        if query_type == 'structured':
            return self.handle_structured_query(query)
        elif query_type == 'rag':
            return self.handle_rag_query(query)
        else:
            return self.handle_hybrid_query(query)


# CLI Commands

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    🚁 SkySense Copilot - Intelligent Drone Flight Log Analysis
    
    Your AI-powered assistant for PX4 flight log analysis.
    """
    pass


@cli.command()
@click.argument('log_file', type=click.Path(exists=True))
def analyze(log_file):
    """
    Analyze a flight log and start interactive session.
    
    Example: copilot analyze logs/flight.ulg
    """
    
    console.print(Panel.fit(
        "[bold blue]SkySense Copilot[/bold blue]\n"
        "Intelligent Flight Log Analysis Assistant",
        border_style="blue"
    ))
    
    copilot = SkySenseCopilot()
    
    # Process and load flight
    if not copilot.load_flight(log_file):
        return
    
    # Interactive session
    console.print("\n[green]Type your questions or 'exit' to quit.[/green]")
    console.print("[dim]Examples: 'Why did battery sag?', 'Show critical events', 'How do I fix vibration?'[/dim]\n")
    
    while True:
        try:
            query = console.input("\n[bold cyan]You:[/bold cyan] ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not query.strip():
                continue
            
            # Process query
            with console.status("[yellow]Thinking...[/yellow]"):
                response = copilot.process_query(query)
            
            # Display response
            console.print(Panel(
                Markdown(response),
                title="[bold green]Copilot[/bold green]",
                border_style="green"
            ))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
def ask():
    """
    Interactive mode with pre-loaded flight data.
    
    Example: copilot ask
    """
    
    console.print(Panel.fit(
        "[bold blue]SkySense Copilot[/bold blue]\n"
        "Interactive Knowledge Assistant",
        border_style="blue"
    ))
    
    copilot = SkySenseCopilot()
    
    # Try to load latest flight
    try:
        copilot.load_flight()
    except:
        console.print("[yellow]No flight data loaded. You can still ask knowledge questions![/yellow]")
    
    console.print("\n[green]Type your questions or 'exit' to quit.[/green]")
    console.print("[dim]Ask about flight logs, troubleshooting, or drone concepts.[/dim]\n")
    
    while True:
        try:
            query = console.input("\n[bold cyan]You:[/bold cyan] ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not query.strip():
                continue
            
            # Process query
            with console.status("[yellow]Thinking...[/yellow]"):
                response = copilot.process_query(query)
            
            # Display response
            console.print(Panel(
                Markdown(response),
                title="[bold green]Copilot[/bold green]",
                border_style="green"
            ))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
def build_kb():
    """
    Build knowledge base from legal sources.
    
    This creates the vector store for RAG retrieval.
    """
    
    console.print("[blue]Building SkySense knowledge base...[/blue]\n")
    
    # Build knowledge base
    from src.copilot.knowledge_builder import KnowledgeBaseBuilder
    
    builder = KnowledgeBaseBuilder()
    knowledge = builder.build_complete_knowledge_base(scrape_web=False)
    kb_path = builder.save()
    
    # Build vector store
    console.print("\n[blue]Creating vector embeddings...[/blue]")
    
    rag = SkySenseRAG()
    rag.build_vector_store(force_rebuild=True)
    
    console.print("\n[green]✓ Knowledge base ready![/green]")
    console.print(f"  Location: {kb_path}")
    console.print("\nYou can now use 'copilot ask' or 'copilot analyze <log>'")


@cli.command()
def info():
    """
    Show copilot information and attributions.
    """
    
    console.print(Panel.fit(
        "[bold]SkySense Copilot v1.0[/bold]\n\n"
        "An intelligent flight log analysis assistant powered by:\n"
        "• Groq (llama-3.3-70b-versatile)\n"
        "• LangChain (RAG framework)\n"
        "• HuggingFace (local embeddings)\n"
        "• FAISS (vector store)",
        title="About",
        border_style="blue"
    ))
    
    console.print("\n[bold]Knowledge Sources (All Legal):[/bold]\n")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Source")
    table.add_column("License")
    table.add_column("Description")
    
    table.add_row(
        "PX4 Documentation",
        "CC BY 4.0",
        "Flight control system reference"
    )
    table.add_row(
        "SkySense Detectors",
        "Proprietary",
        "Detector specifications and logic"
    )
    table.add_row(
        "Manual Curation",
        "Original",
        "Troubleshooting guides and patterns"
    )
    
    console.print(table)
    
    console.print("\n[dim]All sources properly attributed and legally compliant.[/dim]")


if __name__ == "__main__":
    cli()
