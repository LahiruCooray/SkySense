"""
FastAPI server for SkySense insights API
Provides endpoints for retrieving insights and generating plots
"""
import os
import json
from pathlib import Path
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from ..core.models import AnyInsight
from .plotter import InsightPlotter


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="SkySense API",
        description="AI Co-Pilot for Flight Log Analysis",
        version="1.0.0"
    )
    
    plotter = InsightPlotter()
    
    @app.get("/")
    async def root():
        """Health check endpoint"""
        return {"message": "SkySense API is running"}
    
    @app.get("/insights")
    async def get_insights(
        log_name: Optional[str] = Query(None, description="Filter by log name"),
        insight_type: Optional[str] = Query(None, description="Filter by insight type"),
        phase: Optional[str] = Query(None, description="Filter by flight phase"),
        severity: Optional[str] = Query(None, description="Filter by severity")
    ):
        """Get insights with optional filters"""
        
        insights_dir = Path("data/insights")
        if not insights_dir.exists():
            return {"insights": [], "count": 0}
        
        # Load all insight JSON files
        insights = []
        for json_file in insights_dir.glob("ins_*.json"):
            try:
                with open(json_file, 'r') as f:
                    insight_data = json.load(f)
                    insights.append(insight_data)
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
        
        # Apply filters
        filtered_insights = insights
        
        if insight_type:
            filtered_insights = [i for i in filtered_insights if i.get('type') == insight_type]
        
        if phase:
            filtered_insights = [i for i in filtered_insights if i.get('phase') == phase]
        
        if severity:
            filtered_insights = [i for i in filtered_insights if i.get('severity') == severity]
        
        return {
            "insights": filtered_insights,
            "count": len(filtered_insights),
            "total_count": len(insights)
        }
    
    @app.get("/insights/{insight_id}")
    async def get_insight(insight_id: str):
        """Get specific insight by ID"""
        
        insight_file = Path(f"data/insights/{insight_id}.json")
        
        if not insight_file.exists():
            raise HTTPException(status_code=404, detail="Insight not found")
        
        try:
            with open(insight_file, 'r') as f:
                insight_data = json.load(f)
            return insight_data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading insight: {e}")
    
    @app.get("/plot/{insight_id}")
    async def get_plot(insight_id: str):
        """Generate and return plot for specific insight"""
        
        # Check if insight exists
        insight_file = Path(f"data/insights/{insight_id}.json")
        if not insight_file.exists():
            raise HTTPException(status_code=404, detail="Insight not found")
        
        # Load insight
        try:
            with open(insight_file, 'r') as f:
                insight_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading insight: {e}")
        
        # Generate plot
        try:
            plot_path = plotter.generate_plot(insight_data)
            
            if not os.path.exists(plot_path):
                raise HTTPException(status_code=500, detail="Plot generation failed")
            
            return FileResponse(
                plot_path,
                media_type="image/png",
                filename=f"{insight_id}_plot.png"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating plot: {e}")
    
    @app.post("/ask")
    async def ask_insights(query: dict):
        """Query insights using natural language"""
        
        user_query = query.get("question", "")
        time_range = query.get("time_range", {})
        filters = query.get("filters", {})
        
        if not user_query:
            raise HTTPException(status_code=400, detail="Question is required")
        
        # Load insights
        insights_dir = Path("data/insights")
        insights = []
        
        for json_file in insights_dir.glob("ins_*.json"):
            try:
                with open(json_file, 'r') as f:
                    insight_data = json.load(f)
                    insights.append(insight_data)
            except Exception:
                continue
        
        # Simple keyword-based search for MVP
        relevant_insights = []
        query_lower = user_query.lower()
        
        for insight in insights:
            insight_text = insight.get('text', '').lower()
            insight_type = insight.get('type', '').lower()
            
            # Check for keyword matches
            if (any(keyword in insight_text for keyword in query_lower.split()) or
                any(keyword in insight_type for keyword in query_lower.split())):
                
                # Apply time range filter if specified
                if time_range:
                    t_start = insight.get('t_start', 0)
                    t_end = insight.get('t_end', 0)
                    
                    if 'start' in time_range and t_end < time_range['start']:
                        continue
                    if 'end' in time_range and t_start > time_range['end']:
                        continue
                
                relevant_insights.append(insight)
        
        # Sort by relevance (severity, then time)
        severity_order = {'critical': 0, 'warn': 1, 'info': 2}
        relevant_insights.sort(key=lambda x: (
            severity_order.get(x.get('severity', 'info'), 2),
            x.get('t_start', 0)
        ))
        
        # Generate response
        if not relevant_insights:
            response = {
                "answer": "No relevant insights found for your query.",
                "insights": [],
                "query": user_query
            }
        else:
            # Create summary answer
            critical_count = len([i for i in relevant_insights if i.get('severity') == 'critical'])
            warn_count = len([i for i in relevant_insights if i.get('severity') == 'warn'])
            
            answer_parts = [f"Found {len(relevant_insights)} relevant insights."]
            
            if critical_count > 0:
                answer_parts.append(f"{critical_count} critical issues detected.")
            if warn_count > 0:
                answer_parts.append(f"{warn_count} warnings found.")
            
            # Add specific insight details
            if relevant_insights:
                top_insight = relevant_insights[0]
                answer_parts.append(f"Most significant: {top_insight.get('text', '')}")
            
            response = {
                "answer": " ".join(answer_parts),
                "insights": relevant_insights[:10],  # Limit to top 10
                "query": user_query,
                "total_found": len(relevant_insights)
            }
        
        return response
    
    @app.get("/summary")
    async def get_flight_summary():
        """Get flight summary with key metrics"""
        
        insights_dir = Path("data/insights")
        
        # Look for summary insights
        summary_insights = []
        for json_file in insights_dir.glob("ins_summary_*.json"):
            try:
                with open(json_file, 'r') as f:
                    insight_data = json.load(f)
                    summary_insights.append(insight_data)
            except Exception:
                continue
        
        if not summary_insights:
            return {"message": "No flight summary available"}
        
        # Return the most recent summary
        latest_summary = max(summary_insights, key=lambda x: x.get('t_end', 0))
        
        return {
            "summary": latest_summary,
            "kpis": latest_summary.get('kpis', {}),
            "assessment": latest_summary.get('metrics', {}).get('flight_assessment', 'Unknown')
        }
    
    return app