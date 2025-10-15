"""LLM-Powered Query Router for SkySense Copilot"""

from typing import Dict, Any, Optional, List
from enum import Enum
import json


class QueryType(Enum):
    FLIGHT_DATA = "flight_data"
    KNOWLEDGE = "knowledge"
    CONVERSATIONAL = "conversational"


class LLMRouter:
    """Intelligent query router using LLM for classification and prompt structuring."""
    
    def __init__(self, llm, query_engine=None, rag_engine=None):
        self.llm = llm
        self.query_engine = query_engine
        self.rag_engine = rag_engine
        
    def classify_query(self, query: str, flight_loaded: bool) -> Dict[str, Any]:
        """
        Use LLM to classify user query and determine required context.
        
        Returns:
            {
                'type': QueryType,
                'requires_flight_data': bool,
                'requires_knowledge': bool,
                'intent': str,
                'confidence': float
            }
        """
        
        classification_prompt = f"""You are a query classifier for a drone flight log analysis system.

User query: "{query}"

Flight data available: {"Yes" if flight_loaded else "No"}

Classify this query into ONE category:
1. FLIGHT_DATA - User asking about specific flight data/metrics (e.g., "what went wrong?", "show battery sag", "flight duration")
2. KNOWLEDGE - User asking for general drone/PX4 knowledge (e.g., "what is EKF?", "how to fix vibration?")
3. CONVERSATIONAL - Greeting, acknowledgment, or casual chat (e.g., "hi", "ok", "thanks")

Also determine:
- Does this need flight data access? (yes/no)
- Does this need knowledge base access? (yes/no)
- What is the user's intent in one sentence?

Respond ONLY with valid JSON, no other text:
{{
  "category": "FLIGHT_DATA" | "KNOWLEDGE" | "CONVERSATIONAL",
  "needs_flight_data": true | false,
  "needs_knowledge": true | false,
  "intent": "one sentence description",
  "confidence": 0.0-1.0
}}"""

        try:
            response = self.llm.invoke(classification_prompt)
            content = response.content.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            
            # Convert to QueryType enum
            category_map = {
                'FLIGHT_DATA': QueryType.FLIGHT_DATA,
                'KNOWLEDGE': QueryType.KNOWLEDGE,
                'CONVERSATIONAL': QueryType.CONVERSATIONAL
            }
            
            return {
                'type': category_map.get(result['category'], QueryType.CONVERSATIONAL),
                'requires_flight_data': result['needs_flight_data'],
                'requires_knowledge': result['needs_knowledge'],
                'intent': result['intent'],
                'confidence': result.get('confidence', 0.5)
            }
            
        except Exception as e:
            # Fallback to simple classification
            return self._fallback_classification(query, flight_loaded)
    
    def _fallback_classification(self, query: str, flight_loaded: bool) -> Dict[str, Any]:
        """Simple fallback classification if LLM fails"""
        query_lower = query.lower()
        
        # Conversational
        if query_lower in ['hi', 'hello', 'hey', 'ok', 'thanks', 'bye']:
            return {
                'type': QueryType.CONVERSATIONAL,
                'requires_flight_data': False,
                'requires_knowledge': False,
                'intent': 'User greeting or acknowledgment',
                'confidence': 0.9
            }
        
        # Knowledge queries
        knowledge_keywords = ['what is', 'what does', 'explain', 'how to', 'why does']
        if any(kw in query_lower for kw in knowledge_keywords):
            return {
                'type': QueryType.KNOWLEDGE,
                'requires_flight_data': False,
                'requires_knowledge': True,
                'intent': 'User asking for technical knowledge',
                'confidence': 0.7
            }
        
        # Flight data (if loaded)
        if flight_loaded:
            return {
                'type': QueryType.FLIGHT_DATA,
                'requires_flight_data': True,
                'requires_knowledge': False,
                'intent': 'User asking about flight data',
                'confidence': 0.6
            }
        
        # Default
        return {
            'type': QueryType.CONVERSATIONAL,
            'requires_flight_data': False,
            'requires_knowledge': False,
            'intent': 'Unclear query',
            'confidence': 0.3
        }
    
    def gather_context(self, classification: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Gather relevant context based on classification.
        
        Returns:
            {
                'flight_data': {...},
                'knowledge_docs': [...],
                'summary': str
            }
        """
        context = {
            'flight_data': None,
            'knowledge_docs': [],
            'summary': ''
        }
        
        # Gather flight data if needed
        if classification['requires_flight_data'] and self.query_engine:
            context['flight_data'] = self._gather_flight_context(query)
        
        # Gather knowledge if needed
        if classification['requires_knowledge'] and self.rag_engine:
            context['knowledge_docs'] = self._gather_knowledge_context(query)
        
        return context
    
    def _gather_flight_context(self, query: str) -> Dict[str, Any]:
        """Extract relevant flight data based on query"""
        
        if not self.query_engine or not self.query_engine.current_flight_insights:
            return {'error': 'No flight data available'}
        
        # Get basic stats
        severity_counts = self.query_engine.count_by_severity()
        type_counts = self.query_engine.count_by_type()
        
        # Get summary insight if available
        summary_insights = self.query_engine.get_insights_by_type('summary')
        summary_text = summary_insights[0]['text'] if summary_insights else "No summary available"
        
        # Get critical events
        critical_events = self.query_engine.get_critical_events()
        
        # Extract query-specific data
        query_lower = query.lower()
        specific_data = {}
        
        if 'battery' in query_lower:
            specific_data['battery_events'] = self.query_engine.get_insights_by_type('battery_sag')
        
        if 'motor' in query_lower:
            specific_data['motor_events'] = self.query_engine.get_insights_by_type('motor_dropout')
        
        if 'tracking' in query_lower or 'attitude' in query_lower:
            specific_data['tracking_events'] = self.query_engine.get_insights_by_type('tracking_error')
        
        if 'ekf' in query_lower:
            specific_data['ekf_events'] = self.query_engine.get_insights_by_type('ekf_spike')
        
        if 'vibration' in query_lower:
            specific_data['vibration_events'] = self.query_engine.get_insights_by_type('vibration_peak')
        
        return {
            'summary': summary_text[:300],  # Truncate
            'severity_counts': severity_counts,
            'type_counts': type_counts,
            'critical_events': critical_events[:3],  # Top 3
            'specific_data': specific_data,
            'total_insights': len(self.query_engine.current_flight_insights)
        }
    
    def _gather_knowledge_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge base documents"""
        
        if not self.rag_engine or not self.rag_engine.vector_store:
            return []
        
        try:
            docs = self.rag_engine.get_relevant_context(query, k=k)
            return [
                {
                    'content': doc.page_content[:300],  # Truncate
                    'metadata': doc.metadata
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"Knowledge retrieval error: {e}")
            return []
    
    def build_prompt(
        self, 
        query: str, 
        classification: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> str:
        """
        Build structured prompt based on classification and context.
        """
        
        query_type = classification['type']
        
        if query_type == QueryType.CONVERSATIONAL:
            return self._build_conversational_prompt(query, context)
        
        elif query_type == QueryType.FLIGHT_DATA:
            return self._build_flight_data_prompt(query, context)
        
        elif query_type == QueryType.KNOWLEDGE:
            return self._build_knowledge_prompt(query, context)
        
        else:
            return self._build_generic_prompt(query, context)
    
    def _build_conversational_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build prompt for conversational queries"""
        
        has_flight = context.get('flight_data') is not None
        
        prompt = f"""You are SkySense Copilot, a helpful drone flight analysis assistant.

User said: "{query}"

"""
        
        if has_flight:
            flight_data = context['flight_data']
            summary = flight_data.get('summary', '')
            critical_count = flight_data.get('severity_counts', {}).get('critical', 0)
            
            prompt += f"""Flight data is loaded:
- Status: {"⚠️ CRITICAL" if critical_count > 0 else "✅ Good"}
- Issues: {critical_count} critical, {flight_data.get('total_insights', 0)} total

Respond naturally and briefly. Offer to help analyze the flight or answer questions.
"""
        else:
            prompt += """No flight data is loaded yet.

Respond naturally and briefly. Let user know you can help with:
- Flight log analysis (when they provide a log)
- General drone/PX4 knowledge questions
"""
        
        prompt += f"\nYour response (2-3 sentences max):"
        
        return prompt
    
    def _build_flight_data_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build prompt for flight data queries"""
        
        flight_data = context.get('flight_data', {})
        
        if not flight_data or 'error' in flight_data:
            return f"""User asked: "{query}"

No flight data available. Respond: "Please load a flight log first using: python main.py copilot <log_file>"
"""
        
        prompt = f"""You are SkySense Copilot analyzing a drone flight log.

User Query: "{query}"

Flight Data Summary:
{flight_data.get('summary', 'No summary')}

Key Metrics:
- Critical Issues: {flight_data.get('severity_counts', {}).get('critical', 0)}
- Warnings: {flight_data.get('severity_counts', {}).get('warning', 0)}
- Total Insights: {flight_data.get('total_insights', 0)}

Issue Breakdown:
"""
        
        # Add type counts
        for itype, count in flight_data.get('type_counts', {}).items():
            if itype not in ['phase', 'timeline', 'summary'] and count > 0:
                prompt += f"- {itype}: {count}\n"
        
        # Add critical events
        critical = flight_data.get('critical_events', [])
        if critical:
            prompt += f"\nTop Critical Events:\n"
            for event in critical[:3]:
                prompt += f"- {event['type']} at t={event['t_start']:.1f}s: {event['text'][:60]}\n"
        
        # Add specific data if available
        specific = flight_data.get('specific_data', {})
        if specific:
            prompt += f"\nQuery-Specific Data:\n"
            for key, events in specific.items():
                if events:
                    prompt += f"- {key}: {len(events)} event(s)\n"
                    for evt in events[:2]:
                        prompt += f"  • {evt['text'][:60]}\n"
        
        prompt += """

Analyze this data and answer the user's query concisely.
- If no issues found, say "✅ No issues detected in this category"
- If issues found, list them with timestamps
- Keep response under 150 words
- Use bullet points for clarity
- End with a helpful tip

Your response:"""
        
        return prompt
    
    def _build_knowledge_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build prompt for knowledge queries"""
        
        knowledge_docs = context.get('knowledge_docs', [])
        
        prompt = f"""You are SkySense Copilot, expert in PX4 autopilot and drone systems.

User Query: "{query}"

"""
        
        if knowledge_docs:
            prompt += "Relevant Knowledge Base:\n\n"
            for i, doc in enumerate(knowledge_docs, 1):
                prompt += f"--- Document {i} ---\n"
                prompt += f"{doc['content']}\n"
                prompt += f"Source: {doc['metadata'].get('source', 'unknown')}\n\n"
        else:
            prompt += "Note: No specific knowledge base articles found. Use general expertise.\n\n"
        
        prompt += """Answer Guidelines:
- Be concise (under 150 words)
- For terms: Definition + example
- For issues: Causes (2-3) + fixes (2-3)
- For troubleshooting: Actionable steps
- Use bullet points

Your response:"""
        
        return prompt
    
    def _build_generic_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Fallback generic prompt"""
        
        return f"""You are SkySense Copilot for drone flight analysis.

User asked: "{query}"

Provide a helpful, concise response (under 100 words).

Your response:"""
    
    def answer_query(self, query: str, flight_loaded: bool = False) -> Dict[str, Any]:
        """
        Complete query answering pipeline:
        1. Classify query
        2. Gather context
        3. Build structured prompt
        4. Get LLM answer
        
        Returns:
            {
                'answer': str,
                'classification': Dict,
                'confidence': float
            }
        """
        
        # Step 1: Classify
        classification = self.classify_query(query, flight_loaded)
        
        # Step 2: Gather context
        context = self.gather_context(classification, query)
        
        # Step 3: Build prompt
        structured_prompt = self.build_prompt(query, classification, context)
        
        # Step 4: Get answer
        try:
            response = self.llm.invoke(structured_prompt)
            answer = response.content.strip()
            
            return {
                'answer': answer,
                'classification': classification,
                'confidence': classification.get('confidence', 0.5),
                'debug': {
                    'query_type': classification['type'].value,
                    'used_flight_data': context.get('flight_data') is not None,
                    'used_knowledge': len(context.get('knowledge_docs', [])) > 0
                }
            }
            
        except Exception as e:
            return {
                'answer': f"I encountered an error: {e}. Please try rephrasing your question.",
                'classification': classification,
                'confidence': 0.0,
                'debug': {'error': str(e)}
            }
