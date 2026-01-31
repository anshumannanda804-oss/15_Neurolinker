"""
AI Suggestions Engine
Provides intelligent suggestions based on past decisions through conversational interface
"""

from groq import Groq
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import json

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)


def get_available_model():
    """Get available Groq model"""
    # Groq offers several models, using Llama 3.1 as default
    return 'llama-3.1-8b-instant'


class SuggestionEngine:
    """AI-powered suggestion system for improving decisions"""
    
    SUGGESTION_SYSTEM_PROMPT = """You are an expert decision coach and advisor. Your role is to help users improve their decisions by:

1. Analyzing their past decisions and patterns
2. Identifying potential improvements
3. Suggesting alternatives they might not have considered
4. Providing insights based on their documented reasoning
5. Helping them learn from their decisions

Guidelines:
- Be supportive and constructive, not judgmental
- Focus on helping them think better about decisions
- Ask questions to deepen their understanding
- Suggest options based on their constraints and goals
- Reference their past decisions when relevant
- Keep suggestions practical and actionable
- Use their language style
- Never be preachy or condescending
- Acknowledge the complexity of their decisions
- Help them see patterns in their decision-making

Remember: The user owns the decisions. You're here to help them think better, not to tell them what to do."""
    
    def __init__(self, past_decisions: List[Dict] = None):
        self.past_decisions = past_decisions or []
        self.conversation_history = []
        self.model = get_available_model()
        self.suggestions_made = []
    
    def set_past_decisions(self, decisions: List[Dict]):
        """Set the user's past decisions for context"""
        self.past_decisions = decisions
    
    def start_suggestion_session(self, current_decision: Dict = None) -> str:
        """Start a suggestion session"""
        self.conversation_history = []
        self.suggestions_made = []
        
        opening = "🚀 **AI Decision Suggestions**\n\n"
        
        if current_decision:
            opening += f"I can see you're working on: **{current_decision.get('description', 'a decision')}**\n\n"
        
        if self.past_decisions:
            opening += f"I've reviewed your {len(self.past_decisions)} past decisions and can help you:\n"
            opening += "- Improve your current thinking\n"
            opening += "- See patterns in your decision-making\n"
            opening += "- Consider alternatives you might have missed\n"
            opening += "- Build on what's worked before\n\n"
        else:
            opening += "I'm ready to help you think through this decision better.\n\n"
        
        opening += "**What would you like help with?** Tell me about your decision or ask for specific suggestions."
        
        return opening
    
    def get_suggestion_response(self, user_input: str, current_decision: Dict = None) -> str:
        """Get AI suggestion based on conversation and past decisions"""
        
        self.conversation_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Build context with past decisions
        context = self._build_decision_context(current_decision)
        
        # Build the prompt
        prompt = self.SUGGESTION_SYSTEM_PROMPT
        
        if context:
            prompt += f"\n\n**User's Decision Context:**\n{context}"
        
        # Add recent conversation history
        messages = [{"role": "user", "content": prompt}]
        
        # Add last few messages to keep conversation flowing
        for msg in self.conversation_history[-6:]:
            if msg['role'] == 'user':
                messages.append({"role": "user", "content": msg['content']})
            else:
                messages.append({"role": "assistant", "content": msg['content']})
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=600,
                temperature=0.8,
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Add to conversation history
            self.conversation_history.append({
                'role': 'assistant',
                'content': ai_response
            })
            
            # Track suggestions made
            if any(word in ai_response.lower() for word in ['suggest', 'consider', 'option', 'alternative']):
                self.suggestions_made.append({
                    'input': user_input,
                    'suggestion': ai_response
                })
            
            return ai_response
        
        except Exception as e:
            return f"I encountered an issue: {str(e)}. Let me try to help you another way."
    
    def _build_decision_context(self, current_decision: Dict = None) -> str:
        """Build context from past decisions and current decision"""
        context_parts = []
        
        if current_decision:
            context_parts.append("**Current Decision Being Considered:**")
            if current_decision.get('description'):
                context_parts.append(f"- What: {current_decision['description']}")
            if current_decision.get('goal'):
                context_parts.append(f"- Goal: {current_decision['goal']}")
            if current_decision.get('constraints'):
                constraints = ', '.join(current_decision['constraints']) if isinstance(current_decision['constraints'], list) else str(current_decision['constraints'])
                context_parts.append(f"- Constraints: {constraints}")
            context_parts.append("")
        
        if self.past_decisions:
            context_parts.append("**Past Decisions for Reference:**")
            
            # Show last 3 decisions with key details
            for i, decision in enumerate(self.past_decisions[-3:], 1):
                context_parts.append(f"\nDecision {i}: {decision.get('title', 'Untitled')}")
                if decision.get('description'):
                    context_parts.append(f"  - Description: {decision['description'][:100]}...")
                if decision.get('final_choice'):
                    context_parts.append(f"  - Choice: {decision['final_choice']}")
                if decision.get('reasoning'):
                    context_parts.append(f"  - Reasoning: {decision['reasoning'][:100]}...")
                if decision.get('outcome_status'):
                    context_parts.append(f"  - Status: {decision['outcome_status']}")
        
        return '\n'.join(context_parts) if context_parts else ""
    
    def get_decision_analysis(self, decision: Dict) -> str:
        """Get detailed analysis of a single decision"""
        analysis_prompt = f"""Analyze this decision and provide insights:

Title: {decision.get('title', 'Untitled')}
Description: {decision.get('description', '')}
Goal: {decision.get('goal', '')}
Constraints: {', '.join(decision.get('constraints', [])) if isinstance(decision.get('constraints'), list) else decision.get('constraints', '')}
Alternatives: {', '.join(decision.get('alternatives', [])) if isinstance(decision.get('alternatives'), list) else decision.get('alternatives', '')}
Final Choice: {decision.get('final_choice', '')}
Reasoning: {decision.get('reasoning', '')}
Outcome Status: {decision.get('outcome_status', '')}

Provide:
1. Strengths of this decision
2. Potential weaknesses or risks
3. What was done well in the decision process
4. Areas for improvement
5. Lessons learned

Be constructive and supportive."""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=800,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I couldn't analyze that decision: {str(e)}"
    
    def get_pattern_analysis(self) -> str:
        """Analyze patterns across all past decisions"""
        if not self.past_decisions:
            return "You haven't documented enough decisions yet for pattern analysis."
        
        # Prepare decision summaries
        decision_summaries = []
        for d in self.past_decisions[-5:]:  # Analyze last 5
            summary = f"- {d.get('title', 'Decision')}: {d.get('description', '')} (Goal: {d.get('goal', '')})"
            decision_summaries.append(summary)
        
        pattern_prompt = f"""Analyze patterns in these decisions:

{chr(10).join(decision_summaries)}

Identify:
1. Common themes or types of decisions
2. Decision-making patterns (strengths and weaknesses)
3. How constraints are typically handled
4. Evolution of decision-making over time
5. Blind spots or areas to develop
6. Recommendations for better decisions

Be specific and reference the actual decisions."""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": pattern_prompt}],
                max_tokens=800,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I couldn't analyze patterns: {str(e)}"
    
    def update_decision_with_suggestions(self, decision: Dict, new_insights: str) -> Dict:
        """Update decision with new insights from suggestions"""
        updated = decision.copy()
        
        # Add suggestion insights to reflection field
        current_reflection = updated.get('reflection', '')
        
        if current_reflection:
            updated['reflection'] = f"{current_reflection}\n\n[Updated with new suggestions]:\n{new_insights}"
        else:
            updated['reflection'] = f"[AI Suggestions]:\n{new_insights}"
        
        updated['updated_at'] = __import__('datetime').datetime.now().isoformat()
        
        return updated


class ViewDecisionsAssistant:
    """AI assistant for reviewing past decisions"""
    
    def __init__(self, decisions: List[Dict] = None):
        self.decisions = decisions or []
        self.model = get_available_model()
        self.conversation_history = []
    
    def start_review_session(self) -> str:
        """Start decision review session"""
        opening = "📖 **Review Your Past Decisions**\n\n"
        
        if self.decisions:
            opening += f"You have {len(self.decisions)} decisions recorded.\n\n"
            opening += "I can help you:\n"
            opening += "- Explore specific decisions in detail\n"
            opening += "- Understand your decision patterns\n"
            opening += "- Reflect on outcomes and learnings\n"
            opening += "- See how decisions connect\n\n"
            opening += "What would you like to review?"
        else:
            opening += "You haven't recorded any decisions yet. Start by recording a new decision!"
        
        return opening
    
    def get_review_response(self, user_query: str) -> str:
        """Get conversational response about decisions"""
        self.conversation_history.append({
            'role': 'user',
            'content': user_query
        })
        
        # Build context with decision summaries
        decision_context = self._build_decision_summaries()
        
        review_prompt = f"""You are a helpful decision review assistant. Help the user explore and learn from their past decisions.

User's Decisions:
{decision_context}

User Query: {user_query}

Provide thoughtful, conversational responses. Reference specific decisions when relevant. Help them see patterns and learn from their experiences."""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": review_prompt}],
                max_tokens=500,
                temperature=0.7,
            )
            
            ai_response = response.choices[0].message.content
            self.conversation_history.append({
                'role': 'assistant',
                'content': ai_response
            })
            
            return ai_response
        except Exception as e:
            return f"I couldn't process that: {str(e)}"
    
    def _build_decision_summaries(self) -> str:
        """Build summaries of all decisions"""
        summaries = []
        for i, d in enumerate(self.decisions[-5:], 1):
            summary = f"{i}. {d.get('title', 'Decision')} - {d.get('description', '')[:80]}...\n"
            if d.get('outcome_status'):
                summary += f"   Status: {d['outcome_status']}\n"
            summaries.append(summary)
        
        return '\n'.join(summaries) if summaries else "No decisions to review yet."
