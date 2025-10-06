"""
LLM utility functions for quick scope determination and other lightweight tasks.
"""

import json
import logging

logger = logging.getLogger(__name__)


async def quick_llm_call(prompt: str, temperature: float = 0.1, max_tokens: int = 500) -> str:
    """
    Make a quick LLM call for lightweight decisions like scope determination.

    Uses the research_agent's existing client for consistency.
    """
    try:
        # Import here to avoid circular dependencies
        from claude_agent_sdk import ClaudeSDKClient
        from multi_agent_research_system.agents.research_agent import ResearchAgent

        # Try to use actual LLM first
        try:
            client = ClaudeSDKClient()

            # Create a specialized agent for scope determination
            scope_prompt = f"""Analyze this research query and determine the appropriate scope level:

Query: "{prompt}"

Respond with ONLY a JSON object in this exact format:
{{"scope": "brief/default/comprehensive", "reasoning": "brief explanation", "confidence": "high/medium/low"}}

Guidelines:
- "brief": Only if query explicitly asks for summary, overview, or short response
- "comprehensive": Only if query explicitly asks for detailed analysis, extensive research, or in-depth study
- "default": For all other queries (especially news, current events, factual queries)
- Default bias: When uncertain, choose "default"
"""

            response = await client.send_message(scope_prompt, temperature=temperature, max_tokens=max_tokens)

            # Parse the JSON response
            import json
            response_data = json.loads(response.strip())

            # Ensure valid scope value
            if response_data.get("scope") not in ["brief", "default", "comprehensive"]:
                response_data["scope"] = "default"
                response_data["reasoning"] = "Invalid scope detected, using default"
                response_data["confidence"] = "low"

            # Add special requirements check
            prompt_lower = prompt.lower()
            special_req = ""

            if "focus on" in prompt_lower:
                import re
                match = re.search(r"focus on ([^.]+)", prompt_lower)
                if match:
                    special_req = f"Special focus: {match.group(1)}"
            elif "emphasize" in prompt_lower:
                match = re.search(r"emphasize ([^.]+)", prompt_lower)
                if match:
                    special_req = f"Special focus: {match.group(1)}"

            response_data["special_requirements"] = special_req

            logger.info(f"LLM scope determination result: {response_data}")
            return json.dumps(response_data)

        except Exception as llm_error:
            logger.warning(f"LLM call failed, using keyword fallback: {llm_error}")

            # Keyword-based fallback with bias toward default
            prompt_lower = prompt.lower()

            # More restrictive brief detection - removed "latest" and other inappropriate keywords
            if any(keyword in prompt_lower for keyword in ["brief", "summary", "quick", "short", "overview", "report_brief"]):
                scope = "brief"
                reasoning = "Detected explicit brief/summary keywords in query"
            elif any(keyword in prompt_lower for keyword in ["comprehensive", "detailed", "extensive", "thorough", "in-depth", "analysis"]):
                scope = "comprehensive"
                reasoning = "Detected comprehensive/detailed keywords in query"
            else:
                scope = "default"
                reasoning = "No specific scope keywords detected, defaulting to standard research"

            # Check for special requirements
            special_req = ""
            if "focus on" in prompt_lower:
                import re
                match = re.search(r"focus on ([^.]+)", prompt_lower)
                if match:
                    special_req = f"Special focus: {match.group(1)}"
            elif "emphasize" in prompt_lower:
                match = re.search(r"emphasize ([^.]+)", prompt_lower)
                if match:
                    special_req = f"Special focus: {match.group(1)}"

            response = {
                "scope": scope,
                "reasoning": reasoning,
                "special_requirements": special_req,
                "confidence": "medium"  # Fallback confidence
            }

            logger.info(f"Keyword fallback result: {response}")
            return json.dumps(response)

    except Exception as e:
        logger.error(f"Quick LLM call failed: {e}")
        # Return default response
        fallback = {
            "scope": "default",
            "reasoning": "LLM call failed, using default",
            "special_requirements": "",
            "confidence": "low"
        }
        return json.dumps(fallback)
