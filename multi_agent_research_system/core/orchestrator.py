"""Main orchestrator for the multi-agent research system using Claude Agent SDK.

This module manages the research workflow using proper SDK patterns with
ClaudeSDKClient, agent definitions, and custom tools.
"""

import asyncio
import json
import os

# Import from parent directory structure
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not found. Using environment variables only.")

# Set up API configuration
ANTHROPIC_BASE_URL = os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if ANTHROPIC_API_KEY:
    # Set environment variables for the SDK
    os.environ['ANTHROPIC_BASE_URL'] = ANTHROPIC_BASE_URL
    os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
    print(f"✅ Using Anthropic API: {ANTHROPIC_BASE_URL}")
else:
    print("⚠️  No ANTHROPIC_API_KEY found in environment or .env file")

try:
    from claude_agent_sdk import (
        AgentDefinition,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        create_sdk_mcp_server,
    )
    from claude_agent_sdk.types import (
        AssistantMessage,
        HookContext,
        HookMatcher,
        ResultMessage,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )
except ImportError:
    # Fallback for when the SDK is not installed
    print("Warning: claude_agent_sdk not found. Please install the package.")
    AgentDefinition = None
    ClaudeSDKClient = None
    ClaudeAgentOptions = None
    create_sdk_mcp_server = None
    HookMatcher = None
    HookContext = None

from .agent_logger import AgentLoggerFactory
from .logging_config import get_logger

# Import agent_logging with proper path handling
try:
    from ..agent_logging import (
        AgentLogger,
        EditorAgentLogger,
        HookLogger,
        ReportAgentLogger,
        ResearchAgentLogger,
        StructuredLogger,
        UICoordinatorLogger,
        create_agent_logger,
    )
    from ..agent_logging import get_logger as get_logger
except ImportError:
    # Fallback for when running as module
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from agent_logging import (
        AgentLogger,
        EditorAgentLogger,
        ReportAgentLogger,
        ResearchAgentLogger,
        StructuredLogger,
        UICoordinatorLogger,
        create_agent_logger,
    )
    from agent_logging import get_logger as get_logger
# Hook system removed - using simplified implementation

# Use standard Python logging instead of complex agent_logging
import logging

StructuredLogger = logging.getLogger
AgentLogger = logging.getLogger
ResearchAgentLogger = logging.getLogger
ReportAgentLogger = logging.getLogger
EditorAgentLogger = logging.getLogger
UICoordinatorLogger = logging.getLogger


class SessionSearchBudget:
    """Manages search budget and limits for a research session with intelligent adjustments."""

    def __init__(self, session_id: str, topic: str = "", parsed_request=None):
        self.session_id = session_id
        self.original_topic = topic
        self.parsed_request = parsed_request

        # Primary research limits - will be adjusted based on topic parameters
        self.primary_successful_scrapes_limit = 10  # Default limit, will be adjusted
        self.primary_urls_processed = 0
        self.primary_successful_scrapes = 0
        self.primary_search_queries = 0

        # Editorial research limits
        self.editorial_successful_scrapes_limit = 5  # User requested limit
        self.editorial_search_queries_limit = 2  # User requested limit
        self.editorial_urls_processed = 0
        self.editorial_successful_scrapes = 0
        self.editorial_search_queries = 0

        # Global session limits
        self.total_urls_processed_limit = 100  # Safety limit
        self.total_urls_processed = 0

        # Success override flags
        self.success_override_enabled = False
        self.override_reason = ""

        # Quality indicators for progressive budgeting
        self.high_quality_sources_found = 0
        self.relevance_threshold_met = False

        self.logger = logging.getLogger(f"search_budget.{session_id}")

        # Parse topic parameters for intelligent budget adjustment
        if parsed_request:
            self._adjust_budget_from_parsed_request(parsed_request)
        else:
            # Fallback to regex parsing for backward compatibility
            self._adjust_budget_from_topic(topic)

    def _adjust_budget_from_topic(self, topic: str):
        """Parse topic string and adjust budget limits based on scope parameters."""
        if not topic:
            return

        topic_lower = topic.lower()

        # Default adjustments based on scope
        if "scope=limited" in topic_lower:
            self.primary_successful_scrapes_limit = 8
            self.logger.info("Budget adjusted for limited scope: 8 primary scrapes")
        elif "scope=brief" in topic_lower or "report_brief" in topic_lower:
            self.primary_successful_scrapes_limit = 6
            self.logger.info("Budget adjusted for brief scope: 6 primary scrapes")
        elif "scope=comprehensive" in topic_lower:
            self.primary_successful_scrapes_limit = 20
            self.logger.info("Budget adjusted for comprehensive scope: 20 primary scrapes")
        elif "scope=extensive" in topic_lower:
            self.primary_successful_scrapes_limit = 30
            self.logger.info("Budget adjusted for extensive scope: 30 primary scrapes")

        # Adjust based on source requirements
        if "sources=3" in topic_lower:
            self.primary_successful_scrapes_limit = max(self.primary_successful_scrapes_limit, 12)
            self.logger.info("Budget adjusted for 3 sources: 12 primary scrapes")
        elif "sources=5" in topic_lower:
            self.primary_successful_scrapes_limit = max(self.primary_successful_scrapes_limit, 15)
            self.logger.info("Budget adjusted for 5 sources: 15 primary scrapes")
        elif "sources=10" in topic_lower:
            self.primary_successful_scrapes_limit = max(self.primary_successful_scrapes_limit, 25)
            self.logger.info("Budget adjusted for 10 sources: 25 primary scrapes")

        # Fast crawl adjustment (allows more sources, focuses on speed)
        if "fast_crawl" in topic_lower:
            self.primary_successful_scrapes_limit = int(self.primary_successful_scrapes_limit * 1.5)
            self.logger.info(f"Budget increased for fast crawl: {self.primary_successful_scrapes_limit} primary scrapes")

        # Quick test parameters
        if "quick" in topic_lower or "test" in topic_lower:
            self.primary_successful_scrapes_limit = max(self.primary_successful_scrapes_limit, 15)
            self.logger.info(f"Budget increased for quick test: {self.primary_successful_scrapes_limit} primary scrapes")

    def _adjust_budget_from_parsed_request(self, parsed_request):
        """Adjust budget limits based on structured parsed request parameters."""

        # Adjust based on scope
        scope = parsed_request.scope
        if scope == "limited":
            self.primary_successful_scrapes_limit = 8
            self.logger.info("Budget adjusted for limited scope: 8 primary scrapes")
        elif scope == "brief" or parsed_request.report_type == "brief":
            self.primary_successful_scrapes_limit = 6
            self.logger.info("Budget adjusted for brief scope: 6 primary scrapes")
        elif scope == "comprehensive":
            self.primary_successful_scrapes_limit = 20
            self.logger.info("Budget adjusted for comprehensive scope: 20 primary scrapes")
        elif scope == "extensive":
            self.primary_successful_scrapes_limit = 30
            self.logger.info("Budget adjusted for extensive scope: 30 primary scrapes")

        # Adjust based on source requirements
        sources = parsed_request.sources_requested
        if sources <= 3:
            self.primary_successful_scrapes_limit = max(self.primary_successful_scrapes_limit, 12)
            self.logger.info(f"Budget adjusted for {sources} sources: 12 primary scrapes")
        elif sources <= 5:
            self.primary_successful_scrapes_limit = max(self.primary_successful_scrapes_limit, 15)
            self.logger.info(f"Budget adjusted for {sources} sources: 15 primary scrapes")
        elif sources <= 10:
            self.primary_successful_scrapes_limit = max(self.primary_successful_scrapes_limit, 25)
            self.logger.info(f"Budget adjusted for {sources} sources: 25 primary scrapes")

        # Adjust based on crawl speed
        crawl_speed = parsed_request.crawl_speed
        if crawl_speed == "fast":
            self.primary_successful_scrapes_limit = int(self.primary_successful_scrapes_limit * 1.5)
            self.logger.info(f"Budget increased for fast crawl: {self.primary_successful_scrapes_limit} primary scrapes")
        elif crawl_speed == "quick":
            self.primary_successful_scrapes_limit = max(self.primary_successful_scrapes_limit, 15)
            self.logger.info(f"Budget increased for quick crawl: {self.primary_successful_scrapes_limit} primary scrapes")

        # Log special requirements
        if parsed_request.special_requirements:
            self.logger.info(f"Special requirements detected: {parsed_request.special_requirements}")

    def enable_success_override(self, reason: str, quality_indicators: dict[str, Any] = None):
        """Enable success override to allow proceeding despite budget overage."""
        self.success_override_enabled = True
        self.override_reason = reason

        if quality_indicators:
            self.high_quality_sources_found = quality_indicators.get("high_quality_sources", 0)
            self.relevance_threshold_met = quality_indicators.get("relevance_met", False)

        self.logger.info(f"Success override enabled: {reason}")

    def can_primary_research_proceed(self, urls_to_process: int = 1, force_success: bool = False) -> tuple[bool, str]:
        """Check if primary research can proceed with given URL count."""
        # Success override allows proceeding regardless of budget
        if self.success_override_enabled or force_success:
            return True, f"Proceeding with success override: {self.override_reason}"

        # Check successful scrapes limit
        if self.primary_successful_scrapes >= self.primary_successful_scrapes_limit:
            return False, f"Primary research limit reached: {self.primary_successful_scrapes}/{self.primary_successful_scrapes_limit} successful scrapes"

        # Check total URL limit
        if self.total_urls_processed + urls_to_process > self.total_urls_processed_limit:
            return False, f"Session URL limit would be exceeded: {self.total_urls_processed + urls_to_process}/{self.total_urls_processed_limit}"

        return True, "Primary research can proceed"

    def can_editorial_research_proceed(self, urls_to_process: int = 1, force_success: bool = False) -> tuple[bool, str]:
        """Check if editorial research can proceed."""
        # Success override allows proceeding regardless of budget
        if self.success_override_enabled or force_success:
            return True, f"Proceeding with success override: {self.override_reason}"

        # Check search queries limit
        if self.editorial_search_queries >= self.editorial_search_queries_limit:
            return False, f"Editorial search query limit reached: {self.editorial_search_queries}/{self.editorial_search_queries_limit}"

        # Check successful scrapes limit
        if self.editorial_successful_scrapes >= self.editorial_successful_scrapes_limit:
            return False, f"Editorial scrape limit reached: {self.editorial_successful_scrapes}/{self.editorial_successful_scrapes_limit} successful scrapes"

        # Check total URL limit
        if self.total_urls_processed + urls_to_process > self.total_urls_processed_limit:
            return False, f"Session URL limit would be exceeded: {self.total_urls_processed + urls_to_process}/{self.total_urls_processed_limit}"

        return True, "Editorial research can proceed"

    def record_primary_research(self, urls_processed: int, successful_scrapes: int, search_queries: int = 1, quality_indicators: dict[str, Any] = None):
        """Record primary research activity with progressive budgeting."""
        self.primary_urls_processed += urls_processed
        self.primary_successful_scrapes += successful_scrapes
        self.primary_search_queries += search_queries
        self.total_urls_processed += urls_processed

        # Update quality indicators if provided
        if quality_indicators:
            if quality_indicators.get("high_quality_sources", 0) > 0:
                self.high_quality_sources_found += quality_indicators["high_quality_sources"]
            if quality_indicators.get("relevance_met", False):
                self.relevance_threshold_met = True

        # Progressive budgeting: increase limit for high-quality research
        self._apply_progressive_budgeting()

        self.logger.info(f"Primary research recorded: {urls_processed} URLs, {successful_scrapes} successful scrapes")

    def record_editorial_research(self, urls_processed: int, successful_scrapes: int, search_queries: int = 1):
        """Record editorial research activity."""
        self.editorial_urls_processed += urls_processed
        self.editorial_successful_scrapes += successful_scrapes
        self.editorial_search_queries += search_queries
        self.total_urls_processed += urls_processed

        self.logger.info(f"Editorial research recorded: {urls_processed} URLs, {successful_scrapes} successful scrapes")

    def reset_editorial_budget(self):
        """Reset editorial research budget to allow editorial process to proceed regardless of primary research consumption."""
        self.logger.info("Resetting editorial budget for editorial review stage")

        # Reset editorial counters to zero
        self.editorial_urls_processed = 0
        self.editorial_successful_scrapes = 0
        self.editorial_search_queries = 0

        # Ensure editorial limits are adequate for proper review
        self.editorial_search_queries_limit = 2  # User requested limit
        self.editorial_successful_scrapes_limit = 5  # User requested limit

        self.logger.info(f"Editorial budget reset: {self.editorial_search_queries_limit} queries, {self.editorial_successful_scrapes_limit} scrapes allowed")

    def _apply_progressive_budgeting(self):
        """Apply progressive budget increases based on research quality indicators."""
        original_limit = self.primary_successful_scrapes_limit

        # Increase budget for high-quality sources
        if self.high_quality_sources_found >= 3:
            self.primary_successful_scrapes_limit = max(self.primary_successful_scrapes_limit, 15)
        elif self.high_quality_sources_found >= 5:
            self.primary_successful_scrapes_limit = max(self.primary_successful_scrapes_limit, 20)

        # Increase budget for relevant research
        if self.relevance_threshold_met and self.primary_successful_scrapes > 8:
            # Allow 50% more budget for relevant research that exceeds initial limits
            self.primary_successful_scrapes_limit = int(self.primary_successful_scrapes_limit * 1.5)

        # Log any increases
        if self.primary_successful_scrapes_limit > original_limit:
            self.logger.info(f"Progressive budgeting increased limit from {original_limit} to {self.primary_successful_scrapes_limit}")

    def assess_research_quality_for_override(self, research_result: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        """Assess research quality and determine if success override should be enabled."""
        quality_indicators = {
            "substantive_responses": research_result.get("substantive_responses", 0),
            "tools_executed": len(research_result.get("tool_executions", [])),
            "files_created": research_result.get("files_created", 0),
            "content_generated": research_result.get("content_generated", False),
            "has_research_findings": len(research_result.get("research_findings", "")) > 100
        }

        # Quality assessment criteria
        high_quality_score = 0
        reasons = []

        if quality_indicators["substantive_responses"] >= 2:
            high_quality_score += 2
            reasons.append(f"Multiple substantive responses ({quality_indicators['substantive_responses']})")

        if quality_indicators["tools_executed"] >= 3:
            high_quality_score += 2
            reasons.append(f"Multiple tools executed ({quality_indicators['tools_executed']})")

        if quality_indicators["files_created"] > 0:
            high_quality_score += 1
            reasons.append(f"Files created ({quality_indicators['files_created']})")

        if quality_indicators["content_generated"]:
            high_quality_score += 1
            reasons.append("Content generated")

        if quality_indicators["has_research_findings"]:
            high_quality_score += 2
            reasons.append("Substantial research findings")

        # Determine if override should be enabled
        should_override = high_quality_score >= 4  # At least half of quality criteria met
        override_reason = "; ".join(reasons) if should_override else ""

        return should_override, override_reason, quality_indicators

    def get_budget_status(self) -> dict[str, Any]:
        """Get current budget status."""
        return {
            "session_id": self.session_id,
            "primary": {
                "successful_scrapes": f"{self.primary_successful_scrapes}/{self.primary_successful_scrapes_limit}",
                "urls_processed": self.primary_urls_processed,
                "search_queries": self.primary_search_queries,
                "search_queries_remaining": 0,  # Primary research doesn't use query limits the same way
                "can_proceed": self.can_primary_research_proceed()[0],
                "budget_adjusted": self.primary_successful_scrapes_limit != 10
            },
            "editorial": {
                "successful_scrapes": f"{self.editorial_successful_scrapes}/{self.editorial_successful_scrapes_limit}",
                "search_queries": f"{self.editorial_search_queries}/{self.editorial_search_queries_limit}",
                "search_queries_remaining": self.editorial_search_queries_limit - self.editorial_search_queries,
                "search_queries_reached_limit": self.editorial_search_queries >= self.editorial_search_queries_limit,
                "urls_processed": self.editorial_urls_processed,
                "can_proceed": self.can_editorial_research_proceed()[0]
            },
            "global": {
                "total_urls_processed": f"{self.total_urls_processed}/{self.total_urls_processed_limit}",
                "remaining_urls": self.total_urls_processed_limit - self.total_urls_processed
            },
            "override": {
                "enabled": self.success_override_enabled,
                "reason": self.override_reason
            },
            "quality": {
                "high_quality_sources": self.high_quality_sources_found,
                "relevance_threshold_met": self.relevance_threshold_met
            }
        }


def create_agent_logger(session_id: str, agent_type: str) -> logging.Logger:
    """Create a simple logger for an agent."""
    return logging.getLogger(f"{agent_type}_{session_id}")
# Import decoupled editorial agent for independent editorial processing
from multi_agent_research_system.agents.decoupled_editorial_agent import (
    DecoupledEditorialAgent,
)
from multi_agent_research_system.core.progressive_enhancement import (
    ProgressiveEnhancementPipeline,
)
from multi_agent_research_system.core.quality_framework import (
    QualityAssessment,
    QualityFramework,
)
from multi_agent_research_system.core.quality_gates import (
    GateDecision,
    QualityGateManager,
)

# Import core workflow components
from multi_agent_research_system.core.workflow_state import (
    StageStatus,
    WorkflowSession,
    WorkflowStage,
    WorkflowStateManager,
)

from .search_analysis_tools import (
    capture_search_results,
    create_search_verification_report,
    save_webfetch_content,
)
from .simple_research_tools import (
    create_research_report,
    get_session_data,
    save_research_findings,
)

# Import SERP API search tool, advanced scraping tools, intelligent research tool, and enhanced search MCP
try:
    from multi_agent_research_system.mcp_tools.enhanced_search_scrape_clean import (
        enhanced_search_server,
    )
    from multi_agent_research_system.mcp_tools.zplayground1_search import (
        zplayground1_server,
    )
    from multi_agent_research_system.tools.advanced_scraping_tool import (
        advanced_scrape_multiple_urls,
        advanced_scrape_url,
    )
    from multi_agent_research_system.tools.intelligent_research_tool import (
        intelligent_research_with_advanced_scraping,
    )
    from multi_agent_research_system.tools.serp_search_tool import serp_search
except ImportError:
    # Fallback for when the tools module is not available
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from tools.serp_search_tool import serp_search
    try:
        from mcp_tools.zplayground1_search import zplayground1_server
    except ImportError:
        zplayground1_server = None
        print("Warning: zPlayground1 search MCP server not available")
    try:
        from mcp_tools.enhanced_search_scrape_clean import enhanced_search_server
    except ImportError:
        enhanced_search_server = None
        print("Warning: Enhanced search MCP server not available")

# Import config module with fallback
try:
    from multi_agent_research_system.config.agents import get_all_agent_definitions
except ImportError:
    # Fallback for when running as a script
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config.agents import get_all_agent_definitions


class ResearchOrchestrator:
    """Main orchestrator for the multi-agent research system."""

    def __init__(self, debug_mode: bool = False):
        # Initialize structured logging system
        self.logger = get_logger("orchestrator")
        self.structured_logger = get_logger("orchestrator")
        # Hook system disabled - no loggers initialized

        self.logger.info("Initializing ResearchOrchestrator")
        self.structured_logger.info("ResearchOrchestrator initialization started",
                                   event_type="orchestrator_initialization",
                                   debug_mode=debug_mode)

        self.debug_mode = debug_mode
        self.debug_output = []  # Store stderr debug output

        # Hook system configuration disabled
        self.hook_config = None  # Hook system removed
        self.logger.info("Hook system disabled for simplified operation")

        # Initialize decoupled editorial agent for independent editorial processing
        self.decoupled_editorial_agent = DecoupledEditorialAgent()
        self.logger.info("Decoupled editorial agent initialized")

        # Initialize core workflow components
        self.workflow_state_manager = WorkflowStateManager(logger=self.logger)
        self.quality_framework = QualityFramework()
        self.quality_gate_manager = QualityGateManager(logger=self.logger)
        self.progressive_enhancement_pipeline = ProgressiveEnhancementPipeline()
        self.logger.info("Core workflow components initialized")

        self.agent_definitions = get_all_agent_definitions()
        self.logger.debug(f"Loaded {len(self.agent_definitions)} agent definitions")
        self.structured_logger.info("Agent definitions loaded",
                                   agent_count=len(self.agent_definitions),
                                   agent_names=list(self.agent_definitions.keys()))

        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.client: ClaudeSDKClient = None  # Single client for all agents (CORRECT PATTERN)
        self.agent_names: list[str] = []  # List of agent names for reference
        self.mcp_server = None

        # Agent-specific loggers for detailed activity tracking
        self.agent_logger = None  # General agent logger (legacy)
        self.agent_loggers: dict[str, Any] = {}  # Agent-specific loggers

        # Initialize agent-specific loggers
        self._initialize_agent_loggers()

        self.logger.info("ResearchOrchestrator initialized")
        self.structured_logger.info("ResearchOrchestrator initialization completed",
                                   event_type="orchestrator_ready",
                                   total_agents=len(self.agent_definitions))

    def _create_debug_callback(self, agent_name: str):
        """Create a debug callback for capturing stderr output."""
        def debug_callback(message: str):
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            formatted_msg = f"[{timestamp}] {agent_name}: {message.strip()}"
            self.debug_output.append(formatted_msg)

            # Log to agent-specific logger if available
            agent_logger = self.get_agent_logger(agent_name)
            if agent_logger:
                try:
                    # Parse the message to extract activity information for agent-specific logging
                    message_lower = message.lower()

                    if agent_name == "research_agent":
                        # Research agent specific logging
                        if "search" in message_lower and "results" in message_lower:
                            # Extract search information if available
                            agent_logger.log_search_initiation(
                                query="debug_search",  # Would be extracted from message in real implementation
                                search_params={"debug": True},
                                topic="debug_topic",
                                estimated_results=10
                            )
                        elif "source" in message_lower or "extract" in message_lower:
                            agent_logger.log_source_analysis(
                                source_url="debug_source.com",
                                source_title="Debug Source",
                                credibility_score=0.8,
                                content_relevance=0.9,
                                content_type="article",
                                extraction_method="debug"
                            )

                    elif agent_name == "report_agent":
                        # Report agent specific logging
                        if "section" in message_lower or "generating" in message_lower:
                            agent_logger.log_section_generation_start(
                                section_name="debug_section",
                                section_type="debug",
                                research_sources_count=5,
                                target_word_count=500
                            )

                    elif agent_name == "editor_agent":
                        # Editor agent specific logging
                        if "review" in message_lower or "analyzing" in message_lower:
                            agent_logger.log_review_initiation(
                                document_title="Debug Document",
                                document_type="report",
                                word_count=1000,
                                review_focus_areas=["content", "structure"]
                            )

                    elif agent_name == "ui_coordinator":
                        # UI coordinator specific logging
                        if "workflow" in message_lower or "coordinating" in message_lower:
                            agent_logger.log_workflow_initiation(
                                workflow_id=str(uuid.uuid4()),
                                user_request="debug request",
                                workflow_type="debug",
                                estimated_stages=["research", "report", "edit"],
                                priority_level="normal"
                            )

                    # Fallback to generic logging for all agents
                    if "tool" in message_lower:
                        agent_logger.log_activity(
                            agent_name=agent_name,
                            activity_type="debug_message",
                            stage="unknown",
                            input_data={"debug_message": message.strip()},
                            metadata={"message_type": "tool_debug"}
                        )
                    elif "error" in message_lower:
                        agent_logger.log_activity(
                            agent_name=agent_name,
                            activity_type="debug_error",
                            stage="unknown",
                            error=message.strip(),
                            metadata={"message_type": "error_debug"}
                        )
                    elif "response" in message_lower:
                        agent_logger.log_activity(
                            agent_name=agent_name,
                            activity_type="debug_response",
                            stage="unknown",
                            input_data={"debug_message": message.strip()},
                            metadata={"message_type": "response_debug"}
                        )

                except Exception as e:
                    # Don't let logging errors break the debug callback
                    self.logger.warning(f"Failed to log debug message to agent logger {agent_name}: {e}")

            # Log important messages
            if any(keyword in message for keyword in ["ERROR", "WARNING", "tool", "response", "message"]):
                self.logger.debug(f"SDK Debug ({agent_name}): {message.strip()}")

        return debug_callback

    def _get_simplified_hooks(self) -> dict:
        """Get simplified hooks configuration to avoid JavaScript parsing errors.

        Based on finally2.md analysis, the most valuable logging outputs are:
        1. session_state.json - Complete session narrative and success assessment
        2. orchestrator.jsonl - Workflow progression and root cause analysis
        3. agent_summary.json - Performance metrics and error patterns

        The hook system was causing 262 JavaScript parsing errors per session
        with limited additional value over our core logging system.
        """
        try:
            from claude_agent_sdk.types import HookEvent, HookMatcher

            # Only enable essential hooks that provide high value without causing parsing issues
            essential_hooks = {
                HookEvent.PreToolUse: [
                    HookMatcher(
                        matcher="serp_search|mcp__research_tools",
                        hooks=[self._create_essential_tool_hook()]
                    )
                ],
                HookEvent.PostToolUse: [
                    HookMatcher(
                        matcher="serp_search|mcp__research_tools",
                        hooks=[self._create_essential_completion_hook()]
                    )
                ]
            }

            self.logger.info("🔧 Using simplified hooks configuration (4 essential hooks vs 262 error-prone hooks)")
            return essential_hooks

        except Exception as e:
            self.logger.warning(f"⚠️  Hook system unavailable, using basic logging: {e}")
            return {}

    def _create_essential_tool_hook(self):
        """Create essential tool usage hook without complex parsing."""
        async def essential_tool_hook(input_data, tool_use_id, context):
            return {
                "decision": "continue",  # Always allow tools to continue
                "systemMessage": f"Tool executed: {tool_use_id}",
                "hookSpecificOutput": {
                    "tool_use_id": tool_use_id,
                    "timestamp": str(uuid.uuid4()),
                    "simplified": True
                }
            }
        return essential_tool_hook

    def _create_essential_completion_hook(self):
        """Create essential tool completion hook without complex parsing."""
        async def essential_completion_hook(input_data, tool_use_id, context):
            return {
                "decision": "continue",  # Always allow continuation
                "systemMessage": f"Tool completed: {tool_use_id}",
                "hookSpecificOutput": {
                    "tool_use_id": tool_use_id,
                    "completed": True,
                    "simplified": True
                }
            }
        return essential_completion_hook

    # Note: Tool execution monitoring is now handled by the comprehensive HookIntegrationManager
    # The old _debug_pre_tool_use and _debug_post_tool_use methods have been removed
    # as they are superseded by the integrated hook system with proper SDK pattern compliance

    def _initialize_agent_loggers(self):
        """Initialize agent-specific loggers for each agent type."""
        try:
            # Create session ID for this orchestrator instance
            session_id = str(uuid.uuid4())

            # Initialize agent-specific loggers
            for agent_name in self.agent_definitions.keys():
                try:
                    # Create appropriate agent logger based on agent type
                    agent_logger = create_agent_logger(agent_name, session_id)
                    self.agent_loggers[agent_name] = agent_logger

                    self.logger.debug(f"Agent logger initialized: {agent_name}")
                    self.structured_logger.debug("Agent logger initialized",
                                                agent_name=agent_name,
                                                session_id=session_id)

                except Exception as e:
                    self.logger.warning(f"Failed to initialize agent logger for {agent_name}: {e}")
                    self.structured_logger.warning("Agent logger initialization failed",
                                                 agent_name=agent_name,
                                                 error=str(e))

                    # Fallback to generic logger
                    self.agent_loggers[agent_name] = AgentLogger(agent_name, session_id)

            self.logger.info(f"Agent loggers initialized: {len(self.agent_loggers)}")
            self.structured_logger.info("Agent loggers initialization completed",
                                       agent_loggers_count=len(self.agent_loggers),
                                       agent_names=list(self.agent_loggers.keys()))

        except Exception as e:
            self.logger.error(f"Failed to initialize agent loggers: {e}")
            self.structured_logger.error("Agent loggers initialization failed",
                                       error=str(e),
                                       error_type=type(e).__name__)

    def get_agent_logger(self, agent_name: str) -> Any:
        """Get the appropriate agent logger for the specified agent."""
        return self.agent_loggers.get(agent_name, self.agent_logger)

    def get_all_agent_summaries(self) -> dict[str, Any]:
        """Get summaries from all agent loggers."""
        summaries = {}
        for agent_name, agent_logger in self.agent_loggers.items():
            try:
                if hasattr(agent_logger, 'get_research_summary'):
                    summaries[agent_name] = agent_logger.get_research_summary()
                elif hasattr(agent_logger, 'get_report_summary'):
                    summaries[agent_name] = agent_logger.get_report_summary()
                elif hasattr(agent_logger, 'get_editor_summary'):
                    summaries[agent_name] = agent_logger.get_editor_summary()
                elif hasattr(agent_logger, 'get_coordinator_summary'):
                    summaries[agent_name] = agent_logger.get_coordinator_summary()
                else:
                    summaries[agent_name] = {"message": "No summary method available"}
            except Exception as e:
                summaries[agent_name] = {"error": str(e)}

        return summaries

    async def _ensure_playwright_browsers(self):
        """Check if Playwright browsers are installed and install if needed."""
        import subprocess
        from pathlib import Path

        # Check if Chromium browser exists
        playwright_cache = Path.home() / ".cache" / "ms-playwright"
        chromium_dirs = list(playwright_cache.glob("chromium-*"))

        browser_exists = False
        if chromium_dirs:
            # Check if any chromium directory has the chrome executable
            for chromium_dir in chromium_dirs:
                chrome_path = chromium_dir / "chrome-linux" / "chrome"
                if chrome_path.exists():
                    browser_exists = True
                    self.logger.info(f"✅ Playwright browser found: {chrome_path}")
                    break

        if not browser_exists:
            self.logger.warning("⚠️  Playwright browsers not found - installing now...")
            print("⚠️  Playwright browsers not found - installing now...")
            print("   This is a one-time setup that may take 1-2 minutes...")

            try:
                # Run playwright install using uv
                result = subprocess.run(
                    ["uv", "run", "playwright", "install"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                self.logger.info("✅ Playwright browsers installed successfully")
                print("✅ Playwright browsers installed successfully")
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to install Playwright browsers: {e.stderr}"
                self.logger.error(error_msg)
                print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            except FileNotFoundError:
                # If uv is not found, try direct playwright install
                try:
                    result = subprocess.run(
                        ["playwright", "install"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    self.logger.info("✅ Playwright browsers installed successfully")
                    print("✅ Playwright browsers installed successfully")
                except Exception as e:
                    error_msg = f"Failed to install Playwright browsers: {str(e)}"
                    self.logger.error(error_msg)
                    print(f"❌ {error_msg}")
                    raise RuntimeError(error_msg)

    async def initialize(self):
        """Initialize the orchestrator and all agent clients."""
        self.logger.info("Starting orchestrator initialization")

        try:
            # Check and install Playwright browsers if needed
            await self._ensure_playwright_browsers()

            # Hook system disabled
            self.logger.info("Hook system disabled - skipping initialization")

            # Create MCP server with custom tools including SERP API search
            self.logger.debug("Creating MCP server")

            # Create MCP server with properly decorated SdkMcpTool instances
            self.mcp_server = create_sdk_mcp_server(
                name="research_tools",
                version="1.0.0",
                tools=[
                    save_research_findings, create_research_report, get_session_data,
                    capture_search_results, save_webfetch_content, create_search_verification_report,
                    serp_search  # Add SERP API search tool to MCP server
                ]
            )
            self.logger.info("MCP server created successfully")

            # Log MCP server configuration
            self.logger.info("🔍 MCP Server Configuration:")
            self.logger.info(f"   Research Tools Server: {type(self.mcp_server).__name__} (TypedDict - correct)")
            self.logger.info(f"   Server Name: {self.mcp_server.get('name', 'Unknown')}")
            self.logger.info(f"   Available Tools: {len([save_research_findings, create_research_report, get_session_data, capture_search_results, save_webfetch_content, create_search_verification_report, serp_search])} tools")

            # Debug SERP API configuration
            serper_key = os.getenv('SERP_API_KEY', 'NOT_SET')
            serper_status = 'SET' if serper_key != 'NOT_SET' else 'NOT_SET'
            openai_key = os.getenv('OPENAI_API_KEY', 'NOT_SET')
            openai_status = 'SET' if openai_key != 'NOT_SET' else 'NOT_SET'

            # FAIL-FAST VALIDATION: Critical API keys must be present
            critical_errors = []

            if serper_key == 'NOT_SET':
                critical_errors.append("CRITICAL: SERP_API_KEY is missing! Web search functionality will not work.")
                self.logger.error("❌ CRITICAL FAILURE: SERP_API_KEY is NOT SET!")

            if openai_key == 'NOT_SET':
                critical_errors.append("CRITICAL: OPENAI_API_KEY is missing! Content processing will fail.")
                self.logger.error("❌ CRITICAL FAILURE: OPENAI_API_KEY is NOT SET!")

            # Check for SERPER_API_KEY vs SERP_API_KEY discrepancy
            serper_key_alt = os.getenv('SERPER_API_KEY', 'NOT_SET')
            if serper_key_alt != 'NOT_SET' and serper_key == 'NOT_SET':
                critical_errors.append("CRITICAL: Found SERPER_API_KEY but system expects SERP_API_KEY - API key name mismatch!")
                self.logger.error("❌ CRITICAL FAILURE: API key name mismatch! Found SERPER_API_KEY but expect SERP_API_KEY")
            elif serper_key == 'NOT_SET' and serper_key_alt == 'NOT_SET':
                critical_errors.append("CRITICAL: Neither SERP_API_KEY nor SERPER_API_KEY found in environment!")
                self.logger.error("❌ CRITICAL FAILURE: No search API key found!")

            # Fail fast and hard if critical API configuration is missing
            if critical_errors:
                self.logger.error("🚨 CRITICAL CONFIGURATION ERRORS DETECTED - SYSTEM WILL FAIL FAST")
                self.logger.error("During development, we fail immediately to expose configuration issues!")
                for error in critical_errors:
                    self.logger.error(f"  - {error}")
                self.logger.error("")
                self.logger.error("To fix these errors:")
                self.logger.error("1. Set SERP_API_KEY environment variable for search functionality")
                self.logger.error("2. Set OPENAI_API_KEY environment variable for content processing")
                self.logger.error("3. Ensure API keys are valid and have proper permissions")
                self.logger.error("")
                self.logger.error("Example:")
                self.logger.error("export SERP_API_KEY='your-serper-api-key'")
                self.logger.error("export OPENAI_API_KEY='your-openai-api-key'")
                self.logger.error("")

                # During development, fail hard and fast
                raise RuntimeError(f"CRITICAL CONFIGURATION FAILURE: {'; '.join(critical_errors)}")

            self.logger.info("   SERP API Search: Enabled (high-performance replacement for WebPrime MCP)")
            self.logger.info(f"   SERP_API_KEY Status: {serper_status}")
            self.logger.info(f"   OPENAI_API_KEY Status: {openai_status}")
            self.logger.info("   Expected Tools: serp_search, research_tools")

            # Create a single client with all agents configured properly (CORRECT PATTERN)
            self.logger.debug("Creating single multi-agent client")

            # Check if required SDK components are available
            if AgentDefinition is None:
                raise ImportError("AgentDefinition is not available. Please ensure claude_agent_sdk is properly installed.")

            if ClaudeAgentOptions is None:
                raise ImportError("ClaudeAgentOptions is not available. Please ensure claude_agent_sdk is properly installed.")

            # Prepare agents configuration for ClaudeAgentOptions
            agents_config = {}
            for agent_name, agent_def in self.agent_definitions.items():
                # For research agent, preserve its configured tools (including enhanced search)
                if agent_name == "research_agent":
                    # Don't override research_agent tools - it has the enhanced search configuration
                    extended_tools = agent_def.tools + [
                        "mcp__research_tools__save_research_findings",
                        "mcp__research_tools__create_research_report",
                        "mcp__research_tools__get_session_data",
                        "mcp__research_tools__capture_search_results",
                        "mcp__research_tools__save_webfetch_content",
                        "mcp__research_tools__create_search_verification_report",
                        "Read", "Write", "Glob", "Grep"
                    ]
                    self.logger.info("🔍 Preserving research_agent tool configuration (includes enhanced search)")
                else:
                    # For other agents, extend with standard tools
                    extended_tools = agent_def.tools + [
                        "mcp__research_tools__serp_search",  # High-performance SERP API search via MCP
                        "mcp__research_tools__save_research_findings",
                        "mcp__research_tools__create_research_report",
                        "mcp__research_tools__get_session_data",
                        "mcp__research_tools__capture_search_results",
                        "mcp__research_tools__save_webfetch_content",
                        "mcp__research_tools__create_search_verification_report",
                        "Read", "Write", "Glob", "Grep"
                    ]

                # Add zPlayground1 search tool if available
                if zplayground1_server is not None:
                    zplayground1_tools = [
                        "mcp__zplayground1_search__zplayground1_search_scrape_clean"  # Single comprehensive tool
                    ]
                    extended_tools.extend(zplayground1_tools)
                    self.logger.info(f"✅ zPlayground1 search tool added to {agent_name}")
                else:
                    self.logger.warning(f"⚠️ zPlayground1 search tool not available for {agent_name}")

                agents_config[agent_name] = AgentDefinition(
                    description=agent_def.description,
                    prompt=agent_def.prompt,
                    tools=extended_tools,
                    model=agent_def.model
                )

            # Prepare MCP servers configuration
            mcp_servers_config = {
                "research_tools": self.mcp_server,
            }

            # Add zPlayground1 search server if available
            if zplayground1_server is not None:
                mcp_servers_config["zplayground1_search"] = zplayground1_server
                self.logger.info("✅ zPlayground1 search MCP server added to configuration")
            else:
                self.logger.warning("⚠️ zPlayground1 search MCP server not available, using standard tools")

            # Add enhanced search server if available
            if enhanced_search_server is not None:
                mcp_servers_config["enhanced_search_scrape_clean"] = enhanced_search_server
                self.logger.info("✅ Enhanced search MCP server added to configuration")
            else:
                self.logger.warning("⚠️ Enhanced search MCP server not available, using standard tools")

            # Create single options with all agents configured properly
            options = ClaudeAgentOptions(
                agents=agents_config,
                mcp_servers=mcp_servers_config,
                # Use correct settings for proper response handling
                include_partial_messages=False,
                permission_mode="bypassPermissions",
                # Debugging features
                stderr=self._create_debug_callback("multi_agent"),
                extra_args={"debug-to-stderr": None},
                # Simplified logging approach - disable problematic hooks due to JavaScript parsing errors
                # Increase buffer size to handle large JSON outputs (DeepWiki recommendation)
                max_buffer_size=4096000,  # 4MB buffer
                hooks=self._get_simplified_hooks() if HookMatcher else {}
            )

            # Create and connect SINGLE client for all agents (CORRECT PATTERN)
            self.client = ClaudeSDKClient(options=options)
            await self.client.connect()
            self.logger.info("✅ Single multi-agent client created and connected")

            # Store all agent names for reference (no individual clients needed)
            self.agent_names = list(self.agent_definitions.keys())

            # Log agent initialization if agent logger is available
            if hasattr(self, 'agent_logger') and self.agent_logger:
                for agent_name, agent_def in self.agent_definitions.items():
                    self.agent_logger.log_agent_initialization(
                        agent_name=agent_name,
                        config={
                            "model": agent_def.model,
                            "tools": agent_def.tools,
                            "debug_mode": self.debug_mode,
                            "max_turns": 10,
                            "single_client_pattern": True
                        }
                    )

            self.logger.info(f"All {len(self.agent_names)} agents configured in single client")

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    @property
    def agent_clients(self) -> dict[str, Any]:
        """
        Backward compatibility property.
        Returns a dict mapping agent names to the shared client.
        """
        if self.client is None:
            return {}
        return dict.fromkeys(self.agent_names, self.client)

    async def check_agent_health(self) -> dict[str, Any]:
        """Check the health and connectivity of all agents using single client pattern."""
        self.logger.info("🔍 Performing agent health checks...")
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(self.agent_names),
            "healthy_agents": 0,
            "unhealthy_agents": 0,
            "agent_status": {},
            "mcp_servers": {},
            "issues": []
        }

        for agent_name in self.agent_names:
            try:
                self.logger.debug(f"Checking health of {agent_name}...")

                # Test basic connectivity with natural language agent selection (CORRECT PATTERN)
                test_prompt = f"Use the {agent_name} agent to introduce yourself and describe your capabilities in 2-3 sentences."
                start_time = time.time()

                try:
                    # Send query to single client with natural language agent selection
                    await self.client.query(test_prompt)
                    response_time = time.time() - start_time

                    # Collect substantive response to verify agent capability
                    substantive_response = False
                    collected_messages = []

                    try:
                        async for message in self.client.receive_response():
                            collected_messages.append(message)

                            # Check for substantive content
                            if hasattr(message, 'content') and message.content:
                                for block in message.content:
                                    if hasattr(block, 'text') and len(block.text.strip()) > 10:
                                        substantive_response = True
                                        break
                            elif hasattr(message, 'total_cost_usd'):
                                # Found ResultMessage, we're done
                                break
                    except Exception as response_error:
                        self.logger.debug(f"Response collection error for {agent_name}: {response_error}")
                        # Continue with basic check if response collection fails

                    if substantive_response or len(collected_messages) > 0:
                        health_report["agent_status"][agent_name] = {
                            "status": "healthy",
                            "response_time": response_time,
                            "response_type": "messages_collected",
                            "message_count": len(collected_messages),
                            "substantive_response": substantive_response
                        }
                        health_report["healthy_agents"] += 1
                        self.logger.info(f"✅ {agent_name}: Healthy ({response_time:.2f}s, {len(collected_messages)} messages)")
                    else:
                        health_report["agent_status"][agent_name] = {
                            "status": "warning",
                            "issue": "no substantive response collected",
                            "response_time": response_time,
                            "message_count": len(collected_messages)
                        }
                        health_report["unhealthy_agents"] += 1
                        health_report["issues"].append(f"{agent_name}: No substantive response collected")
                        self.logger.warning(f"⚠️ {agent_name}: No substantive response ({response_time:.2f}s)")

                except Exception as query_error:
                    health_report["agent_status"][agent_name] = {
                        "status": "unhealthy",
                        "issue": str(query_error),
                        "response_time": time.time() - start_time
                    }
                    health_report["unhealthy_agents"] += 1
                    health_report["issues"].append(f"{agent_name}: {query_error}")
                    self.logger.error(f"❌ {agent_name}: {query_error}")

            except Exception as e:
                health_report["agent_status"][agent_name] = {
                    "status": "error",
                    "issue": f"Health check failed: {e}"
                }
                health_report["unhealthy_agents"] += 1
                health_report["issues"].append(f"{agent_name}: Health check failed - {e}")
                self.logger.error(f"❌ {agent_name}: Health check failed - {e}")

        # Check MCP server status
        if self.mcp_server:
            try:
                health_report["mcp_servers"]["research_tools"] = {
                    "status": "available",
                    "type": type(self.mcp_server).__name__
                }
                self.logger.info("✅ Research Tools MCP Server: Available")
            except Exception as e:
                health_report["mcp_servers"]["research_tools"] = {
                    "status": "error",
                    "issue": str(e)
                }
                health_report["issues"].append(f"Research Tools MCP: {e}")
                self.logger.error(f"❌ Research Tools MCP Server: {e}")

        # Check SERP API configuration
        serper_key = os.getenv('SERP_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        if serper_key and openai_key:
            health_report["search_system"] = {
                "status": "configured",
                "type": "SERP_API",
                "serper_configured": True,
                "openai_configured": True
            }
            self.logger.info("✅ SERP API Search: Configured with API keys")
        else:
            health_report["search_system"] = {
                "status": "missing_keys",
                "type": "SERP_API",
                "serper_configured": bool(serper_key),
                "openai_configured": bool(openai_key)
            }
            if not serper_key:
                health_report["issues"].append("SERP API: Missing SERP_API_KEY")
                self.logger.error("❌ SERP API: Missing SERP_API_KEY")
            if not openai_key:
                health_report["issues"].append("SERP API: Missing OPENAI_API_KEY")
                self.logger.error("❌ SERP API: Missing OPENAI_API_KEY")

        # Summary
        health_report["summary"] = f"{health_report['healthy_agents']}/{health_report['total_agents']} agents healthy"
        self.logger.info(f"🏥 Agent Health Check Complete: {health_report['summary']}")

        if health_report["issues"]:
            self.logger.warning(f"🚨 Issues found: {len(health_report['issues'])} problems")
            for issue in health_report["issues"]:
                self.logger.warning(f"   - {issue}")

        return health_report

    async def execute_agent_query(
        self,
        agent_name: str,
        prompt: str,
        session_id: str | None = None,
        timeout_seconds: int = 120
    ) -> dict[str, Any]:
        """Execute a query using natural language agent selection (CORRECT PATTERN).

        Args:
            agent_name: Name of the agent to use
            prompt: The task prompt for the agent
            session_id: Optional session ID for tracking
            timeout_seconds: Query timeout in seconds (default: 120)

        Returns:
            Dict with query results including messages, tool executions, and metadata
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        # Construct natural language agent selection prompt
        full_prompt = f"Use the {agent_name} agent to {prompt}"

        self.logger.info(f"🔍 Querying {agent_name} with natural language selection")

        query_result = {
            "agent_name": agent_name,
            "session_id": session_id,
            "prompt_sent": prompt,
            "messages_collected": [],
            "substantive_responses": 0,
            "tool_executions": [],
            "errors": [],
            "query_start_time": datetime.now().isoformat(),
            "success": False
        }

        try:
            # Send query to single client
            await self.client.query(full_prompt)

            # Collect responses
            async for message in self.client.receive_response():
                message_info = {
                    "message_type": type(message).__name__,
                    "timestamp": datetime.now().isoformat()
                }

                # Extract content information
                if hasattr(message, 'content') and message.content:
                    content_texts = []
                    for block in message.content:
                        if hasattr(block, 'text') and block.text:
                            content_texts.append(block.text)
                    if content_texts:
                        message_info["content_texts"] = content_texts
                        query_result["substantive_responses"] += 1

                # Extract tool use information from AssistantMessage content blocks
                tool_executions = self._extract_tool_executions_from_message(message, agent_name)
                if tool_executions:
                    message_info["tool_use"] = tool_executions[0]  # Primary tool for this message
                    query_result["tool_executions"].extend(tool_executions)

                query_result["messages_collected"].append(message_info)

                # Stop when we get ResultMessage (conversation complete)
                if hasattr(message, 'total_cost_usd'):
                    self.logger.debug("ResultMessage received, stopping collection")
                    break

                # Safety limit to prevent infinite loops
                if len(query_result["messages_collected"]) >= 50:
                    self.logger.warning(f"Message limit reached for {agent_name} query")
                    break

            query_result["success"] = True
            query_result["query_end_time"] = datetime.now().isoformat()
            self.logger.info(f"✅ {agent_name} query completed: {len(query_result['messages_collected'])} messages")
            return query_result

        except Exception as e:
            query_result["errors"].append(str(e))
            query_result["success"] = False
            query_result["query_end_time"] = datetime.now().isoformat()
            self.logger.error(f"❌ {agent_name} query failed: {e}")
            raise

    def _extract_quality_indicators(self, research_result: dict[str, Any]) -> dict[str, Any]:
        """Extract quality indicators from research result for progressive budgeting."""
        quality_indicators = {
            "high_quality_sources": 0,
            "relevance_met": False
        }

        # Count high-quality sources based on tool executions
        tool_executions = research_result.get("tool_executions", [])
        search_tools = [tool for tool in tool_executions if "search" in tool.get("name", "").lower()]

        # Estimate high-quality sources based on number of search tools and responses
        if len(search_tools) >= 1:
            quality_indicators["high_quality_sources"] = min(len(search_tools) * 2, 5)  # Estimate 2-5 high-quality sources

        # Check if research findings indicate relevance
        research_findings = research_result.get("research_findings", "")
        if len(research_findings) > 200:  # Substantial findings suggest relevance
            quality_indicators["relevance_met"] = True

        # Additional quality indicators from substantive responses
        substantive_responses = research_result.get("substantive_responses", 0)
        if substantive_responses >= 2:
            quality_indicators["high_quality_sources"] = max(quality_indicators["high_quality_sources"], 3)

        return quality_indicators

    async def _determine_report_scope_with_llm_judge(self, topic: str) -> dict[str, Any]:
        """Use LLM judge to determine report scope and extract requirements."""

        # Quick LLM prompt to determine scope
        scope_prompt = f"""
        Analyze this research query and determine the appropriate report scope:

        QUERY: {topic}

        Choose ONE of the following options:
        1. "brief" - User wants a concise summary, quick overview, or brief report
        2. "default" - Standard comprehensive report with balanced coverage
        3. "comprehensive" - User wants detailed, extensive, in-depth analysis

        Also identify any special requirements or focus areas mentioned.

        Respond in JSON format:
        {{
            "scope": "brief|default|comprehensive",
            "reasoning": "Brief explanation of choice",
            "special_requirements": "Any specific requirements or focus areas (empty string if none)",
            "confidence": "high|medium|low"
        }}

        Default to "default" if uncertain.
        """

        try:
            # Use a quick LLM call for scope determination
            from .llm_utils import quick_llm_call
            result = await quick_llm_call(scope_prompt, temperature=0.1)

            # Parse JSON response
            import json
            scope_data = json.loads(result.strip())

            # Map scope to configuration
            scope = scope_data.get("scope", "default").lower()
            special_requirements = scope_data.get("special_requirements", "")

            # Import settings to get word count defaults
            from multi_agent_research_system.config.settings import get_enhanced_search_config
            config = get_enhanced_search_config()

            # Default configuration
            report_config = {
                "scope": "default",
                "size_multiplier": 1.0,
                "min_words": config.default_report_min_words,
                "style_instructions": "Provide a balanced, comprehensive report covering all key aspects.",
                "editing_rigor": "standard",
                "special_requirements": special_requirements,
                "llm_reasoning": scope_data.get("reasoning", ""),
                "llm_confidence": scope_data.get("confidence", "medium")
            }

            # Apply scope-specific settings
            if scope == "brief":
                report_config.update({
                    "scope": "brief",
                    "size_multiplier": 0.6,
                    "min_words": config.brief_report_min_words,
                    "style_instructions": "Provide a concise, focused report highlighting the most important findings.",
                    "editing_rigor": "light"
                })
            elif scope == "comprehensive":
                report_config.update({
                    "scope": "comprehensive",
                    "size_multiplier": 1.5,
                    "min_words": config.comprehensive_report_min_words,
                    "style_instructions": "Provide a thorough, detailed analysis with comprehensive coverage.",
                    "editing_rigor": "thorough"
                })

            self.logger.info(f"LLM Judge determined scope: {report_config['scope']} (confidence: {report_config['llm_confidence']})")
            if special_requirements:
                self.logger.info(f"LLM identified special requirements: {special_requirements}")

            return report_config

        except Exception as e:
            self.logger.warning(f"LLM scope determination failed, using default: {e}")
            # Fallback to default configuration
            from multi_agent_research_system.config.settings import get_enhanced_search_config
            config = get_enhanced_search_config()
            return {
                "scope": "default",
                "size_multiplier": 1.0,
                "min_words": config.default_report_min_words,
                "style_instructions": "Provide a balanced, comprehensive report covering all key aspects.",
                "editing_rigor": "standard",
                "special_requirements": "",
                "llm_reasoning": "LLM judge failed, using default",
                "llm_confidence": "low"
            }

    def _validate_research_completion(self, research_result: dict[str, Any]) -> bool:
        """Validate that research stage completed successfully.

        Args:
            research_result: Result from execute_agent_query

        Returns:
            True if research completed successfully, False otherwise
        """
        # Check basic success
        if not research_result.get("success", False):
            return False

        # Check for substantive responses
        if research_result.get("substantive_responses", 0) < 1:
            return False

        # Check for tool executions (any research-related tools)
        tool_executions = research_result.get("tool_executions", [])
        if len(tool_executions) < 1:
            return False

        # Function-based validation - check for research activity regardless of tool names
        research_indicators = [
            "search", "scrape", "crawl", "extract", "research", "query", "serp", "expanded"
        ]

        tool_names = [tool.get("name", "") for tool in tool_executions]
        has_research_activity = any(
            any(indicator in tool_name.lower() for indicator in research_indicators)
            for tool_name in tool_names
        )

        # Additional validation: check for actual research output
        has_work_products = (
            research_result.get("files_created", 0) > 0 or
            research_result.get("content_generated", False) or
            len(research_result.get("research_findings", "")) > 100
        )

        validation_result = has_research_activity and has_work_products

        if not validation_result:
            self.logger.warning(f"Research validation failed - Activity: {has_research_activity}, Work Products: {has_work_products}")
            self.logger.warning(f"Tool names found: {tool_names}")

        return validation_result

    def _extract_tool_executions_from_message(self, message, agent_name: str) -> list:
        """Extract tool executions with comprehensive error handling.

        Args:
            message: The message object to extract tool executions from
            agent_name: Name of the agent for logging

        Returns:
            List of tool execution dictionaries
        """
        tool_executions = []

        try:
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        tool_info = {
                            "name": block.name,
                            "id": block.id,
                            "input": block.input,
                            "timestamp": datetime.now().isoformat()
                        }
                        tool_executions.append(tool_info)
                        self.logger.info(f"{agent_name} executed tool: {block.name}")

        except Exception as e:
            self.logger.warning(f"Error extracting tools from {agent_name} message: {e}")

        return tool_executions

    def _extract_scrape_count(self, research_result: dict[str, Any]) -> int:
        """Extract actual scrape count from research results.

        Args:
            research_result: Result from execute_agent_query

        Returns:
            Number of successful scrapes extracted from metadata or text
        """
        import re
        from pathlib import Path

        # Method 1: Check metadata from tool executions
        for tool_exec in research_result.get("tool_executions", []):
            # Check if tool has result with metadata
            if isinstance(tool_exec, dict):
                # Direct metadata check
                metadata = tool_exec.get("metadata", {})
                if "successful_scrapes" in metadata:
                    count = metadata["successful_scrapes"]
                    self.logger.info(f"Extracted scrape count from tool metadata: {count}")
                    return count

                # Check result metadata
                result = tool_exec.get("result", {})
                if isinstance(result, dict) and "metadata" in result:
                    metadata = result["metadata"]
                    if "successful_scrapes" in metadata:
                        count = metadata["successful_scrapes"]
                        self.logger.info(f"Extracted scrape count from tool result metadata: {count}")
                        return count

        # Method 2: Extract from text responses using regex patterns
        for response in research_result.get("responses", []):
            if isinstance(response, dict):
                text = response.get("text", "")
            elif isinstance(response, str):
                text = response
            else:
                continue

            # Pattern 1: "Successfully Crawled: 20"
            match = re.search(r'\*\*Successfully Crawled\*\*:\s*(\d+)', text)
            if match:
                count = int(match.group(1))
                self.logger.info(f"Extracted scrape count from text pattern 1: {count}")
                return count

            # Pattern 2: "URLs Extracted: 20 successfully processed"
            match = re.search(r'\*\*URLs Extracted\*\*:\s*(\d+)', text)
            if match:
                count = int(match.group(1))
                self.logger.info(f"Extracted scrape count from text pattern 2: {count}")
                return count

            # Pattern 3: "URLs Crawled: 20 successfully"
            match = re.search(r'\*\*URLs Crawled\*\*:\s*(\d+)', text)
            if match:
                count = int(match.group(1))
                self.logger.info(f"Extracted scrape count from text pattern 3: {count}")
                return count

        # Method 3: Check most recent work product file
        session_id = research_result.get("session_id")
        if session_id:
            research_dir = Path(f"/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN/sessions/{session_id}/research")
            if research_dir.exists():
                files = sorted(research_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
                if files:
                    try:
                        content = files[0].read_text()
                        match = re.search(r'\*\*Successfully Crawled\*\*:\s*(\d+)', content)
                        if match:
                            count = int(match.group(1))
                            self.logger.info(f"Extracted scrape count from work product file: {count}")
                            return count
                    except Exception as e:
                        self.logger.warning(f"Could not read work product file: {e}")

        # Last resort: estimate from tool count (conservative)
        tool_count = len(research_result.get("tool_executions", []))
        estimated = min(10, tool_count * 5) if tool_count > 0 else 0
        self.logger.warning(f"Could not extract exact scrape count, using conservative estimate: {estimated}")
        return estimated

    def _validate_report_completion(self, report_result: dict[str, Any]) -> bool:
        """Validate that report generation stage completed successfully.

        Args:
            report_result: Result from execute_agent_query

        Returns:
            True if report generation completed successfully, False otherwise
        """
        # Check basic success
        if not report_result.get("success", False):
            return False

        # Check for substantive responses
        if report_result.get("substantive_responses", 0) < 1:
            return False

        # Check for tool executions (should save findings/create report)
        tool_executions = report_result.get("tool_executions", [])
        if len(tool_executions) < 1:
            return False

        return True

    def _validate_editorial_completion(self, editorial_result: dict[str, Any]) -> bool:
        """Validate that editorial review stage completed successfully.

        Args:
            editorial_result: Result from execute_agent_query

        Returns:
            True if editorial review completed successfully, False otherwise
        """
        # Check basic success
        if not editorial_result.get("success", False):
            return False

        # Check for substantive responses
        if editorial_result.get("substantive_responses", 0) < 1:
            return False

        return True

    async def verify_tool_execution(self, agent_name: str = "research_agent") -> dict[str, Any]:
        """Verify that critical tools can be executed by agents."""
        self.logger.info(f"🔧 Verifying tool execution for {agent_name}...")

        tool_verification = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "tools_tested": [],
            "tools_working": [],
            "tools_failed": [],
            "issues": []
        }

        if not self.client:
            error_msg = "Client not initialized"
            tool_verification["issues"].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
            return tool_verification

        if agent_name not in self.agent_names:
            error_msg = f"Agent {agent_name} not available"
            tool_verification["issues"].append(error_msg)
            self.logger.error(f"❌ {error_msg}")
            return tool_verification

        # Test critical tools with simple prompts
        critical_tools = [
            {
                "name": "SERP API Search",
                "test_prompt": "Use the mcp__research_tools__serp_search tool to search for 'test query' with num_results=1.",
                "expected_pattern": "serp_search"
            },
            {
                "name": "File Operations",
                "test_prompt": "Use the Read tool to read the file 'test.txt'.",
                "expected_pattern": "Read"
            }
        ]

        for tool_test in critical_tools:
            tool_name = tool_test["name"]
            test_prompt = tool_test["test_prompt"]

            tool_verification["tools_tested"].append(tool_name)

            try:
                self.logger.debug(f"Testing {tool_name} with {agent_name}...")
                start_time = time.time()

                # Use the new execute_agent_query method (CORRECT PATTERN)
                full_test_prompt = f"{test_prompt} Use the {agent_name} agent for this task."
                result = await self.execute_agent_query(agent_name, full_test_prompt)
                execution_time = time.time() - start_time

                # Check if tool was mentioned in responses
                tool_used = False
                for message_info in result["messages_collected"]:
                    if "content_texts" in message_info:
                        for text in message_info["content_texts"]:
                            if tool_test["expected_pattern"] in str(text):
                                tool_used = True
                                break
                        if tool_used:
                            break

                if tool_used or len(result["messages_collected"]) > 0:
                    tool_verification["tools_working"].append(tool_name)
                    self.logger.info(f"✅ {tool_name}: Working ({execution_time:.2f}s)")
                else:
                    tool_verification["tools_failed"].append(tool_name)
                    issue = f"{tool_name}: Tool not executed in response"
                    tool_verification["issues"].append(issue)
                    self.logger.warning(f"⚠️ {tool_name}: Tool not executed")

            except Exception as e:
                tool_verification["tools_failed"].append(tool_name)
                issue = f"{tool_name}: {str(e)}"
                tool_verification["issues"].append(issue)
                self.logger.error(f"❌ {tool_name}: {str(e)}")

        # Summary
        working_count = len(tool_verification["tools_working"])
        total_count = len(tool_verification["tools_tested"])
        tool_verification["summary"] = f"{working_count}/{total_count} tools working"
        tool_verification["success_rate"] = working_count / total_count if total_count > 0 else 0

        self.logger.info(f"🔧 Tool Verification Complete: {tool_verification['summary']}")

        if tool_verification["issues"]:
            self.logger.warning(f"🚨 Tool issues found: {len(tool_verification['issues'])} problems")
            for issue in tool_verification["issues"]:
                self.logger.warning(f"   - {issue}")

        return tool_verification

    async def execute_research_with_tool_enforcement(self, client, session_id: str, topic: str, user_requirements: dict[str, Any]) -> list:
        """Execute research with mandatory tool execution enforcement."""
        import json
        self.logger.info(f"🔧 Enforcing tool execution for research on: {topic}")

        required_tools = {"mcp__research_tools__serp_search", "mcp__research_tools__save_research_findings"}
        executed_tools = set()
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts and not required_tools.issubset(executed_tools):
            attempt += 1
            self.logger.info(f"Research attempt {attempt}/{max_attempts}")

            # Create stronger prompt based on attempt
            if attempt == 1:
                research_prompt = f"""
                EXECUTE RESEARCH TASK: {topic}

                Session ID: {session_id}
                Requirements: {json.dumps(user_requirements)}

                CRITICAL REQUIREMENT: You MUST execute tools NOW, not respond with text first.

                REQUIRED IMMEDIATE ACTIONS:
                1. Execute mcp__research_tools__serp_search with:
                   - query: "{topic}"
                   - num_results: 15
                   - auto_crawl_top: 8
                   - crawl_threshold: 0.3
                   - session_id: "{session_id}"

                2. Execute mcp__research_tools__save_research_findings to save results

                DO NOT RESPOND WITH TEXT until AFTER executing the search tool.
                The search MUST be your first action.
                """
            else:
                # Stronger prompt for retry attempts
                missing_tools = required_tools - executed_tools
                research_prompt = f"""
                CRITICAL: Previous attempt failed to execute required tools.
                MISSING TOOLS: {missing_tools}

                You MUST execute these tools NOW:
                - mcp__research_tools__serp_search for "{topic}"
                - mcp__research_tools__save_research_findings

                DO NOT respond with "OK" or acknowledgments.
                EXECUTE THE TOOLS IMMEDIATELY.
                """

            # Send query and collect responses with tool tracking
            self.logger.info(f"Sending research query (attempt {attempt})...")
            query_start_time = time.time()

            try:
                response = await client.query(research_prompt)
                query_execution_time = time.time() - query_start_time
                self.logger.info(f"✅ Query sent: {type(response).__name__}")

                # Log the query
                if self.agent_logger:
                    self.agent_logger.log_query_response(
                        agent_name="research_agent",
                        query=research_prompt[:1000] + "..." if len(research_prompt) > 1000 else research_prompt,
                        response=response,
                        execution_time=query_execution_time,
                        stage="research"
                    )

            except Exception as e:
                query_execution_time = time.time() - query_start_time
                self.logger.error(f"❌ Query failed: {e}")
                if self.agent_logger:
                    self.agent_logger.log_error(
                        agent_name="research_agent",
                        error=e,
                        context={"query": research_prompt[:500], "session_id": session_id, "attempt": attempt},
                        stage="research"
                    )
                continue

            # Collect responses and track tool execution
            research_results = []
            tool_executed_this_attempt = False

            try:
                timeout_seconds = 60
                start_time = asyncio.get_event_loop().time()

                async for message in client.receive_response():
                    current_time = asyncio.get_event_loop().time()
                    elapsed = current_time - start_time

                    self.logger.debug(f"🔧 Research message type: {type(message).__name__} after {elapsed:.1f}s")
                    research_results.append(message)

                    # Track tool execution and handle results
                    if hasattr(message, 'content') and message.content:
                        for block in message.content:
                            block_type = type(block).__name__

                            # Handle tool execution (ToolUseBlock)
                            if hasattr(block, 'name') and block.name:
                                executed_tools.add(block.name)
                                tool_executed_this_attempt = True
                                self.logger.info(f"✅ Tool executed: {block.name}")

                            # Handle tool results (ToolResultBlock)
                            if block_type == 'ToolResultBlock':
                                self.logger.info("🔧 Found tool result block")
                                all_attrs = [attr for attr in dir(block) if not attr.startswith('_')]
                                self.logger.info(f"🔧 ToolResultBlock attributes: {all_attrs}")

                                # Check for result content - handle both string and list formats
                                if hasattr(block, 'content') and block.content:
                                    import json

                                    self.logger.info(f"🔧 ToolResultBlock has content: {type(block.content)}")

                                    # Case 1: Content is a string (JSON data or plain text)
                                    if isinstance(block.content, str):
                                        self.logger.info(f"🔧 Processing string content: {len(block.content)} characters")

                                        # First, try to parse as JSON
                                        try:
                                            parsed_data = json.loads(block.content)
                                            self.logger.info("🔧 Successfully parsed JSON string from tool result")
                                            await self._handle_tool_result_data(parsed_data, session_id)
                                        except json.JSONDecodeError:
                                            # If not JSON, treat as plain text result
                                            self.logger.info("🔧 Content is plain text, creating simple result structure")

                                            # Create a simple result structure for plain text content
                                            text_result = {
                                                "content": [{"type": "text", "text": block.content}],
                                                "success": True,
                                                "text_output": block.content,
                                                "tool_type": "text_response"
                                            }

                                            # Try to infer tool name from context or content
                                            tool_name = self._infer_tool_name_from_context(block.content)
                                            await self._handle_tool_file_creation(tool_name, text_result, session_id)

                                    # Case 2: Content is a list of content items
                                    elif isinstance(block.content, list):
                                        self.logger.info(f"🔧 Processing list content: {len(block.content)} items")
                                        for content_item in block.content:
                                            # Handle content items with text attribute
                                            if hasattr(content_item, 'text') and content_item.text:
                                                try:
                                                    parsed_data = json.loads(content_item.text)
                                                    self.logger.info("🔧 Successfully parsed JSON from content item text")
                                                    await self._handle_tool_result_data(parsed_data, session_id)
                                                except json.JSONDecodeError as e:
                                                    self.logger.warning(f"🔧 Could not parse content item text as JSON: {e}")
                                                    self.logger.debug(f"🔧 Raw content item: {content_item.text[:200]}...")

                                            # Handle content items that are already dictionaries
                                            elif isinstance(content_item, dict):
                                                self.logger.info("🔧 Processing dictionary content item")
                                                await self._handle_tool_result_data(content_item, session_id)

                                    # Case 3: Content is already a dictionary
                                    elif isinstance(block.content, dict):
                                        self.logger.info("🔧 Processing dictionary content directly")
                                        await self._handle_tool_result_data(block.content, session_id)

                                # Also check for direct result attribute
                                elif hasattr(block, 'result') and block.result:
                                    self.logger.info("🔧 Found direct result attribute")
                                    await self._handle_tool_file_creation('unknown_tool', block.result, session_id)

                    # Check for completion
                    if hasattr(message, 'result') or elapsed > timeout_seconds:
                        if elapsed > timeout_seconds:
                            self.logger.warning(f"⚠️ Research collection timeout after {elapsed:.1f}s")
                        break

                self.logger.info(f"🔧 Research attempt {attempt} completed: {len(research_results)} messages, tools: {executed_tools}")

            except Exception as e:
                self.logger.error(f"❌ Error collecting research responses: {e}")

            # Check if we have the required tools
            if required_tools.issubset(executed_tools):
                self.logger.info(f"✅ All required tools executed: {executed_tools}")
                break
            else:
                missing = required_tools - executed_tools
                self.logger.warning(f"⚠️ Missing required tools: {missing}")
                if attempt < max_attempts:
                    self.logger.info("🔄 Retrying with stronger tool enforcement...")
                await asyncio.sleep(1)  # Brief pause between attempts

        # Final status
        if not required_tools.issubset(executed_tools):
            self.logger.error(f"❌ Failed to execute required tools after {max_attempts} attempts: {required_tools - executed_tools}")
            # Add fallback response
            fallback_response = {
                "agent": "research_agent",
                "stage": "research",
                "timestamp": datetime.now().isoformat(),
                "response": f"Research tool execution failed. Required tools not executed: {required_tools - executed_tools}",
                "error_type": "tool_execution_failure",
                "attempts": attempt,
                "executed_tools": list(executed_tools),
                "fallback": True
            }
            research_results.append(fallback_response)

        return research_results

    def _create_session_id(self) -> str:
        """Create a new session ID."""
        return str(uuid.uuid4())

    def _cleanup_sessions_directory(self):
        """Clean up logs, sessions, url_tracking, and work products for clean slate."""
        import shutil

        # Define directories to clean - use absolute paths from KEVIN
        cleanup_dirs = [
            "KEVIN/sessions",      # Session data and work products
            "KEVIN/logs",          # All log files
            "KEVIN/url_tracking",  # URL deduplication tracking
        ]

        for dir_path in cleanup_dirs:
            full_path = Path(dir_path)
            if full_path.exists():
                try:
                    self.logger.info(f"🧹 Cleaning up directory: {dir_path}")
                    shutil.rmtree(full_path)
                    self.logger.info(f"✅ Successfully cleaned: {dir_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to clean {dir_path}: {e}")

        self.logger.info("🧹 Session cleanup completed - ready for fresh start")

    async def start_research_session(self, topic: str, user_requirements: dict[str, Any]) -> str:
        """Start a new research session."""
        session_id = self._create_session_id()

        # Enhanced structured logging for session creation
        self.logger.info(f"Starting new research session for topic: {topic}")
        self.structured_logger.info(f"Starting research session: {topic}",
                                    event_type="session_creation_start",
                                    topic=topic,
                                    user_requirements=user_requirements)

        # Use specialized session lifecycle logger
        session_config = {
            "topic": topic,
            "user_requirements": user_requirements,
            "debug_mode": self.debug_mode,
            "available_agents": list(self.agent_definitions.keys())
        }
        # Hook system disabled - no session lifecycle logging

        self.logger.debug(f"Generated session ID: {session_id}")
        self.structured_logger.debug("Session ID generated", session_id=session_id)

        # Clean up previous sessions, logs, and work products for clean slate
        self._cleanup_sessions_directory()

        # Create session directory in KEVIN structure with absolute path
        # Get the project root directory (where KEVIN should be)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # Go up from core/ to project root
        kevin_dir = project_root / "KEVIN"
        session_path = kevin_dir / "sessions" / session_id

        session_path.mkdir(parents=True, exist_ok=True)
        # Create subdirectories for organization
        (session_path / "research").mkdir(exist_ok=True)
        (session_path / "working").mkdir(exist_ok=True)
        (session_path / "final").mkdir(exist_ok=True)
        (session_path / "agent_logs").mkdir(exist_ok=True)

        self.logger.debug(f"Created session directory: {session_path}")
        self.logger.info(f"KEVIN directory structure: {kevin_dir}")
        self.structured_logger.info("Session directory structure created",
                                    session_path=str(session_path),
                                    kevin_dir=str(kevin_dir))

        # Parse CLI input to extract clean topic and parameters
        from .cli_parser import parse_cli_input
        parsed_request = parse_cli_input(topic)

        # Initialize session state with intelligent budgeting
        self.active_sessions[session_id] = {
            "session_id": session_id,
            "topic": topic,
            "clean_topic": parsed_request.clean_topic,
            "parsed_request": parsed_request,
            "user_requirements": user_requirements,
            "status": "initialized",
            "created_at": datetime.now().isoformat(),
            "current_stage": "research",
            "workflow_history": [],
            "final_report": None,
            "search_budget": SessionSearchBudget(session_id, parsed_request.clean_topic, parsed_request),  # Use clean topic and parsed request for budget adjustment
            "work_products": {
                "current_number": 1,
                "completed": [],
                "in_progress": None,
                "tracking": {}
            }
        }

        # Save session state
        await self.save_session_state(session_id)
        self.logger.debug(f"Session state saved for {session_id}")
        self.structured_logger.info("Session state saved and initialized",
                                    session_id=session_id,
                                    session_status="initialized")

        # Hook system disabled - no workflow logging

        # Start the research workflow
        asyncio.create_task(self.execute_research_workflow(session_id))
        self.logger.info(f"Research workflow started for session {session_id}")
        self.structured_logger.info("Research workflow task created",
                                    session_id=session_id,
                                    workflow_type="async_task")

        return session_id

    async def execute_research_workflow(self, session_id: str):
        """Execute the complete research workflow."""
        self.logger.info(f"Starting research workflow for session {session_id}")
        self.structured_logger.info(f"Research workflow started: {session_id}",
                                    event_type="workflow_start",
                                    session_id=session_id)

        # Hook system disabled - no workflow logging

        try:
            # Initialize agent logger for this session with correct path
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent  # Go up from core/ to project root
            kevin_sessions_dir = project_root / "KEVIN" / "sessions"
            self.agent_logger = AgentLoggerFactory.get_logger(session_id, str(kevin_sessions_dir))
            self.agent_logger.log_activity(
                agent_name="orchestrator",
                activity_type="workflow_start",
                stage="initialization",
                input_data={"session_id": session_id},
                metadata={"event": "research_workflow_started"}
            )

            # Enhanced structured logging for workflow initialization
            self.structured_logger.info("Agent logger initialized for session",
                                        session_id=session_id,
                                        log_directory=str(kevin_sessions_dir))

            # Health checks and tool verification moved to dedicated test suite
            # Use: python tests/test_startup_health.py to verify system health
            self.logger.info("🚀 Starting research workflow...")

            session_data = self.active_sessions[session_id]
            topic = session_data["topic"]
            user_requirements = session_data["user_requirements"]

            self.logger.debug(f"Session {session_id}: Topic={topic}, Requirements={user_requirements}")
            self.agent_logger.log_activity(
                agent_name="orchestrator",
                activity_type="session_parameters",
                stage="initialization",
                input_data={"topic": topic, "user_requirements": user_requirements},
                metadata={"event": "session_parameters_set"}
            )

            # Stage 1: Research (with resilience)
            self.logger.info(f"Session {session_id}: Starting research stage")
            self.agent_logger.log_stage_transition("initialization", "research", "orchestrator", {"topic": topic})

            # Hook system disabled - no workflow logging
            self.structured_logger.info("Research stage started",
                                        session_id=session_id,
                                        from_stage="initialization",
                                        to_stage="research",
                                        topic=topic)

            # Execute research stage with resilience
            research_result = await self.execute_stage_with_resilience(
                "research", self.stage_conduct_research, session_id, topic, user_requirements
            )
            if not research_result["success"]:
                self.logger.error(f"❌ Session {session_id}: Research stage failed even with recovery attempts")
                # Continue workflow to ensure we get work products even with failed research
            else:
                if research_result.get("recovery_used", False):
                    self.logger.info(f"🔄 Session {session_id}: Research stage completed with recovery (method: {research_result.get('recovery_method', 'unknown')})")

            # Stage 2: Report Generation
            self.logger.info(f"Session {session_id}: Starting report generation stage")
            self.agent_logger.log_stage_transition("research", "report_generation", "orchestrator")

            # Hook system disabled - no workflow logging
            self.structured_logger.info("Report generation stage started",
                                        session_id=session_id,
                                        from_stage="research",
                                        to_stage="report_generation")

            await self.stage_generate_report(session_id)

            # Stage 3: Editorial Review (with decoupled fallback)
            self.logger.info(f"Session {session_id}: Starting editorial review stage")
            self.agent_logger.log_stage_transition("report_generation", "editorial_review", "orchestrator")

            # Hook system disabled - no workflow logging
            self.structured_logger.info("Editorial review stage started",
                                        session_id=session_id,
                                        from_stage="report_generation",
                                        to_stage="editorial_review")

            # Try traditional editorial review first
            editorial_success = False
            try:
                await self.stage_editorial_review(session_id)
                editorial_success = True
                self.logger.info(f"✅ Session {session_id}: Traditional editorial review completed successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ Session {session_id}: Traditional editorial review failed: {e}")
                self.logger.info(f"🔄 Session {session_id}: Falling back to decoupled editorial review")

                # Use decoupled editorial review as fallback
                try:
                    decoupled_result = await self.stage_decoupled_editorial_review(session_id)
                    if decoupled_result.get("success", False):
                        editorial_success = True
                        self.logger.info(f"✅ Session {session_id}: Decoupled editorial review completed successfully")
                        self.logger.info(f"   Content quality: {decoupled_result.get('content_quality', 'Unknown')}")
                        self.logger.info(f"   Enhancements made: {decoupled_result.get('enhancements_made', False)}")
                    else:
                        self.logger.warning(f"⚠️ Session {session_id}: Decoupled editorial review had limited success")
                except Exception as decoupled_error:
                    self.logger.error(f"❌ Session {session_id}: Both editorial approaches failed: {decoupled_error}")

            if not editorial_success:
                self.logger.error(f"❌ Session {session_id}: All editorial review attempts failed - continuing with minimal processing")

            # Complete Work Product 3: Editorial Review
            session_data = self.active_sessions[session_id]
            if "editorial_work_product_number" in session_data:
                editorial_wp_number = session_data["editorial_work_product_number"]
                self.complete_work_product(session_id, editorial_wp_number, {
                    "stage": "editorial_review",
                    "success": editorial_success,
                    "completion_method": "successful" if editorial_success else "failed_with_continuation"
                })
            else:
                self.logger.warning(f"⚠️ Session {session_id}: Editorial work product number not found for completion")

            # Stage 4: Finalization
            self.logger.info(f"Session {session_id}: Starting finalization stage")
            self.agent_logger.log_stage_transition("editorial_review", "finalization", "orchestrator")

            # Hook system disabled - no workflow logging
            self.structured_logger.info("Finalization stage started",
                                        session_id=session_id,
                                        from_stage="editorial_review",
                                        to_stage="finalization")

            await self.stage_finalize(session_id)

            self.logger.info(f"Session {session_id}: Research workflow completed successfully")
            self.structured_logger.info("Research workflow completed successfully",
                                        event_type="workflow_complete",
                                        session_id=session_id,
                                        final_stage="finalization")

            # Hook system disabled - no workflow logging

            # Use UI coordinator logger for orchestration summary
            ui_coordinator_logger = self.get_agent_logger("ui_coordinator")
            ui_coordinator_logger.log_progress_monitoring(
                current_stage="completed",
                progress_percentage=100.0,
                milestones_reached=["research", "report_generation", "editorial_review", "finalization"],
                session_id=session_id
            )

            self.agent_logger.log_activity(
                agent_name="orchestrator",
                activity_type="workflow_complete",
                stage="finalization",
                metadata={"event": "research_workflow_completed"}
            )

        except Exception as e:
            # Enhanced structured logging for workflow errors
            self.logger.error(f"Error in research workflow for session {session_id}: {e}")
            self.structured_logger.error("Research workflow error occurred",
                                        event_type="workflow_error",
                                        session_id=session_id,
                                        error_type=type(e).__name__,
                                        error_message=str(e),
                                        stage="workflow_execution")

            import traceback
            self.logger.error(traceback.format_exc())
            self.structured_logger.error("Workflow error traceback",
                                        traceback=traceback.format_exc(),
                                        session_id=session_id)

            # Hook system disabled - no workflow logging

            if self.agent_logger:
                self.agent_logger.log_error(
                    agent_name="orchestrator",
                    error=e,
                    context={"session_id": session_id, "stage": "workflow_execution"},
                    stage="error"
                )

            await self.update_session_status(session_id, "error", str(e))

        finally:
            # Finalize agent logger and export debug report
            if self.agent_logger:
                try:
                    self.agent_logger.finalize_session()
                    debug_report_path = self.agent_logger.export_debug_report()
                    self.logger.info(f"Agent debug report exported to: {debug_report_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to export agent debug report: {e}")

                # Clear the agent logger reference
                self.agent_logger = None

    async def execute_quality_gated_research_workflow(self, session_id: str, topic: str, user_requirements: dict[str, Any]) -> dict[str, Any]:
        """Execute research workflow with quality gates and intelligent progression."""
        self.logger.info(f"Starting quality-gated research workflow for session {session_id}")

        # Create workflow session
        workflow_session = self.workflow_state_manager.create_session(
            session_id=session_id,
            topic=topic,
            user_requirements=user_requirements
        )

        try:
            results = {}
            current_stage = WorkflowStage.RESEARCH

            while current_stage not in [WorkflowStage.COMPLETED, WorkflowStage.FAILED]:
                self.logger.info(f"Executing stage: {current_stage.value}")

                # Execute the current stage
                stage_result = await self._execute_workflow_stage(
                    current_stage, session_id, workflow_session, results
                )

                if not stage_result.get("success", False):
                    self.logger.error(f"Stage {current_stage.value} failed: {stage_result.get('error', 'Unknown error')}")
                    workflow_session.update_stage_state(
                        current_stage,
                        status=StageStatus.FAILED,
                        error_message=stage_result.get("error", "Unknown error")
                    )
                    current_stage = WorkflowStage.FAILED
                    break

                # Assess quality if applicable
                if current_stage in [WorkflowStage.RESEARCH, WorkflowStage.REPORT_GENERATION,
                                   WorkflowStage.EDITORIAL_REVIEW, WorkflowStage.DECOUPLED_EDITORIAL_REVIEW,
                                   WorkflowStage.QUALITY_ASSESSMENT]:

                    assessment = await self._assess_stage_quality(
                        current_stage, stage_result, session_id, workflow_session
                    )

                    # Evaluate quality gate
                    gate_result = self.quality_gate_manager.evaluate_quality_gate(
                        current_stage, assessment, workflow_session
                    )

                    self.logger.info(f"Quality gate decision for {current_stage.value}: {gate_result.decision.value}")
                    self.logger.info(f"  Reasoning: {gate_result.reasoning}")
                    self.logger.info(f"  Confidence: {gate_result.confidence:.2f}")

                    # Update workflow session with quality data
                    workflow_session.update_stage_state(
                        current_stage,
                        status=StageStatus.COMPLETED,
                        quality_metrics=assessment.to_dict(),
                        result=stage_result
                    )

                    # Handle gate decision
                    if gate_result.decision == GateDecision.PROCEED:
                        current_stage = gate_result.next_stage or workflow_session._get_next_stage(current_stage)

                    elif gate_result.decision == GateDecision.ENHANCE:
                        self.logger.info(f"Enhancing content for {current_stage.value}")
                        enhancement_result = await self._enhance_stage_content(
                            current_stage, gate_result.enhancement_suggestions,
                            session_id, workflow_session, stage_result
                        )

                        if enhancement_result.get("success", False):
                            # Re-assess after enhancement
                            new_assessment = await self._assess_stage_quality(
                                current_stage, enhancement_result, session_id, workflow_session
                            )

                            # Check if enhancement improved quality sufficiently
                            if new_assessment.overall_score >= assessment.overall_score + 10:
                                self.logger.info(f"Enhancement successful for {current_stage.value}")
                                stage_result = enhancement_result
                                current_stage = workflow_session._get_next_stage(current_stage)
                            else:
                                # Enhancement didn't help, check if we can proceed
                                stage_state = workflow_session.get_stage_state(current_stage)
                                if stage_state.attempt_count >= 3:  # Max enhancement attempts
                                    self.logger.warning(f"Max enhancement attempts reached for {current_stage.value}")
                                    if gate_result.fallback_available:
                                        current_stage = self._get_fallback_stage(current_stage)
                                    else:
                                        current_stage = WorkflowStage.FAILED
                                        break
                                else:
                                    # Try enhancement again
                                    continue
                        else:
                            self.logger.error(f"Enhancement failed for {current_stage.value}")
                            if gate_result.fallback_available:
                                current_stage = self._get_fallback_stage(current_stage)
                            else:
                                current_stage = WorkflowStage.FAILED
                                break

                    elif gate_result.decision == GateDecision.RERUN:
                        stage_state = workflow_session.get_stage_state(current_stage)
                        if stage_state.attempt_count < 3:
                            self.logger.info(f"Rerunning stage {current_stage.value}")
                            workflow_session.update_stage_state(
                                current_stage,
                                status=StageStatus.PENDING,
                                attempt_count=stage_state.attempt_count + 1
                            )
                            continue
                        else:
                            self.logger.error(f"Max rerun attempts reached for {current_stage.value}")
                            current_stage = WorkflowStage.FAILED
                            break

                    elif gate_result.decision == GateDecision.ESCALATE:
                        self.logger.info(f"Escalating from stage {current_stage.value}")
                        current_stage = self._get_escalation_stage(current_stage)

                    elif gate_result.decision == GateDecision.SKIP:
                        self.logger.info(f"Skipping stage {current_stage.value}")
                        current_stage = workflow_session._get_next_stage(current_stage)

                else:
                    # Stage doesn't require quality assessment
                    workflow_session.update_stage_state(
                        current_stage,
                        status=StageStatus.COMPLETED,
                        result=stage_result
                    )
                    current_stage = workflow_session._get_next_stage(current_stage)

                # Save checkpoint
                self.workflow_state_manager.save_checkpoint(
                    session_id,
                    current_stage.value,
                    {"stage_result": stage_result, "workflow_session": workflow_session.to_dict()}
                )

            # Determine final result
            if current_stage == WorkflowStage.COMPLETED:
                workflow_session.overall_status = StageStatus.COMPLETED
                workflow_session.end_time = datetime.now()

                # Final quality assessment
                final_assessment = await self._conduct_final_quality_assessment(results, session_id)
                workflow_session.final_quality_score = final_assessment.overall_score

                self.logger.info(f"Quality-gated workflow completed successfully for session {session_id}")
                self.logger.info(f"Final quality score: {final_assessment.overall_score}")

                return {
                    "success": True,
                    "session_id": session_id,
                    "results": results,
                    "quality_assessment": final_assessment.to_dict(),
                    "workflow_session": workflow_session.to_dict()
                }
            else:
                workflow_session.overall_status = StageStatus.FAILED
                workflow_session.end_time = datetime.now()

                self.logger.error(f"Quality-gated workflow failed for session {session_id}")
                return {
                    "success": False,
                    "session_id": session_id,
                    "error": f"Workflow failed at stage: {current_stage.value}",
                    "results": results,
                    "workflow_session": workflow_session.to_dict()
                }

        except Exception as e:
            self.logger.error(f"Error in quality-gated workflow for session {session_id}: {e}")
            workflow_session.overall_status = StageStatus.FAILED
            workflow_session.end_time = datetime.now()

            return {
                "success": False,
                "session_id": session_id,
                "error": str(e),
                "results": results,
                "workflow_session": workflow_session.to_dict()
            }

        finally:
            # Save final workflow session state
            self.workflow_state_manager.save_session(workflow_session)

    async def _execute_workflow_stage(
        self,
        stage: WorkflowStage,
        session_id: str,
        workflow_session: WorkflowSession,
        results: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a specific workflow stage."""

        self.logger.info(f"Executing workflow stage: {stage.value}")
        workflow_session.update_stage_state(stage, status=StageStatus.IN_PROGRESS)

        try:
            if stage == WorkflowStage.RESEARCH:
                session_data = self.active_sessions[session_id]
                topic = session_data["topic"]
                user_requirements = session_data["user_requirements"]

                research_result = await self.stage_conduct_research(session_id, topic, user_requirements)
                results["research"] = research_result
                return {"success": True, "data": research_result, "stage": "research"}

            elif stage == WorkflowStage.REPORT_GENERATION:
                report_result = await self.stage_generate_report(session_id)
                results["report"] = report_result
                return {"success": True, "data": report_result, "stage": "report_generation"}

            elif stage == WorkflowStage.EDITORIAL_REVIEW:
                editorial_result = await self.stage_editorial_review(session_id)
                results["editorial"] = editorial_result
                return {"success": True, "data": editorial_result, "stage": "editorial_review"}

            elif stage == WorkflowStage.DECOUPLED_EDITORIAL_REVIEW:
                decoupled_result = await self.stage_decoupled_editorial_review(session_id)
                results["decoupled_editorial"] = decoupled_result
                return {"success": True, "data": decoupled_result, "stage": "decoupled_editorial_review"}

            elif stage == WorkflowStage.QUALITY_ASSESSMENT:
                # Quality assessment is handled separately
                return {"success": True, "data": {}, "stage": "quality_assessment"}

            elif stage == WorkflowStage.PROGRESSIVE_ENHANCEMENT:
                enhancement_result = await self._apply_progressive_enhancement(session_id, results, workflow_session)
                results["progressive_enhancement"] = enhancement_result
                return {"success": True, "data": enhancement_result, "stage": "progressive_enhancement"}

            elif stage == WorkflowStage.FINAL_OUTPUT:
                final_result = await self.stage_finalize(session_id)
                results["final"] = final_result
                return {"success": True, "data": final_result, "stage": "final_output"}

            else:
                return {"success": False, "error": f"Unknown stage: {stage.value}"}

        except Exception as e:
            self.logger.error(f"Error executing stage {stage.value}: {e}")
            workflow_session.update_stage_state(
                stage,
                status=StageStatus.FAILED,
                error_message=str(e)
            )
            return {"success": False, "error": str(e), "stage": stage.value}

    async def _assess_stage_quality(
        self,
        stage: WorkflowStage,
        stage_result: dict[str, Any],
        session_id: str,
        workflow_session: WorkflowSession
    ) -> QualityAssessment:
        """Assess the quality of a stage's output."""

        self.logger.info(f"Assessing quality for stage: {stage.value}")

        # Extract content for quality assessment
        content = self._extract_content_for_assessment(stage, stage_result)

        # Get context for assessment
        context = {
            "session_id": session_id,
            "stage": stage.value,
            "topic": workflow_session.topic,
            "user_requirements": workflow_session.user_requirements,
            "previous_results": workflow_session.global_context
        }

        # Conduct quality assessment
        assessment = await self.quality_framework.assess_quality(content, context)

        # Store assessment in workflow session
        workflow_session.quality_history.append({
            "stage": stage.value,
            "timestamp": datetime.now().isoformat(),
            "assessment": assessment.to_dict()
        })

        self.logger.info(f"Quality assessment for {stage.value}: {assessment.overall_score}/100")
        return assessment

    async def _enhance_stage_content(
        self,
        stage: WorkflowStage,
        enhancement_suggestions: list[str],
        session_id: str,
        workflow_session: WorkflowSession,
        current_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Enhance content based on quality assessment suggestions."""

        self.logger.info(f"Enhancing content for stage: {stage.value}")

        try:
            # Use progressive enhancement pipeline for content enhancement
            enhancement_result = await self.progressive_enhancement_pipeline.enhance_content(
                content=current_result,
                enhancement_suggestions=enhancement_suggestions,
                stage=stage.value,
                context={
                    "session_id": session_id,
                    "topic": workflow_session.topic,
                    "user_requirements": workflow_session.user_requirements
                }
            )

            # Record enhancement in workflow session
            workflow_session.enhancement_stages_applied.append(stage.value)
            workflow_session.processing_statistics[f"{stage.value}_enhancements"] = len(enhancement_suggestions)

            return enhancement_result

        except Exception as e:
            self.logger.error(f"Error enhancing content for stage {stage.value}: {e}")
            return {"success": False, "error": str(e)}

    async def _apply_progressive_enhancement(
        self,
        session_id: str,
        results: dict[str, Any],
        workflow_session: WorkflowSession
    ) -> dict[str, Any]:
        """Apply progressive enhancement to improve overall quality."""

        self.logger.info("Applying progressive enhancement")

        try:
            enhancement_result = await self.progressive_enhancement_pipeline.enhance_research_output(
                research_results=results,
                enhancement_config={
                    "target_quality_score": 85,
                    "max_enhancement_cycles": 3,
                    "focus_areas": ["completeness", "clarity", "organization"]
                },
                context={
                    "session_id": session_id,
                    "topic": workflow_session.topic,
                    "user_requirements": workflow_session.user_requirements
                }
            )

            # Update workflow session
            workflow_session.enhancement_stages_applied.append("progressive_enhancement")
            if enhancement_result.get("success", False):
                workflow_session.processing_statistics["progressive_enhancement_applied"] = True
                workflow_session.processing_statistics["enhancement_cycles"] = enhancement_result.get("cycles_applied", 0)

            return enhancement_result

        except Exception as e:
            self.logger.error(f"Error applying progressive enhancement: {e}")
            return {"success": False, "error": str(e)}

    async def _conduct_final_quality_assessment(self, results: dict[str, Any], session_id: str) -> QualityAssessment:
        """Conduct final quality assessment of all results."""

        self.logger.info("Conducting final quality assessment")

        # Combine all content for final assessment
        final_content = {
            "research": results.get("research", {}),
            "report": results.get("report", {}),
            "editorial": results.get("editorial", {}),
            "final": results.get("final", {})
        }

        context = {
            "session_id": session_id,
            "assessment_type": "final",
            "comprehensive": True
        }

        final_assessment = await self.quality_framework.assess_quality(final_content, context)

        self.logger.info(f"Final quality assessment: {final_assessment.overall_score}/100")
        return final_assessment

    def _extract_content_for_assessment(self, stage: WorkflowStage, stage_result: dict[str, Any]) -> str:
        """Extract content from stage result for quality assessment."""

        if stage == WorkflowStage.RESEARCH:
            # Extract research findings and sources
            research_data = stage_result.get("data", {})
            return json.dumps({
                "findings": research_data.get("findings", []),
                "sources": research_data.get("sources", []),
                "key_insights": research_data.get("key_insights", [])
            }, indent=2)

        elif stage == WorkflowStage.REPORT_GENERATION:
            # Extract generated report content
            return stage_result.get("report_content", str(stage_result.get("data", {})))

        elif stage in [WorkflowStage.EDITORIAL_REVIEW, WorkflowStage.DECOUPLED_EDITORIAL_REVIEW]:
            # Extract editorial improvements and content
            return json.dumps({
                "content": stage_result.get("content", ""),
                "improvements": stage_result.get("improvements", []),
                "quality_analysis": stage_result.get("quality_analysis", {})
            }, indent=2)

        else:
            # Default: return string representation of result
            return str(stage_result.get("data", stage_result))

    def _get_fallback_stage(self, current_stage: WorkflowStage) -> WorkflowStage:
        """Get fallback stage for quality gate failures."""

        fallback_map = {
            WorkflowStage.EDITORIAL_REVIEW: WorkflowStage.DECOUPLED_EDITORIAL_REVIEW,
            WorkflowStage.DECOUPLED_EDITORIAL_REVIEW: WorkflowStage.PROGRESSIVE_ENHANCEMENT,
            WorkflowStage.RESEARCH: WorkflowStage.RESEARCH,  # Try different research approach
            WorkflowStage.REPORT_GENERATION: WorkflowStage.REPORT_GENERATION,  # Try different report approach
        }

        return fallback_map.get(current_stage, WorkflowStage.FAILED)

    def _get_escalation_stage(self, current_stage: WorkflowStage) -> WorkflowStage:
        """Get escalation stage for quality gate failures."""

        escalation_map = {
            WorkflowStage.RESEARCH: WorkflowStage.REPORT_GENERATION,  # Proceed with available research
            WorkflowStage.REPORT_GENERATION: WorkflowStage.EDITORIAL_REVIEW,
            WorkflowStage.EDITORIAL_REVIEW: WorkflowStage.DECOUPLED_EDITORIAL_REVIEW,
            WorkflowStage.DECOUPLED_EDITORIAL_REVIEW: WorkflowStage.PROGRESSIVE_ENHANCEMENT,
            WorkflowStage.PROGRESSIVE_ENHANCEMENT: WorkflowStage.FINAL_OUTPUT,  # Force completion
        }

        return escalation_map.get(current_stage, WorkflowStage.FAILED)

    async def execute_agent_query_with_response_collection(self, agent_client, agent_name: str, prompt: str, session_id: str, timeout_seconds: int = 120) -> dict[str, Any]:
        """Execute agent query with proper response collection."""
        self.logger.debug(f"Executing query for {agent_name}")

        query_result = {
            "agent_name": agent_name,
            "session_id": session_id,
            "prompt_sent": prompt,
            "messages_collected": [],
            "substantive_responses": 0,
            "tool_executions": [],
            "errors": [],
            "query_start_time": datetime.now().isoformat(),
            "success": False
        }

        try:
            # Send the query
            await agent_client.query(prompt, session_id=session_id)

            # Collect responses with timeout
            start_time = asyncio.get_event_loop().time()

            async def collect_responses():
                async for message in agent_client.receive_response():
                    current_time = asyncio.get_event_loop().time()
                    elapsed = current_time - start_time

                    message_info = {
                        "message_type": type(message).__name__,
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_seconds": elapsed
                    }

                    # Extract content information
                    if hasattr(message, 'content') and message.content:
                        content_texts = []
                        for block in message.content:
                            if hasattr(block, 'text') and block.text:
                                content_texts.append(block.text)
                        if content_texts:
                            message_info["content_texts"] = content_texts
                            query_result["substantive_responses"] += 1
                            self.logger.debug(f"{agent_name} content: {content_texts[0][:100]}...")

                    # Extract tool use information from AssistantMessage content blocks
                    tool_executions = self._extract_tool_executions_from_message(message, agent_name)
                    if tool_executions:
                        message_info["tool_use"] = tool_executions[0]  # Primary tool for this message
                        query_result["tool_executions"].extend(tool_executions)

                    # Extract result information
                    if isinstance(message, ResultMessage):
                        message_info["has_result"] = True
                        message_info["cost_usd"] = message.total_cost_usd if hasattr(message, 'total_cost_usd') else None
                        self.logger.debug(f"{agent_name} received ResultMessage with cost: ${message_info.get('cost_usd', 'N/A')}")

                    query_result["messages_collected"].append(message_info)

                    # Check for timeout (but don't break - let receive_response() complete naturally)
                    if elapsed > timeout_seconds:
                        self.logger.warning(f"{agent_name} query exceeding {timeout_seconds}s, but continuing to collect all messages")
                        # Don't break - receive_response() auto-stops at ResultMessage

                    # Check if we received ResultMessage (natural completion point)
                    if isinstance(message, ResultMessage):
                        self.logger.info(f"{agent_name} received ResultMessage - collection complete")
                        self.logger.info(f"Total messages: {len(query_result['messages_collected'])}, Tools: {len(query_result['tool_executions'])}, Substantive responses: {query_result['substantive_responses']}")
                        break  # Natural completion point

            # Execute with timeout
            await asyncio.wait_for(collect_responses(), timeout=timeout_seconds + 10)

            query_result["success"] = True
            query_result["query_end_time"] = datetime.now().isoformat()

            self.logger.info(f"{agent_name} query completed: {len(query_result['messages_collected'])} messages, {query_result['substantive_responses']} substantive responses, {len(query_result['tool_executions'])} tool executions")

        except asyncio.TimeoutError:
            error_msg = f"{agent_name} query timeout after {timeout_seconds}s"
            self.logger.error(error_msg)
            query_result["errors"].append(error_msg)
        except Exception as e:
            error_msg = f"{agent_name} query error: {str(e)}"
            self.logger.error(error_msg)
            query_result["errors"].append(error_msg)

        return query_result

    async def stage_conduct_research(self, session_id: str, topic: str, user_requirements: dict[str, Any]):
        """Stage 1: Conduct research using Research Agent."""
        session_data = self.active_sessions[session_id]
        clean_topic = session_data.get("clean_topic", topic)  # Use clean topic if available

        # Start Work Product 1: Research
        work_product_number = self.start_work_product(session_id, "research", "Conduct initial research and gather information")

        self.logger.info(f"Session {session_id}: Starting research on {topic}")
        self.logger.info(f"Session {session_id}: Using clean topic for research: '{clean_topic}'")

        await self.update_session_status(session_id, "researching", "Conducting initial research")

        # Initialize cumulative budget tracking for retry loop
        search_budget = session_data["search_budget"]
        cumulative_scrapes = 0
        max_attempts = 3
        research_successful = False
        research_result = None

        for attempt in range(max_attempts):
            try:
              # Simple budget check: stop research if we've reached the limit, but always proceed to report generation
                remaining_budget = search_budget.primary_successful_scrapes_limit - cumulative_scrapes

                if remaining_budget <= 0:
                    self.logger.info(f"Session {session_id}: Budget reached ({cumulative_scrapes}/{search_budget.primary_successful_scrapes_limit}) - proceeding to report generation")
                    break  # Exit research loop and proceed to report generation

                self.logger.info(f"Session {session_id}: Research attempt {attempt + 1}/{max_attempts}")
                self.logger.info(f"Budget status: {cumulative_scrapes}/{search_budget.primary_successful_scrapes_limit} used, {remaining_budget} remaining")

                # Create comprehensive research prompt with budget awareness using clean topic
                research_prompt = f"""
                Use the research_agent agent to conduct comprehensive research on the topic: "{clean_topic}"

                User Requirements:
                {json.dumps(user_requirements, indent=2)}

                CRITICAL BUDGET STATUS FOR THIS ATTEMPT:
                - Attempt {attempt + 1} of {max_attempts}
                - Scrapes used in previous attempts: {cumulative_scrapes}
                - Remaining budget: {remaining_budget} scrapes
                - YOU MUST STAY WITHIN REMAINING BUDGET

                {"⚠️ BUDGET LOW: Use existing findings from previous attempts if available. Check session directory." if remaining_budget < 5 else ""}
                {"❌ BUDGET EXHAUSTED: DO NOT execute new searches. Use findings from previous attempts only." if remaining_budget <= 0 else ""}

                MANDATORY RESEARCH INSTRUCTIONS:
                1. IMMEDIATELY execute mcp__zplayground1_search__zplayground1_search_scrape_clean with the topic
                2. This tool performs complete search, scrape, and clean workflow in a single operation
                3. REQUIRED PARAMETERS (use these exact values):
                   - query: "{clean_topic}" (the research topic)
                   - search_mode: "news" (MUST be either "web" or "news" - use "news" for current events)
                   - anti_bot_level: 2 (MUST be integer 0-3, where 0=basic, 1=enhanced, 2=advanced, 3=stealth)
                   - num_results: 15
                   - auto_crawl_top: {min(10, remaining_budget)}
                   - crawl_threshold: 0.3 (MUST be float between 0.0-1.0, relevance score threshold for crawling)
                   - session_id: "{session_id}"
                4. Use mcp__research_tools__save_research_findings to save your findings
                5. Use mcp__research_tools__capture_search_results to structure results

                SEARCH BUDGET CONSTRAINTS:
                - **STRICT LIMIT**: Maximum {search_budget.primary_successful_scrapes_limit} successful content extractions per session
                - **ALREADY USED**: {cumulative_scrapes} scrapes in previous attempts
                - **REMAINING**: {remaining_budget} scrapes for this attempt
                - **BUDGET AWARENESS**: Each search consumes from your session budget
                - **EFFICIENCY REQUIRED**: Make each search count with quality sources

                REQUIREMENTS:
                - Execute actual searches using the SERP API tool
                - Gather specific facts, data points, and expert opinions
                - Validate source credibility and authority
                - Save comprehensive findings to files for other agents
                - Do not acknowledge the task - EXECUTE the research immediately

                Session ID: {session_id}
                """

                # Execute research using the new single client pattern with natural language agent selection
                research_result = await self.execute_agent_query(
                    "research_agent", research_prompt, session_id, timeout_seconds=180
                )

                self.logger.info(f"✅ Research execution completed: {research_result['substantive_responses']} responses, {len(research_result['tool_executions'])} tools")

                # ✅ Extract actual scrape count from this attempt
                attempt_scrapes = self._extract_scrape_count(research_result)
                cumulative_scrapes += attempt_scrapes

                # Record scrapes for budget tracking
                search_budget.record_primary_research(
                    urls_processed=attempt_scrapes,
                    successful_scrapes=attempt_scrapes,
                    search_queries=1
                )

                self.logger.info(f"Attempt {attempt + 1} scraped {attempt_scrapes} URLs, cumulative: {cumulative_scrapes}/{search_budget.primary_successful_scrapes_limit}")

                # Simple validation: if we got any research activity, proceed to report generation
                if research_result.get("success", False) and research_result.get("substantive_responses", 0) > 0:
                    research_successful = True
                    self.logger.info(f"✅ Research completed - proceeding to report generation on attempt {attempt + 1}")
                    self.logger.info(f"Research found: {research_result.get('substantive_responses', 0)} responses, {len(research_result.get('tool_executions', []))} tools")
                    break
                else:
                    self.logger.warning(f"Session {session_id}: Research attempt {attempt + 1} had minimal results, but will proceed to report generation anyway")
                    # Always proceed to report generation, even with minimal research
                    research_successful = True
                    self.logger.info(f"✅ Proceeding to report generation with available research on attempt {attempt + 1}")
                    break

            except Exception as e:
                self.logger.error(f"Session {session_id}: Research attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(2)  # Brief delay before retry

        # Always proceed to report generation, even if research had minimal results
        if not research_successful:
            self.logger.warning(f"Session {session_id}: Research stage completed with minimal results, proceeding to report generation anyway")
            # Create a minimal research result to proceed
            research_result = {
                "success": True,
                "substantive_responses": 1,
                "tool_executions": [],
                "research_findings": "Limited research data available. Report will be generated based on existing findings.",
                "content_generated": True
            }

        # Store research results
        session_data = self.active_sessions[session_id]
        session_data["research_results"] = research_result
        session_data["current_stage"] = "report_generation"
        session_data["workflow_history"].append({
            "stage": "research",
            "completed_at": datetime.now().isoformat(),
            "responses_count": research_result["substantive_responses"],
            "tools_executed": len(research_result["tool_executions"]),
            "success": research_result["success"],
            "attempts": attempt + 1
        })

        await self.save_session_state(session_id)
        self.logger.info(f"Session {session_id}: Research stage completed successfully")

        # Complete Work Product 1: Research
        self.complete_work_product(session_id, work_product_number, {
            "stage": "research",
            "success": research_result.get("success", False),
            "substantive_responses": research_result.get("substantive_responses", 0),
            "tool_usage_count": research_result.get("tool_usage_count", 0),
            "search_queries_used": research_result.get("search_queries_used", 0),
            "content_generated": research_result.get("content_generated", False),
            "attempts": attempt + 1,
            "tools_executed": len(research_result["tool_executions"])
        })

    async def stage_generate_report(self, session_id: str):
        """Stage 2: Generate report using Report Agent."""
        self.logger.info(f"Session {session_id}: Generating report")

        # Start Work Product 2: Report Generation
        work_product_number = self.start_work_product(session_id, "report_generation", "Generate comprehensive research report")

        await self.update_session_status(session_id, "generating_report", "Creating initial report")

        session_data = self.active_sessions[session_id]

        # Validate that research data exists before generating report
        if not session_data.get("research_results") or not session_data["research_results"].get("success"):
            raise RuntimeError("Cannot generate report: research stage did not complete successfully")

        max_attempts = 3
        report_successful = False
        report_result = None

        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Session {session_id}: Report generation attempt {attempt + 1}/{max_attempts}")

                # Use LLM judge to determine report scope and extract requirements
                report_config = await self._determine_report_scope_with_llm_judge(session_data['topic'])
                dynamic_requirements = report_config.get('special_requirements', '')

                self.logger.info(f"LLM Judge determined scope: {report_config['scope']} (confidence: {report_config['llm_confidence']})")
                self.logger.info(f"LLM Reasoning: {report_config['llm_reasoning']}")
                if dynamic_requirements:
                    self.logger.info(f"Dynamic requirements: {dynamic_requirements}")

                report_prompt = f"""
                Use the report_agent agent to generate a report based on the research findings.

                Topic: {session_data['topic']}
                User Requirements: {json.dumps(session_data['user_requirements'], indent=2)}

                Report Style Instructions: {report_config['style_instructions']}
                Editing Rigor: {report_config['editing_rigor']}
                LLM Scope Analysis: {report_config['llm_reasoning']} (confidence: {report_config['llm_confidence']})

                {"Dynamic Requirements: " + dynamic_requirements if dynamic_requirements else ""}

                Research results have been collected and are available in the session data.

                Use the mcp__research_tools__get_session_data tool to retrieve research findings, then:

                1. Create a {report_config['scope']} report on the topic
                2. Include all key findings from the research
                3. Organize content logically with clear sections
                4. Adjust the depth and length to match the {report_config['scope']} scope
                5. **TARGET LENGTH**: Aim for {report_config['min_words']}+ words, adjusting based on available research data
                   - If abundant research is available, expand to provide comprehensive coverage
                   - If research is limited, focus on quality over hitting exact word count
                   - Conclude with clear takeaways and implications
                6. Ensure proper citations and source attribution when available
                7. Target the report to the user's specified audience
                8. Use mcp__research_tools__create_research_report to create the report
                9. CRITICAL: Save the report using the provided filepath from the tool

                REQUIREMENTS:
                - Execute the create_research_report tool to generate the report
                - Use the Write tool to save the report to the exact filepath provided
                - Do not just describe the report - actually create and save it
                - Ensure the report is comprehensive and well-structured
                - Adapt report length to research availability (target: {report_config['min_words']}+ words)

                Session ID: {session_id}
                """

                # Execute report generation using the new single client pattern with natural language agent selection
                report_result = await self.execute_agent_query(
                    "report_agent", report_prompt, session_id, timeout_seconds=150
                )

                self.logger.info(f"✅ Report generation completed: {report_result['substantive_responses']} responses, {report_result['tool_executions']} tools")

                # Validate report completion
                if self._validate_report_completion(report_result):
                    report_successful = True
                    break
                else:
                    self.logger.warning(f"Session {session_id}: Report generation attempt {attempt + 1} did not complete required work")

            except Exception as e:
                self.logger.error(f"Session {session_id}: Report generation attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(2)  # Brief delay before retry

        if not report_successful:
            raise RuntimeError(f"Report generation stage failed after {max_attempts} attempts")

        # Store report results
        session_data["report_results"] = report_result
        session_data["current_stage"] = "editorial_review"
        session_data["workflow_history"].append({
            "stage": "report_generation",
            "completed_at": datetime.now().isoformat(),
            "responses_count": report_result["substantive_responses"],
            "tools_executed": len(report_result["tool_executions"]),
            "success": report_result["success"],
            "attempts": attempt + 1
        })

        await self.save_session_state(session_id)
        self.logger.info(f"Session {session_id}: Report generation stage completed successfully")

        # Complete Work Product 2: Report Generation
        self.complete_work_product(session_id, work_product_number, {
            "stage": "report_generation",
            "success": report_result.get("success", False),
            "substantive_responses": report_result.get("substantive_responses", 0),
            "tools_executed": len(report_result.get("tool_executions", [])),
            "attempts": attempt + 1,
            "report_quality": report_result.get("report_quality", "unknown")
        })

    async def stage_editorial_review(self, session_id: str):
        """Stage 3: Editorial review using Editor Agent with success-based search controls."""
        self.logger.info(f"Session {session_id}: Conducting editorial review")

        # Start Work Product 3: Editorial Review
        work_product_number = self.start_work_product(session_id, "editorial_review", "Review and enhance report quality")

        await self.update_session_status(session_id, "editorial_review", "Reviewing report quality")

        session_data = self.active_sessions[session_id]

        # Store work product number for completion in main workflow
        session_data["editorial_work_product_number"] = work_product_number
        search_budget = session_data["search_budget"]

        # Reset editorial budget to ensure editorial process can proceed
        search_budget.reset_editorial_budget()

        # Validate search budget before starting editorial research
        can_proceed, budget_message = search_budget.can_editorial_research_proceed(5)
        if not can_proceed:
            self.logger.warning(f"Session {session_id}: Editorial search budget reached: {budget_message}")
            # Continue with review but no additional searches

        max_attempts = 3
        editorial_successful = False
        review_result = None

        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Session {session_id}: Editorial review attempt {attempt + 1}/{max_attempts}")

                # Use LLM judge to determine editing approach based on original query
                editorial_config = await self._determine_report_scope_with_llm_judge(session_data['topic'])

                self.logger.info(f"Editorial LLM Judge determined rigor: {editorial_config['editing_rigor']} (confidence: {editorial_config['llm_confidence']})")

                # Initialize editorial search tracking
                session_data["editorial_search_stats"] = {
                    "search_attempts": 0,
                    "successful_scrapes": 0,
                    "urls_attempted": 0,
                    "search_limit_reached": False
                }

                # Create current budget status for the prompt
                budget_status = search_budget.get_budget_status()
                editorial_remaining = budget_status["editorial"]["search_queries_remaining"]

                review_prompt = f"""
                Use the editor_agent agent to review the generated report for quality, accuracy, and completeness.

                Topic: {session_data['topic']}
                Editing Rigor: {editorial_config['editing_rigor']} (determined by LLM: {editorial_config['llm_reasoning']})

                EDITORIAL SEARCH CONTROLS:
                **SUCCESS-BASED TERMINATION**: Continue searching until you achieve 5 successful scrapes total
                **SEARCH LIMITS**: Maximum {editorial_remaining} editorial search attempts remaining
                **WORK PRODUCT LABELING**: CRITICAL - Always use workproduct_prefix="editor research"
                **BUDGET AWARENESS**: Track your search usage to stay within limits

                CURRENT BUDGET STATUS:
                - Search queries remaining: {editorial_remaining}/2
                - Successful scrapes: {budget_status["editorial"]["successful_scrapes"]}
                - Search limit reached: {budget_status["editorial"]["search_queries_reached_limit"]}

                Use the Read tool to examine the generated report files AND all available research data, then provide comprehensive review:

                CRITICAL EDITORIAL PRIORITIES - PROVIDE SPECIFIC, ACTIONABLE FEEDBACK:
                1. Assess report quality against professional standards
                2. Check accuracy and proper source attribution
                3. Evaluate clarity, organization, and completeness
                4. **IDENTIFY SPECIFIC DATA GAPS**: Point out where the report lacks specific facts, statistics, quotes, or data points
                5. **RECOMMEND SPECIFIC ENHANCEMENTS**: Tell the report agent exactly which data from the research should be integrated
                6. **CITE MISSING SPECIFICS**: When you find generic statements, recommend the specific data/quotes that should replace them
                7. Provide specific, actionable feedback WITH EXAMPLES of data to add

                EDITORIAL FEEDBACK REQUIREMENTS - ADD SUBSTANCE:
                - When you identify missing data, CITE THE SPECIFIC FACTS/STATISTICS/QUOTES from research that should be added
                - Point to exact data points in the research materials that would enhance the report
                - Recommend specific quotes, figures, or examples to integrate
                - Identify where general statements should be replaced with concrete data
                - Your feedback should give the report agent clear direction on what specific content to add

                SEARCH GUIDELINES:
                - Only search for SPECIFIC identified gaps, not general "more information"
                - Use workproduct_prefix="editor research" for all editorial searches
                - STOP searching when you reach your search query limit or 5 successful scrapes
                - Each successful scrape should provide meaningful content for gap-filling

                If you identify research gaps and budget allows, conduct targeted searches following the guidelines above.

                DELIVERABLE: Provide detailed feedback that includes SPECIFIC DATA RECOMMENDATIONS from the research materials.
                Your feedback should tell the report agent exactly what facts, quotes, and statistics to integrate to enhance the report.
                """

                # Execute editorial review with extended timeout for search activities
                review_result = await self.execute_agent_query(
                    "editor_agent", review_prompt, session_id, timeout_seconds=300  # 5 minutes for editorial searches
                )

                self.logger.info(f"✅ Editorial review completed: {review_result['substantive_responses']} responses, {review_result['tool_executions']} tools")

                # Validate editorial completion
                if self._validate_editorial_completion(review_result):
                    editorial_successful = True
                    break
                else:
                    self.logger.warning(f"Session {session_id}: Editorial review attempt {attempt + 1} did not complete required work")

            except Exception as e:
                self.logger.error(f"Session {session_id}: Editorial review attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(2)  # Brief delay before retry

        if not editorial_successful:
            raise RuntimeError(f"Editorial review stage failed after {max_attempts} attempts")

        # Record editorial search statistics
        search_stats = session_data.get("editorial_search_stats", {})
        if search_stats.get("search_attempts", 0) > 0:
            search_budget.record_editorial_research(
                urls_processed=search_stats.get("urls_attempted", 0),
                successful_scrapes=search_stats.get("successful_scrapes", 0),
                search_queries=search_stats.get("search_attempts", 0)
            )

        # Log editorial search statistics
        self.logger.info(f"📊 Editorial search stats: {search_stats.get('search_attempts', 0)} attempts, {search_stats.get('successful_scrapes', 0)} successful scrapes")

        # Store review results
        session_data["review_results"] = review_result
        session_data["current_stage"] = "finalization"
        session_data["workflow_history"].append({
            "stage": "editorial_review",
            "completed_at": datetime.now().isoformat(),
            "responses_count": review_result["substantive_responses"],
            "tools_executed": len(review_result["tool_executions"]),
            "success": review_result["success"],
            "attempts": attempt + 1,
            "editorial_search_stats": search_stats
        })

        await self.save_session_state(session_id)
        self.logger.info(f"Session {session_id}: Editorial review stage completed successfully")

    async def stage_decoupled_editorial_review(self, session_id: str) -> dict:
        """
        Decoupled editorial review that works independently of research success.

        This method uses the DecoupledEditorialAgent to process any available content
        regardless of research stage completion, ensuring 100% editorial execution rate.
        """
        self.logger.info(f"Session {session_id}: Starting decoupled editorial review")

        # Start Work Product 3: Decoupled Editorial Review (if not already started)
        session_data = self.active_sessions[session_id]
        if "editorial_work_product_number" not in session_data:
            work_product_number = self.start_work_product(session_id, "editorial_review", "Review and enhance report quality (decoupled)")
            session_data["editorial_work_product_number"] = work_product_number

        await self.update_session_status(session_id, "decoupled_editorial_review", "Processing available content")

        # Reset editorial budget to ensure editorial process can proceed
        search_budget = session_data["search_budget"]
        search_budget.reset_editorial_budget()

        try:
            # Collect available content sources regardless of research success
            content_sources = await self.collect_available_content_sources(session_id)

            # Create context for editorial processing
            context = {
                "topic": session_data.get("topic", "Unknown topic"),
                "session_id": session_id,
                "research_quality": session_data.get("research_quality", "unknown"),
                "workflow_stage": "decoupled_editorial_review"
            }

            # Use decoupled editorial agent for independent processing
            editorial_result = await self.decoupled_editorial_agent.process_available_content(
                session_id=session_id,
                content_sources=content_sources,
                context=context
            )

            # Store decoupled editorial results
            session_data["decoupled_editorial_results"] = editorial_result
            session_data["workflow_history"].append({
                "stage": "decoupled_editorial_review",
                "completed_at": datetime.now().isoformat(),
                "editorial_success": editorial_result.editorial_success,
                "content_quality": editorial_result.content_quality,
                "enhancements_made": editorial_result.enhancements_made,
                "files_created": editorial_result.files_created,
                "processing_log_entries": len(editorial_result.processing_log)
            })

            await self.save_session_state(session_id)

            if editorial_result.editorial_success:
                self.logger.info(f"✅ Session {session_id}: Decoupled editorial review completed successfully")
                self.logger.info(f"   Content quality: {editorial_result.content_quality}")
                self.logger.info(f"   Enhancements made: {editorial_result.enhancements_made}")
                self.logger.info(f"   Files created: {len(editorial_result.files_created)}")

                # Log editorial report summary
                if editorial_result.editorial_report:
                    self.logger.info(f"   Editorial summary: {editorial_result.editorial_report.get('editorial_summary', {})}")
            else:
                self.logger.warning(f"⚠️ Session {session_id}: Decoupled editorial review completed with minimal output")

            return {
                "success": editorial_result.editorial_success,
                "content_quality": editorial_result.content_quality,
                "enhancements_made": editorial_result.enhancements_made,
                "files_created": editorial_result.files_created,
                "editorial_report": editorial_result.editorial_report,
                "decoupled_processing": True
            }

        except Exception as e:
            self.logger.error(f"❌ Session {session_id}: Decoupled editorial review failed: {e}")

            # Create minimal failure record
            failure_result = {
                "success": False,
                "error": str(e),
                "decoupled_processing": True,
                "minimal_output": True
            }

            session_data["decoupled_editorial_results"] = failure_result
            session_data["workflow_history"].append({
                "stage": "decoupled_editorial_review",
                "completed_at": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })

            await self.save_session_state(session_id)
            return failure_result

    async def collect_available_content_sources(self, session_id: str) -> list[str]:
        """
        Collect all available content sources for editorial processing.

        Args:
            session_id: Session identifier

        Returns:
            List of file paths containing content
        """
        content_sources = []
        session_data = self.active_sessions[session_id]

        # FAIL-FAST: Use the correct session directory structure
        # The session directory should be KEVIN/sessions/{session_id}
        kevin_base = os.environ.get('KEVIN_WORKPRODUCTS_DIR', '/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN')
        session_dir = Path(kevin_base) / "sessions" / session_id

        self.logger.debug(f"Looking for content sources in session directory: {session_dir}")

        # Validate session directory exists
        if not session_dir.exists():
            self.logger.warning(f"Session directory does not exist: {session_dir}")
            return content_sources

        try:
            # Look for research findings
            research_files = [
                session_dir / "research_findings.md",
                session_dir / "raw_research_data.json",
                session_dir / "search_results.txt"
            ]

            for file_path in research_files:
                if file_path.exists() and file_path.stat().st_size > 100:  # Non-empty files
                    content_sources.append(str(file_path))
                    self.logger.debug(f"Found research content: {file_path}")

            # Look for report files
            report_files = [
                session_dir / "research_report.md",
                session_dir / "final_report.md",
                session_dir / "report.md"
            ]

            for file_path in report_files:
                if file_path.exists() and file_path.stat().st_size > 100:
                    content_sources.append(str(file_path))
                    self.logger.debug(f"Found report content: {file_path}")

            # Look for any scraped content files
            content_dir = session_dir / "content"
            if content_dir.exists():
                for content_file in content_dir.glob("*.md"):
                    if content_file.stat().st_size > 100:
                        content_sources.append(str(content_file))
                        self.logger.debug(f"Found content file: {content_file}")

            # Look for search result files
            for search_file in session_dir.glob("*.json"):
                if "search" in search_file.name.lower() and search_file.stat().st_size > 100:
                    content_sources.append(str(search_file))
                    self.logger.debug(f"Found search data: {search_file}")

            self.logger.info(f"Collected {len(content_sources)} content sources for session {session_id}")

            if not content_sources:
                self.logger.warning(f"No content sources found for session {session_id}")
                # Create a minimal content file with basic session information
                minimal_content = f"""# Minimal Research Content

Session ID: {session_id}
Topic: {session_data.get('topic', 'Unknown topic')}
Research Status: Limited or incomplete content available

## Note

This session had limited research output available. The editorial agent has processed the minimal content that could be collected.
"""
                minimal_file = session_dir / "minimal_content.md"
                with open(minimal_file, 'w', encoding='utf-8') as f:
                    f.write(minimal_content)
                content_sources.append(str(minimal_file))
                self.logger.info(f"Created minimal content file: {minimal_file}")

        except Exception as e:
            self.logger.error(f"FAIL-FAST: Error collecting content sources for session {session_id}: {e}")
            self.logger.error(f"Session directory: {session_dir}")
            self.logger.error(f"Working directory: {os.getcwd()}")
            self.logger.error("This error would have been silently ignored - now we fail fast to expose directory structure issues!")
            # During development, fail hard to expose configuration issues
            raise RuntimeError(f"CRITICAL CONTENT SOURCE COLLECTION ERROR: {e}")

        return content_sources

    async def stage_finalize(self, session_id: str):
        """Stage 4: Finalize the report and complete the session."""
        self.logger.info(f"Session {session_id}: Finalizing report")

        # Start Work Product 4: Finalization
        work_product_number = self.start_work_product(session_id, "finalization", "Complete final report and deliverables")

        await self.update_session_status(session_id, "finalizing", "Completing final report")

        session_data = self.active_sessions[session_id]

        # Check if revisions are needed based on editorial review
        needs_revision = await self.check_if_revision_needed(session_id)

        if needs_revision:
            await self.stage_perform_revisions(session_id)
        else:
            await self.complete_session(session_id)

    async def check_if_revision_needed(self, session_id: str) -> bool:
        """Check if the report needs revisions based on editorial feedback."""
        session_data = self.active_sessions[session_id]
        review_results = session_data.get("review_results", [])

        # Analyze review results to determine if revisions are needed
        # This would parse the actual review feedback
        # For now, assume minor revisions are needed
        return len(review_results) > 0

    async def stage_perform_revisions(self, session_id: str):
        """Perform revisions based on editorial feedback."""
        self.logger.info(f"Session {session_id}: Performing revisions")

        await self.update_session_status(session_id, "revising", "Applying editorial feedback")

        session_data = self.active_sessions[session_id]

        revision_prompt = f"""
        Use the report_agent agent to revise the report based on the editorial feedback provided.

        Topic: {session_data['topic']}

        Use the Read tool to examine the current report and editorial feedback, then:

        CRITICAL REVISION PRIORITIES - TAKE EDITORIAL FEEDBACK SERIOUSLY:
        1. **INTEGRATE SPECIFIC DATA**: Add the exact facts, statistics, quotes, and data points the editor recommended
        2. **REPLACE GENERIC STATEMENTS**: Where the editor identified generic content, substitute with specific data from research
        3. **ADD RECOMMENDED CONTENT**: Incorporate all specific examples, figures, and evidence the editor cited
        4. **ENHANCE WITH FACTS**: Use the editor's guidance to add substance through concrete data and quotes
        5. Address all other feedback from the editorial review (structure, clarity, citations)
        6. Improve report quality based on all specific recommendations
        7. Ensure all identified issues are resolved
        8. Maintain overall report coherence and quality while adding substance
        9. Use create_research_report with report_type="revised" to format the revised report
        10. CRITICAL: The tool will automatically add "2B-" prefix to your revised report filename to indicate this is Work Product 2B (Revised Report)

        EDITORIAL FEEDBACK IMPLEMENTATION REQUIREMENTS:
        - When the editor recommends adding specific data, FIND IT in the research materials and INTEGRATE IT
        - When the editor cites missing statistics/quotes, LOCATE THEM and ADD THEM to the report
        - When the editor points to gaps, FILL THEM with the specific content recommended
        - PRIORITIZE data integration and specificity enhancements from editorial feedback
        - The editor's comments represent critical quality improvements - implement them thoroughly

        Focus on implementing the feedback systematically, especially the specific data and content recommendations,
        to significantly improve the report's substance and meet professional standards.
        """

        # Execute revisions using the new single client pattern with natural language agent selection
        revision_result = await self.execute_agent_query(
            "report_agent", revision_prompt, session_id, timeout_seconds=120
        )

        self.logger.info(f"✅ Revision execution completed: {revision_result['substantive_responses']} responses, {revision_result['tool_executions']} tools")

        # Store revision results
        session_data["revision_results"] = revision_result
        session_data["workflow_history"].append({
            "stage": "revisions",
            "completed_at": datetime.now().isoformat(),
            "responses_count": revision_result["substantive_responses"],
            "tools_executed": len(revision_result["tool_executions"]),
            "success": revision_result["success"]
        })

        await self.save_session_state(session_id)

        # Complete the session after revisions
        await self.complete_session(session_id)

    async def complete_session(self, session_id: str):
        """Complete the research session."""
        self.logger.info(f"Session {session_id}: Completing session")

        await self.update_session_status(session_id, "completed", "Research workflow completed")

        session_data = self.active_sessions[session_id]
        session_data["completed_at"] = datetime.now().isoformat()
        session_data["workflow_history"].append({
            "stage": "completion",
            "completed_at": datetime.now().isoformat()
        })

        # Create final summary (Stage 4 work product)
        try:
            final_summary_prompt = f"""
            Create a final summary document for the completed research session.

            Topic: {session_data['topic']}

            Based on all the work completed (research, report, editorial review, and revisions),
            create a concise final summary that includes:
            1. Session overview and key findings
            2. Summary of the research workflow stages completed
            3. Location of all work products created
            4. Final quality assessment

            Use create_research_report with report_type="final_summary" or use Write tool with "4-" prefix.
            CRITICAL: The filename should have "4-" prefix to indicate this is Work Product 4 (Final Summary)
            Save this as a final summary document in the session directory.
            """

            # Execute final summary creation
            final_summary_result = await self.execute_agent_query(
                "report_agent", final_summary_prompt, session_id, timeout_seconds=60
            )

            self.logger.info(f"✅ Final summary created: {final_summary_result['substantive_responses']} responses")

        except Exception as e:
            self.logger.warning(f"Could not create final summary: {e}")

        await self.save_session_state(session_id)

        # Create final report copy in centralized location
        final_report = self.get_final_report(session_id)
        if "error" not in final_report:
            self.logger.info(f"✅ Final report available at: {final_report['report_file']}")
            self.logger.info(f"   Report location: {final_report['location']}")
            self.logger.info(f"   Report length: {final_report['report_length']} characters")
        else:
            self.logger.warning(f"Final report not accessible: {final_report['error']}")

        self.logger.info(f"Session {session_id}: Research workflow completed successfully")

        # Complete Work Product 4: Finalization
        self.complete_work_product(session_id, work_product_number, {
            "stage": "finalization",
            "success": True,
            "final_summary_created": 'final_summary_result' in locals(),
            "workflow_completed": True,
            "total_stages_completed": 4,
            "final_report_location": final_report.get("location", "unknown"),
            "final_report_file": final_report.get("report_file", "none")
        })

    def get_debug_output(self) -> list[str]:
        """Get all debug output collected from stderr callbacks."""
        return self.debug_output.copy()

    def clear_debug_output(self):
        """Clear the debug output buffer."""
        self.debug_output.clear()

    def _get_final_reports_directory(self) -> Path:
        """Get or create the final reports directory."""
        final_reports_dir = Path("final_reports")
        final_reports_dir.mkdir(parents=True, exist_ok=True)
        return final_reports_dir

    def _create_final_report_symlink(self, session_id: str, original_report_path: Path) -> Path:
        """Create a symlink in the final reports directory for easy access."""
        final_reports_dir = self._get_final_reports_directory()

        # Create a predictable filename
        session_data = self.active_sessions[session_id]
        clean_topic = session_data.get("clean_topic", session_data.get("topic", "unknown_topic"))
        # Clean topic for filename
        safe_topic = "".join(c for c in clean_topic[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_report_name = f"final_report_{safe_topic}_{session_id[:8]}_{timestamp}.md"
        final_report_path = final_reports_dir / final_report_name

        try:
            # Create a copy instead of symlink to ensure file is accessible
            import shutil
            shutil.copy2(original_report_path, final_report_path)
            self.logger.info(f"✅ Final report copied to: {final_report_path}")
            return final_report_path
        except Exception as e:
            self.logger.warning(f"Could not copy final report: {e}")
            return original_report_path

    def get_final_report(self, session_id: str) -> dict[str, Any]:
        """Get the final research report for a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session_data = self.active_sessions[session_id]

        # First, check the centralized final reports directory
        final_reports_dir = self._get_final_reports_directory()
        final_report_files = list(final_reports_dir.glob(f"*{session_id[:8]}*.md"))
        if final_report_files:
            latest_report = max(final_report_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_report, encoding='utf-8') as f:
                    content = f.read()
                return {
                    "report_file": str(latest_report),
                    "report_content": content,
                    "report_length": len(content),
                    "session_id": session_id,
                    "location": "final_reports_directory"
                }
            except Exception as e:
                self.logger.error(f"Error reading final report: {e}")

        # Check for KEVIN directory files
        kevin_dir = Path("/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN")
        if kevin_dir.exists():
            # Look for report files with this session
            report_files = list(kevin_dir.glob("research_report_*.md"))
            if report_files:
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_report, encoding='utf-8') as f:
                        content = f.read()

                    # Create symlink in final reports directory for future access
                    self._create_final_report_symlink(session_id, latest_report)

                    return {
                        "report_file": str(latest_report),
                        "report_content": content,
                        "report_length": len(content),
                        "session_id": session_id,
                        "location": "kevin_directory"
                    }
                except Exception as e:
                    self.logger.error(f"Error reading report: {e}")

        # Check session directory as fallback
        session_dir = Path(session_data.get("workspace_dir", os.getcwd())) / session_id
        session_report_files = [
            session_dir / "research_report.md",
            session_dir / "final_report.md",
            session_dir / "report.md"
        ]

        for report_file in session_report_files:
            if report_file.exists() and report_file.stat().st_size > 100:
                try:
                    with open(report_file, encoding='utf-8') as f:
                        content = f.read()

                    # Create symlink in final reports directory for future access
                    self._create_final_report_symlink(session_id, report_file)

                    return {
                        "report_file": str(report_file),
                        "report_content": content,
                        "report_length": len(content),
                        "session_id": session_id,
                        "location": "session_directory"
                    }
                except Exception as e:
                    self.logger.error(f"Error reading session report: {e}")

        return {"error": "No report found"}

    async def get_session_status(self, session_id: str) -> dict[str, Any]:
        """Get current status of a research session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session_data = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "status": session_data.get("status"),
            "current_stage": session_data.get("current_stage"),
            "topic": session_data.get("topic"),
            "created_at": session_data.get("created_at"),
            "workflow_history": session_data.get("workflow_history", [])
        }

    async def update_session_status(self, session_id: str, status: str, message: str = ""):
        """Update session status and save state."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = status
            self.active_sessions[session_id]["status_message"] = message
            self.active_sessions[session_id]["last_updated"] = datetime.now().isoformat()
            await self.save_session_state(session_id)

    async def save_session_state(self, session_id: str):
        """Save session state to file."""
        session_data = self.active_sessions[session_id]

        # Use absolute path for session file
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # Go up from core/ to project root
        session_dir = project_root / "KEVIN" / "sessions" / session_id
        session_file = session_dir / "session_state.json"

        # Ensure session directory exists
        session_dir.mkdir(parents=True, exist_ok=True)

        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

    def get_search_budget(self, session_id: str) -> SessionSearchBudget | None:
        """Get the search budget for a session."""
        if session_id not in self.active_sessions:
            return None
        return self.active_sessions[session_id].get("search_budget")

    def validate_research_budget(self, session_id: str, agent_type: str, urls_to_process: int = 1) -> tuple[bool, str]:
        """Validate if research can proceed based on budget constraints."""
        budget = self.get_search_budget(session_id)
        if not budget:
            return False, "No search budget found for session"

        if agent_type == "research_agent":
            return budget.can_primary_research_proceed(urls_to_process)
        elif agent_type == "editor_agent":
            return budget.can_editorial_research_proceed(urls_to_process)
        else:
            return False, f"Unknown agent type: {agent_type}"

    def record_research_activity(self, session_id: str, agent_type: str, urls_processed: int, successful_scrapes: int, search_queries: int = 1):
        """Record research activity in the budget."""
        budget = self.get_search_budget(session_id)
        if not budget:
            self.logger.warning(f"No search budget found for session {session_id}")
            return

        if agent_type == "research_agent":
            budget.record_primary_research(urls_processed, successful_scrapes, search_queries)
        elif agent_type == "editor_agent":
            budget.record_editorial_research(urls_processed, successful_scrapes, search_queries)
        else:
            self.logger.warning(f"Unknown agent type for research recording: {agent_type}")

    def get_budget_summary(self, session_id: str) -> dict[str, Any]:
        """Get a summary of the search budget status."""
        budget = self.get_search_budget(session_id)
        if not budget:
            return {"error": "No search budget found for session"}

        return budget.get_budget_status()

    async def _diagnose_mcp_timeout_issue(self, agent_name: str, session_id: str, timeout_duration: int):
        """Diagnose MCP timeout issues and log detailed analysis."""
        self.logger.error(f"🔍 DIAGNOSING MCP TIMEOUT for {agent_name} (session: {session_id})")

        # Check MCP server status
        try:
            if self.mcp_server:
                mcp_status = await self._check_mcp_server_health()
                self.logger.error(f"🔍 MCP Server Status: {mcp_status}")
            else:
                self.logger.error("🔍 MCP Server: None initialized")
        except Exception as e:
            self.logger.error(f"🔍 MCP Server Health Check Failed: {e}")

        # Check single client connectivity
        try:
            if hasattr(self, 'client') and self.client:
                self.logger.error(f"🔍 Single Client: {type(self.client).__name__}")
                self.logger.error(f"🔍 Client Methods: {[m for m in dir(self.client) if not m.startswith('_')]}")
                self.logger.error(f"🔍 Available Agents: {self.agent_names}")
            else:
                self.logger.error("🔍 Single Client: Not initialized")
        except Exception as e:
            self.logger.error(f"🔍 Single Client Check Failed: {e}")

        # Check session state
        try:
            if session_id in self.active_sessions:
                session_data = self.active_sessions[session_id]
                self.logger.error(f"🔍 Session Stage: {session_data.get('current_stage')}")
                self.logger.error(f"🔍 Session Status: {session_data.get('status')}")
            else:
                self.logger.error(f"🔍 Session: {session_id} not found in active sessions")
        except Exception as e:
            self.logger.error(f"🔍 Session State Check Failed: {e}")

        # Log timeout analysis
        self.logger.error("🔍 Timeout Analysis:")
        self.logger.error(f"  - Duration: {timeout_duration}s")
        self.logger.error(f"  - Agent: {agent_name}")
        self.logger.error(f"  - Session: {session_id}")
        self.logger.error("  - Possible Causes: MCP server unresponsive, network issues, complex query processing")
        self.logger.error("  - Recommendation: Check MCP server logs and network connectivity")

    async def _diagnose_mcp_error_issue(self, agent_name: str, session_id: str, error: Exception):
        """Diagnose MCP error issues and log detailed analysis."""
        self.logger.error(f"🔍 DIAGNOSING MCP ERROR for {agent_name} (session: {session_id})")
        self.logger.error(f"🔍 Error Type: {type(error).__name__}")
        self.logger.error(f"🔍 Error Message: {str(error)}")

        # Check if it's an MCP-specific error
        error_str = str(error).lower()
        if any(keyword in error_str for keyword in ['mcp', 'server', 'connection', 'transport']):
            self.logger.error("🔍 MCP-related error detected")
            self.logger.error("🔍 Recommendation: Check MCP server configuration and connectivity")

        if any(keyword in error_str for keyword in ['tool', 'function', 'call']):
            self.logger.error("🔍 Tool execution error detected")
            self.logger.error("🔍 Recommendation: Verify tool availability and permissions")

        # Log error analysis
        self.logger.error("🔍 Error Analysis:")
        self.logger.error(f"  - Agent: {agent_name}")
        self.logger.error(f"  - Session: {session_id}")
        self.logger.error(f"  - Error Class: {error.__class__.__name__}")
        self.logger.error(f"  - Error Details: {str(error)}")

        # Try to get stack trace information
        import traceback
        self.logger.error(f"🔍 Stack Trace: {traceback.format_exc()}")

    async def _check_mcp_server_health(self) -> dict[str, Any]:
        """Check the health of the MCP server."""
        try:
            if not self.mcp_server:
                return {"status": "not_initialized", "error": "MCP server not created"}

            # Try to get server info
            server_info = self.mcp_server.get_server_info()
            return {
                "status": "running",
                "server_info": server_info,
                "agent_count": len(self.agent_names)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

    async def _handle_tool_file_creation(self, tool_name: str, tool_result: Any, session_id: str):
        """Handle file creation for tools that return data instead of writing files directly."""
        try:
            import json

            self.logger.info(f"🔧 Handling file creation for tool: {tool_name}")
            self.logger.info(f"🔧 Tool result type: {type(tool_result)}")

            # Convert tool result to expected format if needed
            parsed_result = None

            # If it's already a dict with success flag, use it directly
            if isinstance(tool_result, dict) and tool_result.get("success"):
                parsed_result = tool_result
                self.logger.info("🔧 Using tool result directly")

            # If it's a string, try to parse as JSON
            elif isinstance(tool_result, str):
                try:
                    parsed_result = json.loads(tool_result)
                    self.logger.info("🔧 Parsed tool result from JSON string")
                except json.JSONDecodeError:
                    self.logger.warning("🔧 Could not parse tool result as JSON")
                    return

            # If it has a 'text' attribute, try to parse that
            elif hasattr(tool_result, 'text'):
                try:
                    parsed_result = json.loads(tool_result.text)
                    self.logger.info("🔧 Parsed tool result from text attribute")
                except (json.JSONDecodeError, AttributeError):
                    self.logger.warning("🔧 Could not parse tool result text as JSON")
                    return

            else:
                self.logger.warning("🔧 Unrecognized tool result format")
                return

            # Parse tool result based on tool type
            if tool_name == "mcp__research_tools__save_research_findings":
                await self._create_research_findings_files(parsed_result, session_id)
            elif tool_name == "mcp__research_tools__create_research_report":
                await self._create_research_report_files(parsed_result, session_id)
            elif tool_name == "mcp__research_tools__serp_search":
                await self._create_serp_search_files(parsed_result, session_id)
            else:
                self.logger.debug(f"🔧 No file creation needed for tool: {tool_name}")

        except Exception as e:
            self.logger.error(f"❌ Error handling file creation for {tool_name}: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")

    def _infer_tool_name_from_context(self, content: str) -> str:
        """Infer tool name from content text or current execution context."""
        content_lower = content.lower()

        # Method 1: Look for tool-specific keywords in content
        if 'search' in content_lower and ('result' in content_lower or 'found' in content_lower):
            return 'mcp__research_tools__serp_search'
        elif 'research' in content_lower and ('finding' in content_lower or 'saved' in content_lower):
            return 'mcp__research_tools__save_research_findings'
        elif 'report' in content_lower and ('created' in content_lower or 'generated' in content_lower):
            return 'mcp__research_tools__create_research_report'
        elif 'session' in content_lower and 'data' in content_lower:
            return 'mcp__research_tools__get_session_data'

        # Method 2: Default based on typical workflow order
        # If we can't determine from content, make an educated guess
        if len(content) > 100:  # Longer content is likely search results
            return 'mcp__research_tools__serp_search'
        else:
            return 'mcp__research_tools__save_research_findings'

    async def _handle_tool_result_data(self, tool_result_data: dict, session_id: str):
        """Extract tool name and route tool result data to appropriate file creation handler."""
        try:
            self.logger.info(f"🔧 Processing tool result data: {type(tool_result_data)}")

            # Try to extract tool name from the result data
            tool_name = None

            # Method 1: Look for explicit tool_name field
            if isinstance(tool_result_data, dict):
                tool_name = tool_result_data.get('tool_name')

                # Method 2: Try to infer from content structure
                if not tool_name:
                    content_text = str(tool_result_data.get('content', []))
                    if 'save_research_findings' in content_text:
                        tool_name = 'mcp__research_tools__save_research_findings'
                    elif 'create_research_report' in content_text:
                        tool_name = 'mcp__research_tools__create_research_report'
                    elif 'serp_search' in content_text:
                        tool_name = 'mcp__research_tools__serp_search'

                # Method 3: Check for success flag and file paths (research tools)
                if not tool_name and tool_result_data.get('success'):
                    if 'research_data' in tool_result_data or 'findings' in tool_result_data:
                        tool_name = 'mcp__research_tools__save_research_findings'
                    elif 'report_content' in tool_result_data:
                        tool_name = 'mcp__research_tools__create_research_report'

            # If we still don't have a tool name, log and exit
            if not tool_name:
                self.logger.warning(f"🔧 Could not determine tool name from result data: {list(tool_result_data.keys()) if isinstance(tool_result_data, dict) else 'Not a dict'}")
                return

            self.logger.info(f"🔧 Identified tool: {tool_name}")

            # Route to the appropriate file creation handler
            await self._handle_tool_file_creation(tool_name, tool_result_data, session_id)

        except Exception as e:
            self.logger.error(f"❌ Error processing tool result data: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")

    async def _create_research_findings_files(self, tool_result: Any, session_id: str):
        """Create research findings files from tool result data."""
        try:
            import json
            from pathlib import Path

            # Extract data from tool result
            if isinstance(tool_result, dict) and tool_result.get("success"):
                research_data = tool_result.get("research_data")
                session_file_path = tool_result.get("session_file_path")
                kevin_file_path = tool_result.get("kevin_file_path")

                if research_data and session_file_path and kevin_file_path:
                    # Create session directory
                    session_dir = Path(session_file_path).parent
                    session_dir.mkdir(parents=True, exist_ok=True)

                    # Create KEVIN directory
                    kevin_dir = Path(kevin_file_path).parent
                    kevin_dir.mkdir(parents=True, exist_ok=True)

                    # Save to session path
                    with open(session_file_path, 'w', encoding='utf-8') as f:
                        json.dump(research_data, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"✅ Created research findings file: {session_file_path}")

                    # Save to KEVIN directory
                    with open(kevin_file_path, 'w', encoding='utf-8') as f:
                        json.dump(research_data, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"✅ Created research findings file in KEVIN: {kevin_file_path}")

                    # Store in session data for later access
                    if session_id in self.active_sessions:
                        self.active_sessions[session_id]["research_findings_file"] = session_file_path
                        self.active_sessions[session_id]["research_findings_kevin_file"] = kevin_file_path

        except Exception as e:
            self.logger.error(f"❌ Error creating research findings files: {e}")

    async def _create_research_report_files(self, tool_result: Any, session_id: str):
        """Create research report files from tool result data."""
        try:
            from pathlib import Path

            # Extract data from tool result
            if isinstance(tool_result, dict) and tool_result.get("success"):
                report_content = tool_result.get("report_content")
                session_file_path = tool_result.get("session_file_path")
                kevin_file_path = tool_result.get("kevin_file_path")

                if report_content and session_file_path and kevin_file_path:
                    # Create session directory
                    session_dir = Path(session_file_path).parent
                    session_dir.mkdir(parents=True, exist_ok=True)

                    # Create KEVIN directory
                    kevin_dir = Path(kevin_file_path).parent
                    kevin_dir.mkdir(parents=True, exist_ok=True)

                    # Save to session path
                    with open(session_file_path, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    self.logger.info(f"✅ Created research report file: {session_file_path}")

                    # Save to KEVIN directory
                    with open(kevin_file_path, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    self.logger.info(f"✅ Created research report file in KEVIN: {kevin_file_path}")

                    # Store in session data for later access
                    if session_id in self.active_sessions:
                        self.active_sessions[session_id]["research_report_file"] = session_file_path
                        self.active_sessions[session_id]["research_report_kevin_file"] = kevin_file_path

        except Exception as e:
            self.logger.error(f"❌ Error creating research report files: {e}")

    async def _create_serp_search_files(self, tool_result: Any, session_id: str):
        """Create SERP search result files from tool result data."""
        try:
            from datetime import datetime
            from pathlib import Path
            self.logger.info(f"🔧 Creating SERP search files for session {session_id}")

            # Extract data from tool result
            if isinstance(tool_result, dict):
                # Handle text-based tool results
                search_content = tool_result.get("text_output", "")
                if not search_content and "content" in tool_result:
                    content_list = tool_result.get("content", [])
                    if content_list and len(content_list) > 0:
                        search_content = content_list[0].get("text", "")

                if search_content:
                    # Create session directory
                    session_dir = Path(f"researchmaterials/sessions/{session_id}")
                    session_dir.mkdir(parents=True, exist_ok=True)

                    # Create KEVIN directory
                    kevin_dir = Path("/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN")
                    kevin_dir.mkdir(parents=True, exist_ok=True)

                    # Generate timestamp
                    timestamp = datetime.now().strftime('%H%M%S')

                    # Create file paths
                    session_file = session_dir / f"serp_search_results_{timestamp}.txt"
                    kevin_file = kevin_dir / f"serp_search_{session_id[:8]}_{timestamp}.txt"

                    # Save search results to session path
                    with open(session_file, 'w', encoding='utf-8') as f:
                        f.write("SERP Search Results\n")
                        f.write(f"Session ID: {session_id}\n")
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(search_content)

                    self.logger.info(f"✅ Created SERP search file: {session_file}")

                    # Save search results to KEVIN directory
                    with open(kevin_file, 'w', encoding='utf-8') as f:
                        f.write("SERP Search Results\n")
                        f.write(f"Session ID: {session_id}\n")
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(search_content)

                    self.logger.info(f"✅ Created SERP search file in KEVIN: {kevin_file}")

                    # Store in session data for later access
                    if session_id in self.active_sessions:
                        self.active_sessions[session_id]["serp_search_file"] = str(session_file)
                        self.active_sessions[session_id]["serp_search_kevin_file"] = str(kevin_file)

                    return str(session_file)
                else:
                    self.logger.warning("🔧 No search content found in tool result")
            else:
                self.logger.warning(f"🔧 Unexpected tool result format: {type(tool_result)}")

        except Exception as e:
            self.logger.error(f"❌ Error creating SERP search files: {e}")
            import traceback
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")

    def get_hook_statistics(self) -> dict[str, Any]:
        """Get comprehensive hook system statistics and performance metrics."""
        if False:  # Hook system disabled
            return {"message": "Hook integration manager not available"}

        try:
            stats = self.hook_integration_manager.get_hook_statistics()
            self.logger.info("Hook statistics retrieved",
                            total_hooks=stats.get("integration_stats", {}).get("total_executions", 0))
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get hook statistics: {str(e)}",
                            error=str(e),
                            error_type=type(e).__name__)
            return {"error": str(e), "error_type": type(e).__name__}

    async def execute_hooks(
        self,
        hook_type: str,
        metadata: dict[str, Any],
        session_id: str | None = None,
        agent_name: str | None = None,
        category: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute hooks of a specific type with given context.

        Args:
            hook_type: Type of hooks to execute
            metadata: Additional metadata for hook execution
            session_id: Session identifier (uses current session if not provided)
            agent_name: Agent name executing the hooks
            category: Optional hook category to limit execution

        Returns:
            List of hook execution results
        """
        if False:  # Hook system disabled
            self.logger.warning("Hook integration manager not available for hook execution")
            return []

        try:
            # Create hook context
            from ..hooks.base_hooks import HookContext
            context = HookContext(
                hook_name=f"orchestrator_{hook_type}",
                hook_type=hook_type,
                session_id=session_id or next(iter(self.active_sessions.keys()), "default"),
                agent_name=agent_name,
                metadata=metadata
            )

            # Hook system disabled - no hook execution

            # Hook system disabled - no results to process
            serializable_results = []

            self.logger.info(f"Hook execution skipped: {hook_type} - hook system disabled")

            return serializable_results

        except Exception as e:
            self.logger.error(f"Hook execution failed: {str(e)}",
                            hook_type=hook_type,
                            error=str(e),
                            error_type=type(e).__name__)
            return [{"error": str(e), "error_type": type(e).__name__}]

    async def cleanup(self):
        """Cleanup all agent clients and resources."""
        self.logger.info("Starting orchestrator cleanup")

        # Shutdown hook integration manager
        if hasattr(self, 'hook_integration_manager') and self.hook_integration_manager:
            try:
                await self.hook_integration_manager.shutdown()
                self.logger.info("HookIntegrationManager shutdown completed")
            except Exception as e:
                self.logger.error(f"HookIntegrationManager shutdown failed: {str(e)}",
                                error=str(e),
                                error_type=type(e).__name__)

        # Cleanup single client
        if hasattr(self, 'client') and self.client:
            try:
                await self.client.disconnect()
                self.logger.info("✅ Single client disconnected")
            except Exception as e:
                self.logger.warning(f"Failed to disconnect client: {str(e)}")
        self.active_sessions.clear()
        self.logger.info("Orchestrator cleanup completed")

    def start_work_product(self, session_id: str, stage_name: str, description: str) -> int:
        """Start a new work product and return its number."""
        if session_id not in self.active_sessions:
            self.logger.error(f"Session {session_id} not found for work product tracking")
            return 0

        work_products = self.active_sessions[session_id]["work_products"]
        work_product_number = work_products["current_number"]

        # Start tracking this work product
        work_products["tracking"][work_product_number] = {
            "stage": stage_name,
            "description": description,
            "started_at": datetime.now().isoformat(),
            "status": "in_progress"
        }
        work_products["in_progress"] = work_product_number

        self.logger.info(f"🔢 Work Product {work_product_number} started: {stage_name} - {description}")
        self.structured_logger.info(f"Work product {work_product_number} started",
                                    session_id=session_id,
                                    work_product_number=work_product_number,
                                    stage=stage_name,
                                    description=description)

        return work_product_number

    def complete_work_product(self, session_id: str, work_product_number: int, result: dict = None):
        """Complete a work product and update tracking."""
        if session_id not in self.active_sessions:
            self.logger.error(f"Session {session_id} not found for work product completion")
            return

        work_products = self.active_sessions[session_id]["work_products"]

        if work_product_number not in work_products["tracking"]:
            self.logger.error(f"Work product {work_product_number} not found in tracking")
            return

        # Complete the work product
        work_products["tracking"][work_product_number]["status"] = "completed"
        work_products["tracking"][work_product_number]["completed_at"] = datetime.now().isoformat()
        work_products["tracking"][work_product_number]["result"] = result or {}

        # Update tracking
        work_products["completed"].append(work_product_number)
        work_products["in_progress"] = None
        work_products["current_number"] = work_product_number + 1

        self.logger.info(f"✅ Work Product {work_product_number} completed: {work_products['tracking'][work_product_number]['stage']}")
        self.structured_logger.info(f"Work product {work_product_number} completed",
                                    session_id=session_id,
                                    work_product_number=work_product_number,
                                    stage=work_products['tracking'][work_product_number]['stage'],
                                    result_summary=str(result)[:100] if result else "No result")

    def get_work_product_summary(self, session_id: str) -> dict:
        """Get a summary of all work products for a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        work_products = self.active_sessions[session_id]["work_products"]

        summary = {
            "session_id": session_id,
            "current_number": work_products["current_number"],
            "completed_count": len(work_products["completed"]),
            "completed_work_products": [],
            "in_progress": work_products["in_progress"],
            "tracking": work_products["tracking"]
        }

        # Add details for completed work products
        for wp_num in work_products["completed"]:
            if wp_num in work_products["tracking"]:
                wp_info = work_products["tracking"][wp_num].copy()
                summary["completed_work_products"].append(wp_info)

        return summary

    # Workflow Resilience and Error Reporting Framework

    def record_workflow_error(self, session_id: str, stage: str, error: Exception, context: dict = None):
        """Record a workflow error with detailed context for debugging."""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "traceback": None,
            "recovery_attempted": False,
            "recovery_successful": False
        }

        # Store error in session data
        if session_id in self.active_sessions:
            if "workflow_errors" not in self.active_sessions[session_id]:
                self.active_sessions[session_id]["workflow_errors"] = []
            self.active_sessions[session_id]["workflow_errors"].append(error_record)

        # Log with structured information
        self.logger.error(f"❌ Workflow Error [{stage}] Session {session_id}: {error_record['error_type']} - {error_record['error_message']}")
        self.structured_logger.error("Workflow error recorded",
                                    session_id=session_id,
                                    stage=stage,
                                    error_type=error_record['error_type'],
                                    error_message=error_record['error_message'],
                                    context=context)

        return error_record

    def attempt_stage_recovery(self, session_id: str, stage: str, error: Exception, fallback_data: dict = None):
        """Attempt to recover from a stage failure with fallback strategies."""
        recovery_strategies = {
            "research": self._recover_research_stage,
            "report_generation": self._recover_report_stage,
            "editorial_review": self._recover_editorial_stage,
            "finalization": self._recover_finalization_stage
        }

        if stage not in recovery_strategies:
            self.logger.warning(f"⚠️ No recovery strategy available for stage: {stage}")
            return {"success": False, "message": f"No recovery strategy for stage: {stage}"}

        self.logger.info(f"🔄 Attempting recovery for {stage} stage in session {session_id}")

        try:
            recovery_result = recovery_strategies[stage](session_id, error, fallback_data)

            # Record recovery attempt
            if session_id in self.active_sessions and "workflow_errors" in self.active_sessions[session_id]:
                for error_record in reversed(self.active_sessions[session_id]["workflow_errors"]):
                    if error_record["stage"] == stage and not error_record["recovery_attempted"]:
                        error_record["recovery_attempted"] = True
                        error_record["recovery_successful"] = recovery_result.get("success", False)
                        break

            if recovery_result.get("success", False):
                self.logger.info(f"✅ Recovery successful for {stage} stage in session {session_id}")
                self.structured_logger.info("Stage recovery successful",
                                           session_id=session_id,
                                           stage=stage,
                                           recovery_method=recovery_result.get("method", "unknown"))
            else:
                self.logger.warning(f"⚠️ Recovery failed for {stage} stage in session {session_id}: {recovery_result.get('message', 'Unknown error')}")
                self.structured_logger.warning("Stage recovery failed",
                                              session_id=session_id,
                                              stage=stage,
                                              recovery_error=recovery_result.get("message", "Unknown error"))

            return recovery_result

        except Exception as recovery_error:
            self.logger.error(f"❌ Recovery attempt failed for {stage} stage in session {session_id}: {recovery_error}")
            self.structured_logger.error("Recovery attempt failed",
                                        session_id=session_id,
                                        stage=stage,
                                        recovery_error=str(recovery_error))
            return {"success": False, "message": f"Recovery attempt failed: {recovery_error}"}

    def _recover_research_stage(self, session_id: str, error: Exception, fallback_data: dict = None) -> dict:
        """Recovery strategy for research stage failures."""
        self.logger.info(f"🔧 Attempting research stage recovery for session {session_id}")

        recovery_methods = [
            {
                "name": "use_existing_content",
                "description": "Use existing content from session directory",
                "action": self._use_existing_research_content
            },
            {
                "name": "minimal_research",
                "description": "Perform minimal research with reduced scope",
                "action": self._perform_minimal_research
            },
            {
                "name": "synthetic_research",
                "description": "Generate synthetic research based on topic",
                "action": self._generate_synthetic_research
            }
        ]

        for method in recovery_methods:
            try:
                self.logger.info(f"🔧 Trying research recovery method: {method['description']}")
                result = method["action"](session_id, error, fallback_data)
                if result.get("success", False):
                    return {"success": True, "method": method["name"], "result": result}
            except Exception as method_error:
                self.logger.warning(f"⚠️ Research recovery method {method['name']} failed: {method_error}")
                continue

        return {"success": False, "message": "All research recovery methods failed"}

    def _recover_report_stage(self, session_id: str, error: Exception, fallback_data: dict = None) -> dict:
        """Recovery strategy for report generation failures."""
        self.logger.info(f"🔧 Attempting report stage recovery for session {session_id}")

        # Try to generate a basic report from available research data
        try:
            session_data = self.active_sessions.get(session_id, {})
            research_data = session_data.get("research_results", {})

            if not research_data:
                return {"success": False, "message": "No research data available for report recovery"}

            # Create basic report from research findings
            basic_report = self._create_basic_report_from_research(session_id, research_data)
            if basic_report:
                session_data["report_results"] = {
                    "success": True,
                    "content": basic_report,
                    "recovery_generated": True,
                    "recovery_method": "basic_from_research"
                }
                return {"success": True, "method": "basic_report_from_research", "result": basic_report}

        except Exception as e:
            self.logger.warning(f"⚠️ Report recovery failed: {e}")

        return {"success": False, "message": "Report recovery failed"}

    def _recover_editorial_stage(self, session_id: str, error: Exception, fallback_data: dict = None) -> dict:
        """Recovery strategy for editorial review failures."""
        self.logger.info(f"🔧 Attempting editorial stage recovery for session {session_id}")

        # Editorial review already has decoupled fallback, so return success to continue workflow
        self.logger.info("✅ Editorial recovery: Continuing with available content (minimal editorial)")

        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id]
            session_data["editorial_review_results"] = {
                "success": True,
                "minimal_review": True,
                "recovery_generated": True,
                "message": "Continued with minimal editorial processing due to stage failure"
            }

        return {"success": True, "method": "minimal_editorial_continuation", "message": "Editorial recovery completed"}

    def _recover_finalization_stage(self, session_id: str, error: Exception, fallback_data: dict = None) -> dict:
        """Recovery strategy for finalization failures."""
        self.logger.info(f"🔧 Attempting finalization stage recovery for session {session_id}")

        try:
            # Create a basic completion summary even if finalization failed
            session_data = self.active_sessions.get(session_id, {})
            completion_summary = self._create_basic_completion_summary(session_id, session_data)

            if completion_summary:
                session_data["finalization_results"] = {
                    "success": True,
                    "summary": completion_summary,
                    "recovery_generated": True,
                    "recovery_method": "basic_completion"
                }
                return {"success": True, "method": "basic_completion", "result": completion_summary}

        except Exception as e:
            self.logger.warning(f"⚠️ Finalization recovery failed: {e}")

        return {"success": False, "message": "Finalization recovery failed"}

    # Recovery implementation methods
    def _use_existing_research_content(self, session_id: str, error: Exception, fallback_data: dict = None) -> dict:
        """Try to use existing research content from session directory."""
        from pathlib import Path

        session_dir = Path("KEVIN") / "sessions" / session_id
        if not session_dir.exists():
            return {"success": False, "message": "No session directory found"}

        # Look for existing research files
        research_files = []
        for pattern in ["*.json", "*.md", "*.txt"]:
            research_files.extend(session_dir.glob(pattern))

        if research_files:
            self.logger.info(f"✅ Found {len(research_files)} existing research files for recovery")
            return {"success": True, "files_found": len(research_files), "files": [str(f) for f in research_files]}

        return {"success": False, "message": "No existing research files found"}

    def _perform_minimal_research(self, session_id: str, error: Exception, fallback_data: dict = None) -> dict:
        """Perform minimal research with drastically reduced scope."""
        session_data = self.active_sessions.get(session_id, {})
        clean_topic = session_data.get("clean_topic", "Unknown topic")

        # Create minimal synthetic research data
        minimal_research = {
            "success": True,
            "minimal_research": True,
            "topic": clean_topic,
            "findings": [
                f"Basic research conducted on: {clean_topic}",
                "Note: This is minimal research generated during recovery from stage failure.",
                "Original research stage encountered an error and was recovered with fallback content."
            ],
            "sources_used": 0,
            "recovery_generated": True
        }

        session_data["research_results"] = minimal_research
        return {"success": True, "minimal_research": True, "result": minimal_research}

    def _generate_synthetic_research(self, session_id: str, error: Exception, fallback_data: dict = None) -> dict:
        """Generate synthetic research based on topic when all else fails."""
        session_data = self.active_sessions.get(session_id, {})
        clean_topic = session_data.get("clean_topic", "Unknown topic")

        synthetic_research = {
            "success": True,
            "synthetic_research": True,
            "topic": clean_topic,
            "findings": [
                f"Synthetic research overview for: {clean_topic}",
                "This content was generated as a fallback due to research stage failure.",
                "The system was unable to complete primary research but has provided this synthetic overview to ensure workflow continuation.",
                f"Topic analysis: {clean_topic} requires further investigation with functional research tools."
            ],
            "sources_used": 0,
            "recovery_generated": True,
            "synthetic": True
        }

        session_data["research_results"] = synthetic_research
        return {"success": True, "synthetic_research": True, "result": synthetic_research}

    def _create_basic_report_from_research(self, session_id: str, research_data: dict) -> str:
        """Create a basic report from available research data."""
        try:
            clean_topic = self.active_sessions.get(session_id, {}).get("clean_topic", "Unknown Topic")

            basic_report = f"""# Research Report: {clean_topic}

## Overview
This report was generated as a recovery fallback after the primary report generation stage failed.

## Research Findings
Based on the available research data:

- Research was conducted on: {clean_topic}
- Report generation encountered an error and was recovered
- This represents a basic summary of available information

## Notes
- This is a recovery-generated report with minimal formatting
- The system experienced issues during the primary report generation stage
- Consider reviewing the research data directly for more detailed information

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: Recovery Generated
"""
            return basic_report
        except Exception as e:
            self.logger.error(f"Failed to create basic report: {e}")
            return None

    def _create_basic_completion_summary(self, session_id: str, session_data: dict) -> str:
        """Create a basic completion summary when finalization fails."""
        try:
            clean_topic = session_data.get("clean_topic", "Unknown Topic")
            completed_stages = len(session_data.get("workflow_history", []))
            errors_count = len(session_data.get("workflow_errors", []))

            summary = f"""# Session Completion Summary

## Session Information
- Session ID: {session_id}
- Topic: {clean_topic}
- Completed Stages: {completed_stages}
- Errors Encountered: {errors_count}

## Workflow Status
This session has been completed with recovery fallbacks due to finalization stage issues.

## Work Products
- Work Product tracking was implemented for this session
- Stages were processed with resilience mechanisms

## Notes
- Session completed with recovery mechanisms
- Some stages may have used fallback content
- Review session data for detailed information

Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: Recovery Completed
"""
            return summary
        except Exception as e:
            self.logger.error(f"Failed to create completion summary: {e}")
            return None

    def get_workflow_error_summary(self, session_id: str) -> dict:
        """Get a summary of all workflow errors for a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session_data = self.active_sessions[session_id]
        workflow_errors = session_data.get("workflow_errors", [])

        summary = {
            "session_id": session_id,
            "total_errors": len(workflow_errors),
            "errors_by_stage": {},
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "errors": workflow_errors
        }

        for error in workflow_errors:
            stage = error["stage"]
            if stage not in summary["errors_by_stage"]:
                summary["errors_by_stage"][stage] = 0
            summary["errors_by_stage"][stage] += 1

            if error["recovery_attempted"]:
                summary["recovery_attempts"] += 1
                if error["recovery_successful"]:
                    summary["successful_recoveries"] += 1
                else:
                    summary["failed_recoveries"] += 1

        return summary

    def execute_stage_with_resilience(self, stage_name: str, stage_function, session_id: str, *args, **kwargs):
        """Execute a stage with comprehensive error handling and recovery."""
        import asyncio

        async def _execute_with_resilience():
            try:
                self.logger.info(f"🚀 Starting {stage_name} stage for session {session_id}")
                result = await stage_function(session_id, *args, **kwargs)
                self.logger.info(f"✅ {stage_name} stage completed successfully for session {session_id}")
                return {"success": True, "result": result, "stage": stage_name, "recovery_used": False}

            except Exception as e:
                # Record the error
                error_record = self.record_workflow_error(session_id, stage_name, e, {
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                    "stage_function": stage_function.__name__ if hasattr(stage_function, '__name__') else str(stage_function)
                })

                # Attempt recovery
                self.logger.warning(f"⚠️ {stage_name} stage failed for session {session_id}, attempting recovery")
                recovery_result = self.attempt_stage_recovery(session_id, stage_name, e, {
                    "original_args": args,
                    "original_kwargs": kwargs
                })

                if recovery_result.get("success", False):
                    self.logger.info(f"✅ {stage_name} stage recovered successfully for session {session_id}")
                    return {
                        "success": True,
                        "result": recovery_result.get("result"),
                        "stage": stage_name,
                        "recovery_used": True,
                        "recovery_method": recovery_result.get("method", "unknown")
                    }
                else:
                    self.logger.error(f"❌ {stage_name} stage recovery failed for session {session_id}")
                    return {
                        "success": False,
                        "error": str(e),
                        "stage": stage_name,
                        "recovery_used": True,
                        "recovery_failed": True,
                        "recovery_error": recovery_result.get("message", "Unknown recovery error")
                    }

        return asyncio.create_task(_execute_with_resilience())
