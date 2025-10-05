"""
zPlayground1 Search, Scrape & Clean MCP Tool

This is a single, comprehensive MCP tool that exactly mirrors the zPlayground1
topic-based search, scrape, and clean functionality. No multiple tool calls -
just one complete implementation that handles everything in Python.

Based on the proven zPlayground1 implementation, fully integrated for standalone operation.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Claude Agent SDK MCP functionality
try:
    from claude_agent_sdk import create_sdk_mcp_server, tool
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Claude Agent SDK not available for MCP integration")
    CLAUDE_SDK_AVAILABLE = False

# Import the exact zPlayground1 search functionality
from utils.z_search_crawl_utils import (
    news_search_and_crawl_direct,
    search_crawl_and_clean_direct,
)

logger = logging.getLogger(__name__)


def create_zplayground1_mcp_server():
    """
    Create the single zPlayground1 MCP server.

    This is ONE tool that does everything - search, scrape, and clean.
    No multiple MCP tool calls, just pure Python processing with thin MCP wrapper.

    Returns:
        MCP server instance or None if Claude SDK unavailable
    """
    if not CLAUDE_SDK_AVAILABLE:
        logger.error("Cannot create MCP server: Claude Agent SDK not available")
        return None

    @tool(
        "zplayground1_search_scrape_clean",
        "Complete zPlayground1 topic-based search, scrape, and clean functionality. Performs Google search/SERP news search, extracts content from multiple URLs in parallel using exact zPlayground1 implementation with progressive anti-bot detection, applies AI content cleaning, and returns comprehensive results within token limits. This is a single tool that handles the complete workflow - no multiple tool calls needed.",
        {
            "query": {
                "type": "string",
                "description": "Search query, topic, or news topic to research"
            },
            "search_mode": {
                "type": "string",
                "enum": ["web", "news"],
                "default": "web",
                "description": "Search mode: 'web' for Google search, 'news' for news search"
            },
            "num_results": {
                "type": "integer",
                "default": 15,
                "minimum": 1,
                "maximum": 50,
                "description": "Number of search results to retrieve"
            },
            "auto_crawl_top": {
                "type": "integer",
                "default": 10,
                "minimum": 0,
                "maximum": 20,
                "description": "Maximum number of URLs to crawl for content extraction"
            },
            "crawl_threshold": {
                "type": "number",
                "default": 0.3,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Minimum relevance score threshold for crawling (0.0-1.0)"
            },
            "anti_bot_level": {
                "type": "integer",
                "default": 1,
                "minimum": 0,
                "maximum": 3,
                "description": "Progressive anti-bot level: 0=basic, 1=enhanced, 2=advanced, 3=stealth"
            },
            "max_concurrent": {
                "type": "integer",
                "default": 15,
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum concurrent crawling operations"
            },
            "session_id": {
                "type": "string",
                "default": "default",
                "description": "Session identifier for tracking and work products"
            },
            "workproduct_prefix": {
                "type": "string",
                "default": "",
                "description": "Optional prefix for work product filenames (e.g., 'editor research' for editorial work)"
            },
            "target_scrapes": {
                "type": "integer",
                "default": 15,
                "minimum": 1,
                "maximum": 50,
                "description": "Target number of successful content extractions - tool will automatically stop when reached"
            }
        }
    )
    async def zplayground1_search_scrape_clean(args: dict[str, Any]) -> dict[str, Any]:
        """
        Complete zPlayground1 search, scrape, and clean functionality.

        This single tool performs the entire zPlayground1 workflow:
        1. Executes Google search or SERP news search based on search_mode
        2. Selects relevant URLs using exact zPlayground1 relevance scoring
        3. Crawls multiple URLs in parallel using exact zPlayground1 SimpleCrawler
        4. Applies progressive anti-bot detection (levels 0-3)
        5. Uses AI content cleaning (GPT-5-nano) via Pydantic AI
        6. Returns cleaned results within token limits
        7. Saves detailed work products for reference

        NO FALLBACKS - Uses exact zPlayground1 implementation that works effectively.
        If it fails, it fails loudly so we know to fix it properly.

        Args:
            args: Dictionary containing all tool parameters

        Returns:
            Dictionary with content and metadata
        """
        try:
            # Extract parameters with FAIL-FAST validation
            query = args.get("query")
            if not query:
                error_msg = "❌ **CRITICAL ERROR**: 'query' parameter is required and cannot be empty!"
                logger.error(f"FAIL-FAST: {error_msg}")
                return {
                    "content": [{"type": "text", "text": error_msg}],
                    "is_error": True
                }

            # Extract and validate parameters with detailed error reporting
            try:
                search_mode = args.get("search_mode", "web")
                if search_mode not in ["web", "news"]:
                    raise ValueError(f"Invalid search_mode '{search_mode}'. Must be 'web' or 'news'")

                num_results = int(args.get("num_results", 15))
                if not (1 <= num_results <= 50):
                    raise ValueError(f"Invalid num_results '{num_results}'. Must be between 1 and 50")

                auto_crawl_top = int(args.get("auto_crawl_top", 10))
                if not (0 <= auto_crawl_top <= 20):
                    raise ValueError(f"Invalid auto_crawl_top '{auto_crawl_top}'. Must be between 0 and 20")

                crawl_threshold = float(args.get("crawl_threshold", 0.3))
                if not (0.0 <= crawl_threshold <= 1.0):
                    raise ValueError(f"Invalid crawl_threshold '{crawl_threshold}'. Must be between 0.0 and 1.0")

                # FAIL-FAST: Fix the parameter validation issue for anti_bot_level
                anti_bot_level_raw = args.get("anti_bot_level", 1)
                logger.debug(f"Raw anti_bot_level parameter: {anti_bot_level_raw} (type: {type(anti_bot_level_raw)})")

                try:
                    anti_bot_level = int(anti_bot_level_raw)
                except (ValueError, TypeError) as e:
                    error_msg = f"FAIL-FAST: Invalid anti_bot_level parameter '{anti_bot_level_raw}' (type: {type(anti_bot_level_raw)}). Must be an integer between 0 and 3!"
                    logger.error(f"❌ {error_msg}")
                    logger.error(f"This is the exact error that was causing the system to fail silently!")
                    logger.error(f"Parameter validation error: {e}")
                    raise ValueError(error_msg)

                if not (0 <= anti_bot_level <= 3):
                    raise ValueError(f"Invalid anti_bot_level '{anti_bot_level}'. Must be between 0 and 3")

                max_concurrent = int(args.get("max_concurrent", 15))
                if not (1 <= max_concurrent <= 20):
                    raise ValueError(f"Invalid max_concurrent '{max_concurrent}'. Must be between 1 and 20")

                target_scrapes = int(args.get("target_scrapes", 15))
                if not (1 <= target_scrapes <= 50):
                    raise ValueError(f"Invalid target_scrapes '{target_scrapes}'. Must be between 1 and 50")

                session_id = args.get("session_id", "default")
                workproduct_prefix = args.get("workproduct_prefix", "")

            except (ValueError, TypeError) as param_error:
                error_msg = f"❌ **CRITICAL PARAMETER VALIDATION ERROR**: {param_error}"
                logger.error(f"FAIL-FAST PARAMETER ERROR: {error_msg}")
                logger.error("This error would have been silently ignored before - now we fail fast!")
                return {
                    "content": [{"type": "text", "text": error_msg}],
                    "is_error": True,
                    "error_details": str(param_error)
                }

            # Parameter validation is now handled above with FAIL-FAST approach

            # NOTE: Do NOT set workproduct_dir here - let search functions use session_id
            # to automatically create proper session-based directory structure and filenames.
            # The search functions check `if workproduct_dir is None` to use session structure.

            logger.info(f"🚀 zPlayground1 executing: query='{query}', search_mode='{search_mode}', anti_bot_level={anti_bot_level}")

            # Execute the exact zPlayground1 search and extract functionality
            # Pass workproduct_dir=None to trigger session-based structure with "1-" prefix
            if search_mode == "news":
                # Use news search and crawl
                result = await news_search_and_crawl_direct(
                    query=query,
                    num_results=num_results,
                    auto_crawl_top=auto_crawl_top,
                    session_id=session_id,
                    anti_bot_level=anti_bot_level,
                    workproduct_dir=None,  # Let function use session-based structure
                    target_scrapes=target_scrapes
                )
            else:
                # Use web search and crawl
                result = await search_crawl_and_clean_direct(
                    query=query,
                    search_type="search",
                    num_results=num_results,
                    auto_crawl_top=auto_crawl_top,
                    crawl_threshold=crawl_threshold,
                    max_concurrent=max_concurrent,
                    session_id=session_id,
                    anti_bot_level=anti_bot_level,
                    workproduct_dir=None,  # Let function use session-based structure
                    target_scrapes=target_scrapes
                )

            # Apply MCP compliance with multi-level content allocation
            from .mcp_compliance_manager import get_mcp_compliance_manager

            mcp_manager = get_mcp_compliance_manager()

            # Prepare metadata for MCP compliance
            # Calculate actual workproduct path for metadata
            session_research_dir = f"/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN/sessions/{session_id}/research"
            base_metadata = {
                "query": query,
                "search_mode": search_mode,
                "anti_bot_level": anti_bot_level,
                "session_id": session_id,
                "workproduct_dir": session_research_dir,
                "implementation": "zPlayground1_exact",
                "parallel_processing": True,
                "ai_content_cleaning": True
            }

            # Context for content allocation
            allocation_context = {
                "query": query,
                "query_terms": query.split(),
                "session_id": session_id,
                "source_count": len([line for line in result.split('\n') if 'URL:' in line]),
                "processing_time": "N/A"  # Would be calculated in production
            }

            # Apply MCP compliance allocation
            allocation = mcp_manager.allocate_content(
                raw_content=result,
                metadata=base_metadata,
                context=allocation_context
            )

            logger.info(f"MCP compliance applied: {allocation.token_usage['total']:,} tokens "
                       f"({allocation.token_usage['utilization']:.1f}% utilization), "
                       f"compression: {allocation.compression_applied}")

            # Combine primary content with metadata
            final_content = f"""{allocation.primary_content}

---

{allocation.metadata_content}
"""

            return {
                "content": [{"type": "text", "text": final_content}],
                "metadata": {
                    **base_metadata,
                    "mcp_compliance": True,
                    "token_usage": allocation.token_usage,
                    "allocation_stats": allocation.allocation_stats,
                    "compression_applied": allocation.compression_applied,
                    "priority_distribution": allocation.priority_distribution
                }
            }

        except Exception as e:
            error_msg = f"""❌ **zPlayground1 Search Error**

Failed to execute search and content extraction using zPlayground1 implementation: {str(e)}

**This uses the exact zPlayground1 implementation that works effectively.**
If this failed, there's an issue with the setup that needs to be addressed.

Please check:
- SERP_API_KEY is configured
- Network connectivity
- Query parameters are valid
- zPlayground1 utilities are properly imported
"""
            logger.error(f"zPlayground1 search failed: {e}")
            return {
                "content": [{"type": "text", "text": error_msg}],
                "is_error": True,
                "error_details": str(e)
            }

    # Create the MCP server with this single comprehensive tool
    server = create_sdk_mcp_server(
        name="zplayground1_search_scrape_clean",
        version="1.0.0",
        tools=[zplayground1_search_scrape_clean]
    )

    logger.info("✅ zPlayground1 Search, Scrape & Clean MCP server created successfully")
    logger.info("📋 Single tool approach - no multiple MCP tool calls needed")
    logger.info("🔧 Exact zPlayground1 implementation with no fallbacks")

    return server


# Create server instance
zplayground1_server = create_zplayground1_mcp_server()

# Export server for integration
__all__ = [
    'zplayground1_server',
    'create_zplayground1_mcp_server'
]
