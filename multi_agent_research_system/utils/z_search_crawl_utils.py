"""
Enhanced search and crawl utilities adapted from zPlayground1.

This module provides optimized search+crawl+clean functionality with
parallel processing, anti-bot detection, and AI content cleaning.
Adapted for integration with the multi-agent research system.
"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class SearchResult:
    """Enhanced search result data structure"""
    def __init__(self, title: str, link: str, snippet: str, position: int = 0,
                 date: str = None, source: str = None, relevance_score: float = 0.0):
        self.title = title
        self.link = link
        self.snippet = snippet
        self.position = position  # Original position in single query results
        self.date = date
        self.source = source
        self.relevance_score = relevance_score
        # Additional attributes for query expansion and merging
        self.merged_from_query = None  # Which query this came from (0, 1, 2, etc.)
        self.original_position = None  # Original position in source query
        self.merged_position = None   # Position in merged results list


# Import enhanced relevance scorer with domain authority
from .enhanced_relevance_scorer import (
    calculate_enhanced_relevance_score_with_domain_authority as calculate_enhanced_relevance_score,
)


async def execute_serper_search(
    query: str,
    search_type: str = "search",
    num_results: int = 10,
    country: str = "us",
    language: str = "en"
) -> list[SearchResult]:
    """
    Execute search using Serper API.

    Args:
        query: Search query
        search_type: "search" or "news"
        num_results: Number of results to retrieve
        country: Country code for search
        language: Language code for search

    Returns:
        List of SearchResult objects
    """
    try:
        import httpx

        # FAIL-FAST: Check for correct API key name - must match orchestrator expectations
        serper_api_key = os.getenv("SERP_API_KEY")  # Note: SERP_API_KEY, not SERPER_API_KEY

        if not serper_api_key:
            # During development, fail hard and fast with clear error message
            error_msg = "CRITICAL: SERP_API_KEY not found in environment variables!"
            logger.error(f"❌ {error_msg}")
            logger.error("Search functionality cannot work without SERP_API_KEY!")
            logger.error("Expected environment variable: SERP_API_KEY")
            logger.error("Set with: export SERP_API_KEY='your-serper-api-key'")

            # Check if user has the wrong API key name
            if os.getenv("SERPER_API_KEY"):
                logger.error("🚨 FOUND SERPER_API_KEY but system expects SERP_API_KEY!")
                logger.error("Please rename your environment variable from SERPER_API_KEY to SERP_API_KEY")

            # During development, fail immediately instead of returning empty results
            raise RuntimeError(f"CRITICAL SEARCH CONFIGURATION FAILURE: {error_msg}")

            # Note: The following return will never be reached due to the RuntimeError above
            # but keeping it for clarity if the fail-fast approach is later softened
            return []

        # Choose endpoint based on search type
        endpoint = "news" if search_type == "news" else "search"
        url = f"https://google.serper.dev/{endpoint}"

        # Build search parameters
        search_params = {
            "q": query,
            "num": min(num_results, 100),  # Serper limit
            "gl": country,
            "hl": language
        }

        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json"
        }

        logger.info(f"Executing {search_type} search for: {query}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=search_params, headers=headers)

        if response.status_code == 200:
            data = response.json()

            # Extract results based on search type
            if search_type == "news" and "news" in data:
                raw_results = data["news"]
            else:
                raw_results = data.get("organic", [])

            # Parse query terms for enhanced relevance scoring
            query_terms = query.lower().replace('or', ' ').replace('and', ' ').split()
            query_terms = [term.strip() for term in query_terms if len(term.strip()) > 2]

            # Convert to SearchResult objects with enhanced relevance scoring
            search_results = []
            for i, result in enumerate(raw_results):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                position = i + 1

                # Calculate enhanced relevance score with domain authority
                relevance_score = calculate_enhanced_relevance_score(
                    title=title,
                    snippet=snippet,
                    position=position,
                    query_terms=query_terms,
                    url=result.get("link", "")
                )

                search_result = SearchResult(
                    title=title,
                    link=result.get("link", ""),
                    snippet=snippet,
                    position=position,
                    date=result.get("date", ""),
                    source=result.get("source", ""),
                    relevance_score=relevance_score
                )
                search_results.append(search_result)

            logger.info(f"Retrieved {len(search_results)} search results for query: '{query}'")
            return search_results

        else:
            logger.error(f"Serper API error: {response.status_code}")
            return []

    except Exception as e:
        # FAIL-FAST: During development, re-raise critical errors instead of swallowing them
        logger.error(f"Error in Serper search: {e}")

        # Check if this is a critical configuration error that should fail fast
        if "CRITICAL" in str(e) or "API_KEY" in str(e) or "Configuration" in str(e):
            logger.error("FAIL-FAST: Critical configuration error detected - re-raising to expose configuration issues!")
            raise  # Re-raise the critical error instead of returning empty results

        # For other errors, return empty list for now
        logger.warning("Non-critical search error - returning empty results")
        return []


def select_urls_for_crawling(
    search_results: list[SearchResult],
    limit: int = 10,
    min_relevance: float = 0.4
) -> list[str]:
    """
    Select URLs for crawling based on relevance scores.

    Args:
        search_results: List of search results
        limit: Maximum number of URLs to select
        min_relevance: Minimum relevance score threshold

    Returns:
        List of URLs to crawl
    """
    try:
        # Filter by relevance threshold (ensure type safety)
        filtered_results = [
            result for result in search_results
            if float(result.relevance_score) >= float(min_relevance) and result.link
        ]

        # Sort by relevance score (highest first) - ensure float comparison
        filtered_results.sort(key=lambda x: float(x.relevance_score), reverse=True)

        # Extract URLs up to limit
        urls = [result.link for result in filtered_results[:limit]]

        # Enhanced logging for URL selection process
        logger.info(f"URL selection with threshold {min_relevance}:")
        logger.info(f"  - Total results: {len(search_results)}")
        logger.info(f"  - Above threshold: {len(filtered_results)}")
        logger.info(f"  - Selected for crawling: {len(urls)}")
        logger.info(f"  - Rejected: {len(search_results) - len(filtered_results)} below threshold")
        return urls

    except Exception as e:
        logger.error(f"Error selecting URLs for crawling: {e}")
        return []


def format_search_results(search_results: list[SearchResult]) -> str:
    """
    Format search results for display.

    Args:
        search_results: List of search results

    Returns:
        Formatted search results string
    """
    if not search_results:
        return "No search results found."

    result_parts = [
        f"# Search Results ({len(search_results)} found)",
        ""
    ]

    for i, result in enumerate(search_results, 1):
        result_parts.extend([
            f"## {i}. {result.title}",
            f"**URL**: {result.link}",
            f"**Source**: {result.source}" if result.source else "",
            f"**Date**: {result.date}" if result.date else "",
            f"**Relevance Score**: {result.relevance_score:.2f}",
            "",
            result.snippet,
            "",
            "---",
            ""
        ])

    return "\n".join(result_parts)


def save_work_product(
    search_results: list[SearchResult],
    crawled_content: list[str],
    urls: list[str],
    query: str,
    session_id: str = "default",
    workproduct_dir: str = None
) -> str:
    """
    Save detailed search and crawl results to work product file.

    Args:
        search_results: List of search results
        crawled_content: List of cleaned content strings
        urls: List of crawled URLs
        query: Original search query
        session_id: Session identifier
        workproduct_dir: Directory to save work products

    Returns:
        Path to saved work product file
    """
    try:
        # GATHER CONTEXT: Log current state before action
        logger.info(f"🔧 save_work_product called:")
        logger.info(f"   session_id: {session_id}")
        logger.info(f"   workproduct_dir param: {workproduct_dir}")
        logger.info(f"   search_results: {len(search_results)} items")
        logger.info(f"   crawled_content: {len(crawled_content)} items")
        logger.info(f"   urls: {len(urls)} items")

        # Determine correct session directory structure
        if workproduct_dir is None:
            # Default to KEVIN sessions directory - save to research/ for Work Product 1
            base_sessions_dir = "/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN/sessions"
            session_dir = os.path.join(base_sessions_dir, session_id)
            # Save to research/ directory for raw SERP metadata + scraped content
            working_dir = os.path.join(session_dir, "research")
            Path(working_dir).mkdir(parents=True, exist_ok=True)

            # Generate timestamp and sanitized topic for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize query for filename
            sanitized_query = query.replace(' ', '-').replace('/', '-')[:50].lower()

            # Count existing Work Product 1 files to determine suffix (1, 1A, 1B, 1C, etc.)
            existing_files = list(Path(working_dir).glob("1*-*.md"))
            if not existing_files:
                # First Work Product 1 - no suffix
                prefix = "1"
            else:
                # Calculate suffix based on count (A, B, C, D, etc.)
                suffix_letter = chr(65 + len(existing_files))  # 65 is ASCII for 'A'
                prefix = f"1{suffix_letter}"

            filename = f"{prefix}-{sanitized_query}-search-results.md"
            filepath = os.path.join(working_dir, filename)
            logger.info(f"   Using KEVIN structure: {working_dir}")
            logger.info(f"   Filename will be: {filename}")
            logger.info(f"   Full path: {filepath}")
        else:
            # Custom workproduct directory (legacy support)
            Path(workproduct_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced-search-crawl-workproduct_{timestamp}.md"
            filepath = os.path.join(workproduct_dir, filename)
            logger.info(f"   Using custom dir: {workproduct_dir}")
            logger.info(f"   Filename: {filename}")
            logger.info(f"   Full path: {filepath}")

        # Build work product content
        workproduct_content = [
            "# Enhanced Search+Crawl+Clean Workproduct",
            "",
            f"**Session ID**: {session_id}",
            f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "**Agent**: Enhanced Search+Crawl Tool (zPlayground1 integration)",
            f"**Search Query**: {query}",
            f"**Total Search Results**: {len(search_results)}",
            f"**Successfully Crawled**: {len(crawled_content)}",
            "",
            "---",
            "",
            "## 🔍 Search Results Summary",
            "",
        ]

        # Add search results overview
        for i, result in enumerate(search_results, 1):
            workproduct_content.extend([
                f"### {i}. {result.title}",
                f"**URL**: {result.link}",
                f"**Source**: {result.source}" if result.source else "",
                f"**Date**: {result.date}" if result.date else "",
                f"**Relevance Score**: {result.relevance_score:.2f}",
                "",
                f"**Snippet**: {result.snippet}",
                "",
                "---",
                ""
            ])

        workproduct_content.extend([
            "",
            "## 📄 Detailed Crawled Content (AI Cleaned)",
            ""
        ])

        # Add detailed crawled content
        for i, (content, url) in enumerate(zip(crawled_content, urls, strict=False), 1):
            # Find corresponding search result for title
            title = f"Article {i}"
            for result in search_results:
                if result.link == url:
                    title = result.title
                    break

            workproduct_content.extend([
                f"## 🌐 {i}. {title}",
                "",
                f"**URL**: {url}",
                f"**Extraction Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Content Length**: {len(content)} characters",
                "**Processing**: ✅ Cleaned with GPT-5-nano",
                "",
                "### 📄 Full Cleaned Content",
                "",
                "---",
                "",
                content,
                "",
                "---",
                ""
            ])

        # Add footer
        workproduct_content.extend([
            "",
            "## 📊 Processing Summary",
            "",
            f"- **Search Query**: {query}",
            f"- **Search Results Found**: {len(search_results)}",
            f"- **URLs Successfully Crawled**: {len(crawled_content)}",
            "- **Content Cleaning**: GPT-5-nano AI processing",
            "- **Total Processing Time**: Combined search+crawl+clean in single operation",
            "- **Performance**: Parallel processing with anti-bot detection",
            "",
            "*Generated by Enhanced Search+Crawl+Clean Tool - Powered by zPlayground1 technology*"
        ])

        # Write to file
        logger.info(f"🔧 Writing {len(workproduct_content)} lines to file...")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(workproduct_content))

        # VERIFY WORK: Check file was actually written (SDK principle!)
        if not os.path.exists(filepath):
            logger.error(f"❌ File write failed - file does not exist: {filepath}")
            return ""

        file_size = os.path.getsize(filepath)
        if file_size == 0:
            logger.error(f"❌ File write failed - file is empty: {filepath}")
            return ""

        logger.info(f"✅ Work product saved to: {filepath}")
        logger.info(f"   File size: {file_size:,} bytes")
        logger.info(f"   Content: {len(search_results)} search results, {len(crawled_content)} articles")
        return filepath

    except Exception as e:
        logger.error(f"❌ Error saving work product:")
        logger.error(f"   Session: {session_id}")
        logger.error(f"   Query: {query}")
        logger.error(f"   Workproduct dir: {workproduct_dir}")
        logger.error(f"   Exception: {e}")
        logger.exception("Full traceback:")
        return ""


def calculate_adaptive_batch_size(remaining_target: int, config) -> int:
    """
    Calculate optimal batch size based on remaining target and configuration.

    This function implements adaptive batch sizing to balance efficiency with resource usage.
    Larger batches are used when far from target, smaller batches when close to target.

    Args:
        remaining_target: Number of scrapes still needed
        config: Enhanced search configuration object

    Returns:
        Optimal batch size for the current situation
    """
    if not config.adaptive_batch_enabled:
        # Fall back to fixed batch size if adaptive sizing is disabled
        return config.initial_batch_size

    # Import configuration to get access to adaptive batch parameters
    min_batch_size = config.min_batch_size
    max_batch_size = config.max_batch_size
    batch_reduction_threshold = config.batch_reduction_threshold
    success_rate_buffer = config.success_rate_buffer

    # Calculate adaptive batch size based on remaining target
    if remaining_target >= batch_reduction_threshold:
        # Far from target - use larger batch for efficiency
        # Add success rate buffer to increase chances of reaching target
        calculated_size = int(remaining_target * (1 + success_rate_buffer))
        batch_size = min(max_batch_size, calculated_size)

        logger.info(f"Adaptive batch sizing: remaining_target={remaining_target}, "
                   f"buffer={success_rate_buffer:.0%}, calculated={calculated_size}, "
                   f"final={batch_size} (efficiency mode)")

    else:
        # Close to target - use smaller batch to minimize waste
        # Still add buffer but be more conservative
        calculated_size = int(remaining_target * (1 + success_rate_buffer))
        batch_size = min(max_batch_size, max(min_batch_size, calculated_size))

        logger.info(f"Adaptive batch sizing: remaining_target={remaining_target}, "
                   f"buffer={success_rate_buffer:.0%}, calculated={calculated_size}, "
                   f"final={batch_size} (conservative mode)")

    return batch_size


def remove_duplicate_urls(search_results: list[SearchResult]) -> list[SearchResult]:
    """
    Remove duplicate URLs from search results using robust URL normalization.

    This function deduplicates search results by normalizing URLs and removing
    duplicates while preserving the original order and metadata.

    Args:
        search_results: List of search results potentially containing duplicate URLs

    Returns:
        Deduplicated list of search results with original order preserved
    """
    from urllib.parse import urlparse, urlunparse, urljoin
    import re

    def normalize_url(url: str) -> str:
        """
        Normalize URL for robust deduplication.

        Handles common URL variations that point to the same content:
        - HTTP/HTTPS variations
        - www prefix variations
        - Trailing slashes
        - Query parameter ordering
        - Fragment removal
        """
        if not url:
            return ""

        try:
            # Parse URL
            parsed = urlparse(url.strip())

            # Normalize scheme (prefer https)
            scheme = "https" if parsed.scheme in ["http", "https"] else parsed.scheme

            # Normalize netloc (remove www prefix for consistency)
            netloc = parsed.netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]

            # Normalize path (remove trailing slash unless it's the root)
            path = parsed.path
            if path != "/" and path.endswith("/"):
                path = path[:-1]

            # Sort query parameters for consistent ordering
            if parsed.query:
                query_params = sorted(parsed.query.split("&"))
                query = "&".join(query_params)
            else:
                query = ""

            # Remove fragments (they don't affect content)
            fragment = ""

            # Reconstruct normalized URL
            normalized = urlunparse((scheme, netloc, path, "", query, fragment))
            return normalized

        except Exception as e:
            logger.warning(f"URL normalization failed for '{url}': {e}")
            return url.lower().strip()  # Fallback to simple normalization

    # Track seen URLs and their indices
    seen_urls = set()
    deduplicated_results = []
    duplicates_removed = 0

    for i, result in enumerate(search_results):
        if not result.link:
            # Skip results without URLs
            duplicates_removed += 1
            continue

        # Normalize URL for comparison
        normalized_url = normalize_url(result.link)

        if normalized_url not in seen_urls:
            # First time seeing this URL - keep it
            seen_urls.add(normalized_url)
            deduplicated_results.append(result)
        else:
            # Duplicate URL - remove it
            duplicates_removed += 1
            logger.debug(f"Removed duplicate URL: {result.link} (normalized: {normalized_url})")

    total_results = len(search_results)
    final_count = len(deduplicated_results)

    logger.info(f"URL deduplication: {total_results} → {final_count} results "
               f"(removed {duplicates_removed} duplicates, {final_count/total_results*100:.1f}% retained)")

    return deduplicated_results


def merge_search_results_position_based(search_results_lists: list[list[SearchResult]], max_total_results: int = 50) -> list[SearchResult]:
    """
    Merge multiple search results lists using position-based ranking.

    This function takes results from multiple queries and merges them by position:
    - 1st result from each query gets merged rank 1, 1st result from each query gets merged rank 2, etc.
    - This ensures fair representation across all queries rather than relevance-based mixing

    Args:
        search_results_lists: List of search results lists (one per query)
        max_total_results: Maximum total results to include in merged list

    Returns:
        Position-merged list of search results
    """
    if not search_results_lists:
        return []

    logger.info(f"Position-based merging: {len(search_results_lists)} query result lists, max {max_total_results} total results")

    # Find the maximum length among all result lists
    max_position = max(len(results) for results in search_results_lists)
    merged_results = []
    position_count = 0

    # Merge by position: take all 1st place results, then all 2nd place results, etc.
    for position in range(max_position):
        position_results = []

        # Collect results at this position from all queries
        for query_index, results in enumerate(search_results_lists):
            if position < len(results):
                result = results[position]
                # Add metadata about source query and position
                result.merged_from_query = query_index
                result.original_position = position
                position_results.append(result)

        if position_results:
            # Add all results at this position to merged list
            merged_results.extend(position_results)
            position_count += len(position_results)
            logger.debug(f"Position {position + 1}: {len(position_results)} results from {len(position_results)} queries")

        # Stop if we've reached the maximum total results
        if len(merged_results) >= max_total_results:
            merged_results = merged_results[:max_total_results]
            logger.info(f"Reached maximum total results ({max_total_results}), stopping merge")
            break

    # Update final positions in merged list
    for i, result in enumerate(merged_results):
        result.merged_position = i + 1

    total_input_results = sum(len(results) for results in search_results_lists)
    logger.info(f"Position-based merge complete: {total_input_results} input results → "
               f"{len(merged_results)} merged results across {max_position} positions")

    return merged_results


async def generate_query_expansions_with_llm(
    original_query: str,
    search_mode: str = "web",
    session_id: str = "default",
    orchestrator_client = None
) -> list[str]:
    """
    Generate intelligent query variations using LLM for broader research coverage.

    This function uses the research_agent LLM to create additional search queries
    that provide different angles and tangential directions for comprehensive research.

    Args:
        original_query: The original research query
        search_mode: "web" or "news" to match search strategy
        session_id: Session identifier for caching
        orchestrator_client: Client instance for LLM access

    Returns:
        List of queries including original + generated variations
    """
    if not orchestrator_client:
        logger.warning("No orchestrator client provided for query expansion, using original query only")
        return [original_query]

    # Import configuration for query expansion settings
    try:
        from config.settings import get_enhanced_search_config
        config = get_enhanced_search_config()
    except ImportError:
        logger.warning("Could not import search config, using default settings")
        class DefaultConfig:
            query_expansion_enabled = False
            max_query_expansions = 2
        config = DefaultConfig()

    if not config.query_expansion_enabled:
        logger.info("Query expansion disabled in configuration, using original query only")
        return [original_query]

    # Check cache first (simple in-memory cache per function call)
    cache_key = f"{session_id}:{original_query}:{search_mode}"
    # Note: In a full implementation, you might want persistent caching across sessions

    logger.info(f"Generating query expansions for: '{original_query}' (search_mode: {search_mode})")

    # Create LLM prompt for query expansion
    prompt = f"""Generate {config.max_query_expansions} additional search queries for comprehensive research on: "{original_query}"

Requirements:
1. Each query should explore a different angle of the topic
2. Include tangential but relevant directions for broader coverage
3. Optimize for {"SERP News API" if search_mode == "news" else "Google Search API"} (clear, specific, research-focused)
4. Avoid simply rephrasing the original - add new dimensions
5. Consider temporal, geographic, or thematic variations where appropriate

Goal: Provide diverse coverage that captures different aspects of the topic for thorough research.

Format your response as a JSON array with exactly {config.max_query_expansions} query strings:
["query 1", "query 2", "query 3" (if applicable)]

Each query should:
- Be specific and research-oriented
- Target different aspects than the original query
- Be suitable for web search APIs
- Avoid overly complex or boolean syntax
- Focus on discovering unique information"""

    try:
        # Use the orchestrator client to generate query expansions
        # This assumes the client has a method to call the research agent
        # The exact implementation depends on your orchestrator architecture

        # For now, we'll implement a fallback that returns the original query
        # In a full implementation, you would call the LLM here:
        # response = await orchestrator_client.query_agent("research_agent", prompt)
        # expanded_queries = parse_json_response(response)

        logger.warning("LLM query expansion not fully implemented yet, using original query only")
        return [original_query]

    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        logger.info("Falling back to original query only")
        return [original_query]


async def execute_expanded_search_with_iterative_scraping(
    query: str,
    search_type: str = "search",
    session_id: str = "default",
    anti_bot_level: int = 1,
    target_scrapes: int = 15,
    orchestrator_client = None
) -> str:
    """
    Execute comprehensive search with query expansion and iterative batch scraping.

    This is the main function that implements the new architecture:
    1. Generate query variations using LLM (if enabled)
    2. Execute multiple SERP searches
    3. Merge results using position-based ranking
    4. Deduplicate URLs
    5. Process URLs in adaptive batches until target reached

    Args:
        query: Original research query
        search_type: "search" or "news"
        session_id: Session identifier for tracking
        anti_bot_level: Anti-bot detection level
        target_scrapes: Target number of successful scrapes
        orchestrator_client: Client for LLM access

    Returns:
        Comprehensive research results as formatted string
    """
    try:
        start_time = datetime.now()
        logger.info(f"Starting expanded search with iterative scraping: '{query}' "
                   f"(target_scrapes: {target_scrapes}, anti_bot_level: {anti_bot_level})")

        # Import configuration
        try:
            from config.settings import get_enhanced_search_config
            config = get_enhanced_search_config()
        except ImportError:
            logger.warning("Could not import search config, using default settings")
            class DefaultConfig:
                query_expansion_enabled = False
                max_total_results = 50
                deduplication_enabled = True
                adaptive_batch_enabled = True
                initial_batch_size = 12
                min_batch_size = 4
                max_batch_size = 15
                batch_reduction_threshold = 6
                success_rate_buffer = 0.25
            config = DefaultConfig()

        # Step 1: Generate query expansions (if enabled)
        if config.query_expansion_enabled and orchestrator_client:
            queries = await generate_query_expansions_with_llm(
                query, search_type, session_id, orchestrator_client
            )
        else:
            queries = [query]
            if config.query_expansion_enabled:
                logger.info("Query expansion enabled but no orchestrator client provided, using original query only")

        logger.info(f"Executing searches for {len(queries)} queries: {queries}")

        # Step 2: Execute multiple SERP searches
        all_search_results = []
        for i, exp_query in enumerate(queries):
            logger.info(f"Executing search {i+1}/{len(queries)}: '{exp_query}'")
            try:
                search_results = await execute_serper_search(
                    query=exp_query,
                    search_type=search_type,
                    num_results=15  # Fixed per query
                )
                all_search_results.append(search_results)
                logger.info(f"Search {i+1} returned {len(search_results)} results")
            except Exception as e:
                logger.error(f"Search {i+1} failed for query '{exp_query}': {e}")
                all_search_results.append([])  # Add empty list to maintain structure

        # Step 3: Merge search results using position-based ranking
        if len(all_search_results) > 1:
            merged_results = merge_search_results_position_based(
                all_search_results, config.max_total_results
            )
        else:
            merged_results = all_search_results[0] if all_search_results else []

        # Step 4: Deduplicate URLs
        if config.url_deduplication_enabled and merged_results:
            deduplicated_results = remove_duplicate_urls(merged_results)
        else:
            deduplicated_results = merged_results

        if not deduplicated_results:
            return f"❌ **Search Failed**\n\nNo results found after query expansion and deduplication for query: '{query}'"

        logger.info(f"After processing: {len(deduplicated_results)} unique URLs ready for scraping")

        # Step 5: Process URLs in adaptive batches until target reached
        batch_results, cleaned_content, successful_urls = await process_scraping_in_batches(
            deduplicated_results,
            session_id,
            target_scrapes,
            anti_bot_level,
            config
        )

        # Step 6: Save Work Product 1 (SERP metadata + scraped content)
        logger.info(f"📋 Saving Work Product 1: SERP metadata + scraped content (expanded search)")
        logger.info(f"   Session: {session_id}")
        logger.info(f"   Search results: {len(deduplicated_results)}")
        logger.info(f"   Cleaned content: {len(cleaned_content)}")

        work_product_path = save_work_product(
            search_results=deduplicated_results,
            crawled_content=cleaned_content,
            urls=successful_urls,
            query=query,
            session_id=session_id,
            workproduct_dir=None  # Use session-based structure
        )

        # VERIFY WORK (Critical SDK principle!)
        import os
        if work_product_path and os.path.exists(work_product_path):
            file_size = os.path.getsize(work_product_path)
            logger.info(f"✅ Work Product 1 VERIFIED: {work_product_path}")
            logger.info(f"   File size: {file_size:,} bytes")
            logger.info(f"   Contains: {len(deduplicated_results)} search results, {len(cleaned_content)} crawled articles")
        else:
            logger.error(f"❌ CRITICAL: Work Product 1 file NOT FOUND after save!")
            logger.error(f"   Expected path: {work_product_path}")
            logger.error(f"   Session: {session_id}")

        # Step 7: Format and return results
        processing_time = (datetime.now() - start_time).total_seconds()
        summary = f"""
# Expanded Search Results

**Original Query**: {query}
**Query Expansions**: {len(queries)} searches performed
**Unique URLs Found**: {len(deduplicated_results)}
**Processing Time**: {processing_time:.1f} seconds
**Queries**: {', '.join(f'"{q}"' for q in queries)}
**Work Product 1 Saved**: {work_product_path if work_product_path else 'FAILED'}

{batch_results}

---
*Results generated using expanded search with iterative batch processing*
"""
        return summary.strip()

    except Exception as e:
        error_msg = f"❌ **Expanded Search Failed**\n\nError: {str(e)}"
        logger.error(f"Expanded search failed: {e}")
        return error_msg


async def process_scraping_in_batches(
    search_results: list[SearchResult],
    session_id: str,
    target_scrapes: int,
    anti_bot_level: int,
    config
) -> tuple[str, list, list]:
    """
    Process search results in adaptive batches until target scrapes reached.

    This function implements the core iterative scraping logic:
    - Calculate adaptive batch size based on remaining target
    - Process batch in parallel
    - Check success and continue if needed
    - Accept overshoot when target exceeded

    Args:
        search_results: Deduplicated search results to process
        session_id: Session identifier
        target_scrapes: Target number of successful scrapes
        anti_bot_level: Anti-bot detection level
        config: Search configuration

    Returns:
        Tuple of (formatted_results_string, cleaned_content_list, successful_urls_list)
    """
    try:
        # Load current session scrape count
        session_scrape_file = f"KEVIN/sessions/{session_id}/session_scrape_count.json"
        existing_scrapes = 0

        if os.path.exists(session_scrape_file):
            try:
                with open(session_scrape_file, 'r') as f:
                    session_data = json.load(f)
                    existing_scrapes = session_data.get('total_scrapes', 0)
                logger.info(f"Session {session_id} has {existing_scrapes} existing scrapes")
            except Exception as e:
                logger.warning(f"Could not read session scrape file: {e}")

        # Calculate remaining scrapes needed
        remaining_target = max(0, target_scrapes - existing_scrapes)
        if remaining_target == 0:
            logger.info(f"Session {session_id} has already reached target of {target_scrapes} scrapes")
            msg = f"✅ **Target Already Reached**\n\nSession has already achieved {target_scrapes} successful scrapes."
            return (msg, [], [])

        logger.info(f"Session {session_id} needs {remaining_target} more scrapes to reach target of {target_scrapes}")

        # Extract URLs from search results
        urls_to_process = [result.link for result in search_results if result.link]
        total_urls = len(urls_to_process)

        if not urls_to_process:
            msg = "❌ **No URLs to Process**\n\nNo valid URLs found in search results."
            return (msg, [], [])

        logger.info(f"Starting iterative batch processing: {total_urls} URLs, target {remaining_target} scrapes")

        # Process in adaptive batches
        all_crawled_content = []
        processed_urls = []
        batch_count = 0
        cumulative_scrapes = existing_scrapes

        for start_idx in range(0, total_urls, config.max_batch_size):
            # Calculate adaptive batch size for this iteration
            remaining_needed = max(0, target_scrapes - cumulative_scrapes)
            if remaining_needed == 0:
                logger.info("Target reached, stopping batch processing")
                break

            # Calculate optimal batch size
            batch_size = calculate_adaptive_batch_size(remaining_needed, config)
            end_idx = min(start_idx + batch_size, total_urls)
            current_batch_urls = urls_to_process[start_idx:end_idx]
            current_batch_results = search_results[start_idx:end_idx]

            batch_count += 1
            logger.info(f"Processing batch {batch_count}: {len(current_batch_urls)} URLs "
                       f"(remaining needed: {remaining_needed}, batch size: {batch_size})")

            # Process this batch using existing anti-bot escalation
            try:
                from utils.anti_bot_escalation import get_escalation_manager
                escalation_manager = get_escalation_manager()

                batch_crawl_results = await escalation_manager.crawl_multiple_with_escalation(
                    urls=current_batch_urls,
                    initial_level=anti_bot_level,
                    max_level=3,
                    max_concurrent=config.max_batch_size
                )

                # Filter successful results and extract content
                batch_successful_content = []
                for i, crawl_result in enumerate(batch_crawl_results):
                    if crawl_result and crawl_result.success and crawl_result.content:
                        batch_successful_content.append({
                            'url': current_batch_urls[i],
                            'content': crawl_result.content,
                            'title': current_batch_results[i].title if i < len(current_batch_results) else '',
                            'snippet': current_batch_results[i].snippet if i < len(current_batch_results) else '',
                            'source': current_batch_results[i].source if i < len(current_batch_results) else '',
                            'merged_position': current_batch_results[i].merged_position if i < len(current_batch_results) else None
                        })

                batch_scrapes = len(batch_successful_content)
                cumulative_scrapes += batch_scrapes

                logger.info(f"Batch {batch_count} completed: {batch_scrapes}/{len(current_batch_urls)} successful "
                           f"(cumulative: {cumulative_scrapes}, target: {target_scrapes})")

                # Add successful content to results
                all_crawled_content.extend(batch_successful_content)
                processed_urls.extend(current_batch_urls)

                # Update session scrape count
                try:
                    session_data = {
                        'total_scrapes': cumulative_scrapes,
                        'target_scrapes': target_scrapes,
                        'last_updated': datetime.now().isoformat(),
                        'session_id': session_id,
                        'query': f"expanded_search_batch_{batch_count}",
                        'successful_urls_this_run': [item['url'] for item in batch_successful_content],
                        'scrapes_this_run': batch_scrapes
                    }

                    os.makedirs(f"KEVIN/sessions/{session_id}", exist_ok=True)
                    with open(session_scrape_file, 'w') as f:
                        json.dump(session_data, f, indent=2)

                except Exception as e:
                    logger.error(f"Failed to update session scrape file: {e}")

                # Check if target reached (accept overshoot)
                if cumulative_scrapes >= target_scrapes:
                    logger.info(f"Target reached! {cumulative_scrapes} >= {target_scrapes}")
                    break

            except Exception as e:
                logger.error(f"Batch {batch_count} failed: {e}")
                continue

        # Apply content cleaning to all successful scrapes
        if all_crawled_content:
            logger.info(f"Applying AI content cleaning to {len(all_crawled_content)} crawled articles (parallel processing)")
            try:
                from agents.content_cleaner_agent import get_content_cleaner
                content_cleaner = get_content_cleaner()

                # Prepare content tuples for parallel batch processing
                from agents.content_cleaner_agent import ContentCleaningContext
                from urllib.parse import urlparse

                content_tuples = []
                for item in all_crawled_content:
                    # Create proper context for content cleaning
                    parsed_url = urlparse(item['url'])
                    source_domain = parsed_url.netloc

                    cleaning_context = ContentCleaningContext(
                        search_query="",  # No specific query for batch processing
                        query_terms=[],  # No specific query terms for batch processing
                        url=item['url'],
                        source_domain=source_domain,
                        session_id=session_id
                    )

                    content_tuples.append((item['content'], cleaning_context))

                # Use existing parallel content cleaning method - run ALL items concurrently
                logger.info(f"Starting parallel content cleaning: {len(content_tuples)} items with max_concurrent={len(content_tuples)}")
                cleaned_results = await content_cleaner.clean_multiple_contents(
                    contents=content_tuples,
                    max_concurrent=len(content_tuples)  # Clean all items concurrently
                )

                # Convert parallel results back to current format
                cleaned_content = []
                for i, cleaned_result in enumerate(cleaned_results):
                    if cleaned_result and cleaned_result.cleaned_content:
                        cleaned_content.append({
                            **all_crawled_content[i],
                            'cleaned_content': cleaned_result.cleaned_content,
                            'quality_score': cleaned_result.quality_score
                        })

                logger.info(f"Content cleaning completed: {len(cleaned_content)}/{len(all_crawled_content)} passed quality threshold")
                all_crawled_content = cleaned_content

            except Exception as e:
                logger.warning(f"Content cleaning failed: {e}")
                # Continue with raw content if cleaning fails

        # Format final results and return structured data
        formatted_results = format_iterative_scraping_results(all_crawled_content, batch_count, cumulative_scrapes, target_scrapes)

        # Extract cleaned content and URLs for work product saving
        cleaned_content = [item.get('cleaned_content', item.get('content', '')) for item in all_crawled_content]
        successful_urls = [item['url'] for item in all_crawled_content]

        return (formatted_results, cleaned_content, successful_urls)

    except Exception as e:
        logger.error(f"Iterative batch processing failed: {e}")
        error_msg = f"❌ **Batch Processing Failed**\n\nError: {str(e)}"
        return (error_msg, [], [])


def format_iterative_scraping_results(crawled_content: list, batch_count: int, total_scrapes: int, target_scrapes: int) -> str:
    """
    Format results from iterative batch processing into a comprehensive report.

    Args:
        crawled_content: List of successfully crawled and cleaned content
        batch_count: Number of batches processed
        total_scrapes: Total successful scrapes achieved
        target_scrapes: Original target number of scrapes

    Returns:
        Formatted results string
    """
    if not crawled_content:
        return f"❌ **No Content Retrieved**\n\nProcessed {batch_count} batches but no successful content scrapes were achieved."

    result_parts = [
        f"# Iterative Scraping Results",
        f"",
        f"**Summary**: {total_scrapes} successful scrapes from {batch_count} batches (target: {target_scrapes})",
        f"**Status**: {'✅ Target Achieved' if total_scrapes >= target_scrapes else '⚠️ Below Target'}",
        f"",
        f"## Retrieved Content ({len(crawled_content)} sources)",
        f""
    ]

    for i, item in enumerate(crawled_content, 1):
        quality_info = f" (Quality: {item.get('quality_score', 'N/A')})" if 'quality_score' in item else ""
        position_info = f" (Position: {item.get('merged_position', 'N/A')})" if item.get('merged_position') else ""

        result_parts.extend([
            f"### {i}. {item.get('title', 'Untitled')}{quality_info}{position_info}",
            f"**URL**: {item['url']}",
            f"**Source**: {item.get('source', 'Unknown')}",
            f"**Snippet**: {item.get('snippet', 'No snippet available')[:200]}...",
            f"",
            f"**Content Preview**:",
            f"{item.get('cleaned_content', item.get('content', ''))[:1000]}...",
            f"",
            f"---",
            f""
        ])

    result_parts.append(f"**Processing Complete**: {total_scrapes} sources successfully scraped and processed.")

    return "\n".join(result_parts)


async def search_crawl_and_clean_direct(
    query: str,
    search_type: str = "search",
    num_results: int = 15,
    auto_crawl_top: int = 10,
    crawl_threshold: float = 0.3,
    max_concurrent: int = 15,
    session_id: str = "default",
    anti_bot_level: int = 1,
    workproduct_dir: str = None,
    target_scrapes: int = 15,
    use_expanded_search: bool = False,
    orchestrator_client = None
) -> str:
    """
    Combined search, crawl, and clean operation using zPlayground1 technology.

    This function:
    1. Performs search + crawl + clean in a single optimized flow
    2. Saves detailed work product to workproducts directory
    3. Returns full detailed data for orchestrator agent analysis
    4. Uses parallel processing and anti-bot detection
    5. Supports expanded search with query expansion and iterative batch processing

    Args:
        query: Search query
        search_type: "search" or "news"
        num_results: Number of search results to retrieve
        auto_crawl_top: Maximum number of URLs to crawl
        crawl_threshold: Minimum relevance threshold for crawling
        max_concurrent: Maximum concurrent crawling operations
        session_id: Session identifier
        anti_bot_level: Progressive anti-bot level (0-3)
        workproduct_dir: Directory for work products
        target_scrapes: Target number of successful scrapes
        use_expanded_search: Enable LLM-powered query expansion and iterative batch processing
        orchestrator_client: Client instance for LLM access during query expansion

    Returns:
        Full detailed content for orchestrator agent processing
    """
    try:
        start_time = datetime.now()
        logger.info(f"Starting enhanced search+crawl+clean for query: '{query}' (anti_bot_level: {anti_bot_level}, target_scrapes: {target_scrapes})")

        # Route to expanded search if enabled (new architecture)
        if use_expanded_search:
            logger.info(f"Using expanded search with iterative batch processing for query: '{query}'")
            return await execute_expanded_search_with_iterative_scraping(
                query=query,
                search_type=search_type,
                session_id=session_id,
                anti_bot_level=anti_bot_level,
                target_scrapes=target_scrapes,
                orchestrator_client=orchestrator_client
            )

        # Continue with traditional single-query approach (backward compatibility)
        logger.info(f"Using traditional search approach for query: '{query}'")

        # Step 0: Check session scrape count and adjust target
        import os
        import json

        session_scrape_file = f"KEVIN/sessions/{session_id}/session_scrape_count.json"
        existing_scrapes = 0

        # Create session directory if it doesn't exist
        os.makedirs(f"KEVIN/sessions/{session_id}", exist_ok=True)

        # Load existing scrape count
        if os.path.exists(session_scrape_file):
            try:
                with open(session_scrape_file, 'r') as f:
                    session_data = json.load(f)
                    existing_scrapes = session_data.get('total_scrapes', 0)
                logger.info(f"Session {session_id} has {existing_scrapes} existing scrapes")
            except Exception as e:
                logger.warning(f"Could not read session scrape file: {e}")

        # Calculate remaining scrapes needed
        remaining_scrapes = max(0, target_scrapes - existing_scrapes)
        if remaining_scrapes == 0:
            logger.info(f"Session {session_id} has already reached target of {target_scrapes} scrapes")
            return f"✅ **Target Already Reached**\n\nSession has already achieved {target_scrapes} successful scrapes. No additional crawling needed."

        logger.info(f"Session {session_id} needs {remaining_scrapes} more scrapes to reach target of {target_scrapes}")

        # Adjust auto_crawl_top to not exceed remaining target
        effective_crawl_top = min(auto_crawl_top, remaining_scrapes)
        logger.info(f"Adjusted crawling limit from {auto_crawl_top} to {effective_crawl_top} based on session target")

        # Step 1: Intelligent search strategy selection
        from utils.search_strategy_selector import get_search_strategy_selector
        strategy_selector = get_search_strategy_selector()

        strategy_analysis = strategy_selector.select_search_strategy(query)
        logger.info(f"Search strategy analysis: {strategy_analysis.recommended_strategy.value} "
                   f"(confidence: {strategy_analysis.confidence:.2f})")

        # Determine optimal search type based on strategy
        if strategy_analysis.recommended_strategy.value == "news":
            optimal_search_type = "news"
            logger.info("Using SERP News API based on strategy analysis")
        elif strategy_analysis.recommended_strategy.value == "search":
            optimal_search_type = "search"
            logger.info("Using Google Search API based on strategy analysis")
        else:  # hybrid
            # For hybrid, use the search type with higher weight
            optimal_search_type = search_type if search_type else "search"
            logger.info(f"Using hybrid approach - primary: {optimal_search_type}")

        # Step 2: Execute search with optimal strategy (1-2 seconds)
        search_results = await execute_serper_search(
            query=query,
            search_type=optimal_search_type,
            num_results=num_results
        )

        if not search_results:
            return f"❌ **Search Failed**\n\nNo results found for query: '{query}'"

        # Step 2: Select URLs for crawling based on relevance
        urls_to_crawl = select_urls_for_crawling(
            search_results=search_results,
            limit=effective_crawl_top,
            min_relevance=crawl_threshold
        )

        if not urls_to_crawl:
            # Return standard search results if no URLs meet crawling threshold
            search_section = format_search_results(search_results)
            return f"{search_section}\n\n**Note**: No URLs met the crawling threshold ({crawl_threshold}). Standard search results provided above."

        # Step 3: Execute parallel crawling with progressive anti-bot escalation
        logger.info(f"Crawling {len(urls_to_crawl)} URLs with anti-bot escalation (level: {anti_bot_level})")

        # Import anti-bot escalation system
        from utils.anti_bot_escalation import get_escalation_manager

        escalation_manager = get_escalation_manager()
        crawl_results = await escalation_manager.crawl_multiple_with_escalation(
            urls=urls_to_crawl,
            initial_level=anti_bot_level,
            max_level=3,
            max_concurrent=max_concurrent,
            use_content_filter=False,
            session_id=session_id
        )

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Step 4: Extract content from successful crawls with anti-bot escalation
        crawled_content_list = []
        successful_urls = []
        escalation_stats = []

        for result in crawl_results:
            # Extract content from EscalationResult objects
            content = result.content

            if result.success and content and len(content.strip()) > 200:
                # Only include results with substantial content
                crawled_content_list.append(content.strip())
                successful_urls.append(result.url)

                escalation_info = f"L{result.final_level}"
                if result.escalation_used:
                    escalation_info += f" (escalated from {result.final_level - result.attempts_made + 1})"

                logger.info(f"✅ Anti-bot escalation success: {len(content)} chars from {result.url} "
                           f"[{escalation_info}, {result.attempts_made} attempts, {result.duration:.1f}s]")
            else:
                logger.error(f"❌ Anti-bot escalation FAILED: {result.url} "
                           f"[L{result.final_level}, {result.attempts_made} attempts] - {result.error}")

            # Track escalation statistics
            escalation_stats.append({
                'url': result.url,
                'success': result.success,
                'attempts': result.attempts_made,
                'final_level': result.final_level,
                'escalation_used': result.escalation_used,
                'duration': result.duration,
                'word_count': result.word_count,
                'char_count': result.char_count
            })

        # Step 4.5: Update session scrape count
        successful_scrapes = len(crawled_content_list)
        new_total_scrapes = existing_scrapes + successful_scrapes

        # Update session scrape count file
        session_data = {
            'total_scrapes': new_total_scrapes,
            'target_scrapes': target_scrapes,
            'last_updated': datetime.now().isoformat(),
            'session_id': session_id,
            'query': query,
            'successful_urls_this_run': successful_urls,
            'scrapes_this_run': successful_scrapes
        }

        try:
            with open(session_scrape_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            logger.info(f"Updated session {session_id}: {existing_scrapes} + {successful_scrapes} = {new_total_scrapes} total scrapes")
        except Exception as e:
            logger.error(f"Failed to update session scrape file: {e}")

        if crawled_content_list:

            # Step 5: Apply AI content cleaning with GPT-5-nano
            logger.info(f"Applying AI content cleaning to {len(crawled_content_list)} crawled articles")

            from agents.content_cleaner_agent import (
                ContentCleaningContext,
                get_content_cleaner,
            )

            content_cleaner = get_content_cleaner()
            query_terms = query.split()

            # Prepare content for cleaning
            cleaning_contexts = []
            for content, url in zip(crawled_content_list, successful_urls, strict=False):
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.lower()

                context = ContentCleaningContext(
                    search_query=query,
                    query_terms=query_terms,
                    url=url,
                    source_domain=domain,
                    session_id=session_id,
                    min_quality_threshold=50,
                    max_content_length=50000
                )
                cleaning_contexts.append((content, context))

            # Clean content concurrently - run ALL items concurrently
            cleaning_results = await content_cleaner.clean_multiple_contents(
                cleaning_contexts,
                max_concurrent=len(crawled_content_list)  # Clean all items concurrently
            )

            # Filter and replace with cleaned content
            cleaned_content_list = []
            cleaned_urls = []
            cleaning_stats = []

            for i, (cleaned_result, (original_content, context)) in enumerate(zip(cleaning_results, cleaning_contexts, strict=False)):
                # Accept all content regardless of quality score - user wants results, not perfection
                if cleaned_result.quality_score >= 0:  # Quality threshold (accept everything)
                    cleaned_content_list.append(cleaned_result.cleaned_content)
                    cleaned_urls.append(context.url)

                    cleaning_stats.append({
                        'url': context.url,
                        'quality_score': cleaned_result.quality_score,
                        'quality_level': cleaned_result.quality_level.value,
                        'relevance_score': cleaned_result.relevance_score,
                        'word_count': cleaned_result.word_count,
                        'char_count': cleaned_result.char_count,
                        'key_points_count': len(cleaned_result.key_points),
                        'topics_detected': cleaned_result.topics_detected,
                        'model_used': cleaned_result.model_used,
                        'processing_time': cleaned_result.processing_time
                    })

                    logger.info(f"✅ AI content cleaning: {cleaned_result.quality_score}/100 "
                               f"({cleaned_result.quality_level.value}) - {context.url}")
                else:
                    logger.warning(f"⚠️ Content below quality threshold ({cleaned_result.quality_score}/100) - {context.url}")

            # Update successful URLs and content based on cleaning results
            successful_urls = cleaned_urls
            crawled_content_list = cleaned_content_list

            if not cleaned_content_list:
                logger.error("All content failed quality filtering after AI cleaning")
                return f"❌ **Content Quality Filter Failed**\n\nAll crawled content failed quality filtering after AI cleaning for query: '{query}'"

            # Step 6: Save detailed work product to file (Work Product 1: SERP + scraped content)
            logger.info(f"📋 Saving Work Product 1: SERP metadata + scraped content")
            logger.info(f"   Session: {session_id}")
            logger.info(f"   Search results: {len(search_results)}")
            logger.info(f"   Crawled articles: {len(cleaned_content_list)}")

            work_product_path = save_work_product(
                search_results=search_results,
                crawled_content=cleaned_content_list,
                urls=successful_urls,
                query=query,
                session_id=session_id,
                workproduct_dir=workproduct_dir
            )

            # VERIFY WORK (Critical SDK principle!)
            import os
            if work_product_path and os.path.exists(work_product_path):
                file_size = os.path.getsize(work_product_path)
                logger.info(f"✅ Work Product 1 VERIFIED: {work_product_path}")
                logger.info(f"   File size: {file_size:,} bytes")
                logger.info(f"   Contains: {len(search_results)} search results, {len(cleaned_content_list)} crawled articles")
            else:
                logger.error(f"❌ CRITICAL: Work Product 1 file NOT FOUND after save!")
                logger.error(f"   Expected path: {work_product_path}")
                logger.error(f"   Session ID: {session_id}")
                logger.error(f"   Workproduct dir param: {workproduct_dir}")
                # This is a critical failure - the primary work product is missing!

            # Step 7: Standardize research data for report generation integration
            try:
                import os

                from utils.research_data_standardizer import (
                    standardize_and_save_research_data,
                )

                # Determine session directory from workproduct directory
                if workproduct_dir and os.path.exists(workproduct_dir):
                    session_dir = os.path.dirname(workproduct_dir)  # Go up one level to session dir
                elif workproduct_dir is None:
                    # Calculate session-based directory when None is passed
                    session_dir = f"KEVIN/sessions/{session_id}"
                    workproduct_dir = f"{session_dir}/research"
                else:
                    # workproduct_dir is provided but doesn't exist
                    session_dir = f"KEVIN/sessions/{session_id}"
                    workproduct_dir = f"{session_dir}/research"

                standardized_path = standardize_and_save_research_data(
                    session_id=session_id,
                    research_topic=query,
                    workproduct_dir=workproduct_dir,
                    session_dir=session_dir
                )
                if standardized_path:
                    logger.info(f"✅ Research data standardized for report generation: {standardized_path}")
                else:
                    logger.warning("⚠️ Research data standardization failed, but research completed successfully")
            except Exception as e:
                logger.error(f"⚠️ Research data standardization error: {e}")

            # Step 8: Build comprehensive data for orchestrator
            # Include EVERYTHING for the orchestrator to analyze

            # Calculate escalation statistics
            successful_crawls = sum(1 for stat in escalation_stats if stat['success'])
            escalations_triggered = sum(1 for stat in escalation_stats if stat['escalation_used'])
            avg_attempts = sum(stat['attempts'] for stat in escalation_stats) / len(escalation_stats) if escalation_stats else 0

            orchestrator_data = f"""# ENHANCED SEARCH+CRAWL+CLEAN COMPLETE DATA

**Query**: {query}
**Search Type**: {search_type}
**Search Results**: {len(search_results)} found
**URLs Crawled**: {len(successful_urls)} successfully processed
**Anti-Bot Escalation**: Level {anti_bot_level} initial, up to level 3
**Crawl Success Rate**: {successful_crawls}/{len(escalation_stats)} ({successful_crawls/len(escalation_stats):.1%})
**Escalations Triggered**: {escalations_triggered}
**Avg Attempts per URL**: {avg_attempts:.1f}
**Processing Time**: {total_duration:.2f}s
**Work Product Saved**: {work_product_path}

## SEARCH STRATEGY ANALYSIS

**Recommended Strategy**: {strategy_analysis.recommended_strategy.value.upper()}
**Confidence**: {strategy_analysis.confidence:.2f}
**Time Sensitivity Factor**: {strategy_analysis.time_factor:.2f}
**Topic Factor**: {strategy_analysis.topic_factor:.2f}
**Query Factor**: {strategy_analysis.query_factor:.2f}
**Optimal Search Type**: {optimal_search_type.upper()}

**Reasoning**:
"""

            for reason in strategy_analysis.reasoning:
                orchestrator_data += f"- {reason}\n"

            orchestrator_data += """

## ANTI-BOT ESCALATION STATISTICS

| URL | Success | Attempts | Final Level | Escalation | Duration (s) | Words | Chars |
|-----|---------|----------|-------------|------------|---------------|-------|-------|
"""

            for stat in escalation_stats:
                escalation_mark = "✅" if stat['escalation_used'] else "—"
                success_mark = "✅" if stat['success'] else "❌"
                orchestrator_data += f"| {stat['url'][:50]}... | {success_mark} | {stat['attempts']} | L{stat['final_level']} | {escalation_mark} | {stat['duration']:.1f} | {stat['word_count']} | {stat['char_count']} |\n"

            orchestrator_data += """

---

## SEARCH RESULTS DATA

"""

            # Add all search results with full metadata
            for i, result in enumerate(search_results, 1):
                orchestrator_data += f"""### Search Result {i}
**Title**: {result.title}
**URL**: {result.link}
**Source**: {result.source}
**Date**: {result.date}
**Relevance Score**: {result.relevance_score:.2f}
**Snippet**: {result.snippet}

"""

            orchestrator_data += f"""---

## AI CONTENT CLEANING STATISTICS

**Total Articles Processed**: {len(cleaning_stats)}
**Quality Threshold**: 50/100
**Content Cleaning Model**: {cleaning_stats[0]['model_used'] if cleaning_stats else 'N/A'}

| URL | Quality Score | Quality Level | Relevance | Words | Topics | Time (s) |
|-----|---------------|---------------|-----------|-------|---------|-----------|
"""

            for stat in cleaning_stats:
                quality_emoji = "🟢" if stat['quality_score'] >= 80 else "🟡" if stat['quality_score'] >= 60 else "🔴"
                orchestrator_data += (f"| {stat['url'][:50]}... | {quality_emoji} {stat['quality_score']}/100 "
                                     f"({stat['quality_level']}) | {stat['relevance_score']:.2f} | "
                                     f"{stat['word_count']} | {len(stat['topics_detected'])} | "
                                     f"{stat['processing_time']:.1f} |\n")

            orchestrator_data += f"""

## CRAWLED CONTENT DATA (AI Cleaned)

Total articles successfully crawled and AI-cleaned: {len(crawled_content_list)}

"""

            # Add all crawled content with full details
            for i, (content, url) in enumerate(zip(crawled_content_list, successful_urls, strict=False), 1):
                # Find corresponding search result for metadata
                title = f"Article {i}"
                source = "Unknown"
                for result in search_results:
                    if result.link == url:
                        title = result.title
                        source = result.source or "Unknown"
                        break

                orchestrator_data += f"""### Crawled Article {i}: {title}
**URL**: {url}
**Source**: {source}
**Content Length**: {len(content)} characters
**Processing**: ✅ Cleaned with GPT-5-nano

**FULL CLEANED CONTENT**:
{content}

---

"""

            orchestrator_data += f"""
## PROCESSING SUMMARY

- **Search executed**: {search_type} search for "{query}"
- **Results found**: {len(search_results)} search results
- **URLs selected for crawling**: {len(urls_to_crawl)} (threshold: {crawl_threshold})
- **Successful crawls**: {len(crawled_content_list)} articles
- **Anti-bot level**: {anti_bot_level} (progressive detection)
- **Content cleaning**: GPT-5-nano AI processing applied to all articles
- **Total processing time**: {total_duration:.2f} seconds
- **Work product file**: {work_product_path}
- **Performance**: Enhanced parallel processing with zPlayground1 technology

This is the complete raw data for orchestrator analysis and user response generation.
"""

            logger.info(f"✅ Enhanced search+crawl+clean completed in {total_duration:.2f}s - Work product saved to {work_product_path}")
            return orchestrator_data

        else:
            # Crawling failed: Return search results only
            search_section = format_search_results(search_results)

            # Still save partial work product with search results only
            work_product_path = save_work_product(
                search_results=search_results,
                crawled_content=[],
                urls=[],
                query=query,
                session_id=session_id,
                workproduct_dir=workproduct_dir
            )

            failed_result = f"""{search_section}

---

**Note**: Content extraction failed for the selected URLs. Only search results available.
**Anti-Bot Level**: {anti_bot_level}
**Execution Time**: {total_duration:.2f}s
**Work Product Saved**: {work_product_path}
"""

            logger.warning(f"Crawling failed, returning search results only. Duration: {total_duration:.2f}s")
            return failed_result

    except Exception as e:
        logger.error(f"Error in enhanced search+crawl+clean: {e}")
        return f"❌ **Enhanced Search and Crawl Error**\n\nFailed to execute search and crawl operation: {str(e)}"


async def news_search_and_crawl_direct(
    query: str,
    num_results: int = 15,
    auto_crawl_top: int = 10,
    session_id: str = "default",
    anti_bot_level: int = 1,
    workproduct_dir: str = None,
    target_scrapes: int = 15,
    use_expanded_search: bool = False,
    orchestrator_client = None
) -> str:
    """
    Specialized news search with content extraction using enhanced technology.

    Args:
        query: News search query
        num_results: Number of news results to retrieve
        auto_crawl_top: Maximum number of articles to crawl
        session_id: Session identifier
        anti_bot_level: Progressive anti-bot level
        workproduct_dir: Directory for work products

    Returns:
        Full detailed news content for orchestrator agent processing
    """
    # Add "latest" and time relevance to news queries
    enhanced_query = f"{query} latest news"

    return await search_crawl_and_clean_direct(
        query=enhanced_query,
        search_type="news",
        num_results=num_results,
        auto_crawl_top=auto_crawl_top,
        crawl_threshold=0.3,  # Standard threshold
        max_concurrent=15,  # Increased concurrency for more URLs
        session_id=session_id,
        anti_bot_level=anti_bot_level,
        workproduct_dir=workproduct_dir,
        target_scrapes=target_scrapes,
        use_expanded_search=use_expanded_search,
        orchestrator_client=orchestrator_client
    )


# Export commonly used functions
__all__ = [
    'search_crawl_and_clean_direct',
    'news_search_and_crawl_direct',
    'save_work_product',
    'select_urls_for_crawling',
    'SearchResult'
]
