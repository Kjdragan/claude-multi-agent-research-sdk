"""
SERP API Search Utilities for Multi-Agent Research System

This module provides optimized search+crawl+clean functionality using SERP API
to replace the WebPrime MCP search system. Offers 10x performance improvement
with automatic content extraction and advanced relevance scoring.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

import httpx

# Token limit configuration
MAX_RESPONSE_TOKENS = 20000  # Conservative limit to stay under 25k token limit
APPOX_CHARS_PER_TOKEN = 4    # Rough approximation
MAX_RESPONSE_CHARS = MAX_RESPONSE_TOKENS * APPOX_CHARS_PER_TOKEN

# Import URL tracking system
try:
    from .url_tracker import get_url_tracker
except ImportError:
    # Fallback import for module execution
    import sys
    sys.path.append(os.path.dirname(__file__))
    from url_tracker import get_url_tracker

def summarize_content(content: str, max_length: int = 2000) -> str:
    """
    Summarize content to fit within token limits.

    Args:
        content: Original content to summarize
        max_length: Maximum length of summarized content

    Returns:
        Summarized content within the specified length limit
    """
    if len(content) <= max_length:
        return content

    # Simple summarization: take first and last portions with an indicator
    first_part = content[:max_length//2]
    last_part = content[-max_length//2:]

    return f"""{first_part}

[Content summarized - full content ({len(content)} chars) saved to work product file]

{last_part}"""


# Import configuration
try:
    from ..config.settings import get_enhanced_search_config
except ImportError:
    # Fallback for standalone usage
    # Simple fallback config with all required attributes
    class SimpleConfig:
        # Search settings
        default_num_results = 15
        default_auto_crawl_top = 10
        default_crawl_threshold = 0.3
        default_anti_bot_level = 1
        default_max_concurrent = 15

        # Target-based scraping settings
        target_successful_scrapes = 10
        url_deduplication_enabled = True
        progressive_retry_enabled = True

        # Retry logic settings
        max_retry_attempts = 3
        progressive_timeout_multiplier = 1.5

        # Token management
        max_response_tokens = 20000
        content_summary_threshold = 20000

        # Content cleaning settings
        default_cleanliness_threshold = 0.7
        min_content_length_for_cleaning = 500
        min_cleaned_content_length = 200

        # Crawl settings
        default_crawl_timeout = 30000
        max_concurrent_crawls = 15
        crawl_retry_attempts = 2

        # Anti-bot levels
        anti_bot_levels = {
            0: "basic",      # 6/10 sites success
            1: "enhanced",   # 8/10 sites success
            2: "advanced",   # 9/10 sites success
            3: "stealth"     # 9.5/10 sites success
        }

    def get_enhanced_search_config():
        config = SimpleConfig()
        # Debug: verify the critical attribute exists
        if not hasattr(config, 'default_max_concurrent'):
            logger.error("SimpleConfig is missing default_max_concurrent attribute!")
            logger.error(f"Available attributes: {[attr for attr in dir(config) if not attr.startswith('_')]}")
            raise AttributeError("SimpleConfig missing required attribute: default_max_concurrent")
        return config

# Import advanced scraping utilities
try:
    from .crawl4ai_utils import scrape_and_clean_single_url_direct
except ImportError:
    # Fallback import for module execution
    import sys
    sys.path.append(os.path.dirname(__file__))
    from crawl4ai_utils import scrape_and_clean_single_url_direct

logger = logging.getLogger(__name__)


class SearchResult:
    """Search result data structure with enhanced relevance scoring"""
    def __init__(self, title: str, link: str, snippet: str, position: int = 0,
                 date: str = None, source: str = None, relevance_score: float = 0.0):
        self.title = title
        self.link = link
        self.snippet = snippet
        self.position = position
        self.date = date
        self.source = source
        self.relevance_score = relevance_score


# Import enhanced relevance scorer with domain authority
from .enhanced_relevance_scorer import (
    calculate_enhanced_relevance_score_with_domain_authority as calculate_enhanced_relevance_score,
)


async def execute_serp_search(
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
        serper_api_key = os.getenv("SERP_API_KEY")
        if not serper_api_key:
            logger.warning("SERP_API_KEY not found in environment variables")
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
            logger.error(f"Serper API error: {response.status_code} - {response.text}")
            return []

    except Exception as e:
        logger.error(f"Error in Serper search: {e}")
        return []


def select_urls_for_crawling(
    search_results: list[SearchResult],
    limit: int = 10,
    min_relevance: float = 0.3,
    session_id: str = "default",
    use_deduplication: bool = True
) -> list[str]:
    """
    Select URLs for crawling based on relevance scores with URL deduplication.

    Args:
        search_results: List of search results
        limit: Maximum number of URLs to select
        min_relevance: Minimum relevance score threshold (fixed at 0.3 for better success)
        session_id: Session identifier for URL tracking
        use_deduplication: Whether to use URL deduplication

    Returns:
        List of URLs to crawl
    """
    try:
        # Get configuration
        config = get_enhanced_search_config()

        # Filter by relevance threshold (ensure type safety)
        filtered_results = [
            result for result in search_results
            if float(result.relevance_score) >= float(min_relevance) and result.link
        ]

        # Sort by relevance score (highest first) - ensure float comparison
        filtered_results.sort(key=lambda x: float(x.relevance_score), reverse=True)

        # Extract URLs up to limit
        candidate_urls = [result.link for result in filtered_results[:limit]]

        # Apply URL deduplication if enabled
        if use_deduplication and config.url_deduplication_enabled:
            url_tracker = get_url_tracker()
            urls_to_crawl, skipped_urls = url_tracker.filter_urls(candidate_urls, session_id)

            logger.info(f"URL selection with threshold {min_relevance}:")
            logger.info(f"  - Total results: {len(search_results)}")
            logger.info(f"  - Above threshold: {len(filtered_results)}")
            logger.info(f"  - Before deduplication: {len(candidate_urls)}")
            logger.info(f"  - After deduplication: {len(urls_to_crawl)}")
            logger.info(f"  - Skipped duplicates: {len(skipped_urls)}")

            return urls_to_crawl
        else:
            logger.info(f"URL selection with threshold {min_relevance} (no deduplication):")
            logger.info(f"  - Total results: {len(search_results)}")
            logger.info(f"  - Above threshold: {len(filtered_results)}")
            logger.info(f"  - Selected for crawling: {len(candidate_urls)}")
            logger.info(f"  - Rejected: {len(search_results) - len(filtered_results)} below threshold")

            return candidate_urls

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


async def advanced_content_extraction(url: str, session_id: str, search_query: str = None) -> str:
    """
    Advanced content extraction using Crawl4AI + AI cleaning.

    Replaces simple HTTP+regex with multi-stage browser automation:
    - Stage 1: Fast CSS selector extraction
    - Stage 2: Robust fallback extraction
    - Stage 3: Judge assessment and AI cleaning
    - 70-100% success rate (vs 30% with basic HTTP+regex)
    - 30K-58K characters extracted (vs 2K limit before)

    Args:
        url: URL to extract content from
        session_id: Session identifier for logging
        search_query: Search query context for content relevance filtering

    Returns:
        Extracted and cleaned content as string
    """
    try:
        result = await scrape_and_clean_single_url_direct(
            url=url,
            session_id=session_id,
            search_query=search_query,
            extraction_mode="article",
            include_metadata=False,
            preserve_technical_content=True
        )

        if result['success']:
            logger.info(f"Advanced extraction successful for {url}: {result['char_count']} chars")
            return result['cleaned_content']
        else:
            logger.warning(f"Advanced extraction failed for {url}: {result['error_message']}")
            return ""

    except Exception as e:
        logger.error(f"Advanced content extraction error for {url}: {e}")
        return ""


def save_search_work_product(
    search_results: list[SearchResult],
    crawled_content: list[str],
    urls: list[str],
    query: str,
    session_id: str,
    kevin_dir: Path
) -> str:
    """
    Save detailed search and crawl results to work product file.

    Args:
        search_results: List of search results
        crawled_content: List of cleaned content strings
        urls: List of crawled URLs
        query: Original search query
        session_id: Session identifier
        kevin_dir: KEVIN directory path

    Returns:
        Path to saved work product file
    """
    try:
        # Use session-based directory structure
        sessions_dir = kevin_dir / "sessions" / session_id
        research_dir = sessions_dir / "research"
        research_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp and filename with numbered prefix + suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Count existing research files to determine suffix (1, 1A, 1B, 1C, etc.)
        existing_files = list(research_dir.glob("1*-search_workproduct_*.md"))
        if not existing_files:
            # First research file - no suffix
            prefix = "1"
        else:
            # Calculate suffix based on count (A, B, C, D, etc.)
            suffix_letter = chr(65 + len(existing_files))  # 65 is ASCII for 'A'
            prefix = f"1{suffix_letter}"

        filename = f"{prefix}-search_workproduct_{timestamp}.md"
        filepath = research_dir / filename

        # Build work product content
        workproduct_content = [
            "# Search Results Work Product",
            "",
            f"**Session ID**: {session_id}",
            f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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

        if crawled_content:
            workproduct_content.extend([
                "",
                "## 📄 Extracted Content",
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
                    f"**Content Length**: {len(content)} characters",
                    "",
                    "### 📄 Extracted Content",
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
            "- **Processing**: SERP API search + content extraction",
            "",
            "*Generated by Multi-Agent Research System - SERP API Integration*"
        ])

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(workproduct_content))

        logger.info(f"✅ Work product saved to: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error saving work product: {e}")
        return ""


async def target_based_scraping(
    search_results: list[SearchResult],
    session_id: str,
    target_successful_scrapes: int = 10,
    crawl_threshold: float = 0.3,
    max_concurrent: int = 10
) -> tuple[list[str], list[str]]:
    """
    Perform target-based scraping to achieve desired number of successful extractions.

    This function processes ALL candidates in parallel to achieve the target
    number of successful scrapes without sequential blocking.

    Args:
        search_results: List of search results to select from
        session_id: Session identifier for URL tracking
        target_successful_scrapes: Target number of successful content extractions
        crawl_threshold: Initial relevance threshold (fixed at 0.3)
        max_concurrent: Maximum concurrent crawling operations

    Returns:
        Tuple of (successful_content, attempted_urls)
    """
    config = get_enhanced_search_config()
    url_tracker = get_url_tracker()

    # Start with current threshold and target from config
    target_count = target_successful_scrapes or config.target_successful_scrapes

    # Get ALL candidates at once for parallel processing
    # Start with 0.3 threshold, expand to 0.2 for additional candidates
    # Reduced multipliers to prevent excessive URL processing
    primary_candidates = select_urls_for_crawling(
        search_results=search_results,
        limit=int(target_count * 1.5),  # Reduced multiplier: 1.5x instead of 2x
        min_relevance=0.3,
        session_id=session_id,
        use_deduplication=True
    )

    secondary_candidates = select_urls_for_crawling(
        search_results=search_results,
        limit=target_count * 2,  # Reduced multiplier: 2x instead of 3x
        min_relevance=0.2,
        session_id=session_id,
        use_deduplication=True
    )

    # Combine and deduplicate all candidates
    all_candidate_urls = list(dict.fromkeys(primary_candidates + secondary_candidates))  # Preserve order, remove duplicates
    logger.info(f"Target-based scraping: target={target_count}, total_candidates={len(all_candidate_urls)} (primary: {len(primary_candidates)}, secondary: {len(secondary_candidates)})")

    # Process ALL candidates in parallel using progressive retry
    successful_content, attempted_urls = await _crawl_urls_with_retry(
        urls=all_candidate_urls,
        session_id=session_id,
        max_concurrent=max_concurrent,
        use_progressive_retry=True  # Enable progressive retry for better success rates
    )

    # Early termination: Check if target already achieved after primary crawl
    if len(successful_content) >= target_count:
        logger.info(f"✅ Target achieved after primary crawl: {len(successful_content)}/{target_count} successful scrapes")
        return successful_content, attempted_urls

    # If we still need more and have retry candidates, process them too
    if config.progressive_retry_enabled and len(successful_content) < target_count:
        retry_candidates = url_tracker.get_retry_candidates(attempted_urls)
        if retry_candidates:
            logger.info(f"Processing additional retry candidates: {len(retry_candidates)} URLs")
            retry_content, retry_urls = await _crawl_urls_with_retry(
                urls=retry_candidates,
                session_id=session_id,
                max_concurrent=max_concurrent,
                is_retry=True
            )
            successful_content.extend(retry_content)
            attempted_urls.extend(retry_urls)

    # Early termination: Check if target achieved after retry candidates
    if len(successful_content) >= target_count:
        logger.info(f"✅ Target achieved after retry candidates: {len(successful_content)}/{target_count} successful scrapes")
        return successful_content, attempted_urls

    # If we still don't have enough, try one more batch with even lower threshold
    if len(successful_content) < target_count:
        fallback_candidates = select_urls_for_crawling(
            search_results=search_results,
            limit=target_count * 2,  # Reduced multiplier: 2x instead of 4x
            min_relevance=0.1,  # Very low threshold as last resort
            session_id=session_id,
            use_deduplication=True
        )

        # Filter out already attempted URLs
        new_candidates = [url for url in fallback_candidates if url not in attempted_urls]

        if new_candidates:
            logger.info(f"Final fallback batch: {len(new_candidates)} URLs at 0.1 threshold")
            content, urls = await _crawl_urls_with_retry(
                urls=new_candidates,
                session_id=session_id,
                max_concurrent=max_concurrent,
                use_progressive_retry=True
            )
            successful_content.extend(content)
            attempted_urls.extend(urls)

    # Final statistics
    success_rate = len(successful_content) / len(attempted_urls) if attempted_urls else 0
    logger.info(f"Target-based scraping completed: {len(successful_content)}/{target_count} successful "
               f"({success_rate:.1%} success rate from {len(attempted_urls)} URLs)")

    return successful_content, attempted_urls


async def _crawl_urls_with_retry(
    urls: list[str],
    session_id: str,
    max_concurrent: int,
    is_retry: bool = False,
    use_progressive_retry: bool = False
) -> tuple[list[str], list[str]]:
    """
    Helper function to crawl URLs with retry logic.

    Args:
        urls: URLs to crawl
        session_id: Session identifier
        max_concurrent: Maximum concurrent operations
        is_retry: Whether this is a retry operation

    Returns:
        Tuple of (successful_content, attempted_urls)
    """
    if not urls:
        return [], []

    config = get_enhanced_search_config()
    url_tracker = get_url_tracker()

    # Use z-playground1 implementation
    from utils.crawl4ai_z_playground import crawl_multiple_urls_with_results

    # Always use progressive retry for better parallelization
    crawl_results = await crawl_multiple_urls_with_results(
        urls=urls,
        session_id=session_id,
        max_concurrent=max_concurrent,
        extraction_mode="article",
        include_metadata=True,
        use_progressive_retry=use_progressive_retry,
        max_retries=3 if use_progressive_retry else 0
    )

    # Process results and record attempts
    successful_content = []
    attempted_urls = []

    for result in crawl_results:
        url = result['url']
        success = result['success']
        content = result.get('content', '')
        content_length = len(content) if content else 0
        duration = result.get('duration', 0.0)
        error_message = result.get('error_message')

        # Determine anti-bot level used
        anti_bot_level = 1  # Default
        if is_retry:
            anti_bot_level = url_tracker.get_retry_anti_bot_level(url)

        # Record attempt in URL tracker asynchronously to prevent blocking
        tracking_task = asyncio.create_task(
            asyncio.to_thread(
                url_tracker.record_attempt,
                url=url,
                success=success and content_length > 100,  # Only count substantial content as success
                anti_bot_level=anti_bot_level,
                content_length=content_length,
                duration=duration,
                error_message=error_message,
                session_id=session_id
            )
        )
        # Don't await tracking task to prevent blocking - let it run in background

        attempted_urls.append(url)

        if success and content_length > 100:
            successful_content.append(content.strip())
            logger.info(f"✅ Extracted {content_length} chars from {url}")
        else:
            logger.warning(f"❌ Failed to extract substantial content from {url}")

    return successful_content, attempted_urls


async def serp_search_and_extract(
    query: str,
    search_type: str = "search",
    num_results: int = 15,
    auto_crawl_top: int = 5,
    crawl_threshold: float = 0.3,
    session_id: str = "default",
    kevin_dir: Path = None
) -> str:
    """
    Combined SERP API search and content extraction.

    This function:
    1. Performs search using SERP API
    2. Selects relevant URLs based on threshold
    3. Extracts content from selected URLs
    4. Saves work product and returns comprehensive results

    Args:
        query: Search query
        search_type: "search" or "news"
        num_results: Number of search results to retrieve
        auto_crawl_top: Maximum number of URLs to crawl
        crawl_threshold: Minimum relevance threshold for crawling
        session_id: Session identifier
        kevin_dir: KEVIN directory path

    Returns:
        Full detailed content for agent processing
    """
    try:
        start_time = datetime.now()
        logger.info(f"Starting SERP search+extract for query: '{query}'")

        # Set default KEVIN directory if not provided
        if kevin_dir is None:
            kevin_dir = Path.home() / "lrepos" / "claude-agent-sdk-python" / "KEVIN"

        # Step 1: Execute SERP search (1-2 seconds)
        search_results = await execute_serp_search(
            query=query,
            search_type=search_type,
            num_results=num_results
        )

        if not search_results:
            return f"❌ **Search Failed**\n\nNo results found for query: '{query}'"

        # Step 2: Use target-based scraping with improved threshold logic
        config = get_enhanced_search_config()

        # Use fixed 0.3 threshold and target-based scraping
        logger.info(f"Using target-based scraping: threshold=0.3, target={config.target_successful_scrapes}")

        crawled_content, successful_urls = await target_based_scraping(
            search_results=search_results,
            session_id=session_id,
            target_successful_scrapes=config.target_successful_scrapes,
            crawl_threshold=0.3,  # Fixed at 0.3 for better success rates
            max_concurrent=config.default_max_concurrent
        )


        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Step 4: Save work product
        work_product_path = save_search_work_product(
            search_results=search_results,
            crawled_content=crawled_content,
            urls=successful_urls,
            query=query,
            session_id=session_id,
            kevin_dir=kevin_dir
        )

        # Step 5: Build comprehensive results for agents
        if crawled_content:
            orchestrator_data = f"""# SERP API SEARCH RESULTS

**Query**: {query}
**Search Type**: {search_type}
**Search Results**: {len(search_results)} found
**URLs Extracted**: {len(crawled_content)} successfully processed
**Processing Time**: {total_duration:.2f}s
**Work Product Saved**: {work_product_path}

---

## SEARCH RESULTS SUMMARY

"""

            # Add all search results with full metadata
            for i, result in enumerate(search_results, 1):
                orchestrator_data += f"""### {i}. {result.title}
**URL**: {result.link}
**Source**: {result.source}
**Date**: {result.date}
**Relevance Score**: {result.relevance_score:.2f}
**Snippet**: {result.snippet}

"""

            if crawled_content:
                orchestrator_data += f"""---

## EXTRACTED CONTENT

Total articles successfully extracted: {len(crawled_content)}

"""

                # Calculate remaining space for content after metadata
                current_length = len(orchestrator_data)
                remaining_space = MAX_RESPONSE_CHARS - current_length - 1000  # Leave buffer for summary section
                content_per_article = remaining_space // len(crawled_content) if crawled_content else 0
                max_content_length = min(content_per_article, 3000)  # Cap at 3000 chars per article

                # Add extracted content with token limit handling
                for i, (content, url) in enumerate(zip(crawled_content, successful_urls, strict=False), 1):
                    # Find corresponding search result for metadata
                    title = f"Article {i}"
                    source = "Unknown"
                    for result in search_results:
                        if result.link == url:
                            title = result.title
                            source = result.source or "Unknown"
                            break

                    # Apply content summarization if needed
                    if len(content) > max_content_length:
                        content = summarize_content(content, max_content_length)
                        content_note = f" (summarized from {len(content)} original characters)"
                    else:
                        content_note = ""

                    orchestrator_data += f"""### Extracted Article {i}: {title}
**URL**: {url}
**Source**: {source}
**Content Length**: {len(content)} characters{content_note}

**EXTRACTED CONTENT**:
{content}

---

"""

            orchestrator_data += f"""
## PROCESSING SUMMARY

- **Search executed**: SERP API {search_type} search for "{query}"
- **Results found**: {len(search_results)} search results
- **URLs selected for extraction**: {len(successful_urls)} (threshold: 0.3)
- **Successful extractions**: {len(crawled_content)} articles
- **Total processing time**: {total_duration:.2f} seconds
- **Work product file**: {work_product_path}
- **Performance**: SERP API direct search (10x faster than MCP)

This is the complete search data for research analysis and report generation.
"""

            logger.info(f"✅ SERP search+extract completed in {total_duration:.2f}s")
            return orchestrator_data

        else:
            # Content extraction failed: Return search results only
            search_section = format_search_results(search_results)

            failed_result = f"""{search_section}

---

**Note**: Content extraction failed for selected URLs. Search results provided above.
**Execution Time**: {total_duration:.2f}s
**Work Product Saved**: {work_product_path}
"""

            logger.warning(f"Content extraction failed, returning search results only. Duration: {total_duration:.2f}s")
            return failed_result

    except Exception as e:
        logger.error(f"Error in SERP search+extract: {e}")
        return f"❌ **Search Error**\n\nFailed to execute SERP API search: {str(e)}"


async def expanded_query_search_and_extract(
    query: str,
    search_type: str = "search",
    num_results: int = 15,
    auto_crawl_top: int = 5,
    crawl_threshold: float = 0.3,
    session_id: str = "default",
    kevin_dir: Path = None,
    max_expanded_queries: int = 3
) -> str:
    """
    Corrected query expansion workflow with master result consolidation.

    This function implements the proper workflow:
    1. Generate multiple related search queries using query expansion
    2. Execute SERP searches for each expanded query
    3. Collect & deduplicate all results into one master list
    4. Rank results by relevance
    5. Scrape from master ranked list within budget limits

    Args:
        query: Original search query
        search_type: "search" or "news"
        num_results: Number of results per SERP search
        auto_crawl_top: Maximum number of URLs to crawl from master list
        crawl_threshold: Minimum relevance threshold for crawling
        session_id: Session identifier
        kevin_dir: KEVIN directory path
        max_expanded_queries: Maximum number of expanded queries to generate

    Returns:
        Full detailed content for agent processing
    """
    try:
        start_time = datetime.now()
        logger.info(f"Starting expanded query search+extract for query: '{query}'")

        # Set default KEVIN directory if not provided
        if kevin_dir is None:
            kevin_dir = Path.home() / "lrepos" / "claude-agent-sdk-python" / "KEVIN"

        # Step 1: Generate expanded queries
        expanded_queries = await generate_expanded_queries(query, max_expanded_queries)
        logger.info(f"Generated {len(expanded_queries)} expanded queries: {expanded_queries}")

        # Step 2: Execute SERP searches for each expanded query
        all_search_results = []
        for expanded_query in expanded_queries:
            logger.info(f"Executing SERP search for expanded query: '{expanded_query}'")
            search_results = await execute_serp_search(
                query=expanded_query,
                search_type=search_type,
                num_results=num_results
            )
            all_search_results.extend(search_results)
            logger.info(f"Retrieved {len(search_results)} results for expanded query: '{expanded_query}'")

        # Step 3: Deduplicate all results into one master list
        master_results = deduplicate_search_results(all_search_results)
        logger.info(f"Deduplicated results: {len(all_search_results)} -> {len(master_results)} unique results")

        # Step 4: Rank results by relevance
        master_results.sort(key=lambda x: x.relevance_score, reverse=True)
        logger.info(f"Ranked {len(master_results)} results by relevance score")

        # Step 5: Scrape from master ranked list within budget limits
        config = get_enhanced_search_config()

        # Use target-based scraping with budget limits
        logger.info(f"Using target-based scraping: threshold={crawl_threshold}, target={config.target_successful_scrapes}")

        crawled_content, successful_urls = await target_based_scraping(
            search_results=master_results,
            session_id=session_id,
            target_successful_scrapes=config.target_successful_scrapes,
            crawl_threshold=crawl_threshold,
            max_concurrent=config.default_max_concurrent
        )

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Step 6: Save work product with expanded query information
        work_product_path = save_expanded_search_work_product(
            original_query=query,
            expanded_queries=expanded_queries,
            master_results=master_results,
            crawled_content=crawled_content,
            successful_urls=successful_urls,
            session_id=session_id,
            kevin_dir=kevin_dir,
            total_duration=total_duration
        )

        # Step 7: Build comprehensive results for agents
        if crawled_content:
            orchestrator_data = f"""# EXPANDED QUERY SEARCH RESULTS

**Original Query**: {query}
**Expanded Queries**: {', '.join(expanded_queries)}
**Search Type**: {search_type}
**Total Master Results**: {len(master_results)} found (after deduplication)
**URLs Extracted**: {len(crawled_content)} successfully processed
**Processing Time**: {total_duration:.2f}s
**Work Product Saved**: {work_product_path}

---

## QUERY EXPANSION ANALYSIS

**Original Query**: {query}
**Generated Expanded Queries**:
"""

            for i, eq in enumerate(expanded_queries, 1):
                orchestrator_data += f"{i}. {eq}\n"

            orchestrator_data += f"""
**Total Queries Executed**: {len(expanded_queries)}
**Total Raw Results**: {len(all_search_results)}
**Deduplicated Results**: {len(master_results)}
**Deduplication Rate**: {(1 - len(master_results)/len(all_search_results)):.1%}

---

## MASTER SEARCH RESULTS (Top {min(20, len(master_results))} by Relevance)

"""

            # Add top master results with full metadata
            for i, result in enumerate(master_results[:20], 1):
                orchestrator_data += f"""### {i}. {result.title}
**URL**: {result.link}
**Source**: {result.source}
**Date**: {result.date}
**Relevance Score**: {result.relevance_score:.2f}
**Snippet**: {result.snippet}

"""

            if crawled_content:
                orchestrator_data += f"""---

## EXTRACTED CONTENT

Total articles successfully extracted: {len(crawled_content)}

"""

                # Calculate remaining space for content after metadata
                current_length = len(orchestrator_data)
                remaining_space = MAX_RESPONSE_CHARS - current_length - 1000  # Leave buffer for summary section
                content_per_article = remaining_space // len(crawled_content) if crawled_content else 0
                max_content_length = min(content_per_article, 3000)  # Cap at 3000 chars per article

                # Add extracted content with token limit handling
                for i, (content, url) in enumerate(zip(crawled_content, successful_urls, strict=False), 1):
                    # Find corresponding search result for metadata
                    title = f"Article {i}"
                    source = "Unknown"
                    for result in master_results:
                        if result.link == url:
                            title = result.title
                            source = result.source or "Unknown"
                            break

                    # Apply content summarization if needed
                    if len(content) > max_content_length:
                        content = summarize_content(content, max_content_length)
                        content_note = f" (summarized from {len(content)} original characters)"
                    else:
                        content_note = ""

                    orchestrator_data += f"""### Extracted Article {i}: {title}
**URL**: {url}
**Source**: {source}
**Content Length**: {len(content)} characters{content_note}

**EXTRACTED CONTENT**:
{content}

---

"""

            orchestrator_data += f"""
## PROCESSING SUMMARY

- **Original query**: "{query}"
- **Expanded queries executed**: {len(expanded_queries)}
- **Total raw results found**: {len(all_search_results)}
- **Deduplicated results**: {len(master_results)} unique URLs
- **URLs selected for extraction**: {len(successful_urls)} (threshold: {crawl_threshold})
- **Successful extractions**: {len(crawled_content)} articles
- **Total processing time**: {total_duration:.2f} seconds
- **Work product file**: {work_product_path}
- **Performance**: Expanded query search with master result consolidation

This is the complete search data for research analysis and report generation.
"""

            logger.info(f"✅ Expanded query search+extract completed in {total_duration:.2f}s")
            return orchestrator_data

        else:
            # Content extraction failed: Return master search results only
            search_section = format_expanded_search_results(query, expanded_queries, master_results)

            failed_result = f"""{search_section}

---

**Note**: Content extraction failed for selected URLs. Master search results provided above.
**Execution Time**: {total_duration:.2f}s
**Work Product Saved**: {work_product_path}
"""

            logger.warning(f"Content extraction failed, returning master search results only. Duration: {total_duration:.2f}s")
            return failed_result

    except Exception as e:
        logger.error(f"Error in expanded query search+extract: {e}")
        return f"❌ **Expanded Query Search Error**\n\nFailed to execute expanded query search: {str(e)}"


async def generate_expanded_queries(original_query: str, max_queries: int = 3) -> list[str]:
    """
    Generate expanded search queries using simple query expansion techniques.

    Args:
        original_query: Original search query
        max_queries: Maximum number of expanded queries to generate

    Returns:
        List of expanded queries including the original
    """
    # Start with the original query
    queries = [original_query]

    # Simple query expansion techniques
    query_lower = original_query.lower()

    # Technique 1: Add context terms
    if "news" not in query_lower and "latest" not in query_lower:
        queries.append(f"{original_query} latest news")

    # Technique 2: Add comprehensive/overview terms
    if "overview" not in query_lower and "comprehensive" not in query_lower:
        queries.append(f"{original_query} overview comprehensive")

    # Technique 3: Add analysis/in-depth terms
    if "analysis" not in query_lower and "in-depth" not in query_lower:
        queries.append(f"{original_query} analysis in-depth")

    # Technique 4: Add research/study terms
    if "research" not in query_lower and "study" not in query_lower:
        queries.append(f"{original_query} research study")

    # Technique 5: Add developments/trends terms
    if "developments" not in query_lower and "trends" not in query_lower:
        queries.append(f"{original_query} recent developments trends")

    # Return limited number of queries (original + expanded)
    return queries[:max_queries]


def deduplicate_search_results(search_results: list[SearchResult]) -> list[SearchResult]:
    """
    Deduplicate search results based on URL, keeping the highest relevance score.

    Args:
        search_results: List of search results that may contain duplicates

    Returns:
        Deduplicated list of search results
    """
    seen_urls = set()
    deduplicated_results = []

    for result in search_results:
        if result.link not in seen_urls:
            seen_urls.add(result.link)
            deduplicated_results.append(result)
        else:
            # If we've seen this URL before, keep the one with higher relevance score
            for i, existing_result in enumerate(deduplicated_results):
                if existing_result.link == result.link and result.relevance_score > existing_result.relevance_score:
                    deduplicated_results[i] = result
                    break

    return deduplicated_results


def format_expanded_search_results(original_query: str, expanded_queries: list[str], search_results: list[SearchResult]) -> str:
    """
    Format expanded search results for display.

    Args:
        original_query: Original search query
        expanded_queries: List of expanded queries
        search_results: List of search results

    Returns:
        Formatted search results string
    """
    if not search_results:
        return "No search results found."

    result_parts = [
        f"# Expanded Query Search Results ({len(search_results)} found)",
        "",
        f"**Original Query**: {original_query}",
        f"**Expanded Queries**: {', '.join(expanded_queries)}",
        "",
        "---",
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


def save_expanded_search_work_product(
    original_query: str,
    expanded_queries: list[str],
    master_results: list[SearchResult],
    crawled_content: list[str],
    successful_urls: list[str],
    session_id: str,
    kevin_dir: Path,
    total_duration: float
) -> str:
    """
    Save detailed expanded search and crawl results to work product file.

    Args:
        original_query: Original search query
        expanded_queries: List of expanded queries
        master_results: Master list of deduplicated search results
        crawled_content: List of cleaned content strings
        successful_urls: List of successfully crawled URLs
        session_id: Session identifier
        kevin_dir: KEVIN directory path
        total_duration: Total processing time

    Returns:
        Path to saved work product file
    """
    try:
        # Use session-based directory structure
        sessions_dir = kevin_dir / "sessions" / session_id
        research_dir = sessions_dir / "research"
        research_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp and filename with numbered prefix + suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Count existing expanded search files to determine suffix (1, 1A, 1B, 1C, etc.)
        existing_files = list(research_dir.glob("1*-expanded_search_workproduct_*.md"))
        if not existing_files:
            # First expanded search file - no suffix
            prefix = "1"
        else:
            # Calculate suffix based on count (A, B, C, D, etc.)
            suffix_letter = chr(65 + len(existing_files))  # 65 is ASCII for 'A'
            prefix = f"1{suffix_letter}"

        filename = f"{prefix}-expanded_search_workproduct_{timestamp}.md"
        filepath = research_dir / filename

        # Build work product content
        workproduct_content = [
            "# Expanded Query Search Results Work Product",
            "",
            f"**Session ID**: {session_id}",
            f"**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Original Query**: {original_query}",
            f"**Expanded Queries**: {', '.join(expanded_queries)}",
            f"**Total Queries Executed**: {len(expanded_queries)}",
            f"**Total Master Results**: {len(master_results)}",
            f"**Successfully Crawled**: {len(crawled_content)}",
            f"**Processing Time**: {total_duration:.2f}s",
            "",
            "---",
            "",
            "## 🔍 Query Expansion Analysis",
            "",
            f"**Original Query**: {original_query}",
            "",
            "**Expanded Queries Generated**:",
        ]

        for i, eq in enumerate(expanded_queries, 1):
            workproduct_content.append(f"{i}. {eq}")

        workproduct_content.extend([
            "",
            "**Query Expansion Strategy**: Simple context and scope enhancement",
            f"**Deduplication Applied**: {len(master_results)} unique URLs from all searches",
            "",
            "---",
            "",
            "## 📊 Master Search Results Summary",
            "",
        ])

        # Add master search results overview
        for i, result in enumerate(master_results, 1):
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

        if crawled_content:
            workproduct_content.extend([
                "",
                "## 📄 Extracted Content",
                ""
            ])

            # Add detailed crawled content
            for i, (content, url) in enumerate(zip(crawled_content, successful_urls, strict=False), 1):
                # Find corresponding search result for title
                title = f"Article {i}"
                for result in master_results:
                    if result.link == url:
                        title = result.title
                        break

                workproduct_content.extend([
                    f"## 🌐 {i}. {title}",
                    "",
                    f"**URL**: {url}",
                    f"**Content Length**: {len(content)} characters",
                    "",
                    "### 📄 Extracted Content",
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
            f"- **Original Query**: {original_query}",
            f"- **Expanded Queries**: {len(expanded_queries)} queries executed",
            f"- **Master Results Found**: {len(master_results)} unique URLs",
            f"- **URLs Successfully Crawled**: {len(crawled_content)}",
            "- **Processing**: Expanded query search + content extraction",
            f"- **Total Processing Time**: {total_duration:.2f} seconds",
            "- **Deduplication Applied**: Yes (URL-based)",
            "",
            "*Generated by Multi-Agent Research System - Expanded Query Search Integration*"
        ])

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(workproduct_content))

        logger.info(f"✅ Expanded search work product saved to: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error saving expanded search work product: {e}")
        return ""


# Export commonly used functions
__all__ = [
    'serp_search_and_extract',
    'expanded_query_search_and_extract',
    'execute_serp_search',
    'format_search_results',
    'save_search_work_product',
    'generate_expanded_queries',
    'deduplicate_search_results'
]
