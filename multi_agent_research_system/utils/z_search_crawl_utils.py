"""
Enhanced search and crawl utilities adapted from zPlayground1.

This module provides optimized search+crawl+clean functionality with
parallel processing, anti-bot detection, and AI content cleaning.
Adapted for integration with the multi-agent research system.
"""

import logging
import os
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
        self.position = position
        self.date = date
        self.source = source
        self.relevance_score = relevance_score


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
        # Determine correct session directory structure
        if workproduct_dir is None:
            # Default to KEVIN sessions directory with proper categorical organization
            base_sessions_dir = "/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN/sessions"
            session_dir = os.path.join(base_sessions_dir, session_id)
            research_dir = os.path.join(session_dir, "research")
            Path(research_dir).mkdir(parents=True, exist_ok=True)

            # Generate timestamp and filename with numbered prefix + suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Count existing research files to determine suffix (1, 1A, 1B, 1C, etc.)
            existing_files = list(Path(research_dir).glob("1*-search_workproduct_*.md"))
            if not existing_files:
                # First research file - no suffix
                prefix = "1"
            else:
                # Calculate suffix based on count (A, B, C, D, etc.)
                suffix_letter = chr(65 + len(existing_files))  # 65 is ASCII for 'A'
                prefix = f"1{suffix_letter}"

            filename = f"{prefix}-search_workproduct_{timestamp}.md"
            filepath = os.path.join(research_dir, filename)
        else:
            # Custom workproduct directory (legacy support)
            Path(workproduct_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced-search-crawl-workproduct_{timestamp}.md"
            filepath = os.path.join(workproduct_dir, filename)

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
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(workproduct_content))

        logger.info(f"✅ Work product saved to: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Error saving work product: {e}")
        return ""


async def search_crawl_and_clean_direct(
    query: str,
    search_type: str = "search",
    num_results: int = 15,
    auto_crawl_top: int = 10,
    crawl_threshold: float = 0.3,
    max_concurrent: int = 15,
    session_id: str = "default",
    anti_bot_level: int = 1,
    workproduct_dir: str = None
) -> str:
    """
    Combined search, crawl, and clean operation using zPlayground1 technology.

    This function:
    1. Performs search + crawl + clean in a single optimized flow
    2. Saves detailed work product to workproducts directory
    3. Returns full detailed data for orchestrator agent analysis
    4. Uses parallel processing and anti-bot detection

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

    Returns:
        Full detailed content for orchestrator agent processing
    """
    try:
        start_time = datetime.now()
        logger.info(f"Starting enhanced search+crawl+clean for query: '{query}' (anti_bot_level: {anti_bot_level})")

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
            limit=auto_crawl_top,
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

            # Clean content concurrently
            cleaning_results = await content_cleaner.clean_multiple_contents(
                cleaning_contexts,
                max_concurrent=min(5, len(crawled_content_list))
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

            # Step 6: Save detailed work product to file
            work_product_path = save_work_product(
                search_results=search_results,
                crawled_content=cleaned_content_list,
                urls=successful_urls,
                query=query,
                session_id=session_id,
                workproduct_dir=workproduct_dir
            )

            # Step 7: Standardize research data for report generation integration
            try:
                import os

                from utils.research_data_standardizer import (
                    standardize_and_save_research_data,
                )

                # Determine session directory from workproduct directory
                if workproduct_dir and os.path.exists(workproduct_dir):
                    session_dir = os.path.dirname(workproduct_dir)  # Go up one level to session dir
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
                else:
                    logger.warning(f"⚠️ Cannot standardize research data - workproduct directory not found: {workproduct_dir}")
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
    workproduct_dir: str = None
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
        workproduct_dir=workproduct_dir
    )


# Export commonly used functions
__all__ = [
    'search_crawl_and_clean_direct',
    'news_search_and_crawl_direct',
    'save_work_product',
    'select_urls_for_crawling',
    'SearchResult'
]
