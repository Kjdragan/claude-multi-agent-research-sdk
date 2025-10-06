"""
Configuration settings for the multi-agent research system.

This module contains configuration for enhanced search, crawling,
and content cleaning functionality.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EnhancedSearchConfig:
    """Configuration for enhanced search functionality."""

    # Search settings
    default_num_results: int = 15
    default_auto_crawl_top: int = 10
    default_crawl_threshold: float = 0.3  # Fixed at 0.3 for better success rates
    default_anti_bot_level: int = 1
    default_max_concurrent: int = 15

    # Anti-bot levels
    anti_bot_levels = {
        0: "basic",      # 6/10 sites success
        1: "enhanced",   # 8/10 sites success
        2: "advanced",   # 9/10 sites success
        3: "stealth"     # 9.5/10 sites success
    }

    # Target-based scraping settings
    target_successful_scrapes: int = 15  # Target number of successful scrapes per search
    url_deduplication_enabled: bool = True  # Prevent duplicate URL crawling
    progressive_retry_enabled: bool = True  # Retry failed URLs with higher anti-bot levels

    # Adaptive batch sizing configuration
    adaptive_batch_enabled: bool = True  # Enable adaptive batch sizing
    initial_batch_size: int = 12  # Starting batch size for efficiency
    max_batch_size: int = 15  # Upper limit for resource management
    min_batch_size: int = 4  # Lower limit for parallelization benefits
    success_rate_buffer: float = 0.25  # Add 25% buffer for expected failures
    batch_reduction_threshold: int = 6  # Switch to smaller batches when close to target
    max_total_results: int = 50  # Total URLs to consider from search merging

    # Query expansion configuration
    query_expansion_enabled: bool = False  # Enable LLM-powered query expansion
    max_query_expansions: int = 2  # Number of additional queries to generate (total 3)
    query_expansion_cache_enabled: bool = True  # Cache query expansions per session

    # Retry logic settings
    max_retry_attempts: int = 3
    progressive_timeout_multiplier: float = 1.5

    # Token management
    max_response_tokens: int = 20000
    content_summary_threshold: int = 20000

    # Report generation settings
    brief_report_min_words: int = 800      # Brief report minimum target
    default_report_min_words: int = 2500   # Default report minimum target
    comprehensive_report_min_words: int = 5000  # Comprehensive report minimum target

    # Work product directories
    default_workproduct_dir: str = "/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN/work_products"
    kevin_workproducts_dir: str | None = None

    # Content cleaning settings
    default_cleanliness_threshold: float = 0.7
    min_content_length_for_cleaning: int = 500
    min_cleaned_content_length: int = 200

    # Crawl settings
    default_crawl_timeout: int = 30000
    max_concurrent_crawls: int = 15
    crawl_retry_attempts: int = 2

    def __post_init__(self):
        """Initialize derived settings."""
        # Set KEVIN workproducts directory from environment if available
        self.kevin_workproducts_dir = os.getenv('KEVIN_WORKPRODUCTS_DIR')
        if self.kevin_workproducts_dir:
            self.default_workproduct_dir = self.kevin_workproducts_dir


class SettingsManager:
    """Manages configuration settings for the research system."""

    def __init__(self):
        self._enhanced_search_config = EnhancedSearchConfig()
        self._load_environment_overrides()

    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""

        # Enhanced search settings
        if os.getenv('ENHANCED_SEARCH_NUM_RESULTS'):
            try:
                self._enhanced_search_config.default_num_results = int(os.getenv('ENHANCED_SEARCH_NUM_RESULTS'))
            except ValueError:
                pass

        if os.getenv('ENHANCED_SEARCH_AUTO_CRAWL_TOP'):
            try:
                self._enhanced_search_config.default_auto_crawl_top = int(os.getenv('ENHANCED_SEARCH_AUTO_CRAWL_TOP'))
            except ValueError:
                pass

        if os.getenv('ENHANCED_SEARCH_CRAWL_THRESHOLD'):
            try:
                self._enhanced_search_config.default_crawl_threshold = float(os.getenv('ENHANCED_SEARCH_CRAWL_THRESHOLD'))
            except ValueError:
                pass

        if os.getenv('ENHANCED_SEARCH_ANTI_BOT_LEVEL'):
            try:
                level = int(os.getenv('ENHANCED_SEARCH_ANTI_BOT_LEVEL'))
                if 0 <= level <= 3:
                    self._enhanced_search_config.default_anti_bot_level = level
            except ValueError:
                pass

        if os.getenv('ENHANCED_SEARCH_MAX_CONCURRENT'):
            try:
                self._enhanced_search_config.default_max_concurrent = int(os.getenv('ENHANCED_SEARCH_MAX_CONCURRENT'))
            except ValueError:
                pass

        # Adaptive batch sizing settings
        if os.getenv('ADAPTIVE_BATCH_ENABLED'):
            self._enhanced_search_config.adaptive_batch_enabled = os.getenv('ADAPTIVE_BATCH_ENABLED').lower() in ('true', '1', 'yes')

        if os.getenv('INITIAL_BATCH_SIZE'):
            try:
                self._enhanced_search_config.initial_batch_size = int(os.getenv('INITIAL_BATCH_SIZE'))
            except ValueError:
                pass

        if os.getenv('MAX_BATCH_SIZE'):
            try:
                self._enhanced_search_config.max_batch_size = int(os.getenv('MAX_BATCH_SIZE'))
            except ValueError:
                pass

        if os.getenv('MIN_BATCH_SIZE'):
            try:
                self._enhanced_search_config.min_batch_size = int(os.getenv('MIN_BATCH_SIZE'))
            except ValueError:
                pass

        if os.getenv('SUCCESS_RATE_BUFFER'):
            try:
                buffer = float(os.getenv('SUCCESS_RATE_BUFFER'))
                if 0.0 <= buffer <= 1.0:
                    self._enhanced_search_config.success_rate_buffer = buffer
            except ValueError:
                pass

        if os.getenv('BATCH_REDUCTION_THRESHOLD'):
            try:
                self._enhanced_search_config.batch_reduction_threshold = int(os.getenv('BATCH_REDUCTION_THRESHOLD'))
            except ValueError:
                pass

        if os.getenv('MAX_TOTAL_RESULTS'):
            try:
                self._enhanced_search_config.max_total_results = int(os.getenv('MAX_TOTAL_RESULTS'))
            except ValueError:
                pass

        # Query expansion settings
        if os.getenv('QUERY_EXPANSION_ENABLED'):
            self._enhanced_search_config.query_expansion_enabled = os.getenv('QUERY_EXPANSION_ENABLED').lower() in ('true', '1', 'yes')

        if os.getenv('MAX_QUERY_EXPANSIONS'):
            try:
                self._enhanced_search_config.max_query_expansions = int(os.getenv('MAX_QUERY_EXPANSIONS'))
            except ValueError:
                pass

    @property
    def enhanced_search(self) -> EnhancedSearchConfig:
        """Get enhanced search configuration."""
        return self._enhanced_search_config

    def get_anti_bot_description(self, level: int) -> str:
        """Get description for anti-bot level."""
        return self._enhanced_search_config.anti_bot_levels.get(level, "unknown")

    def validate_anti_bot_level(self, level: int) -> int:
        """Validate and clamp anti-bot level to valid range."""
        return max(0, min(3, level))

    def validate_crawl_threshold(self, threshold: float) -> float:
        """Validate and clamp crawl threshold to valid range."""
        return max(0.0, min(1.0, threshold))

    def ensure_workproduct_directory(self, custom_dir: str = None, session_id: str = None, category: str = "research") -> Path:
        """Ensure workproduct directory exists and return path with session-based organization."""
        if custom_dir:
            workproduct_dir = Path(custom_dir)
        elif session_id:
            # Use session-based directory structure
            base_sessions_dir = Path("/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN/sessions")
            session_dir = base_sessions_dir / session_id
            workproduct_dir = session_dir / category
        elif self._enhanced_search_config.kevin_workproducts_dir:
            workproduct_dir = Path(self._enhanced_search_config.kevin_workproducts_dir)
        else:
            workproduct_dir = Path(self._enhanced_search_config.default_workproduct_dir)

        workproduct_dir.mkdir(parents=True, exist_ok=True)
        return workproduct_dir

    def get_default_search_params(self) -> dict[str, Any]:
        """Get default search parameters."""
        return {
            "num_results": self._enhanced_search_config.default_num_results,
            "auto_crawl_top": self._enhanced_search_config.default_auto_crawl_top,
            "crawl_threshold": self._enhanced_search_config.default_crawl_threshold,
            "anti_bot_level": self._enhanced_search_config.default_anti_bot_level,
            "max_concurrent": self._enhanced_search_config.default_max_concurrent,
            "session_id": "default"
        }

    def get_debug_info(self) -> dict[str, Any]:
        """Get debug information about current configuration."""
        return {
            "enhanced_search_config": {
                "default_num_results": self._enhanced_search_config.default_num_results,
                "default_auto_crawl_top": self._enhanced_search_config.default_auto_crawl_top,
                "default_crawl_threshold": self._enhanced_search_config.default_crawl_threshold,
                "default_anti_bot_level": self._enhanced_search_config.default_anti_bot_level,
                "default_max_concurrent": self._enhanced_search_config.default_max_concurrent,
                "workproduct_dir": self._enhanced_search_config.default_workproduct_dir,
                "kevin_workproducts_dir": self._enhanced_search_config.kevin_workproducts_dir
            },
            "environment_variables": {
                "SERP_API_KEY": "SET" if os.getenv('SERP_API_KEY') else "NOT_SET",
                "OPENAI_API_KEY": "SET" if os.getenv('OPENAI_API_KEY') else "NOT_SET",
                "ANTHROPIC_API_KEY": "SET" if os.getenv('ANTHROPIC_API_KEY') else "NOT_SET",
                "KEVIN_WORKPRODUCTS_DIR": os.getenv('KEVIN_WORKPRODUCTS_DIR', 'NOT_SET')
            }
        }


# Global settings instance
_settings_manager = None


def get_settings() -> SettingsManager:
    """Get the global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager


def get_enhanced_search_config() -> EnhancedSearchConfig:
    """Get enhanced search configuration."""
    return get_settings().enhanced_search


# Export functions
__all__ = [
    'EnhancedSearchConfig',
    'SettingsManager',
    'get_settings',
    'get_enhanced_search_config'
]
