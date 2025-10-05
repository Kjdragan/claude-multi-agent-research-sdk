"""
CLI Input Parser - Extracts research topic and parameters from raw CLI input.

This module handles parsing of raw CLI input strings to separate the core research topic
from configuration parameters, replacing regex-based parsing with structured parsing.
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedResearchRequest:
    """Structured representation of a parsed research request."""
    clean_topic: str
    raw_input: str
    parameters: Dict[str, Any]
    scope: str = "default"
    report_type: str = "default"
    sources_requested: int = 10
    crawl_speed: str = "normal"
    special_requirements: str = ""

    def __post_init__(self):
        """Validate and set derived properties."""
        # Ensure scope is valid
        valid_scopes = ["limited", "brief", "default", "comprehensive", "extensive"]
        if self.scope not in valid_scopes:
            self.scope = "default"

        # Ensure report_type is valid
        valid_report_types = ["brief", "default", "comprehensive"]
        if self.report_type not in valid_report_types:
            self.report_type = "default"

        # Ensure sources is reasonable
        if not isinstance(self.sources_requested, int) or self.sources_requested < 1:
            self.sources_requested = 10
        elif self.sources_requested > 50:
            self.sources_requested = 50


class CLIInputParser:
    """Parses CLI input to extract clean research topic and structured parameters."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Define parameter patterns
        self.parameter_patterns = {
            'scope': r'scope=(\w+)',
            'report_brief': r'report_brief',
            'sources': r'sources=(\d+)',
            'fast_crawl': r'fast_crawl',
            'quick_search': r'quick_search',
            'focus_on': r'focus on ([^.]+)',
            'emphasize': r'emphasize ([^.]+)'
        }

    def parse_input(self, raw_input: str) -> ParsedResearchRequest:
        """
        Parse raw CLI input into structured research request.

        Args:
            raw_input: Raw CLI input string

        Returns:
            ParsedResearchRequest with clean topic and structured parameters
        """
        self.logger.info(f"Parsing CLI input: {raw_input}")

        # Initialize parameters
        parameters = {}
        clean_topic = raw_input

        # Extract parameters using regex patterns
        for param_name, pattern in self.parameter_patterns.items():
            matches = re.findall(pattern, raw_input, re.IGNORECASE)
            if matches:
                if param_name == 'scope':
                    parameters['scope'] = matches[0].lower()
                elif param_name == 'report_brief':
                    parameters['report_type'] = 'brief'
                elif param_name == 'sources':
                    parameters['sources_requested'] = int(matches[0])
                elif param_name == 'fast_crawl':
                    parameters['crawl_speed'] = 'fast'
                elif param_name == 'quick_search':
                    parameters['crawl_speed'] = 'quick'
                elif param_name in ['focus_on', 'emphasize']:
                    parameters['special_requirements'] = matches[0].strip()

        # Clean the topic by removing parameter strings
        clean_topic = self._clean_topic(raw_input, parameters)

        # Apply intelligent topic analysis if no explicit parameters provided
        if not parameters.get('sources_requested') and not parameters.get('scope'):
            intelligent_params = self._analyze_topic_intelligently(clean_topic)
            parameters.update(intelligent_params)
            self.logger.info(f"Applied intelligent analysis: {intelligent_params}")

        # Create parsed request
        parsed_request = ParsedResearchRequest(
            clean_topic=clean_topic.strip(),
            raw_input=raw_input,
            parameters=parameters,
            scope=parameters.get('scope', 'default'),
            report_type=parameters.get('report_type', 'default'),
            sources_requested=parameters.get('sources_requested', 15),
            crawl_speed=parameters.get('crawl_speed', 'normal'),
            special_requirements=parameters.get('special_requirements', '')
        )

        self.logger.info(f"Parsed topic: '{parsed_request.clean_topic}'")
        self.logger.info(f"Extracted parameters: {parameters}")

        return parsed_request

    def _clean_topic(self, raw_input: str, parameters: Dict[str, Any]) -> str:
        """
        Remove parameter strings from the raw input to get clean topic.

        Args:
            raw_input: Raw CLI input
            parameters: Extracted parameters

        Returns:
            Clean topic string
        """
        clean_topic = raw_input

        # Remove parameter patterns from topic
        clean_topic = re.sub(r'scope=\w+', '', clean_topic, flags=re.IGNORECASE)
        clean_topic = re.sub(r'report_brief', '', clean_topic, flags=re.IGNORECASE)
        clean_topic = re.sub(r'sources=\d+', '', clean_topic, flags=re.IGNORECASE)
        clean_topic = re.sub(r'fast_crawl', '', clean_topic, flags=re.IGNORECASE)
        clean_topic = re.sub(r'quick_search', '', clean_topic, flags=re.IGNORECASE)
        clean_topic = re.sub(r'focus on [^.]+', '', clean_topic, flags=re.IGNORECASE)
        clean_topic = re.sub(r'emphasize [^.]+', '', clean_topic, flags=re.IGNORECASE)

        # Clean up extra commas and whitespace
        clean_topic = re.sub(r',+', ',', clean_topic)  # Multiple commas to single
        clean_topic = re.sub(r'\s+', ' ', clean_topic)  # Multiple spaces to single
        clean_topic = re.sub(r',\s*$', '', clean_topic)  # Trailing comma
        clean_topic = re.sub(r'^\s*,\s*', '', clean_topic)  # Leading comma
        clean_topic = re.sub(r'^\s*,+\s*', '', clean_topic)  # Multiple leading commas
        clean_topic = re.sub(r',\s*$', '', clean_topic)  # Trailing comma
        clean_topic = re.sub(r',\s*$', '', clean_topic)  # Remove trailing comma
        clean_topic = clean_topic.strip()

        return clean_topic

    def get_topic_for_search(self, parsed_request: ParsedResearchRequest) -> str:
        """
        Get the clean topic optimized for search engines.

        Args:
            parsed_request: Parsed research request

        Returns:
            Search-optimized topic string
        """
        # Use the clean topic directly - this is what should be sent to search APIs
        return parsed_request.clean_topic

    def _analyze_topic_intelligently(self, topic: str) -> Dict[str, Any]:
        """
        Analyze topic to determine intelligent default parameters.

        This eliminates the need for users to specify sources=N or scope parameters
        by automatically determining appropriate values based on topic analysis.

        Args:
            topic: Clean topic string

        Returns:
            Dictionary of intelligent parameters
        """
        topic_lower = topic.lower()

        # Default values
        intelligent_params = {
            'sources_requested': 15,  # Default target
            'scope': 'standard'
        }

        # Topic complexity analysis
        topic_indicators = {
            # Brief/scope indicators
            'brief': {'scope': 'limited', 'sources': 8},
            'summary': {'scope': 'limited', 'sources': 8},
            'overview': {'scope': 'limited', 'sources': 8},
            'quick': {'scope': 'brief', 'sources': 6},
            'basic': {'scope': 'limited', 'sources': 8},

            # Comprehensive indicators
            'comprehensive': {'scope': 'comprehensive', 'sources': 20},
            'detailed': {'scope': 'comprehensive', 'sources': 20},
            'in-depth': {'scope': 'comprehensive', 'sources': 20},
            'thorough': {'scope': 'comprehensive', 'sources': 20},
            'extensive': {'scope': 'extensive', 'sources': 30},
            'complete': {'scope': 'comprehensive', 'sources': 20},

            # Academic indicators
            'academic': {'scope': 'comprehensive', 'sources': 25},
            'research': {'scope': 'comprehensive', 'sources': 20},
            'scholarly': {'scope': 'comprehensive', 'sources': 25},
            'paper': {'scope': 'comprehensive', 'sources': 20},
            'thesis': {'scope': 'extensive', 'sources': 30},
            'dissertation': {'scope': 'extensive', 'sources': 30},

            # Business indicators
            'business': {'scope': 'standard', 'sources': 12},
            'market': {'scope': 'comprehensive', 'sources': 18},
            'industry': {'scope': 'comprehensive', 'sources': 18},
            'company': {'scope': 'standard', 'sources': 12},
            'startup': {'scope': 'standard', 'sources': 12},

            # Technical indicators
            'technical': {'scope': 'comprehensive', 'sources': 15},
            'engineering': {'scope': 'comprehensive', 'sources': 18},
            'technology': {'scope': 'comprehensive', 'sources': 15},
            'software': {'scope': 'standard', 'sources': 12},
            'programming': {'scope': 'standard', 'sources': 12},

            # News indicators
            'news': {'scope': 'standard', 'sources': 12},
            'latest': {'scope': 'standard', 'sources': 10},
            'recent': {'scope': 'standard', 'sources': 10},
            'current': {'scope': 'standard', 'sources': 10},
            'breaking': {'scope': 'limited', 'sources': 8},

            # Analysis indicators
            'analysis': {'scope': 'comprehensive', 'sources': 18},
            'evaluate': {'scope': 'comprehensive', 'sources': 18},
            'compare': {'scope': 'comprehensive', 'sources': 15},
            'review': {'scope': 'comprehensive', 'sources': 18},
            'assessment': {'scope': 'comprehensive', 'sources': 18}
        }

        # Check for topic indicators
        for indicator, params in topic_indicators.items():
            if indicator in topic_lower:
                intelligent_params['scope'] = params['scope']
                intelligent_params['sources_requested'] = params['sources']
                self.logger.info(f"Topic analysis: found '{indicator}' → scope={params['scope']}, sources={params['sources']}")
                break

        # Length-based adjustment
        word_count = len(topic.split())
        if word_count > 15:  # Long, complex topics
            intelligent_params['scope'] = 'comprehensive'
            intelligent_params['sources_requested'] = max(intelligent_params['sources_requested'], 18)
        elif word_count < 5:  # Short, simple topics
            intelligent_params['scope'] = 'limited'
            intelligent_params['sources_requested'] = min(intelligent_params['sources_requested'], 10)

        # Question-based analysis
        if '?' in topic:
            question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
            if any(qword in topic_lower for qword in question_words):
                intelligent_params['scope'] = 'standard'
                intelligent_params['sources_requested'] = 12

        # Time-based indicators
        time_indicators = ['2024', '2023', '2022', '2021', '2020', '2019', '2018']
        if any(year in topic for year in time_indicators):
            intelligent_params['scope'] = 'comprehensive'
            intelligent_params['sources_requested'] = max(intelligent_params['sources_requested'], 15)

        # Multiple topics analysis (indicated by 'and', 'vs', 'versus', 'compare')
        if any(separator in topic_lower for separator in [' and ', ' vs ', ' versus ', ' compare ']):
            intelligent_params['scope'] = 'comprehensive'
            intelligent_params['sources_requested'] = max(intelligent_params['sources_requested'], 20)

        self.logger.info(f"Intelligent topic analysis result: {intelligent_params}")
        return intelligent_params

    def get_parameter_summary(self, parsed_request: ParsedResearchRequest) -> str:
        """
        Get a human-readable summary of the parsed parameters.

        Args:
            parsed_request: Parsed research request

        Returns:
            Parameter summary string
        """
        summary_parts = []

        if parsed_request.scope != "default":
            summary_parts.append(f"scope={parsed_request.scope}")

        if parsed_request.report_type != "default":
            summary_parts.append(f"report_type={parsed_request.report_type}")

        if parsed_request.sources_requested != 15:
            summary_parts.append(f"sources={parsed_request.sources_requested}")

        if parsed_request.crawl_speed != "normal":
            summary_parts.append(f"crawl_speed={parsed_request.crawl_speed}")

        if parsed_request.special_requirements:
            summary_parts.append(f"focus={parsed_request.special_requirements}")

        return ", ".join(summary_parts) if summary_parts else "default parameters"


# Global parser instance
_cli_parser = None


def get_cli_parser() -> CLIInputParser:
    """Get the global CLI parser instance."""
    global _cli_parser
    if _cli_parser is None:
        _cli_parser = CLIInputParser()
    return _cli_parser


def parse_cli_input(raw_input: str) -> ParsedResearchRequest:
    """
    Parse CLI input using the global parser instance.

    Args:
        raw_input: Raw CLI input string

    Returns:
        ParsedResearchRequest
    """
    parser = get_cli_parser()
    return parser.parse_input(raw_input)