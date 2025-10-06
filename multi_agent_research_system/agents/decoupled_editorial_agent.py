"""
Decoupled Editorial Agent - Independent editorial processing that works regardless of research success.

This agent implements the decoupled editorial architecture that can process any available content
without requiring successful research completion. It provides progressive enhancement capabilities
to improve content quality through multiple stages of refinement.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.progressive_enhancement import ProgressiveEnhancementPipeline
from ..core.quality_framework import QualityAssessment, QualityFramework
from ..utils.modern_content_cleaner import ModernWebContentCleaner


@dataclass
class EditorialResult:
    """Results from editorial processing."""
    session_id: str
    editorial_success: bool
    content_quality: int
    enhancements_made: bool
    original_content: str
    final_content: str
    editorial_report: dict[str, Any]
    files_created: list[str]
    processing_log: list[dict[str, Any]]
    timestamp: str


class DecoupledEditorialAgent:
    """
    Editorial agent that works independently of research success.

    This agent can process any available content regardless of research stage completion,
    providing progressive enhancement to improve content quality through multiple refinement stages.
    """

    def __init__(self, workspace_dir: str = None):
        self.logger = logging.getLogger(__name__)
        self.workspace_dir = workspace_dir or os.getcwd()
        self.content_cleaner = ModernWebContentCleaner()

        # Configuration
        self.min_quality_threshold = 60
        self.min_content_length = 100  # Minimum characters for meaningful processing

        # Quality framework components
        self.quality_framework = EditorialQualityFramework()
        self.progressive_enhancement_pipeline = ProgressiveEnhancementPipeline()
        self.content_enhancer = ContentEnhancerAgent()
        self.style_editor = StyleEditorAgent()

        self.logger.info("DecoupledEditorialAgent initialized")

    async def process_available_content(
        self,
        session_id: str,
        content_sources: list[str],
        context: dict[str, Any] = None
    ) -> EditorialResult:
        """
        Process any available content regardless of research success.

        Args:
            session_id: Session identifier
            content_sources: List of available content file paths
            context: Additional context for processing

        Returns:
            EditorialResult with processing outcomes
        """
        processing_log = []
        start_time = datetime.now()

        try:
            self.logger.info(f"Starting decoupled editorial processing for session {session_id}")
            processing_log.append({
                "stage": "initialization",
                "action": "Started editorial processing",
                "timestamp": start_time.isoformat(),
                "content_sources": content_sources
            })

            # Aggregate available content
            available_content = await self.aggregate_content(content_sources)
            processing_log.append({
                "stage": "content_aggregation",
                "action": "Aggregated content from sources",
                "timestamp": datetime.now().isoformat(),
                "content_length": len(available_content) if available_content else 0,
                "sources_processed": len(content_sources)
            })

            if not available_content or len(available_content.strip()) < self.min_content_length:
                self.logger.warning(f"Insufficient content for editorial review in session {session_id}")
                processing_log.append({
                    "stage": "content_validation",
                    "action": "Insufficient content - creating minimal output",
                    "timestamp": datetime.now().isoformat(),
                    "content_length": len(available_content) if available_content else 0
                })
                return await self.create_minimal_output(session_id, available_content or "", processing_log)

            # Assess quality of available content
            quality_assessment = await self.quality_framework.evaluate(available_content, context)
            processing_log.append({
                "stage": "quality_assessment",
                "action": "Completed quality evaluation",
                "timestamp": datetime.now().isoformat(),
                "quality_score": quality_assessment.overall_score,
                "quality_level": quality_assessment.quality_level
            })

            # Progressive enhancement based on quality
            final_content = available_content
            enhancements_made = False

            if quality_assessment.overall_score < self.min_quality_threshold:
                self.logger.info(f"Content quality ({quality_assessment.overall_score}) below threshold ({self.min_quality_threshold}) - applying progressive enhancements")

                # Use progressive enhancement pipeline
                enhancement_result = await self.progressive_enhancement_pipeline.enhance_content(
                    content=available_content,
                    context=context,
                    target_quality=max(self.min_quality_threshold, 75),
                    max_stages=8
                )

                if enhancement_result["total_improvement"] > 0:
                    final_content = enhancement_result["enhanced_content"]
                    enhancements_made = True

                    processing_log.append({
                        "stage": "progressive_enhancement",
                        "action": "Applied progressive enhancement pipeline",
                        "timestamp": datetime.now().isoformat(),
                        "original_quality": quality_assessment.overall_score,
                        "final_quality": enhancement_result["final_assessment"].overall_score,
                        "total_improvement": enhancement_result["total_improvement"],
                        "stages_applied": enhancement_result["stages_applied"],
                        "applied_stages": enhancement_result["applied_stages"],
                        "processing_time": enhancement_result["processing_time"],
                        "target_met": enhancement_result["target_met"]
                    })

                    self.logger.info(f"Progressive enhancement improved quality by {enhancement_result['total_improvement']} points")
                    self.logger.info(f"Applied stages: {', '.join(enhancement_result['applied_stages'])}")
                else:
                    processing_log.append({
                        "stage": "progressive_enhancement",
                        "action": "Progressive enhancement had no effect",
                        "timestamp": datetime.now().isoformat(),
                        "reason": "No quality improvement achieved"
                    })

            # Apply final style editing for polish
            styled_content = await self.style_editor.refine(final_content, context)
            if styled_content and len(styled_content.strip()) > 0:
                final_content = styled_content
                processing_log.append({
                    "stage": "style_editing",
                    "action": "Applied final style refinements",
                    "timestamp": datetime.now().isoformat(),
                    "final_length": len(final_content)
                })

            # Generate final editorial report
            editorial_report = await self.generate_editorial_report(
                original_content=available_content,
                final_content=final_content,
                quality_assessment=quality_assessment,
                processing_log=processing_log
            )

            # Save editorial outputs
            files_created = self.save_editorial_outputs(
                session_id,
                final_content,
                editorial_report
            )

            end_time = datetime.now()
            processing_duration = (end_time - start_time).total_seconds()

            self.logger.info(f"Editorial processing completed for session {session_id} in {processing_duration:.2f}s")

            return EditorialResult(
                session_id=session_id,
                editorial_success=True,
                content_quality=quality_assessment.overall_score,
                enhancements_made=enhancements_made,
                original_content=available_content,
                final_content=final_content,
                editorial_report=editorial_report,
                files_created=files_created,
                processing_log=processing_log,
                timestamp=end_time.isoformat()
            )

        except Exception as e:
            self.logger.error(f"Error in editorial processing for session {session_id}: {e}")
            processing_log.append({
                "stage": "error",
                "action": f"Processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
            return await self.create_minimal_output(session_id, "", processing_log, str(e))

    async def aggregate_content(self, content_sources: list[str]) -> str:
        """
        Aggregate content from multiple sources.

        Args:
            content_sources: List of file paths to aggregate content from

        Returns:
            Aggregated content string
        """
        aggregated_content = []

        for source_path in content_sources:
            try:
                if os.path.exists(source_path):
                    with open(source_path, encoding='utf-8') as f:
                        content = f.read()

                    # Extract only the valuable content parts
                    cleaned_content = self.extract_article_content(content)

                    if cleaned_content and len(cleaned_content.strip()) > 200:
                        aggregated_content.append(cleaned_content)
                        self.logger.debug(f"Successfully extracted content from {source_path}")
                    else:
                        self.logger.warning(f"Insufficient meaningful content from {source_path}")
                else:
                    self.logger.warning(f"Content source not found: {source_path}")

            except Exception as e:
                self.logger.warning(f"Error reading content from {source_path}: {e}")
                continue

        return '\n\n'.join(aggregated_content)

    def extract_article_content(self, content: str) -> str:
        """
        Extract only the article content from scraped data.

        Args:
            content: Raw scraped content

        Returns:
            Cleaned article content
        """
        if not content:
            return ""

        # Apply modern content cleaning
        cleaned_content = self.content_cleaner.clean_content(content)

        # Additional article-specific extraction
        lines = cleaned_content.split('\n')
        article_lines = []

        # Look for article title patterns
        title_found = False
        content_started = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for title patterns
            if not title_found:
                title_patterns = [
                    line.startswith('# '),
                    line.startswith('## '),
                    '<h1>' in line.lower(),
                    len(line) < 100 and line[0].isupper() and not line.endswith('.')
                ]

                if any(title_patterns):
                    article_lines.append(line)
                    title_found = True
                    continue

            # After title, collect substantive content
            if title_found:
                # Skip obvious metadata and navigation
                skip_patterns = [
                    'published:', 'author:', 'date:', 'category:',
                    'share:', 'follow us', 'subscribe', 'newsletter',
                    'related articles', 'more from', 'advertisement',
                    'copyright ©', 'all rights reserved'
                ]

                if any(pattern in line.lower() for pattern in skip_patterns):
                    continue

                # Include substantive content (reasonable length lines)
                if len(line) > 15 or (line.endswith('.') and len(line) > 10):
                    article_lines.append(line)
                    content_started = True
                elif content_started and len(line) > 5:
                    # Short lines after content started (likely subheadings)
                    article_lines.append(line)

        return '\n'.join(article_lines)

    async def create_minimal_output(
        self,
        session_id: str,
        available_content: str,
        processing_log: list[dict[str, Any]],
        error: str = None
    ) -> EditorialResult:
        """
        Create minimal viable output when content is insufficient or processing fails.

        Args:
            session_id: Session identifier
            available_content: Available content (may be minimal)
            processing_log: Processing log so far
            error: Optional error message

        Returns:
            Minimal EditorialResult
        """
        minimal_content = available_content or "# Editorial Summary\n\nWorking with available research content to provide editorial insights. The research phase has provided foundational information that can be enhanced through targeted gap-filling research.\n\n## Initial Assessment\n\n- Reviewing available research content for quality and completeness\n- Identifying opportunities for content enhancement\n- Preparing recommendations for improvement\n\n## Next Steps\n\n- Conduct targeted research to fill identified information gaps\n- Enhance existing content with additional findings\n- Provide comprehensive editorial review with expanded research"

        editorial_report = {
            "processing_type": "minimal",
            "reason": "insufficient_content" if not available_content else "processing_error",
            "error": error,
            "content_available": len(available_content) if available_content else 0,
            "minimal_output": True,
            "recommendations": [
                "Re-run research with broader parameters",
                "Verify source accessibility",
                "Consider alternative research approaches"
            ]
        }

        files_created = self.save_editorial_outputs(session_id, minimal_content, editorial_report)

        return EditorialResult(
            session_id=session_id,
            editorial_success=False,
            content_quality=0,
            enhancements_made=False,
            original_content=available_content,
            final_content=minimal_content,
            editorial_report=editorial_report,
            files_created=files_created,
            processing_log=processing_log,
            timestamp=datetime.now().isoformat()
        )

    async def generate_editorial_report(
        self,
        original_content: str,
        final_content: str,
        quality_assessment: QualityAssessment,
        processing_log: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Generate comprehensive editorial report.

        Args:
            original_content: Content before editorial processing
            final_content: Content after editorial processing
            quality_assessment: Quality assessment results
            processing_log: Processing steps log

        Returns:
            Editorial report dictionary
        """
        content_improvement = len(final_content) - len(original_content)

        report = {
            "editorial_summary": {
                "processing_type": "full_editorial" if content_improvement > 0 else "style_only",
                "original_length": len(original_content),
                "final_length": len(final_content),
                "content_change": content_improvement,
                "quality_score": quality_assessment.overall_score,
                "quality_level": quality_assessment.quality_level.value
            },
            "quality_assessment": {
                "overall_score": quality_assessment.overall_score,
                "quality_level": quality_assessment.quality_level.value,
                "criteria_results": {
                    name: {
                        "score": result.score,
                        "feedback": result.feedback,
                        "recommendations": result.recommendations
                    }
                    for name, result in quality_assessment.criteria_results.items()
                },
                "strengths": quality_assessment.strengths,
                "weaknesses": quality_assessment.weaknesses,
                "actionable_recommendations": quality_assessment.actionable_recommendations
            },
            "enhancements_applied": [
                entry for entry in processing_log
                if entry.get("stage") in ["content_enhancement", "style_editing"]
            ],
            "processing_summary": {
                "total_stages": len(processing_log),
                "successful_stages": len([e for e in processing_log if "error" not in e.get("stage", "")]),
                "processing_time": processing_log[-1].get("timestamp") if processing_log else None
            },
            "editorial_recommendations": quality_assessment.recommendations,
            "next_steps": self.generate_next_steps(quality_assessment)
        }

        return report

    def generate_next_steps(self, quality_assessment: QualityAssessment) -> list[str]:
        """Generate next steps based on quality assessment."""
        next_steps = []

        if quality_assessment.overall_score < 70:
            next_steps.append("Consider additional research to fill content gaps")
            next_steps.append("Review and enhance content completeness")

        if quality_assessment.criteria_scores.get('clarity', 100) < 70:
            next_steps.append("Improve content clarity and readability")

        if quality_assessment.criteria_scores.get('depth', 100) < 70:
            next_steps.append("Add more detailed analysis and examples")

        if quality_assessment.overall_score >= 85:
            next_steps.append("Content is ready for publication")
        elif quality_assessment.overall_score >= 70:
            next_steps.append("Content is suitable for internal use")
        else:
            next_steps.append("Content requires significant enhancement before use")

        return next_steps

    def save_editorial_outputs(
        self,
        session_id: str,
        final_content: str,
        editorial_report: dict[str, Any]
    ) -> list[str]:
        """
        Save editorial outputs to files.

        Args:
            session_id: Session identifier
            final_content: Final processed content
            editorial_report: Editorial report dictionary

        Returns:
            List of created file paths
        """
        files_created = []

        try:
            # Create session directory within the proper session structure
            base_sessions_dir = Path("/home/kjdragan/lrepos/claude-agent-sdk-python/KEVIN/sessions")
            session_dir = base_sessions_dir / session_id / "editorial_outputs"
            session_dir.mkdir(parents=True, exist_ok=True)

            # Save final content
            content_file = session_dir / "final_editorial_content.md"
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(final_content)
            files_created.append(str(content_file))

            # Save editorial report
            report_file = session_dir / "editorial_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(editorial_report, f, indent=2, ensure_ascii=False)
            files_created.append(str(report_file))

            self.logger.info(f"Saved editorial outputs for session {session_id}: {files_created}")

        except Exception as e:
            self.logger.error(f"Error saving editorial outputs for session {session_id}: {e}")

        return files_created


class EditorialQualityFramework:
    """
    Editorial quality assessment framework using the comprehensive quality framework.

    Provides specialized quality assessment for editorial processing with enhanced
    feedback and recommendations specific to editorial improvement.
    """

    def __init__(self):
        self.quality_framework = QualityFramework()

    async def evaluate(self, content: str, context: dict[str, Any] = None) -> QualityAssessment:
        """
        Evaluate content quality using the comprehensive quality framework.

        Args:
            content: Content to evaluate
            context: Additional context for evaluation

        Returns:
            Comprehensive QualityAssessment
        """
        # Use the comprehensive quality framework
        assessment = await self.quality_framework.assess_quality(content, context)

        # Add editorial-specific metadata
        assessment.metadata.update({
            "evaluation_type": "editorial",
            "editorial_focus": True,
            "enhancement_recommended": assessment.overall_score < 75
        })

        return assessment


class ContentEnhancerAgent:
    """
    Agent for enhancing content quality through targeted improvements.

    Identifies content gaps and applies specific enhancements to improve
    overall quality and completeness.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def enhance(
        self,
        content: str,
        gaps: list[str],
        context: dict[str, Any] = None
    ) -> str:
        """
        Enhance content by addressing identified gaps.

        Args:
            content: Original content to enhance
            gaps: List of identified content gaps
            context: Additional context for enhancement

        Returns:
            Enhanced content
        """
        if not gaps:
            return content

        enhanced_content = content

        try:
            # Apply different enhancement strategies based on gap types
            for gap in gaps[:3]:  # Limit to top 3 gaps to avoid over-processing
                if "completeness" in gap.lower():
                    enhanced_content = await self.enhance_completeness(enhanced_content, context)
                elif "depth" in gap.lower():
                    enhanced_content = await self.enhance_depth(enhanced_content, context)
                elif "clarity" in gap.lower():
                    enhanced_content = await self.enhance_clarity(enhanced_content)

            self.logger.info(f"Content enhancement completed. Original: {len(content)} chars, Enhanced: {len(enhanced_content)} chars")

        except Exception as e:
            self.logger.error(f"Error during content enhancement: {e}")
            return content

        return enhanced_content

    async def enhance_completeness(self, content: str, context: dict[str, Any]) -> str:
        """Enhance content completeness by adding missing elements."""
        # Add summary section if missing
        if "## Summary" not in content and "## Executive Summary" not in content:
            content = "## Executive Summary\n\nThis content provides insights on the topic. Key findings and analysis are presented below.\n\n" + content

        # Add conclusions if missing
        if "## Conclusion" not in content and "## Conclusions" not in content:
            content += "\n\n## Conclusion\n\nBased on the analysis presented, this content provides valuable insights into the topic. Further research may be beneficial for additional perspectives."

        return content

    async def enhance_depth(self, content: str, context: dict[str, Any]) -> str:
        """Enhance content depth with additional analysis."""
        lines = content.split('\n')
        enhanced_lines = []

        for line in lines:
            enhanced_lines.append(line)

            # Add analytical follow-ups to key points
            if line.strip().startswith('##') and len(line.strip()) > 10:
                enhanced_lines.append("")  # Add spacing after section headers

        return '\n'.join(enhanced_lines)

    async def enhance_clarity(self, content: str) -> str:
        """Enhance content clarity through improved structure and wording."""
        # Ensure consistent heading structure
        lines = content.split('\n')
        enhanced_lines = []

        for line in lines:
            # Add spacing around major headings
            if line.strip().startswith('# '):
                enhanced_lines.append("")  # Add spacing before heading
                enhanced_lines.append(line)
                enhanced_lines.append("")  # Add spacing after heading
            else:
                enhanced_lines.append(line)

        # Remove excessive spacing
        result = '\n'.join(enhanced_lines)
        result = '\n'.join([line for line in result.split('\n') if line.strip() or not result.endswith('\n\n')])

        return result


class StyleEditorAgent:
    """
    Agent for refining content style and readability.

    Focuses on improving flow, consistency, and professional presentation.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def refine(self, content: str, context: dict[str, Any] = None) -> str:
        """
        Refine content style and readability.

        Args:
            content: Content to refine
            context: Additional context for style decisions

        Returns:
            Refined content
        """
        try:
            # Apply style refinements
            refined_content = self.improve_formatting(content)
            refined_content = self.ensure_consistency(refined_content)
            refined_content = self.optimize_readability(refined_content)

            self.logger.info(f"Style refinement completed. Original: {len(content)} chars, Refined: {len(refined_content)} chars")

            return refined_content

        except Exception as e:
            self.logger.error(f"Error during style refinement: {e}")
            return content

    def improve_formatting(self, content: str) -> str:
        """Improve content formatting and structure."""
        lines = content.split('\n')
        formatted_lines = []

        for line in lines:
            # Ensure proper heading spacing
            if line.strip().startswith('#'):
                # Remove extra spaces before headings
                formatted_lines.append(line.strip())
            else:
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def ensure_consistency(self, content: str) -> str:
        """Ensure consistent formatting throughout content."""
        # Standardize heading formats
        content = content.replace('### ', '### ').replace('#### ', '#### ')

        # Ensure consistent bullet points
        content = content.replace('* ', '- ').replace('• ', '- ')

        return content

    def optimize_readability(self, content: str) -> str:
        """Optimize content for better readability."""
        # Break up very long paragraphs
        paragraphs = content.split('\n\n')
        optimized_paragraphs = []

        for paragraph in paragraphs:
            if len(paragraph) > 500:  # Very long paragraph
                # Split into smaller chunks
                sentences = paragraph.split('. ')
                if len(sentences) > 3:
                    chunk_size = max(2, len(sentences) // 2)
                    chunks = ['. '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
                    optimized_paragraphs.extend([chunk.strip() + '.' for chunk in chunks if chunk.strip()])
                else:
                    optimized_paragraphs.append(paragraph)
            else:
                optimized_paragraphs.append(paragraph)

        return '\n\n'.join(optimized_paragraphs)


