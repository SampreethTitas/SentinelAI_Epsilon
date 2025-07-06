"""
AdShield Document Processing Utilities
Handles various document formats for marketing content analysis
"""

import json
import docx
import mammoth
from datetime import datetime
import pandas as pd
from typing import Dict, List, Union, Optional
from pathlib import Path
import logging
from .adshield_core_part1 import AdShieldCore, ThreatLevel, AttackType, ContentType #, DetectionResult


logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process various document formats for AdShield analysis"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.docx', '.json', '.csv', '.html']
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, str]:
        """
        Process uploaded file and extract text content
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary with extracted content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        try:
            if file_path.suffix.lower() == '.txt':
                return self._process_text_file(file_path)
            elif file_path.suffix.lower() == '.docx':
                return self._process_docx_file(file_path)
            elif file_path.suffix.lower() == '.json':
                return self._process_json_file(file_path)
            elif file_path.suffix.lower() == '.csv':
                return self._process_csv_file(file_path)
            elif file_path.suffix.lower() == '.html':
                return self._process_html_file(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def _process_text_file(self, file_path: Path) -> Dict[str, str]:
        """Process plain text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return {
            'content': content,
            'filename': file_path.name,
            'format': 'text',
            'size': len(content),
            'metadata': {
                'line_count': content.count('\n') + 1,
                'word_count': len(content.split())
            }
        }
    
    def _process_docx_file(self, file_path: Path) -> Dict[str, str]:
        """Process Word document"""
        try:
            # Try using mammoth for better formatting
            with open(file_path, 'rb') as f:
                result = mammoth.extract_raw_text(f)
                content = result.value
        except Exception:
            # Fallback to python-docx
            doc = docx.Document(file_path)
            content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        return {
            'content': content,
            'filename': file_path.name,
            'format': 'docx',
            'size': len(content),
            'metadata': {
                'paragraph_count': content.count('\n') + 1,
                'word_count': len(content.split())
            }
        }
    
    def _process_json_file(self, file_path: Path) -> Dict[str, str]:
        """Process JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text content from JSON
        content = self._extract_text_from_json(data)
        
        return {
            'content': content,
            'filename': file_path.name,
            'format': 'json',
            'size': len(content),
            'metadata': {
                'json_keys': list(data.keys()) if isinstance(data, dict) else [],
                'json_structure': type(data).__name__
            }
        }
    
    def _process_csv_file(self, file_path: Path) -> Dict[str, str]:
        """Process CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Extract text content from all string columns
            text_content = []
            for column in df.columns:
                if df[column].dtype == 'object':  # String columns
                    text_content.extend(df[column].dropna().astype(str).tolist())
            
            content = '\n'.join(text_content)
            
            return {
                'content': content,
                'filename': file_path.name,
                'format': 'csv',
                'size': len(content),
                'metadata': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist()
                }
            }
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            raise
    
    def _process_html_file(self, file_path: Path) -> Dict[str, str]:
        """Process HTML file"""
        try:
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            content = soup.get_text()
            
            # Clean up whitespace
            content = '\n'.join(line.strip() for line in content.splitlines() if line.strip())
            
            return {
                'content': content,
                'filename': file_path.name,
                'format': 'html',
                'size': len(content),
                'metadata': {
                    'html_tags': len(soup.find_all()),
                    'links': len(soup.find_all('a')),
                    'images': len(soup.find_all('img'))
                }
            }
        except ImportError:
            logger.warning("BeautifulSoup not available, using basic HTML processing")
            # Basic HTML processing without BeautifulSoup
            import re
            
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Remove script and style tags
            html_content = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            html_content = re.sub(r'<style.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove all HTML tags
            content = re.sub(r'<[^>]+>', '', html_content)
            
            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            
            return {
                'content': content,
                'filename': file_path.name,
                'format': 'html',
                'size': len(content),
                'metadata': {
                    'processed_with': 'basic_regex'
                }
            }
    
    def _extract_text_from_json(self, data, max_depth=5, current_depth=0) -> str:
        """Recursively extract text content from JSON data"""
        if current_depth > max_depth:
            return ""
        
        text_content = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    text_content.append(value)
                elif isinstance(value, (dict, list)):
                    text_content.append(self._extract_text_from_json(value, max_depth, current_depth + 1))
        
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    text_content.append(item)
                elif isinstance(item, (dict, list)):
                    text_content.append(self._extract_text_from_json(item, max_depth, current_depth + 1))
        
        elif isinstance(data, str):
            text_content.append(data)
        
        return '\n'.join(filter(None, text_content))

class BatchProcessor:
    """Process multiple files in batch"""
    
    def __init__(self, adshield_analyzer):
        self.analyzer = adshield_analyzer
        self.processor = DocumentProcessor()
    
    def process_directory(self, directory_path: Union[str, Path], 
                         recursive: bool = False) -> List[Dict]:
        """
        Process all supported files in a directory
        
        Args:
            directory_path: Path to directory containing files
            recursive: Whether to process subdirectories
            
        Returns:
            List of analysis results
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        results = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.processor.supported_formats:
                try:
                    # Process file
                    processed_file = self.processor.process_file(file_path)
                    
                    # Analyze content
                    analysis_result = self.analyzer.analyze_content(processed_file['content'])
                    
                    # Combine results
                    results.append({
                        'file_info': processed_file,
                        'analysis': analysis_result,
                        'file_path': str(file_path)
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results.append({
                        'file_path': str(file_path),
                        'error': str(e)
                    })
        
        return results
    
    def generate_report(self, results: List[Dict], output_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive report from batch analysis results
        
        Args:
            results: List of analysis results
            output_path: Optional path to save report
            
        Returns:
            Report dictionary
        """
        report = {
            'summary': {
                'total_files': len(results),
                'successful_analyses': len([r for r in results if 'analysis' in r]),
                'failed_analyses': len([r for r in results if 'error' in r]),
                'timestamp': datetime.now().isoformat()
            },
            'threat_distribution': {},
            'attack_type_distribution': {},
            'compliance_scores': [],
            'marketing_scores': [],
            'high_risk_files': [],
            'recommendations': []
        }
        
        # Process successful analyses
        for result in results:
            if 'analysis' not in result:
                continue
            
            analysis = result['analysis']
            
            # Threat level distribution
            threat_level = analysis.threat_level.value
            report['threat_distribution'][threat_level] = report['threat_distribution'].get(threat_level, 0) + 1
            
            # Attack type distribution
            for attack_type in analysis.attack_types:
                attack_name = attack_type.value
                report['attack_type_distribution'][attack_name] = report['attack_type_distribution'].get(attack_name, 0) + 1
            
            # Scores
            report['compliance_scores'].append(analysis.compliance_score)
            report['marketing_scores'].append(analysis.marketing_score)
            
            # High risk files
            if analysis.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                report['high_risk_files'].append({
                    'file_path': result['file_path'],
                    'threat_level': threat_level,
                    'attack_types': [at.value for at in analysis.attack_types],
                    'recommendation': analysis.recommendation
                })
        
        # Generate recommendations
        if report['high_risk_files']:
            report['recommendations'].append("Immediate attention required for high-risk files")
        
        if report['compliance_scores']:
            avg_compliance = sum(report['compliance_scores']) / len(report['compliance_scores'])
            if avg_compliance < 70:
                report['recommendations'].append("Overall compliance score is below recommended threshold")
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_path}")
        
        return report
