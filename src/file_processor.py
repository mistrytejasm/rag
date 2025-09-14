import os
from pathlib import Path
from typing import Dict
import pandas as pd
import PyPDF2
import docx
from bs4 import BeautifulSoup
import json
from src.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.support_extensions = {
            '.txt': self._process_text,
            '.md': self._process_text,
            '.pdf': self._process_pdf_with_pages,
            '.docx': self._process_docx_with_pages,
            '.csv': self._process_csv,
            '.xlsx': self._process_excel,
            '.html': self._process_html,
            '.json': self._process_json,
        }

    def process_document(self, file_path: str) -> Dict:
        """Process document and extract content with page-aware metadata"""
        extension = Path(file_path).suffix.lower()

        if extension not in self.support_extensions:
            logger.error(f"Unsupported file type: {extension}")
            raise ValueError(f"Unsupported file type: {extension}")
        
        # Get content with page information
        result = self.support_extensions[extension](file_path)
        
        # Handle both page-aware and simple content
        if isinstance(result, dict) and 'pages' in result:
            content = result['content']
            pages = result['pages']
        else:
            content = result
            pages = None
        
        metadata = {
            'filename': Path(file_path).name,
            'file_path': file_path,
            'file_type': extension,
            'file_size': os.path.getsize(file_path),
            'content_length': len(content),
            'has_page_info': pages is not None,
            'total_pages': len(pages) if pages else None
        }
        
        logger.info(f"Processed {file_path}; length: {len(content)} chars, pages: {len(pages) if pages else 'N/A'}")
        
        return {
            'content': content, 
            'metadata': metadata,
            'pages': pages  
        }
    

    def _process_text(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            return ""

    def _process_pdf_with_pages(self, file_path: str) -> Dict:
        """Extract PDF content with page tracking"""
        full_text = ""
        pages = []
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text() or ""
                    
                    # Store page information
                    page_info = {
                        'page_number': page_num,
                        'start_char': len(full_text),
                        'end_char': len(full_text) + len(page_text),
                        'content': page_text,
                        'char_count': len(page_text)
                    }
                    pages.append(page_info)
                    
                    full_text += page_text + "\n" 
                
            return {
                'content': full_text,
                'pages': pages
            }
            
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {e}")
            return {'content': "", 'pages': []}

    def _process_docx_with_pages(self, file_path: str) -> Dict:
        """Extract DOCX content with page tracking (approximated)"""
        full_text = ""
        pages = []
        
        try:
            doc = docx.Document(file_path)
            
            # Approximate page breaks (DOCX doesn't have clear page boundaries)
            current_page = 1
            chars_per_page = 2500  # Approximate characters per page
            page_start = 0
            
            for para in doc.paragraphs:
                para_text = para.text + "\n"
                
                # Check if we should start a new page (rough approximation)
                if len(full_text) - page_start > chars_per_page and para_text.strip():
                    # Finalize current page
                    pages.append({
                        'page_number': current_page,
                        'start_char': page_start,
                        'end_char': len(full_text),
                        'content': full_text[page_start:],
                        'char_count': len(full_text) - page_start
                    })
                    
                    # Start new page
                    current_page += 1
                    page_start = len(full_text)
                
                full_text += para_text
            
            # Add final page
            if len(full_text) > page_start:
                pages.append({
                    'page_number': current_page,
                    'start_char': page_start,
                    'end_char': len(full_text),
                    'content': full_text[page_start:],
                    'char_count': len(full_text) - page_start
                })
            
            return {
                'content': full_text,
                'pages': pages
            }
            
        except Exception as e:
            logger.error(f"Failed to process DOCX {file_path}: {e}")
            return {'content': "", 'pages': []}

    def _process_csv(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
            return df.to_csv(index=False)
        except Exception as e:
            logger.error(f"Failed to process CSV {file_path}: {e}")
            return ""

    def _process_excel(self, file_path: str) -> str:
        try:
            df = pd.read_excel(file_path)
            return df.to_csv(index=False)
        except Exception as e:
            logger.error(f"Failed to process Excel {file_path}: {e}")
            return ""

    def _process_html(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                return soup.get_text()
        except Exception as e:
            logger.error(f"Failed to process HTML {file_path}: {e}")
            return ""

    def _process_json(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"Failed to process JSON {file_path}: {e}")
            return ""
