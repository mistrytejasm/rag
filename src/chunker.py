import re
from typing import List, Dict, Optional
import tiktoken
from src.logger import get_logger

logger = get_logger(__name__)

class Chunker:
    def __init__(self, strategy="semantic", chunk_size=200, overlap=50, model_name="gpt-3.5-turbo"):
        self.strategy = strategy
        self.chunk_size = chunk_size  # in tokens, not words
        self.overlap = overlap        # in tokens
        self.model_name = model_name
        try:
            self.enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Model {model_name} not found, using cl100k_base encoding")
            self.enc = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str, metadata: Dict, pages: Optional[List[Dict]] = None) -> List[Dict]:
        """Main chunking method with page awareness"""
        if self.strategy == "semantic":
            return self.semantic_chunking(text, metadata, pages)
        else:
            return self.fixed_size_chunking(text, metadata, pages)
        
    def _find_page_for_position(self, char_position: int, pages: List[Dict]) -> Dict:
        """Find which page contains a given character position"""
        if not pages:
            return {'page_number': 1, 'start_char': 0, 'end_char': len(text)}
        
        for page in pages:
            if page['start_char'] <= char_position < page['end_char']:
                return page
        
        # Fallback to last page if position is beyond
        return pages[-1]

    def _count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))
    
    def _tokenizer(self, text: str) -> List[int]:
        return self.enc.encode(text)
    
    def _detokenize(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)
    
    def semantic_chunking(self, text: str, metadata: Dict, pages: Optional[List[Dict]] = None) -> List[Dict]:
        """Token-Aware Semantic Chunking with Page Tracking"""
        logger.info(f"Starting Semantic chunking for {metadata.get('filename', 'unknown')}")

        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks, current_tokens, chunk_index = [], [], 0
        current_char_position = 0

        for sentence in sentences:
            sent_tokens = self._tokenizer(sentence)

            if len(current_tokens) + len(sent_tokens) > self.chunk_size:
                # Finalize current chunk with page info
                chunk_content = self._detokenize(current_tokens)
                
                # Find page information for this chunk
                chunk_start_pos = current_char_position - len(chunk_content)
                chunk_end_pos = current_char_position
                
                chunks.append(self._create_chunk(
                    content=chunk_content, 
                    chunk_index=chunk_index, 
                    metadata=metadata,
                    pages=pages,
                    char_start=chunk_start_pos,
                    char_end=chunk_end_pos
                ))
                chunk_index += 1

                # Start new chunk with overlap
                overlap_tokens = current_tokens[-self.overlap:] if self.overlap else []
                current_tokens = overlap_tokens + sent_tokens
            else:
                current_tokens.extend(sent_tokens)
            
            current_char_position += len(sentence) + 1  # +1 for space/punctuation

        # Handle leftover
        if current_tokens:
            chunk_content = self._detokenize(current_tokens)
            chunk_start_pos = current_char_position - len(chunk_content)
            
            chunks.append(self._create_chunk(
                content=chunk_content,
                chunk_index=chunk_index,
                metadata=metadata,
                pages=pages,
                char_start=chunk_start_pos,
                char_end=current_char_position
            ))

        logger.info(f"Created {len(chunks)} semantic chunks (token-aware) with page tracking")
        return chunks
    
    def fixed_size_chunking(self, text: str, metadata: Dict) -> List[Dict]:  
        """Token Aware fixed-sized chunking"""
        logger.info(f"starting fixed-size chunking for {metadata.get('filename', 'unknown')}")

        tokens = self._tokenizer(text)
        chunks, start_idx, chunk_index = [], 0, 0

        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx : end_idx]
            chunk_content = self._detokenize(chunk_tokens)

            chunks.append(self._create_chunk(
                content=chunk_content,
                chunk_index=chunk_index,
                metadata=metadata,
                start_token_idx = start_idx,
                end_token_idx = end_idx
            ))

            start_idx += (self.chunk_size - self.overlap)
            chunk_index += 1

            if end_idx >= len(tokens):
                break

        logger.info(f"Created {len(chunks)} fixed-size chunks (token-aware)")
        return chunks
    
    def _create_chunk(self, content: str, chunk_index: int, metadata: Dict,
                     pages: Optional[List[Dict]] = None,
                     char_start: Optional[int] = None,
                     char_end: Optional[int] = None,
                     start_token_idx: Optional[int] = None,
                     end_token_idx: Optional[int] = None) -> Dict:
        """Create chunk dictionary with page-aware metadata"""
        
        chunk_metadata = {
            **metadata,
            'chunk_index': chunk_index,
            'chunk_token_count': self._count_tokens(content),
            'chunk_char_count': len(content),
            'chunking_strategy': self.strategy,
            'chunk_size_config': self.chunk_size,
            'overlap_config': self.overlap,
            'model_name': self.model_name
        }

        # Add page information if available
        if pages and char_start is not None:
            start_page = self._find_page_for_position(char_start, pages)
            end_page = self._find_page_for_position(char_end or char_start, pages)
            
            chunk_metadata.update({
                'start_page_number': start_page['page_number'],
                'end_page_number': end_page['page_number'],
                'spans_multiple_pages': start_page['page_number'] != end_page['page_number'],
                'char_start_in_document': char_start,
                'char_end_in_document': char_end or char_start + len(content)
            })

        # Add token positions if available
        if start_token_idx is not None:
            chunk_metadata['start_token_index'] = start_token_idx
            chunk_metadata['end_token_index'] = end_token_idx
        
        return {
            'content': content,
            'metadata': chunk_metadata
        }