"""
Logging utilities for AutoGEO.
"""
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class AutoGEOLogger:
    """Logger for AutoGEO operations with file and console output."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        task_name: Optional[str] = None,
        log_level: int = logging.INFO
    ):
        """Initialize logger.
        
        Args:
            log_dir: Directory to save log files
            task_name: Name of the task (used in log filename)
            log_level: Logging level (default: INFO)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if task_name:
            log_filename = f"{task_name}_{timestamp}.log"
        else:
            log_filename = f"autogeo_{timestamp}.log"
        
        self.log_file = self.log_dir / log_filename
        self.start_time = datetime.now()
        
        # Setup logger
        self.logger = logging.getLogger(f"AutoGEO_{task_name or 'default'}")
        self.logger.setLevel(log_level)
        self.logger.propagate = False  # Prevent propagation to root logger to avoid duplicate logs
        self.logger.handlers.clear()  # Remove existing handlers
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logging initialized. Log file: {self.log_file}")
        self.logger.info(f"Task started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def log_progress(self, current: int, total: int, item_name: str = "items"):
        """Log progress information.
        
        Args:
            current: Current progress
            total: Total items
            item_name: Name of items being processed
        """
        percentage = (current / total * 100) if total > 0 else 0
        elapsed = datetime.now() - self.start_time
        if current > 0:
            avg_time = elapsed.total_seconds() / current
            remaining = (total - current) * avg_time
            eta_str = f", ETA: {remaining:.0f}s"
        else:
            eta_str = ""
        
        self.logger.info(
            f"Progress: {current}/{total} {item_name} ({percentage:.1f}%)"
            f" | Elapsed: {elapsed.total_seconds():.0f}s{eta_str}"
        )
    
    def log_document(self, question_id: str, original: str, rewritten: str, method: str):
        """Log rewritten document for inspection.
        
        Args:
            question_id: ID of the question
            original: Original document text
            rewritten: Rewritten document text
            method: Rewrite method used
        """
        doc_log_file = self.log_dir / f"rewritten_docs_{self.start_time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        doc_entry = {
            "timestamp": datetime.now().isoformat(),
            "question_id": question_id,
            "method": method,
            "original_length": len(original),
            "rewritten_length": len(rewritten),
            "original_preview": original[:200] + "..." if len(original) > 200 else original,
            "rewritten_preview": rewritten[:200] + "..." if len(rewritten) > 200 else rewritten,
            "original": original,
            "rewritten": rewritten
        }
        
        with open(doc_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(doc_entry, ensure_ascii=False) + '\n')
        
        self.logger.debug(
            f"Document logged: {question_id} | "
            f"Original: {len(original)} chars | "
            f"Rewritten: {len(rewritten)} chars"
        )
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_name: str):
        """Save checkpoint for resuming.
        
        Args:
            checkpoint_data: Data to save
            checkpoint_name: Name of the checkpoint
        """
        checkpoint_file = self.log_dir / f"checkpoint_{checkpoint_name}.json"
        checkpoint_data["timestamp"] = datetime.now().isoformat()
        checkpoint_data["elapsed_seconds"] = (datetime.now() - self.start_time).total_seconds()
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint for resuming.
        
        Args:
            checkpoint_name: Name of the checkpoint
            
        Returns:
            Checkpoint data if found, None otherwise
        """
        checkpoint_file = self.log_dir / f"checkpoint_{checkpoint_name}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            self.logger.info(f"Checkpoint loaded: {checkpoint_file}")
            return checkpoint_data
        return None
    
    def get_elapsed_time(self) -> str:
        """Get elapsed time since start.
        
        Returns:
            Formatted elapsed time string
        """
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def close(self):
        """Close logger and log final summary."""
        elapsed = datetime.now() - self.start_time
        self.logger.info(f"Task completed. Total time: {self.get_elapsed_time()}")
        self.logger.info(f"Log file saved to: {self.log_file}")


def get_logger(
    log_dir: str = "logs",
    task_name: Optional[str] = None,
    log_level: int = logging.INFO
) -> AutoGEOLogger:
    """Get or create a logger instance.
    
    Args:
        log_dir: Directory to save log files
        task_name: Name of the task
        log_level: Logging level
        
    Returns:
        AutoGEOLogger instance
    """
    return AutoGEOLogger(log_dir=log_dir, task_name=task_name, log_level=log_level)

