import logging
import sys
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from threading import Lock
from config import Config


class MemoryHandler(logging.Handler):
    """Thread-safe custom logging handler that stores logs in memory"""
    
    def __init__(self, max_logs: int = 1000):
        super().__init__()
        self.logs = []
        self.max_logs = max_logs
        self._lock = Lock()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record and store it in memory"""
        try:
            with self._lock:
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': self.format(record),
                    'module': getattr(record, 'module', 'unknown'),
                    'function': getattr(record, 'funcName', 'unknown'),
                    'line': getattr(record, 'lineno', 0),
                    'pathname': getattr(record, 'pathname', 'unknown')
                }
                
                # Add exception info if present
                if record.exc_info:
                    log_entry['exception'] = ''.join(traceback.format_exception(*record.exc_info))
                
                self.logs.append(log_entry)
                
                # Maintain max logs limit
                if len(self.logs) > self.max_logs:
                    self.logs = self.logs[-self.max_logs:]
                    
        except Exception as e:
            # Avoid infinite recursion in logging
            print(f"Error in MemoryHandler.emit: {e}", file=sys.stderr)
    
    def get_logs(self, level: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get logs with optional filtering"""
        with self._lock:
            logs = self.logs.copy()
        
        # Filter by level if specified
        if level:
            level_upper = level.upper()
            logs = [log for log in logs if log['level'] == level_upper]
        
        # Limit results if specified
        if limit and limit > 0:
            logs = logs[-limit:]
        
        return logs
    
    def clear_logs(self) -> None:
        """Clear all stored logs"""
        with self._lock:
            self.logs.clear()
    
    def get_log_count(self) -> int:
        """Get total number of stored logs"""
        with self._lock:
            return len(self.logs)
    
    def get_logs_by_level(self) -> Dict[str, int]:
        """Get count of logs by level"""
        with self._lock:
            level_counts = {}
            for log in self.logs:
                level = log['level']
                level_counts[level] = level_counts.get(level, 0) + 1
            return level_counts


class LogManager:
    """Centralized logging manager with enhanced error handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_handler = MemoryHandler(max_logs=config.MAX_LOG_ENTRIES)
        self.logger = None
        self._initialized = False
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Setup comprehensive logging configuration"""
        try:
            # Get root logger
            root_logger = logging.getLogger()
            
            # Set log level
            log_level = getattr(logging, self.config.LOG_LEVEL.upper(), logging.INFO)
            root_logger.setLevel(log_level)
            
            # Clear existing handlers to avoid duplicates
            root_logger.handlers.clear()
            
            # Setup console handler with error handling
            console_handler = self._create_console_handler()
            if console_handler:
                root_logger.addHandler(console_handler)
            
            # Setup memory handler
            memory_formatter = logging.Formatter(
                fmt=self.config.LOG_FORMAT,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self.memory_handler.setFormatter(memory_formatter)
            root_logger.addHandler(self.memory_handler)
            
            # Setup file handler if specified
            if self.config.LOG_FILE:
                file_handler = self._create_file_handler()
                if file_handler:
                    root_logger.addHandler(file_handler)
            
            # Create application logger
            self.logger = logging.getLogger(__name__)
            self.logger.info("Logging system initialized successfully")
            self._initialized = True
            
        except Exception as e:
            print(f"Failed to setup logging: {e}", file=sys.stderr)
            self._setup_fallback_logging()
    
    def _create_console_handler(self) -> Optional[logging.StreamHandler]:
        """Create console handler with error handling"""
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            console_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            
            # Test the handler
            test_record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="Test message", args=(), exc_info=None
            )
            console_handler.handle(test_record)
            
            return console_handler
            
        except Exception as e:
            print(f"Failed to create console handler: {e}", file=sys.stderr)
            return None
    
    def _create_file_handler(self) -> Optional[logging.FileHandler]:
        """Create file handler with error handling"""
        try:
            file_handler = logging.FileHandler(self.config.LOG_FILE, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            file_formatter = logging.Formatter(
                fmt=self.config.LOG_FORMAT,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            
            return file_handler
            
        except Exception as e:
            print(f"Failed to create file handler: {e}", file=sys.stderr)
            return None
    
    def _setup_fallback_logging(self) -> None:
        """Setup minimal fallback logging if main setup fails"""
        try:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                stream=sys.stdout
            )
            self.logger = logging.getLogger(__name__)
            self.logger.warning("Using fallback logging configuration")
            self._initialized = True
            
        except Exception as e:
            print(f"Even fallback logging failed: {e}", file=sys.stderr)
            self._initialized = False
    
    def get_logs(self, level: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get logs from memory handler with error handling"""
        try:
            return self.memory_handler.get_logs(level=level, limit=limit)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to retrieve logs: {e}")
            return []
    
    def clear_logs(self) -> bool:
        """Clear logs from memory handler"""
        try:
            self.memory_handler.clear_logs()
            if self.logger:
                self.logger.info("Memory logs cleared")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to clear logs: {e}")
            return False
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get comprehensive log statistics"""
        try:
            return {
                'total_logs': self.memory_handler.get_log_count(),
                'logs_by_level': self.memory_handler.get_logs_by_level(),
                'max_capacity': self.memory_handler.max_logs,
                'initialized': self._initialized,
                'log_level': self.config.LOG_LEVEL,
                'file_logging_enabled': bool(self.config.LOG_FILE)
            }
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get log statistics: {e}")
            return {'error': str(e)}
    
    def is_initialized(self) -> bool:
        """Check if logging system is properly initialized"""
        return self._initialized
    
    def log_exception(self, exception: Exception, context: str = "") -> None:
        """Log an exception with full context"""
        try:
            if self.logger:
                context_msg = f" in {context}" if context else ""
                self.logger.error(
                    f"Exception occurred{context_msg}: {str(exception)}", 
                    exc_info=True
                )
            else:
                print(f"Exception{' in ' + context if context else ''}: {exception}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to log exception: {e}", file=sys.stderr)
    
    def log_performance(self, operation: str, duration: float, details: Optional[Dict] = None) -> None:
        """Log performance metrics"""
        try:
            if self.logger:
                message = f"Performance: {operation} took {duration:.3f}s"
                if details:
                    message += f" - Details: {details}"
                self.logger.info(message)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to log performance: {e}")
    
    def create_child_logger(self, name: str) -> logging.Logger:
        """Create a child logger for specific modules"""
        try:
            return logging.getLogger(name)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to create child logger '{name}': {e}")
            return logging.getLogger()  # Return root logger as fallback