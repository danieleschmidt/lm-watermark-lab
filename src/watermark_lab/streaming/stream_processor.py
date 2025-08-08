"""High-performance streaming processor for real-time watermark detection and analysis."""

import time
import asyncio
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from collections import deque
import json

try:
    import asyncio
    ASYNCIO_AVAILABLE = True
except ImportError:
    ASYNCIO_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import kafka
    from kafka import KafkaProducer, KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.exceptions import StreamingError, ProcessingError
from ..utils.metrics import MetricsCollector
from ..core.factory import WatermarkFactory
from ..core.detector import WatermarkDetector

logger = get_logger("streaming.processor")


@dataclass
class StreamConfig:
    """Configuration for stream processing."""
    
    # Processing settings
    batch_size: int = 100
    max_concurrent_batches: int = 10
    processing_timeout: float = 30.0
    max_queue_size: int = 10000
    
    # Input/Output
    input_sources: List[str] = None
    output_targets: List[str] = None
    
    # Watermark detection
    detection_methods: List[str] = None
    detection_threshold: float = 0.8
    
    # Performance
    enable_async: bool = True
    worker_threads: int = 4
    
    # Monitoring
    metrics_interval: int = 60
    log_frequency: int = 1000
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    dead_letter_queue: bool = True
    
    def __post_init__(self):
        """Initialize default values."""
        if self.input_sources is None:
            self.input_sources = ["queue"]
        if self.output_targets is None:
            self.output_targets = ["stdout"]
        if self.detection_methods is None:
            self.detection_methods = ["kirchenbauer"]


@dataclass
class StreamMessage:
    """Message in the stream processing pipeline."""
    
    id: str
    content: str
    metadata: Dict[str, Any] = None
    timestamp: float = None
    source: str = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamMessage":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ProcessingResult:
    """Result of stream message processing."""
    
    message_id: str
    success: bool
    detection_results: List[Dict[str, Any]] = None
    processing_time: float = 0.0
    error_message: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.detection_results is None:
            self.detection_results = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class StreamSource(ABC):
    """Abstract base class for stream input sources."""
    
    @abstractmethod
    async def read_messages(self) -> AsyncGenerator[StreamMessage, None]:
        """Read messages from the stream source."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the stream source."""
        pass


class StreamSink(ABC):
    """Abstract base class for stream output sinks."""
    
    @abstractmethod
    async def write_result(self, result: ProcessingResult):
        """Write processing result to the sink."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the stream sink."""
        pass


class QueueSource(StreamSource):
    """Queue-based stream source."""
    
    def __init__(self, queue_obj: queue.Queue):
        self.queue = queue_obj
        self.running = True
    
    async def read_messages(self) -> AsyncGenerator[StreamMessage, None]:
        """Read messages from queue."""
        
        while self.running:
            try:
                # Use asyncio to make queue access non-blocking
                message_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.queue.get, True, 0.1  # 100ms timeout
                )
                
                if isinstance(message_data, dict):
                    message = StreamMessage.from_dict(message_data)
                else:
                    message = StreamMessage(
                        id=f"msg_{int(time.time() * 1000)}",
                        content=str(message_data)
                    )
                
                yield message
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Queue source error: {e}")
                await asyncio.sleep(1.0)
    
    async def close(self):
        """Close queue source."""
        self.running = False


class WebSocketSource(StreamSource):
    """WebSocket stream source."""
    
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None
        
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library required for WebSocket source")
    
    async def read_messages(self) -> AsyncGenerator[StreamMessage, None]:
        """Read messages from WebSocket."""
        
        try:
            async with websockets.connect(self.uri) as websocket:
                self.websocket = websocket
                
                async for message_data in websocket:
                    try:
                        data = json.loads(message_data)
                        message = StreamMessage.from_dict(data)
                        yield message
                    except json.JSONDecodeError:
                        # Treat as plain text
                        message = StreamMessage(
                            id=f"ws_{int(time.time() * 1000)}",
                            content=message_data,
                            source="websocket"
                        )
                        yield message
                        
        except Exception as e:
            logger.error(f"WebSocket source error: {e}")
            raise StreamingError(f"WebSocket connection failed: {e}")
    
    async def close(self):
        """Close WebSocket connection."""
        if self.websocket:
            await self.websocket.close()


class ConsoleSource(StreamSource):
    """Console input stream source for testing."""
    
    def __init__(self):
        self.running = True
    
    async def read_messages(self) -> AsyncGenerator[StreamMessage, None]:
        """Read messages from console input."""
        
        message_id = 0
        
        while self.running:
            try:
                # Simulate console input
                await asyncio.sleep(5.0)  # Wait 5 seconds between messages
                
                sample_texts = [
                    "This is a test message for watermark detection.",
                    "Analyzing streaming text data in real-time processing.",
                    "Machine learning models can detect statistical watermarks.",
                    "Natural language processing enables content analysis."
                ]
                
                content = sample_texts[message_id % len(sample_texts)]
                message = StreamMessage(
                    id=f"console_{message_id}",
                    content=content,
                    source="console"
                )
                
                message_id += 1
                yield message
                
            except Exception as e:
                logger.error(f"Console source error: {e}")
                await asyncio.sleep(1.0)
    
    async def close(self):
        """Close console source."""
        self.running = False


class ConsoleSink(StreamSink):
    """Console output stream sink."""
    
    async def write_result(self, result: ProcessingResult):
        """Write result to console."""
        
        print(f"[{time.strftime('%H:%M:%S')}] Message {result.message_id}")
        print(f"  Success: {result.success}")
        print(f"  Processing time: {result.processing_time:.3f}s")
        
        if result.detection_results:
            for detection in result.detection_results:
                method = detection.get('method', 'unknown')
                confidence = detection.get('confidence', 0.0)
                is_watermarked = detection.get('is_watermarked', False)
                print(f"  {method}: {'WATERMARKED' if is_watermarked else 'CLEAN'} (confidence: {confidence:.3f})")
        
        if result.error_message:
            print(f"  Error: {result.error_message}")
        
        print()
    
    async def close(self):
        """Close console sink."""
        pass


class FileSink(StreamSink):
    """File-based stream sink."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.file_handle = None
    
    async def write_result(self, result: ProcessingResult):
        """Write result to file."""
        
        if not self.file_handle:
            self.file_handle = open(self.filename, 'a', encoding='utf-8')
        
        result_line = json.dumps(result.to_dict()) + '\n'
        self.file_handle.write(result_line)
        self.file_handle.flush()
    
    async def close(self):
        """Close file sink."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None


class StreamProcessor:
    """High-performance stream processor for real-time watermark detection."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.logger = get_logger("stream_processor")
        self.metrics = MetricsCollector()
        
        # Processing state
        self.running = False
        self.processing_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.result_queue = asyncio.Queue(maxsize=config.max_queue_size)
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'messages_failed': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'throughput_per_second': 0.0,
            'start_time': None
        }
        
        # Sources and sinks
        self.sources: List[StreamSource] = []
        self.sinks: List[StreamSink] = []
        
        # Watermark detectors
        self.detectors: Dict[str, WatermarkDetector] = {}
        self._initialize_detectors()
        
        # Background tasks
        self.background_tasks = []
    
    def _initialize_detectors(self):
        """Initialize watermark detectors."""
        
        for method in self.config.detection_methods:
            try:
                watermarker = WatermarkFactory.create(method)
                detector = WatermarkDetector(watermarker.get_config())
                self.detectors[method] = detector
                self.logger.info(f"Initialized detector: {method}")
            except Exception as e:
                self.logger.error(f"Failed to initialize detector {method}: {e}")
    
    def add_source(self, source: StreamSource):
        """Add input stream source."""
        self.sources.append(source)
        self.logger.info(f"Added stream source: {type(source).__name__}")
    
    def add_sink(self, sink: StreamSink):
        """Add output stream sink."""
        self.sinks.append(sink)
        self.logger.info(f"Added stream sink: {type(sink).__name__}")
    
    def add_queue_source(self, queue_obj: queue.Queue):
        """Add queue-based source."""
        self.add_source(QueueSource(queue_obj))
    
    def add_websocket_source(self, uri: str):
        """Add WebSocket source."""
        if WEBSOCKETS_AVAILABLE:
            self.add_source(WebSocketSource(uri))
        else:
            self.logger.error("WebSocket source not available")
    
    def add_console_source(self):
        """Add console source for testing."""
        self.add_source(ConsoleSource())
    
    def add_console_sink(self):
        """Add console sink."""
        self.add_sink(ConsoleSink())
    
    def add_file_sink(self, filename: str):
        """Add file sink."""
        self.add_sink(FileSink(filename))
    
    async def start_processing(self):
        """Start the stream processing pipeline."""
        
        if self.running:
            self.logger.warning("Stream processor already running")
            return
        
        if not self.sources:
            self.logger.error("No stream sources configured")
            raise StreamingError("No stream sources configured")
        
        if not self.sinks:
            self.logger.warning("No stream sinks configured, adding console sink")
            self.add_console_sink()
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        self.logger.info("Starting stream processing pipeline")
        
        try:
            # Start input readers
            for source in self.sources:
                task = asyncio.create_task(self._read_from_source(source))
                self.background_tasks.append(task)
            
            # Start message processors
            for i in range(self.config.worker_threads):
                task = asyncio.create_task(self._process_messages())
                self.background_tasks.append(task)
            
            # Start result writers
            task = asyncio.create_task(self._write_results())
            self.background_tasks.append(task)
            
            # Start metrics collector
            task = asyncio.create_task(self._collect_metrics())
            self.background_tasks.append(task)
            
            # Wait for completion
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Stream processing failed: {e}")
            raise StreamingError(f"Stream processing failed: {e}")
        finally:
            await self.stop_processing()
    
    async def stop_processing(self):
        """Stop the stream processing pipeline."""
        
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping stream processing pipeline")
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to finish
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close sources and sinks
        for source in self.sources:
            await source.close()
        
        for sink in self.sinks:
            await sink.close()
        
        self.logger.info("Stream processing stopped")
    
    async def _read_from_source(self, source: StreamSource):
        """Read messages from a stream source."""
        
        try:
            async for message in source.read_messages():
                if not self.running:
                    break
                
                try:
                    await self.processing_queue.put(message)
                except asyncio.QueueFull:
                    self.logger.warning("Processing queue full, dropping message")
                    self.stats['messages_failed'] += 1
                    
        except Exception as e:
            self.logger.error(f"Source reader error: {e}")
    
    async def _process_messages(self):
        """Process messages from the processing queue."""
        
        while self.running:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=1.0
                )
                
                # Process the message
                result = await self._process_single_message(message)
                
                # Put result in output queue
                try:
                    await self.result_queue.put(result)
                except asyncio.QueueFull:
                    self.logger.warning("Result queue full, dropping result")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
    
    async def _process_single_message(self, message: StreamMessage) -> ProcessingResult:
        """Process a single message."""
        
        start_time = time.time()
        
        try:
            detection_results = []
            
            # Run watermark detection with all configured methods
            for method_name, detector in self.detectors.items():
                try:
                    detection = detector.detect(message.content)
                    
                    result = {
                        'method': method_name,
                        'is_watermarked': detection.is_watermarked,
                        'confidence': detection.confidence,
                        'p_value': getattr(detection, 'p_value', None),
                        'processing_time': time.time() - start_time
                    }
                    
                    detection_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Detection failed for {method_name}: {e}")
                    detection_results.append({
                        'method': method_name,
                        'error': str(e),
                        'is_watermarked': False,
                        'confidence': 0.0
                    })
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['messages_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['avg_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['messages_processed']
            )
            
            return ProcessingResult(
                message_id=message.id,
                success=True,
                detection_results=detection_results,
                processing_time=processing_time,
                metadata={
                    'message_timestamp': message.timestamp,
                    'message_source': message.source
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['messages_failed'] += 1
            
            self.logger.error(f"Message processing failed: {e}")
            
            return ProcessingResult(
                message_id=message.id,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    async def _write_results(self):
        """Write results to configured sinks."""
        
        while self.running:
            try:
                # Get result with timeout
                result = await asyncio.wait_for(
                    self.result_queue.get(),
                    timeout=1.0
                )
                
                # Write to all sinks
                for sink in self.sinks:
                    try:
                        await sink.write_result(result)
                    except Exception as e:
                        self.logger.error(f"Sink write error: {e}")
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Result writer error: {e}")
    
    async def _collect_metrics(self):
        """Collect and log performance metrics."""
        
        while self.running:
            try:
                await asyncio.sleep(self.config.metrics_interval)
                
                # Calculate throughput
                if self.stats['start_time']:
                    elapsed_time = time.time() - self.stats['start_time']
                    self.stats['throughput_per_second'] = (
                        self.stats['messages_processed'] / max(1, elapsed_time)
                    )
                
                # Log metrics
                self.logger.info(
                    f"Stream metrics: "
                    f"processed={self.stats['messages_processed']}, "
                    f"failed={self.stats['messages_failed']}, "
                    f"throughput={self.stats['throughput_per_second']:.2f}/s, "
                    f"avg_time={self.stats['avg_processing_time']:.3f}s"
                )
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        
        current_stats = dict(self.stats)
        current_stats.update({
            'processing_queue_size': self.processing_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'active_sources': len(self.sources),
            'active_sinks': len(self.sinks),
            'configured_detectors': list(self.detectors.keys()),
            'running': self.running
        })
        
        return current_stats


# Convenience functions
async def process_stream_from_queue(
    input_queue: queue.Queue,
    output_file: Optional[str] = None,
    detection_methods: List[str] = None
) -> Dict[str, Any]:
    """Process stream from a queue with simple configuration."""
    
    config = StreamConfig(
        detection_methods=detection_methods or ["kirchenbauer"],
        batch_size=50,
        worker_threads=2
    )
    
    processor = StreamProcessor(config)
    processor.add_queue_source(input_queue)
    
    if output_file:
        processor.add_file_sink(output_file)
    else:
        processor.add_console_sink()
    
    try:
        await processor.start_processing()
    except KeyboardInterrupt:
        await processor.stop_processing()
    
    return processor.get_stats()


# Export main classes
__all__ = [
    "StreamProcessor",
    "StreamConfig",
    "StreamMessage",
    "ProcessingResult",
    "StreamSource",
    "StreamSink",
    "QueueSource",
    "WebSocketSource",
    "ConsoleSource",
    "ConsoleSink",
    "FileSink",
    "process_stream_from_queue"
]