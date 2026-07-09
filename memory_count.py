import psutil
import os
import threading
import time
class FunctionPerformanceMonitor:
    def __init__(self, sampling_interval=0.001):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.process = psutil.Process(os.getpid())

    def _monitor_loop(self):
        while self.monitoring:
            try:
                cpu_percent = self.process.cpu_percent(interval=None)
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = self.process.memory_percent()

                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append({
                    'mb': memory_mb,
                    'percent': memory_percent
                })

                time.sleep(self.sampling_interval)
            except Exception:
                pass

    def monitor_function(self, func, *args, **kwargs):
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self.monitoring = True

        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()

        try:
            result = func(*args, **kwargs)
        finally:
            self.monitoring = False
            monitor_thread.join()

        max_cpu = max(self.cpu_samples) if self.cpu_samples else 0
        max_mem_mb = max(m['mb'] for m in self.memory_samples) if self.memory_samples else 0
        max_mem_percent = max(m['percent'] for m in self.memory_samples) if self.memory_samples else 0

        return {
            'result': result,
            'max_cpu_percent': max_cpu,
            'max_memory_mb': max_mem_mb,
            'max_memory_percent': max_mem_percent
        }

