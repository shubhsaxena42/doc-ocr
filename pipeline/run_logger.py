
import logging
import os
from datetime import datetime

class RunLogger:
    def __init__(self):
        self.logger = None
        self.run_id = None
    
    def start_new_run(self):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f"pipeline_run_{self.run_id}.log")
        
        # Configure logging
        self.logger = logging.getLogger(f"Run_{self.run_id}")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        self.logger.addHandler(fh)
        self.logger.info(f"Started new pipeline run: {self.run_id}")
        
    def log_ocr_extraction(self, field_name, engine, value, confidence):
        if self.logger:
            self.logger.info(f"OCR_EXTRACTION | Field: {field_name} | Engine: {engine} | Value: {value} | Conf: {confidence:.4f}")

    def log_consensus_check(self, field_name, has_conflict, value1, value2):
        if self.logger:
            status = "CONFLICT" if has_conflict else "AGREEMENT"
            self.logger.info(f"CONSENSUS_CHECK | Field: {field_name} | Status: {status} | V1: {value1} | V2: {value2}")

    def log_adjudication_start(self, field_name, tier):
        if self.logger:
             self.logger.info(f"ADJUDICATION_START | Field: {field_name} | Tier: {tier}")

    def log_adjudication_result(self, field_name, tier, method, value, confidence):
        if self.logger:
             self.logger.info(f"ADJUDICATION_RESULT | Field: {field_name} | Tier: {tier} | Method: {method} | Value: {value} | Conf: {confidence:.4f}")

    def log_final_result(self, field_name, value, confidence, source):
        if self.logger:
             self.logger.info(f"FINAL_RESULT | Field: {field_name} | Value: {value} | Conf: {confidence:.4f} | Source: {source}")

# Global instance
run_logger = RunLogger()
