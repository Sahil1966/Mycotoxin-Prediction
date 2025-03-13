import sys
import traceback
from src.utils.logger import get_logger

logger = get_logger("ExceptionHandler")

class CustomException(Exception):
    """
    Custom Exception class to handle errors consistently in the pipeline.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = CustomException.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):
        """
        Extracts detailed error information including filename and line number.
        """
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        detailed_message = f"Error in script: [{file_name}] at line [{line_number}] - {error_message}"
        return detailed_message

    def __str__(self):
        return self.error_message

# Example usage of logging exception
def test_exception_handling():
    try:
        1 / 0  # Intentional error (division by zero)
    except Exception as e:
        logger.error(CustomException(str(e), sys))
