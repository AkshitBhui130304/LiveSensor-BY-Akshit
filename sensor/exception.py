import sys
import os

def error_message_detail(error, error_detail:sys):
        _,_,exc_tb =  error_detail.exc_info() 
        filename = exc_tb.tb_frame.f_code.co_filename
        error_message = f"Error occurred in script: {filename} at line number: {exc_tb.tb_lineno} with error message: {str(error)}"
        return error_message
        

class SensorException(Exception):
    def __init__(self, message, err_detail:sys):
        super().__init__(message)
        
        self.message = error_message_detail(message, error_detail =err_detail)

    def __str__(self):
        return self.message    