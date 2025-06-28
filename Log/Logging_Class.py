#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##
# @brief    [TaskLogger] Logging Class for controlling our command and log file in Autonomous System Laboratory
# @brief    [ClientLogger] Logging Class for controlling our command and log file in Autonomous System Laboratory
# @author   Hyuk Jun Yoo (yoohj9475@kist.re.kr)   
# @version  1_2   
# TEST 2021-09-28
# TEST 2022-04-11

import os
import time
import logging

class Logger(object):
    """
    [Logger] Logging Class for controlling our command and log file to upgrade

    # Variable
    :param user_name (str) : use your initial of first name. 
        ex) Hyuk Jun --> HJ
    
    # function
    - _setLoggingLevel(level)
    - _get_ClientLogger(total_path)
    - debug(debug_msg="debug!")
    - info(info_msg="info!")
    - warning(warning_msg="warning!")
    - error(error_msg="error!")

    """   
    # def __init__(self, platform_name):
        
    #     dir_name=time.strftime("%Y%m%d")
    #     time_str=time.strftime("%Y%m%d")
    #     TOTAL_LOG_FOLDER = "{}/{}/{}".format("Log", platform_name, dir_name)

    #     if os.path.isdir(TOTAL_LOG_FOLDER) == False:
    #         os.makedirs(TOTAL_LOG_FOLDER)

    #     self.__TOTAL_LOG_FILE = os.path.join(TOTAL_LOG_FOLDER, "{}.log".format(time_str))

    #     self.myLogger = logging.getLogger(platform_name)
    #     self._setLoggingLevel("INFO")
    #     formatter_string = '%(asctime)s - %(name)s::%(levelname)s -- %(message)s'
    #     self._setFileHandler(formatter_string, total_path=self.__TOTAL_LOG_FILE)
    #     self._setStreamHandler(formatter_string)

    def getLogFilePath(self):
        """:return: self.__TOTAL_LOG_FILE"""
        return self.__TOTAL_LOG_FILE

    def _setLoggingLevel(self,level="INFO"):
        """
        Set Logging Level

        :param level (str) : "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        :return: None
        """
        try:
            if level == "DEBUG":
                self.myLogger.setLevel(logging.DEBUG)
            elif level == "INFO":
                self.myLogger.setLevel(logging.INFO)
            elif level == "WARNING":
                self.myLogger.setLevel(logging.WARNING)
            elif level == "ERROR":
                self.myLogger.setLevel(logging.ERROR)
            elif level == "CRITICAL":
                self.myLogger.setLevel(logging.CRITICAL)
            else:
                raise Exception
        except Exception as e:
            self.info("[Basic ClientLogger] : setLevel is incorrect word!")

    def _setStreamHandler(self, formatter_string):
        """
        Sets up the ClientLogger object for logging messages
        
        :param formatter_string (str) : logging.Formatter(formatter_string)
        :param total_path (str) : "/home/sdl-pc/catkin_ws/src/doosan-robot/Log/{$present_time}" + a
        :return: None
        """
        logging.basicConfig(format=formatter_string)

    def _setFileHandler(self, formatter_string, total_path):
        """
        Sets up the ClientLogger object for logging messages

        formatter_string
        :param total_path (str) : "/home/sdl-pc/catkin_ws/src/doosan-robot/Log/{$present_time}" + a
        :return: None
        """
        formatter=logging.Formatter(formatter_string)
        total_file_handler = logging.FileHandler(filename=total_path)
        total_file_handler.setFormatter(formatter)
        self.myLogger.addHandler(total_file_handler)

    def debug(self, user_name, debug_msg="debug!"):
        """
        write infomration log message in total.log with debug message and show command

        :param debug_msg (str) : Message to log
        :return: True
        """

        msg = "[{}] : {}".format(user_name, debug_msg)
        self.myLogger.debug(msg)

        return msg

    def info(self, user_name, info_msg="info!"):
        """
        write infomration log message in total.log with info message and show command

        :param info_msg (str) : Message to log
        :return: msg
        """

        msg = "[{}] : {}".format(user_name, info_msg)
        self.myLogger.info(msg)

        return msg

    def warning(self, user_name, warning_msg="warning!"):
        """
        write warning log message in total.log with warning log and show command

        :param warning_msg (str) : Message to log
        :return: msg
        """

        msg = "[{}] : {}".format(user_name, warning_msg)
        self.myLogger.warning(msg)
        
        return msg

    def error(self, user_name, error_msg="error!"):
        """
        write error log message in error.log and show command

        :param error_msg (str) : Message to log
        :return: msg
        """

        msg = "[{}] : {}".format(user_name, error_msg)
        self.myLogger.error(msg)

        return msg


class ModelLogger(Logger):
    """
    [JobLogger] Logging Class for controlling our job and log file to upgrade

    # Variable
    :param user_name (str) : use your initial of first name. 
        ex) Hyuk Jun --> HJ

    """   
    def __init__(self, platform_name, log_level, TOTAL_LOG_FOLDER):

        Logger.__init__(self)
        
        dir_name=time.strftime("%Y%m%d")
        filename=time.strftime("%Y%m%d_%H%M%S")

        self.__TOTAL_LOG_FILE = os.path.join(TOTAL_LOG_FOLDER, "{}.log".format(filename))

        self.myLogger = logging.getLogger(platform_name)
        self._setLoggingLevel(log_level)
        formatter_string = '%(asctime)s - %(name)s::%(levelname)s -- %(message)s'
        self._setFileHandler(formatter_string, total_path=self.__TOTAL_LOG_FILE)
        self._setStreamHandler(formatter_string)
