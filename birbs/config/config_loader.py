"""
Config loader module
"""

# Default imports
import os
import json
import logging

config_logger = logging.getLogger("ConfigLoader")


class ConfigLoader:
    """
    This class is responsible for loading the configuration from a JSON file.
    """

    def __init__(self) -> None:
        # Empty initialization
        self.yacy_settings = None
        self.flask_settings = None
        self.socket_settings = None

        config_logger.info("ConfigLoader initialized, loading configuration...")

        try:
            # Load the configuration
            self.config = self.load_config("config.json")
            self.initialize_settings()

            # Check if the configuration is valid
            if (
                self.yacy_settings is None
                or self.flask_settings is None
                or self.socket_settings is None
            ):
                raise ValueError("Invalid configuration file")

        except Exception as e:
            config_logger.error(
                "An error occurred while loading the configuration: %s", e
            )
            raise e

    def load_config(self, config_file_path: str) -> dict:
        """
        This function loads the configuration from a JSON file.
        """

        # Check if the file exists
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        # Load the configuration
        with open(config_file_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        return config

    def initialize_settings(self):
        """
        This function initializes the settings from the loaded configuration.
        """
        self.yacy_settings = self.config["yacy"]
        self.flask_settings = self.config["flask"]
        self.socket_settings = self.config["socket"]
