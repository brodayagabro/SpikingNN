"""
JSON Schema validation for SpikingNN configuration files.

This module provides validation for network and simulation configurations
using JSON Schema. It ensures that configuration files meet the required
format and constraints before being used by the simulation engine.
"""

import jsonschema


class ConfigurationError(Exception):
    """
    Raised when a configuration file fails validation.
    
    Attributes:
        message: Human-readable error message
        path: JSON path to the invalid field
        schema_path: Path in the schema that failed
    """
    def __init__(self, message: str, path: tuple = None, schema_path: tuple = None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.schema_path = schema_path


class ValidationError(Exception):
    """
    Raised when signal parameters are invalid.
    
    Attributes:
        message: Human-readable error message
        field: The invalid field name
    """
    def __init__(self, message: str, field: str = None):
        super().__init__(message)
        self.message = message
        self.field = field


class SimulationError(Exception):
    """
    Raised when a simulation encounters a runtime error.
    
    Attributes:
        message: Human-readable error message
        timestep: The timestep where the error occurred (if known)
    """
    def __init__(self, message: str, timestep: int = None):
        super().__init__(message)
        self.message = message
        self.timestep = timestep


SCHEMA = {
    "type": "object",
    "properties": {
        "network": {
            "type": "object",
            "properties": {
                "neuron_count": {"type": "integer", "minimum": 1},
                "neuron_types": {"type": "array", "items": {"type": "string"}},
                "connectivity": {"type": "array", "items": {"type": "array"}},
                "weights": {"type": "array", "items": {"type": "array"}},
                "tau_syn": {"type": "array", "items": {"type": "array"}}
            },
            "required": ["neuron_count"],
            "additionalProperties": False
        },
        "simulation": {
            "type": "object",
            "properties": {
                "dt": {"type": "number", "minimum": 0.001},
                "duration": {"type": "number", "exclusiveMinimum": 0},
                "input_current": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["constant", "sine", "square", "ramp", "noise"]},
                        "amplitude": {"type": "number"},
                        "frequency": {"type": "number", "minimum": 0},
                        "neurons": {"type": "array", "items": {"type": "integer", "minimum": 0}}
                    },
                    "required": ["type", "amplitude"]
                }
            },
            "required": ["dt", "duration"],
            "additionalProperties": False
        }
    },
    "required": ["network", "simulation"],
    "additionalProperties": False
}


def validate_config(config: dict) -> None:
    """
    Validate a configuration dictionary against the SpikingNN schema.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    try:
        jsonschema.validate(config, SCHEMA)
    except jsonschema.ValidationError as e:
        raise ConfigurationError(
            message=f"Invalid configuration: {e.message}",
            path=e.absolute_path,
            schema_path=e.absolute_schema_path
        )