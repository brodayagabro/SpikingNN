import jsonschema


class ConfigurationError(Exception):
    pass


SCHEMA = {
    "type": "object",
    "properties": {
        "network": {
            "type": "object",
            "properties": {
                "neuron_count": {"type": "integer", "minimum": 1},
                "neuron_types": {"type": "array", "items": {"type": "string"}},
                "connectivity": {"type": "array"},
                "weights": {"type": "array"},
                "tau_syn": {"type": "array"}
            },
            "required": ["neuron_count"]
        },
        "simulation": {
            "type": "object",
            "properties": {
                "dt": {"type": "number", "minimum": 0.001},
                "duration": {"type": "number", "minimum": 0}
            },
            "required": ["dt", "duration"]
        }
    },
    "required": ["network", "simulation"]
}


def validate_config(config: dict) -> None:
    try:
        jsonschema.validate(config, SCHEMA)
    except jsonschema.ValidationError as e:
        raise ConfigurationError(f"Invalid configuration: {e.message}")
