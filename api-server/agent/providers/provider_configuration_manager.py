"""Configuration classes for LLM providers."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMProviderConfig:
    """Configuration for LLM provider API calls.

    This class centralizes all common parameters used across different LLM providers
    to address the "too many parameters" (PLR0913) issue in the provider interface.

    Attributes:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model identifier to use
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        top_p: Nucleus sampling parameter
        stop_sequences: List of sequences that will stop generation
        stream: Whether to stream the response
        timeout: Request timeout in seconds
        extra_params: Additional provider-specific parameters
    """

    messages: list[dict[str, str]]
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[list[str]] = None
    stream: bool = False
    timeout: float = 60.0
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to a dictionary for API requests.

        Returns:
            Dictionary containing all parameters, suitable for API requests
        """
        result = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "stream": self.stream,
        }

        # Include optional parameters only if they are set
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens

        if self.top_p is not None:
            result["top_p"] = self.top_p

        if self.stop_sequences:
            result["stop"] = self.stop_sequences

        # Include any extra provider-specific parameters
        result.update(self.extra_params)

        return result


@dataclass
class OpenAIConfig(LLMProviderConfig):
    """OpenAI-specific configuration.

    Extends the base configuration with parameters specific to OpenAI API.

    Attributes:
        frequency_penalty: Penalty for token frequency (OpenAI specific)
        presence_penalty: Penalty for token presence (OpenAI specific)
        logit_bias: Modify likelihood of specific tokens appearing (OpenAI specific)
        user: End-user identifier for OpenAI usage monitoring
    """

    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to an OpenAI-specific dictionary.

        Returns:
            Dictionary containing all parameters for OpenAI API
        """
        result = super().to_dict()

        # Add OpenAI-specific parameters
        if self.frequency_penalty is not None:
            result["frequency_penalty"] = self.frequency_penalty

        if self.presence_penalty is not None:
            result["presence_penalty"] = self.presence_penalty

        if self.logit_bias:
            result["logit_bias"] = self.logit_bias

        if self.user:
            result["user"] = self.user

        return result


@dataclass
class AnthropicConfig(LLMProviderConfig):
    """Anthropic-specific configuration.

    Extends the base configuration with parameters specific to Anthropic API.

    Attributes:
        system: System prompt for Claude models
        metadata: Additional metadata for Anthropic API
    """

    system: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to an Anthropic-specific dictionary.

        Returns:
            Dictionary containing all parameters for Anthropic API
        """
        result = super().to_dict()

        # Add Anthropic-specific parameters
        if self.system:
            result["system"] = self.system

        if self.metadata:
            result["metadata"] = self.metadata

        # Anthropic uses "stop_sequences" instead of "stop"
        if "stop" in result and self.stop_sequences:
            result["stop_sequences"] = result.pop("stop")

        return result


@dataclass
class OllamaConfig(LLMProviderConfig):
    """Ollama-specific configuration.

    Extends the base configuration with parameters specific to Ollama API.

    Attributes:
        repeat_penalty: Penalty for repeated tokens (Ollama specific)
        repeat_last_n: Number of tokens to look back for repetitions
        seed: Random seed for deterministic sampling
        num_predict: Number of tokens to predict (similar to max_tokens)
        mirostat: Enable Mirostat sampling algorithm (0, 1, or 2)
        mirostat_tau: Mirostat target entropy
        mirostat_eta: Mirostat learning rate
    """

    repeat_penalty: Optional[float] = None
    repeat_last_n: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to an Ollama-specific dictionary.

        Returns:
            Dictionary containing all parameters for Ollama API
        """
        result = super().to_dict()

        # Ollama uses "options" object for many parameters
        options = {}

        if self.repeat_penalty is not None:
            options["repeat_penalty"] = self.repeat_penalty

        if self.repeat_last_n is not None:
            options["repeat_last_n"] = self.repeat_last_n

        if self.seed is not None:
            options["seed"] = self.seed

        if self.num_predict is not None:
            options["num_predict"] = self.num_predict

        if self.mirostat is not None:
            options["mirostat"] = self.mirostat

        if self.mirostat_tau is not None:
            options["mirostat_tau"] = self.mirostat_tau

        if self.mirostat_eta is not None:
            options["mirostat_eta"] = self.mirostat_eta

        # Add options if any are set
        if options:
            result["options"] = options

        return result


@dataclass
class OpenRouterConfig(LLMProviderConfig):
    """OpenRouter-specific configuration.

    Extends the base configuration with parameters specific to OpenRouter API.

    Attributes:
        transforms: Transformation options for OpenRouter
        route: Routing preferences for OpenRouter
        prompt_format: Format for parsing prompt (e.g., "openai", "anthropic")
    """

    transforms: Optional[list[str]] = None
    route: Optional[str] = None
    prompt_format: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to an OpenRouter-specific dictionary.

        Returns:
            Dictionary containing all parameters for OpenRouter API
        """
        result = super().to_dict()

        # Add OpenRouter-specific parameters
        if self.transforms:
            result["transforms"] = self.transforms

        if self.route:
            result["route"] = self.route

        if self.prompt_format:
            result["prompt_format"] = self.prompt_format

        return result


# Provider Manager System

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from .llm_provider_interface import LLMProvider
from .openai_api_client import OpenAIProvider
from .claude_api_client import AnthropicProvider
from .local_ollama_client import OllamaProvider
from .openrouter_gateway_client import OpenRouterProvider

logger = logging.getLogger(__name__)


class ProviderManager:
    """Manages LLM providers and their configurations"""
    
    def __init__(self, config_file: str = "provider_configs.json"):
        self.config_file = Path(config_file)
        self.providers: Dict[str, LLMProvider] = {}
        self.active_provider: Optional[str] = None
        self.configurations: Dict[str, Dict[str, Any]] = {}
        
        # Initialize available provider classes FIRST
        self.provider_classes = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "ollama": OllamaProvider,
            "openrouter": OpenRouterProvider
        }
        
        # Load configurations if file exists
        if self.config_file.exists():
            self.load_configurations()
    
    def load_configurations(self):
        """Load provider configurations from file"""
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.configurations = data.get("providers", {})
                self.active_provider = data.get("active_provider")
                
                # Initialize configured providers
                for name, config in self.configurations.items():
                    self._initialize_provider(name, config)
                    
        except Exception as e:
            logger.error(f"Failed to load provider configurations: {e}")
    
    def save_configurations(self):
        """Save provider configurations to file"""
        try:
            data = {
                "providers": self.configurations,
                "active_provider": self.active_provider
            }
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save provider configurations: {e}")
    
    def _initialize_provider(self, name: str, config: Dict[str, Any]) -> bool:
        """Initialize a provider with configuration"""
        provider_class = self.provider_classes.get(name)
        if not provider_class:
            logger.error(f"Unknown provider: {name}")
            return False
        
        try:
            # Extract provider-specific parameters
            api_key = config.get("api_key")
            model = config.get("model", config.get("default_model"))
            base_url = config.get("base_url")
            
            # Create provider instance
            if name == "openai":
                provider = provider_class(api_key=api_key)
            elif name == "anthropic":
                provider = provider_class(api_key=api_key)
            elif name == "ollama":
                provider = provider_class(model=model, base_url=base_url)
            elif name == "openrouter":
                provider = provider_class(api_key=api_key, base_url=base_url)
            else:
                provider = provider_class()
            
            self.providers[name] = provider
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize provider {name}: {e}")
            return False
    
    def get_active_provider(self) -> Optional[Dict[str, Any]]:
        """Get the active provider information"""
        if not self.active_provider or self.active_provider not in self.providers:
            return None
        
        return {
            "name": self.active_provider,
            "provider": self.providers[self.active_provider],
            "config": self.configurations.get(self.active_provider, {})
        }
    
    def set_active_provider(self, name: str) -> Dict[str, Any]:
        """Set the active provider"""
        if name not in self.providers:
            return {"success": False, "error": f"Provider {name} not configured"}
        
        self.active_provider = name
        self.save_configurations()
        
        return {"success": True, "provider": name}
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """List all configured providers"""
        providers = []
        for name, provider in self.providers.items():
            providers.append({
                "name": name,
                "active": name == self.active_provider,
                "configured": provider.is_configured() if hasattr(provider, 'is_configured') else True,
                "model": self.configurations.get(name, {}).get("model", "default")
            })
        return providers
    
    def add_provider(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add or update a provider configuration"""
        if name not in self.provider_classes:
            return {"success": False, "error": f"Unknown provider type: {name}"}
        
        # Store configuration
        self.configurations[name] = config
        
        # Initialize provider
        if self._initialize_provider(name, config):
            self.save_configurations()
            return {"success": True, "provider": name}
        else:
            return {"success": False, "error": f"Failed to initialize provider {name}"}
    
    def remove_provider(self, name: str) -> Dict[str, Any]:
        """Remove a provider configuration"""
        if name in self.providers:
            del self.providers[name]
        if name in self.configurations:
            del self.configurations[name]
        if self.active_provider == name:
            self.active_provider = None
        
        self.save_configurations()
        return {"success": True}
    
    async def generate(
        self,
        provider_name: Optional[str] = None,
        messages: List[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response using specified or active provider"""
        # Use specified provider or active provider
        if provider_name:
            provider = self.providers.get(provider_name)
            if not provider:
                raise ValueError(f"Provider {provider_name} not configured")
        else:
            active = self.get_active_provider()
            if not active:
                raise ValueError("No active provider configured")
            provider = active["provider"]
        
        # Create config object based on provider type - use the correct interface
        provider_name = provider_name or self.active_provider
        
        # Use MessageConfig which is what the provider expects
        from .llm_provider_interface import MessageConfig
        
        model = kwargs.get("model", self.configurations.get(provider_name, {}).get("model"))
        if not model:
            # Set default models per provider
            model_defaults = {
                "openai": "gpt-3.5-turbo",
                "anthropic": "claude-3-sonnet-20240229", 
                "ollama": "llama2",
                "openrouter": "openai/gpt-3.5-turbo"
            }
            model = model_defaults.get(provider_name, "default")
        
        config = MessageConfig(
            messages=messages,
            model=model,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens"),
            top_p=kwargs.get("top_p"),
            stop_sequences=kwargs.get("stop_sequences"),
            kwargs=kwargs  # Pass any extra provider-specific parameters
        )
        
        # Generate response using the correct provider interface
        response = await provider.send_message(config)
        
        # Add provider info to response
        response["provider"] = provider_name
        if "model" not in response:
            response["model"] = config.model
        
        return response
    
    # Compatibility methods for legacy code
    def get_agent(self):
        """Get active provider (legacy compatibility)"""
        active = self.get_active_provider()
        if active:
            return active["provider"]
        return None


# Global provider manager instance
_provider_manager: Optional[ProviderManager] = None


def get_provider_manager() -> ProviderManager:
    """Get global provider manager instance"""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = ProviderManager()
    return _provider_manager
