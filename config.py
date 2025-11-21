"""Configuration for HITL test case."""

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

# Test Configuration
TEST_THREAD_ID_PREFIX = "test_hitl_"
INTERRUPT_WAIT_SECONDS = 2.0  # Reduced for demo (set to 60.0 for full test)

# Agent Configuration
AGENT_TEMPERATURE = 0.7


