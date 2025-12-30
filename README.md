# trainer_agent

A lightweight multi-agent system for trainee task management using A2A and MCP.

This repository provides a Python-based agent framework and a minimal MCP server used by a Curator and Coordinator agent. It includes a small FastAPI-based stack and integrations with LLM providers (OpenAI, Google GenAI) as specified in pyproject.toml.

Table of Contents

- [About](#about)
- [Features](#features)
- [Built with](#built-with)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running](#running)
- [Example usage](#example-usage)
- [Project structure](#project-structure)
- [Tests](#tests)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)
- [Contact](#contact)

## About

trainer_agent is designed to coordinate agents that create and manage learning tasks for trainees. It demonstrates an MCP server component, curator and coordinator agents, and integrations with external LLMs. The project is intended as a developer-focused starter framework; adapt components and configs to your needs.

## Features

- Coordinator and Curator agents
- MCP server to expose Curator functionality
- Integration-ready for OpenAI and Google Generative AI
- Async-ready using asyncio

## Built with

- Python (>=3.9)
- FastAPI (for server components)
- asyncio
- Popular ML / LLM packages (see dependencies in pyproject.toml)

## Requirements

- Python 3.9 or later
- pip
- (Optional) CUDA-enabled GPU if using torch with GPU support

## Installation

Clone the repository:

```bash
git clone https://github.com/jajos12/trainer_ai_agent.git
cd trainer_ai_agent
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate     # Windows (PowerShell)
```

Install the package and runtime dependencies:

```bash
# Install in editable mode (recommended for development)
pip install -e .
# Install dev/test dependencies
pip install -e .[dev]
```

Note: This project uses a pyproject.toml with hatch build backend — if you prefer to build a wheel or install via pip wheel, use your preferred workflow.

## Configuration

This project uses environment variables (and a .env file) for secrets and runtime config. A .env.example file is present in the repo — copy it to .env and fill in the required values.

Required environment variables (observed in src/main.py):

- OPENAI_API_KEY — API key for OpenAI (or set credentials for other LLM providers used in your configs)

## Running

The repository includes an entrypoint at src/main.py. Run the application with:

```bash
# Ensure .env is configured (see .env.example) and virtualenv is activated
python -m src.main
```

The main script will:

- start a Curator MCP server (mcp_servers.curator_mcp.server.CuratorMCPServer)
- start a CuratorAgent (agents.curator.CuratorAgent)
- create a CoordinatorAgent (agents.coordinator.CoordinatorAgent) which requires OPENAI_API_KEY

You can stop the process with CTRL+C; the script attempts a graceful shutdown.

## Example usage

Once started, main.py demonstrates a sample interaction where the coordinator handles a user request (see src/main.py). Adapt or extend the agents/ and mcp_servers/ packages to build workflows for your training and task-management use cases.

## Project structure

- pyproject.toml            # Project metadata and dependencies
- .env.example              # Example environment variables
- src/
  - main.py                 # Primary application entrypoint
  - agents/                 # Agent implementations (curator, coordinator, ...)
  - mcp_servers/            # MCP server implementations (curator_mcp)

## Tests

Run tests with pytest (dev dependencies):

```bash
pip install -e .[dev]
pytest -q
```

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repo
2. Create a feature branch: git checkout -b feat/your-feature
3. Open a PR with a clear description and tests
4. Run linters and tests locally

## Roadmap

- Add more agent examples and configs
- Add documentation for MCP message formats
- Add CI and code-quality checks

## License

This project is licensed under the MIT License. See LICENSE for full text.

## Contact

Maintained by: jajos12 (https://github.com/jajos12)
