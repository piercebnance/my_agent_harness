# My Own Agent Harness

Building my own personal agent harness

## Setup

From the repository root, install the dependencies and the project into the
active Python environment:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Then start the agent with:

```powershell
python -m harness.cli
```

The editable install makes the `harness` package available regardless of the
current working directory. Use the same Python environment for installation
and for running the agent.
