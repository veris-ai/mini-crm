# Mini CRM Lead Qualifier

Demo CRM agent wired for Veris simulations. 

## Quickstart

Prerequisites: Python ≥3.11, OpenAI key, Veris key, ngrok (authtoken configured)

### 1) Clone and change dir
```bash
git clone https://github.com/veris-ai/mini-crm.git
cd mini-crm
```

### 2) Download dependencies
```bash
uv sync && source .venv/bin/activate
```

### 3) Setup env variables (replace placeholders with your secrets)
Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:
```bash
OPENAI_API_KEY=<your-openai-key>
VERIS_API_KEY=<your-veris-api-key>
```

### 4) Run veris setup (keep this terminal open)
> **⚠️ Make sure ngrok is configured on your CLI!**
>
> You must have an [ngrok](https://ngrok.com/) account and your authtoken set up locally.  
> To verify, run:
> ```bash
> ngrok config check
> ```
> To get started you may follow the [ngrok getting started guide](https://ngrok.com/docs/getting-started/) (steps 1 and 2 are sufficient).

```bash
veris setup-local start --local-url http://localhost:8000
```

The setup command launches the FastAPI server and makes it publicly accessible via ngrok. The simulator command watches and runs scenarios.

### 5) Publish an agent card & launch simulator
Open a new terminal, then:

```bash
cd mini-crm
source .venv/bin/activate
```

Create a new agent card:
```bash
veris sim create --agent-path agent_card.json --agent-version v1.0.0
```

Launch the simulation:
```bash
veris sim v3launch --watch 
```

Watch the simulation run, you can check the status by going to the correspoding .veris/runs/RUN_ID file prompted in the terminal.

Give it 3-4 minutes to run, if scenarios are not yet finished, you can kill the simulation by running:
```bash
veris sim kill <simulation_id>
```

### 7) Conclusion
To learn more explore the [`veris-ai/veris-cli`](https://github.com/veris-ai/veris-cli/) docs.

## Local Checks (optional)

```bash
uvicorn app.main:app --reload
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Qualify Acme Capital."}'
```

Direct runs mutate `db/leads.json`; simulator runs capture the same tool calls without touching disk.

## Key Files

- `app/agent.py` – agent definition and tools
- `app/main.py` – FastAPI surface instrumented by Veris
- `db/leads.json` – sample TinyDB store
- `.veris/` – generated simulator config after `veris init`

Need more depth? See TROUBLESHOOTING.md
