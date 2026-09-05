<div align="center">

# Maxwell Mensah

### Production: AI, Agents & ML

</div>

**$$ \color{#2563eb}{\textsf{Agentic Systems Architecture $\cdot$ Model Engineering $\cdot$ LLMOps}} $$**

<div align="center">

I train models, build agentic systems, and ship the evaluation
and infrastructure <br>layers that keep AI reliable in production.

📍 Ghana, West Africa &nbsp;&nbsp;·&nbsp;&nbsp; 🌍 Worldwide

</div>

<hr />

<div align="center">

<table>
  <tr>
    <td align="center"><a href="#agents"><code>[01] AGENTIC SYSTEMS</code></a><br/><sub>LangGraph · Strands · MCP</sub></td>
    <td align="center"><a href="#finetuning"><code>[02] MODEL ENGINEERING</code></a><br/><sub>LoRA · GGUF · Unsloth</sub></td>
    <td align="center"><a href="#evaluation"><code>[03] EVALUATION</code></a><br/><sub>LLM-as-Judge · RAG Evals</sub></td>
    <td align="center"><a href="#production"><code>[04] PRODUCTION</code></a><br/><sub>GCP · Docker · Observability</sub></td>
  </tr>
</table>

**Engineering stance:** Agents fail in production differently than in demos.   
I build for failure first. Evaluation, observability, and guardrails before features.

</div>

<hr />

<a id="agents"></a>
<details>
<summary><b>[01] AGENTIC SYSTEMS</b> &nbsp;—&nbsp; LangGraph · Strands · CrewAI · MCP</summary>
<br/>

**→ [`agentic_engineering`](https://github.com/MaxwellMensah/agentic_engineering)**

Production-grade AI agent design patterns, autonomous system architectures, and multi-framework orchestration.

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#eff6ff',
    'primaryTextColor': '#1f2937',
    'primaryBorderColor': '#3b82f6',
    'lineColor': '#2563eb',
    'secondaryColor': '#dbeafe',
    'tertiaryColor': '#ffffff'
  }
}}%%
graph TD
    A["<b>REACT CORE</b><br/><br/>LangGraph<br/>State Graph<br/>Tool Schema<br/>react_agent"]
    B["<b>MULTI-AGENT</b><br/><br/>Strands<br/>CrewAI<br/>Swarm · Graph<br/>Task Delegation"]
    C["<b>CONTEXT LAYER</b><br/><br/>Input Guardrails<br/>Model Routing<br/>Role-Based Policy<br/>Production Telemetry"]
    D["<b>[Agentic RAG + Vector Search]</b><br/>Weaviate · Graph Query Engine"]

    A --- B
    B --- C
    B --> D

    style A fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1f2937
    style B fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1f2937
    style C fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1f2937
    style D fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#1f2937
```

* **Orchestration**: LangGraph graph-based state machines, Strands multi-agent swarms, CrewAI task pipelines
* **Context Engineering**: Input guardrails, dynamic model routing, role-based tool policy, production telemetry
* **Architecture Analysis**: Comparative breakdown — Strands vs CrewAI across task delegation, memory, and tool-use APIs

</details>

<hr />

<a id="finetuning"></a>
<details>
<summary><b>[02] MODEL ENGINEERING</b> &nbsp;—&nbsp; LoRA · GGUF · Unsloth · PyTorch · XGBoost</summary>
<br/>

**→ [`fine_tuning_modeling`](https://github.com/MaxwellMensah/fine_tuning_modeling)**

End-to-end model engineering across two paradigms — LLM domain adaptation and classical ML research pipelines.

**▸ LLM Domain Adaptation**

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#dcfce7',
    'primaryTextColor': '#1f2937',
    'primaryBorderColor': '#22c55e',
    'lineColor': '#16a34a',
    'secondaryColor': '#bbf7d0',
    'tertiaryColor': '#ffffff'
  }
}}%%
graph TD
    D1[(Dataset JSONL <br/> train · val)] --> D2[SFT Training <br/> Unsloth · QLoRA]
    D2 --> D3([Checkpoint-350])
    D3 --> D4[Merge Weights <br/> export_model.py]
    D3 --> D5[8-bit Quantize <br/> transform_gguf.py]
    D4 --> D6[(Fraud Model v7)] --> D7[push_to_huggingface.py] --> D8([HuggingFace Hub])
    D5 --> D9[(GGUF Binary)] --> D10[Ollama Engine]
    D5 --> D11[Edge Case Testing]
    D5 --> D12[Benchmark Run]

    style D1 fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#1f2937
    style D3 fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#1f2937
    style D6 fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#1f2937
    style D8 fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#1f2937
    style D9 fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#1f2937
```

**▸ Traditional ML Research & Training**

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#ffedd5',
    'primaryTextColor': '#1f2937',
    'primaryBorderColor': '#f97316',
    'lineColor': '#fb923c',
    'secondaryColor': '#fed7aa',
    'tertiaryColor': '#ffffff'
  }
}}%%
graph LR
    M1[(Raw Data)] --> M2[EDA & Features] --> M3[Model Training <br/> XGBoost · sklearn] --> M4[Validation & Metrics <br/> AUC · F1 · CV] --> M5[Serialization <br/> joblib · ONNX] --> M6[Deployment <br/> FastAPI · Docker]
    M3 --> MW[Weights & Biases <br/> Loss · Scores · Versions]

    style M1 fill:#ffedd5,stroke:#f97316,stroke-width:2px,color:#1f2937
    style M6 fill:#ffedd5,stroke:#f97316,stroke-width:2px,color:#1f2937
    style MW fill:#fff7ed,stroke:#fb923c,stroke-width:2px,color:#1f2937
```

* **LLM Fine-Tuning**: PEFT via QLoRA/LoRA using Unsloth — dataset curation, SFT, checkpoint management
* **Export**: 16-bit merged weights + 8-bit GGUF quantization for local inference via Ollama
* **Classical ML**: End-to-end research pipelines — feature engineering, model selection, cross-validation, serialization
* **Evaluation**: Edge case testing suite and base vs. fine-tuned model comparison on domain-specific metrics

</details>

<hr />

<a id="evaluation"></a>
<details>
<summary><b>[03] EVALUATION HARNESS</b> &nbsp;—&nbsp; LLM-as-Judge · RAG Evals · Plotly</summary>
<br/>

**→ [`LLM-evaluations`](https://github.com/MaxwellMensah/LLM-evaluations)**

Local multi-dimensional evaluation framework — the bridge between raw model output and business logic reliability.

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#fef3c7',
    'primaryTextColor': '#1f2937',
    'primaryBorderColor': '#d97706',
    'lineColor': '#f59e0b',
    'secondaryColor': '#fde68a',
    'tertiaryColor': '#ffffff'
  }
}}%%
graph LR
    E1[Base Model] --> E3[Eval Framework <br/> Pydantic Schema]
    E2[Fine-Tuned Model] --> E3
    E3 --> E4[LLM-as-Judge <br/> temp=0] --> E5([Plotly Dashboard <br/> Score Distribution])

    style E5 fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#1f2937
```

* **Metrics**: LLM-as-a-Judge faithfulness scoring, embedding-based relevancy, hallucination reduction
* **Reliability**: Deterministic grading with Pydantic schema enforcement and `temperature=0` judge controls
* **Observability**: Interactive Plotly dashboards for score distribution and regression monitoring

</details>

<hr />

<a id="production"></a>
<details>
<summary><b>[04] PRODUCTION ENGINEERING</b> &nbsp;—&nbsp; GCP · Docker · Kubernetes · Observability</summary>
<br/>

**→ [`production_engineering`](https://github.com/MaxwellMensah/production_engineering)**

LLMOps infrastructure, containerized AI microservices, and agent observability pipelines.

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#fee2e2',
    'primaryTextColor': '#1f2937',
    'primaryBorderColor': '#ef4444',
    'lineColor': '#f87171',
    'secondaryColor': '#fecaca',
    'tertiaryColor': '#ffffff'
  }
}}%%
graph LR
    P1[FastAPI Service] --> P2[Docker Container] --> P3[GCP Cloud Run] --> P4([Observability Pipeline <br/> Traces · Metrics])

    style P4 fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#1f2937
```

* **Infrastructure**: GCP Cloud Run autoscaling, Docker containerization, Kubernetes pod management
* **Reliability**: Cost/latency optimization, circuit breakers, graceful degradation strategies
* **Observability**: Distributed tracing, eval harnesses, agent monitoring pipelines

</details>

<hr />

## TECHNICAL STACK

| Layer | Technologies |
| --- | --- |
| **Agentic Frameworks** | LangGraph · Langsmith · Strands · CrewAI · MCP |
| **ML Stack** | PyTorch · Scikit-learn · XGBoost · Pandas · NumPy · Matplotlib · Weights & Biases |
| **Model Engineering** | Unsloth · HuggingFace · LoRA/QLoRA · GGUF · Ollama |
| **LLM APIs** | Gemini · Claude · OpenAI · Bedrock |
| **Production & Infra** | GCP · Docker · Kubernetes · Cloud Run · FastAPI |
| **Evaluation** | LLM-as-Judge · RAG Evals · Plotly · Pydantic |

<hr />

<div align="center">
<sub>Committing daily · Building in public</sub>
</div>
