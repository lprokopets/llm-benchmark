"""Interactive CLI for LLM Benchmark Suite."""

import os
import sys
from datetime import datetime
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt

from .prompts import PROMPTS
from .providers import ModelConfig, ProviderConfig, check_health, resolve_api_key
from .tester import run_tests, save_results, TestResult
from .reporter import generate_scorecard, generate_running_scorecard

console = Console()

# ───────────────────── Config Loading ─────────────────────

def get_base_dir() -> Path:
    """Find the llm-benchmark directory."""
    # Check relative to this file
    d = Path(__file__).parent.parent
    if (d / "config.yaml").exists():
        return d
    return Path.cwd()


def load_config(base_dir: Path = None) -> dict:
    """Load config.yaml."""
    base_dir = base_dir or get_base_dir()
    config_path = base_dir / "config.yaml"
    if not config_path.exists():
        console.print("[red]No config.yaml found. Run from llm-benchmark/ directory.[/red]")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_env(base_dir: Path = None):
    """Load .env file."""
    base_dir = base_dir or get_base_dir()
    env_path = base_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def parse_providers(config: dict) -> dict:
    """Parse provider configs from yaml."""
    providers = {}
    for name, pconf in config.get("providers", {}).items():
        providers[name] = ProviderConfig(
            name=name,
            base_url=pconf["base_url"],
            api_key_env=pconf.get("api_key_env"),
            api_key=pconf.get("api_key"),
        )
    return providers


def parse_models(config: dict) -> list:
    """Parse model configs from yaml."""
    models = []
    for m in config.get("models", []):
        models.append(ModelConfig(
            name=m["name"],
            provider=m["provider"],
            model_id=m.get("model_id", m["name"]),
            description=m.get("description", ""),
            type=m.get("type", "cloud"),
            port=m.get("port"),
        ))
    return models


# ───────────────────── Display Helpers ─────────────────────

def show_models(models: list, providers: dict, verbose: bool = False):
    """Display available models in a table."""
    table = Table(title="Available Models", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Name", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Description")
    table.add_column("Status", style="bold")

    for i, m in enumerate(models, 1):
        provider = providers.get(m.provider)
        status = ""

        if m.type == "local":
            if provider:
                healthy = check_health(provider, m)
                status = "[green]ONLINE[/green]" if healthy else "[red]OFFLINE[/red]"
            else:
                status = "[red]NO PROVIDER[/red]"
        else:
            key = resolve_api_key(provider) if provider else None
            if key and key != "not-needed":
                status = "[green]READY[/green]"
            else:
                status = f"[yellow]NO KEY ({provider.api_key_env if provider else '?'})[/yellow]"

        table.add_row(str(i), m.name, m.provider, m.description, status)

    console.print(table)


def show_prompts():
    """Display available test prompts."""
    table = Table(title="Test Prompts")
    table.add_column("#", style="dim", width=3)
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Preview", style="dim", max_width=60)

    for i, p in enumerate(PROMPTS, 1):
        preview = p.text[:80] + "..." if len(p.text) > 80 else p.text
        table.add_row(str(i), p.name, p.description, preview)

    console.print(table)


def show_results_table(results: dict):
    """Display test results in a formatted table."""
    if not results:
        console.print("[yellow]No results to display.[/yellow]")
        return

    prompt_names = [p.name for p in PROMPTS]
    model_names = list(results.keys())

    # Quality table
    table = Table(title="Quality Scores (0-10)")
    table.add_column("Test", style="bold")
    for m in model_names:
        table.add_column(m, justify="center")

    for pname in prompt_names:
        row = [pname]
        for mname in model_names:
            model_results = results.get(mname, [])
            score = next((r.score for r in model_results if r.prompt_name == pname), None)
            if score is None:
                row.append("-")
            elif score >= 9:
                row.append(f"[bold green]{score}[/bold green]")
            elif score <= 3:
                row.append(f"[bold red]{score}[/bold red]")
            else:
                row.append(str(score))
        table.add_row(*row)

    # Total row
    total_row = ["[bold]TOTAL[/bold]"]
    for mname in model_names:
        model_results = results.get(mname, [])
        total = sum(r.score for r in model_results)
        total_row.append(f"[bold]{total}/{len(prompt_names) * 10}[/bold]")
    table.add_row(*total_row)

    console.print(table)

    # Speed table
    speed_table = Table(title="Latency & Speed")
    speed_table.add_column("Model", style="bold")
    speed_table.add_column("Avg tok/s", justify="right")
    speed_table.add_column("Avg Latency", justify="right")
    speed_table.add_column("Output Tokens", justify="right")

    for mname in model_names:
        model_results = results.get(mname, [])
        valid = [r for r in model_results if r.response.error is None and r.response.elapsed > 0]
        if valid:
            avg_tps = sum(r.response.tok_per_sec for r in valid) / len(valid)
            avg_lat = sum(r.response.elapsed for r in valid) / len(valid)
            total_tok = sum(r.response.output_tokens for r in valid)
            speed_table.add_row(mname, f"{avg_tps:.1f}", f"{avg_lat:.1f}s", str(total_tok))
        else:
            speed_table.add_row(mname, "[red]ERROR[/red]", "-", "-")

    console.print(speed_table)


# ───────────────────── CLI Commands ─────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """LLM Benchmark Suite — test local and cloud models."""
    ctx.ensure_object(dict)
    base_dir = get_base_dir()
    load_env(base_dir)
    ctx.obj["base_dir"] = base_dir

    if ctx.invoked_subcommand is None:
        # Default: show interactive menu
        interactive(base_dir)


@cli.command()
@click.pass_context
def models(ctx):
    """List available models and their status."""
    base_dir = ctx.obj.get("base_dir", get_base_dir())
    config = load_config(base_dir)
    providers = parse_providers(config)
    model_list = parse_models(config)
    show_models(model_list, providers)


@cli.command()
@click.pass_context
def prompts(ctx):
    """List available test prompts."""
    show_prompts()


@cli.command()
@click.option("--models", "-m", "model_spec", default=None,
              help="Models to test (comma-separated numbers or 'all')")
@click.option("--prompts", "-p", "prompt_spec", default=None,
              help="Prompts to run (comma-separated names or 'all')")
@click.option("--output", "-o", default=None, help="Output directory for results")
@click.pass_context
def test(ctx, model_spec, prompt_spec, output):
    """Run quality + latency tests."""
    base_dir = ctx.obj.get("base_dir", get_base_dir())
    config = load_config(base_dir)
    providers = parse_providers(config)
    model_list = parse_models(config)
    defaults = config.get("defaults", {})

    # Resolve models
    if model_spec and model_spec != "all":
        indices = [int(x.strip()) - 1 for x in model_spec.split(",")]
        selected_models = [model_list[i] for i in indices if 0 <= i < len(model_list)]
    else:
        selected_models = model_list

    # Resolve prompts
    prompt_names = None
    if prompt_spec and prompt_spec != "all":
        prompt_names = [x.strip() for x in prompt_spec.split(",")]

    if not selected_models:
        console.print("[red]No models selected.[/red]")
        return

    console.print(Panel(
        f"Models: {', '.join(m.name for m in selected_models)}\n"
        f"Prompts: {', '.join(prompt_names or ['all'])}",
        title="Running Tests",
        border_style="cyan",
    ))

    results = run_tests(
        models=selected_models,
        providers=providers,
        prompt_names=prompt_names,
        max_tokens=defaults.get("max_tokens", 4096),
        temperature=defaults.get("temperature", 0.3),
        timeout=defaults.get("timeout", 300),
        base_dir=str(base_dir),
    )

    if not results:
        console.print("[red]No results collected.[/red]")
        return

    # Display results
    show_results_table(results)

    # Save
    results_dir = output or str(base_dir / "results")
    run_dir = save_results(results, results_dir)

    # Generate scorecard
    model_descs = {m.name: m.description for m in selected_models}
    scorecard = generate_scorecard(results, model_descs)
    scorecard = scorecard.replace("$DATE", datetime.now().strftime("%Y-%m-%d"))

    scorecard_path = run_dir / "scorecard.md"
    with open(scorecard_path, "w") as f:
        f.write(scorecard)

    console.print(f"\n[green]Results saved to: {run_dir}[/green]")
    console.print(f"[green]Scorecard: {scorecard_path}[/green]")


@cli.command()
@click.argument("path", required=False)
@click.pass_context
def results(ctx, path):
    """View results from a previous run."""
    base_dir = ctx.obj.get("base_dir", get_base_dir())
    results_dir = base_dir / "results"

    if path:
        scorecard = Path(path) / "scorecard.md"
    else:
        # Find latest
        quality_dirs = sorted(results_dir.glob("quality_*"))
        if not quality_dirs:
            console.print("[yellow]No results found.[/yellow]")
            return
        scorecard = quality_dirs[-1] / "scorecard.md"

    if not scorecard.exists():
        console.print(f"[red]No scorecard at: {scorecard}[/red]")
        return

    # Read and display (strip markdown formatting for console)
    content = scorecard.read_text()
    # Show as plain text in console, or use rich markdown
    console.print(Panel(content, title=str(scorecard.parent.name), border_style="green"))


@cli.command()
@click.pass_context
def history(ctx):
    """Show running scorecard across all results to date."""
    base_dir = ctx.obj.get("base_dir", get_base_dir())
    results_dir = str(base_dir / "results")
    scorecard = generate_running_scorecard(results_dir)
    console.print(Panel(scorecard, title="All-Time Scorecard", border_style="magenta"))


# ───────────────────── Interactive Mode ─────────────────────

def interactive(base_dir: Path):
    """Interactive menu-driven mode."""
    config = load_config(base_dir)
    providers = parse_providers(config)
    model_list = parse_models(config)

    console.print(Panel(
        "[bold cyan]LLM Benchmark Suite[/bold cyan]\n"
        f"Config: {base_dir / 'config.yaml'}",
        border_style="cyan",
    ))

    while True:
        console.print("\n[bold]Commands:[/bold]")
        console.print("  [cyan]models[/cyan]   — list models & status")
        console.print("  [cyan]prompts[/cyan]   — list test prompts")
        console.print("  [cyan]test[/cyan]      — run tests (interactive selection)")
        console.print("  [cyan]results[/cyan]   — view last scorecard")
        console.print("  [cyan]history[/cyan]   — running scorecard (all results to date)")
        console.print("  [cyan]quit[/cyan]      — exit")
        console.print()

        cmd = Prompt.ask("[bold]>", default="test")

        if cmd in ("quit", "q", "exit"):
            break
        elif cmd == "models":
            show_models(model_list, providers)
        elif cmd == "prompts":
            show_prompts()
        elif cmd == "results":
            results_dir = base_dir / "results"
            quality_dirs = sorted(results_dir.glob("quality_*"))
            if not quality_dirs:
                console.print("[yellow]No results found. Run tests first.[/yellow]")
            else:
                scorecard = quality_dirs[-1] / "scorecard.md"
                if scorecard.exists():
                    content = scorecard.read_text()
                    console.print(Panel(content, title=str(scorecard.parent.name), border_style="green"))
                else:
                    console.print(f"[yellow]No scorecard in latest run.[/yellow]")
        elif cmd == "history":
            scorecard = generate_running_scorecard(str(base_dir / "results"))
            console.print(Panel(scorecard, title="All-Time Scorecard", border_style="magenta"))
        elif cmd == "test":
            show_models(model_list, providers)
            console.print()
            model_input = Prompt.ask(
                "[bold]Select models[/bold] (numbers, comma-separated, or 'all')",
                default="1",
            )

            # Allow bailing out with a top-level command
            COMMANDS = {"models", "prompts", "results", "history", "quit", "q", "exit"}
            if model_input.strip().lower() in COMMANDS:
                # Fall through to next loop iteration — the command will be lost,
                # so just print a hint and restart
                console.print("[dim]Tip: type commands at the main > prompt, not during test selection.[/dim]")
                continue

            if model_input == "all":
                selected = model_list
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in model_input.split(",")]
                    selected = [model_list[i] for i in indices if 0 <= i < len(model_list)]
                except (ValueError, IndexError):
                    console.print("[red]Invalid selection.[/red]")
                    continue

            if not selected:
                console.print("[red]No models selected.[/red]")
                continue

            # Filter to available models
            available = []
            managed_ports = {11435, 8091, 8092, 8093, 11436, 11437, 8094, 8095, 8096, 8097, 8098}
            for m in selected:
                provider = providers.get(m.provider)
                if m.type == "local":
                    if m.port in managed_ports:
                        available.append(m)
                    elif check_health(provider, m):
                        available.append(m)
                    else:
                        console.print(f"[yellow]Skipping {m.name}: server offline[/yellow]")
                else:
                    key = resolve_api_key(provider) if provider else None
                    if key and key != "not-needed":
                        available.append(m)
                    else:
                        console.print(f"[yellow]Skipping {m.name}: no API key[/yellow]")

            if not available:
                console.print("[red]No available models. Check servers and API keys.[/red]")
                continue

            show_prompts()
            prompt_input = Prompt.ask(
                "[bold]Select prompts[/bold] (numbers, comma-separated, or 'all')",
                default="all",
            )

            if prompt_input == "all":
                prompt_names = None
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in prompt_input.split(",")]
                    prompt_names = [PROMPTS[i].name for i in indices if 0 <= i < len(PROMPTS)]
                except (ValueError, IndexError):
                    console.print("[red]Invalid selection.[/red]")
                    continue

            defaults = config.get("defaults", {})
            console.print(Panel(
                f"Models: {', '.join(m.name for m in available)}\n"
                f"Prompts: {', '.join(prompt_names or [p.name for p in PROMPTS])}",
                title="Running Tests",
                border_style="cyan",
            ))

            test_results = run_tests(
                models=available,
                providers=providers,
                prompt_names=prompt_names,
                max_tokens=defaults.get("max_tokens", 4096),
                temperature=defaults.get("temperature", 0.3),
                timeout=defaults.get("timeout", 300),
                base_dir=str(base_dir),
            )

            if test_results:
                show_results_table(test_results)

                run_dir = save_results(test_results, str(base_dir / "results"))
                model_descs = {m.name: m.description for m in available}
                scorecard = generate_scorecard(test_results, model_descs)
                scorecard = scorecard.replace("$DATE", datetime.now().strftime("%Y-%m-%d"))
                scorecard_path = run_dir / "scorecard.md"
                with open(scorecard_path, "w") as f:
                    f.write(scorecard)

                console.print(f"\n[green]Scorecard: {scorecard_path}[/green]")
        else:
            console.print(f"[red]Unknown command: {cmd}[/red]")
