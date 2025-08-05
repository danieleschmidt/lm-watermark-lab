"""Main CLI entry point."""

import click
import json
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def main():
    """LM Watermark Lab CLI - Comprehensive watermarking toolkit."""
    pass


@main.command()
@click.option("--method", default="kirchenbauer", help="Watermark method to use")
@click.option("--model", default="gpt2", help="Model to use for generation")
@click.option("--prompt", required=True, help="Input prompt")
@click.option("--output", help="Output file path")
@click.option("--max-length", default=100, type=int, help="Maximum text length")
def generate(method, model, prompt, output, max_length):
    """Generate watermarked text."""
    console.print(f"[green]Generating watermarked text using {method}[/green]")
    console.print(f"Model: {model}")
    console.print(f"Prompt: {prompt}")
    
    try:
        from ..core.factory import WatermarkFactory
        
        # Create watermark instance
        watermark = WatermarkFactory.create(method)
        result = watermark.generate(prompt, max_length=max_length)
        
        if output:
            with open(output, "w") as f:
                f.write(result)
            console.print(f"[green]Output saved to {output}[/green]")
        else:
            console.print(f"[cyan]{result}[/cyan]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print(f"[yellow]Available methods: {', '.join(WatermarkFactory.list_methods())}[/yellow]")


@main.command()
@click.option("--text", required=True, help="Text to analyze")
@click.option("--method", default="kirchenbauer", help="Detection method to use")
@click.option("--config", help="Watermark configuration file")
def detect(text, method, config):
    """Detect watermark in text."""
    console.print("[green]Analyzing text for watermarks...[/green]")
    
    try:
        from ..core.detector import WatermarkDetector
        
        # Create detector
        detector_config = {"method": method}
        if config:
            import json
            with open(config, 'r') as f:
                detector_config.update(json.load(f))
        
        detector = WatermarkDetector(detector_config)
        result = detector.detect(text)
        
        table = Table(title="Detection Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        status = "Yes" if result.is_watermarked else "No"
        table.add_row("Watermark Detected", status)
        table.add_row("Confidence", f"{result.confidence:.2%}")
        table.add_row("P-value", f"{result.p_value:.6f}")
        table.add_row("Test Statistic", f"{result.test_statistic:.3f}")
        table.add_row("Method", result.method)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error during detection: {e}[/red]")


@main.command()
def methods():
    """List available watermark methods."""
    console.print("[green]Available watermark methods:[/green]")
    
    methods_list = [
        ("kirchenbauer", "Kirchenbauer et al. watermarking"),
        ("markllm", "MarkLLM toolkit integration"),
        ("aaronson", "Aaronson cryptographic approach"),
        ("zhao", "Zhao et al. robust watermarking"),
    ]
    
    table = Table(title="Watermark Methods")
    table.add_column("Method", style="cyan")
    table.add_column("Description", style="white")
    
    for method, description in methods_list:
        table.add_row(method, description)
    
    console.print(table)


@main.command()
@click.option("--methods", multiple=True, help="Methods to compare")
@click.option("--prompts", default=5, type=int, help="Number of test prompts")
@click.option("--output", help="Output file for results")
def benchmark(methods, prompts, output):
    """Run benchmark comparison of watermark methods."""
    console.print("[green]Running watermark benchmark...[/green]")
    
    try:
        from ..core.benchmark import WatermarkBenchmark
        
        if not methods:
            methods = ["kirchenbauer", "markllm", "aaronson"]
        
        benchmark_suite = WatermarkBenchmark(num_samples=prompts)
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Running benchmark...", total=None)
            
            results = benchmark_suite.compare(
                list(methods), 
                benchmark_suite.test_prompts[:prompts],
                ["detectability", "quality", "robustness"]
            )
        
        # Display results
        table = Table(title="Benchmark Results")
        table.add_column("Method", style="cyan")
        table.add_column("Detectability", style="green")
        table.add_column("Quality", style="yellow")
        table.add_column("Robustness", style="red")
        
        for method, metrics in results.items():
            table.add_row(
                method,
                f"{metrics.get('detectability', 0):.3f}",
                f"{metrics.get('quality', 0):.3f}",
                f"{metrics.get('robustness', 0):.3f}"
            )
        
        console.print(table)
        
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Results saved to {output}[/green]")
            
    except Exception as e:
        console.print(f"[red]Benchmark error: {e}[/red]")


@main.command()
@click.option("--text", required=True, help="Text to attack")
@click.option("--attack", default="paraphrase", help="Attack type")
@click.option("--strength", default="medium", help="Attack strength")
def attack(text, attack, strength):
    """Test attack resistance of watermarked text."""
    console.print(f"[green]Running {attack} attack with {strength} strength...[/green]")
    
    try:
        from ..core.attacks import AttackSimulator
        
        simulator = AttackSimulator()
        result = simulator.run_attack(text, attack, strength=strength)
        
        console.print(Panel.fit(
            f"[cyan]Original:[/cyan] {result.original_text[:100]}...\n\n"
            f"[red]Attacked:[/red] {result.attacked_text[:100]}...",
            title="Attack Results"
        ))
        
        table = Table(title="Attack Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Attack Type", result.attack_type)
        table.add_row("Success", "Yes" if result.success else "No")
        table.add_row("Quality Score", f"{result.quality_score:.3f}")
        table.add_row("Similarity Score", f"{result.similarity_score:.3f}")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Attack error: {e}[/red]")


@main.command()
@click.option("--config-file", help="Configuration file path")
def config(config_file):
    """Manage configuration settings."""
    if config_file:
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            console.print(Panel(json.dumps(config_data, indent=2), title="Configuration"))
        else:
            console.print(f"[red]Configuration file not found: {config_file}[/red]")
    else:
        # Show default configuration
        default_config = {
            "watermark": {
                "default_method": "kirchenbauer",
                "kirchenbauer": {"gamma": 0.25, "delta": 2.0},
                "markllm": {"algorithm": "KGW", "strength": 2.0},
                "aaronson": {"threshold": 0.5},
                "zhao": {"message_bits": "101010", "redundancy": 3}
            },
            "detection": {
                "threshold": 0.05,
                "confidence_level": 0.95
            },
            "attacks": {
                "default_strength": "medium",
                "available": ["paraphrase", "truncation", "insertion", "substitution"]
            }
        }
        console.print(Panel(json.dumps(default_config, indent=2), title="Default Configuration"))


if __name__ == "__main__":
    main()