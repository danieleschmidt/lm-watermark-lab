"""Main CLI entry point."""

import click
from rich.console import Console
from rich.table import Table

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


if __name__ == "__main__":
    main()