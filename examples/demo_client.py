#!/usr/bin/env python3
"""
Professional CLI Client for Two-Tier Document Parser.
Demonstrates how to interact with the API using a rich interface.
"""
import argparse
import json
import time
import requests
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table

console = Console()

FAST_API_URL = "http://localhost:8004/parse"
ACCURATE_API_URL = "http://localhost:8005/parse"

def parse_file(file_path: Path, mode: str, timeout: int = 600):
    """Parse a PDF file using the specified mode."""
    url = FAST_API_URL if mode == "fast" else ACCURATE_API_URL
    
    console.print(f"[bold blue]🚀 Starting {mode.upper()} parsing job...[/bold blue]")
    console.print(f"📂 File: [cyan]{file_path}[/cyan]")
    console.print(f"🔗 Endpoint: [dim]{url}[/dim]")

    start_time = time.time()
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                progress.add_task(f"Processing ({mode})...", total=None)

                # Send request
                response = requests.post(url, files=files, timeout=timeout)
                
                if response.status_code != 200:
                    progress.stop()
                    console.print(f"[bold red]❌ Error {response.status_code}:[/bold red] {response.text}")
                    return

                result = response.json()
                
    except requests.exceptions.Timeout:
        console.print("[bold red]⏰ Request timed out![/bold red] Increase timeout for large files.")
        return
    except Exception as e:
        console.print(f"[bold red]❌ Connection failed:[/bold red] {str(e)}")
        return

    elapsed = time.time() - start_time
    
    # Display Summary
    console.print("\n[bold green]✅ Parsing Complete![/bold green]")
    
    # Metadata Table
    table = Table(title="Parsing Results Metadata", show_header=False)
    table.add_row("Time Taken", f"{elapsed:.2f}s")
    table.add_row("Pages", str(result["metadata"].get("pages", "N/A")))
    table.add_row("Parser", result["metadata"].get("parser", "N/A"))
    
    if mode == "accurate":
        table.add_row("Images Extracted", str(len(result.get("images", []))))
        table.add_row("Tables Extracted", str(len(result.get("tables", []))))
        table.add_row("Formulas Extracted", str(len(result.get("formulas", []))))
    
    console.print(table)

    # Show Snippet
    markdown_content = result.get("markdown", "")
    snippet = markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content
    
    console.print(Panel(
        Syntax(snippet, "markdown", theme="monokai", line_numbers=True),
        title="Markdown Preview",
        border_style="blue"
    ))
    
    # Save output
    output_file = file_path.with_suffix(f".{mode}.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    console.print(f"[dim]💾 Full result saved to: {output_file}[/dim]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-Tier Parser CLI Client")
    parser.add_argument("file", type=Path, help="Path to PDF file")
    parser.add_argument("--mode", choices=["fast", "accurate"], default="fast", help="Parsing mode")
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    if not args.file.exists():
        console.print(f"[bold red]File not found: {args.file}[/bold red]")
        exit(1)
        
    parse_file(args.file, args.mode, args.timeout)

