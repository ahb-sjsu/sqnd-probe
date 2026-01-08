"""
Command-line interface for Dear Ethicist.

A text-based version of the advice column game.
"""

import os
import random
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from dear_ethicist.letters import LetterBank, create_default_letter_bank, get_response_options
from dear_ethicist.models import (
    GameState,
    Letter,
    Protocol,
    Response,
    TrialRecord,
    Verdict,
)
from dear_ethicist.reactions import generate_engagement_stats, generate_reactions
from dear_ethicist.telemetry import TelemetryLogger

console = Console()


def clear_screen():
    """Clear screen - works reliably on Windows."""
    os.system("cls" if os.name == "nt" else "clear")


@click.group()
@click.version_option()
def cli():
    """Dear Ethicist - An advice column game for measuring moral reasoning."""
    pass


@cli.command()
@click.option(
    "--output-dir", default="./data", type=click.Path(), help="Telemetry output directory"
)
def play(output_dir: str):
    """Start a new game session."""
    clear_screen()
    console.print(
        Panel.fit(
            "[bold]THE MORNING CHRONICLE[/bold]\n\n"
            "Welcome to your first day as the new advice columnist.\n\n"
            "Letters will arrive from readers seeking guidance.\n"
            "Read them carefully. Offer your wisdom.\n\n"
            "[dim]Press Enter to begin...[/dim]",
            title="Dear Ethicist",
            border_style="blue",
        )
    )
    input()

    # Initialize game
    game_state = GameState()
    letter_bank = create_default_letter_bank()
    telemetry = TelemetryLogger(Path(output_dir), game_state.session_id)

    # Main game loop
    try:
        run_game_loop(game_state, letter_bank, telemetry)
    except KeyboardInterrupt:
        console.print("\n[dim]Game saved. See you tomorrow.[/dim]")


def run_game_loop(game_state: GameState, letter_bank: LetterBank, telemetry: TelemetryLogger):
    """Main game loop."""
    # Get all letters and randomize order
    letters_to_process = letter_bank.list_ids()
    random.shuffle(letters_to_process)

    for letter_id in letters_to_process:
        letter = letter_bank.get(letter_id)
        if not letter:
            continue

        clear_screen()
        display_letter(letter, game_state)

        # Get player response
        response = get_player_response(letter, game_state)

        if response:
            # Log telemetry
            record = TrialRecord(
                session_id=game_state.session_id,
                letter_id=letter.letter_id,
                day=game_state.current_day,
                week=game_state.current_week,
                protocol=letter.protocol,
                protocol_params=letter.protocol_params,
                verdicts=response.verdicts,
                career_stage=game_state.career_stage,
            )
            telemetry.log_trial(record)

            # Show reactions
            display_reactions(letter, response)

            game_state.letters_answered += 1

            if not Confirm.ask("\n[dim]Continue to next letter?[/dim]", default=True):
                break

    # End of session
    display_session_end(game_state, telemetry)


def display_letter(letter: Letter, game_state: GameState):
    """Display a letter on screen."""
    # Header
    console.print(
        Panel(f"[bold]THE MORNING CHRONICLE[/bold] — Day {game_state.current_day}", style="blue")
    )

    # Letter content - extract body after "Dear Ethicist," and before signoff
    body_content = letter.body.split("Dear Ethicist,")[-1].strip()
    # Remove signoff if it's already in the body
    signoff_marker = f"— {letter.signoff}"
    if signoff_marker in body_content:
        body_content = body_content.replace(signoff_marker, "").strip()

    letter_text = Text()
    letter_text.append("DEAR ETHICIST,\n\n", style="bold")
    letter_text.append(body_content)
    letter_text.append(f"\n\n— {letter.signoff}", style="italic")

    console.print(
        Panel(
            letter_text,
            title=f"[bold]{letter.subject.upper()}[/bold]",
            border_style="white",
            padding=(1, 2),
        )
    )


def get_player_response(letter: Letter, _game_state: GameState) -> Response | None:
    """Get the player's response to a letter."""
    console.print("\n[bold]YOUR VERDICT[/bold] [dim](for your records)[/dim]\n")

    verdicts = []
    options = get_response_options(letter)

    for option_group in options:
        console.print(f"[cyan]{option_group['category']}:[/cyan]")
        for i, choice in enumerate(option_group["choices"], 1):
            console.print(f"  [{i}] {choice['label']}")

        while True:
            selection = Prompt.ask("Select", choices=["1", "2", "skip"], default="1")
            if selection == "skip":
                break
            idx = int(selection) - 1
            if 0 <= idx < len(option_group["choices"]):
                choice = option_group["choices"][idx]
                # Find expected state for this party
                expected = None
                for party in letter.parties:
                    if party.name == choice["party"]:
                        expected = party.expected_state
                        break

                verdicts.append(
                    Verdict(
                        party_name=choice["party"],
                        state=choice["state"],
                        expected=expected,
                    )
                )
                break

        console.print()

    if not verdicts:
        return None

    # Publish
    console.print("\n" + "═" * 60)
    if Confirm.ask("[bold]PUBLISH TO COLUMN?[/bold]", default=True):
        console.print("\n[green]✓ Published[/green]")
        return Response(
            letter_id=letter.letter_id,
            verdicts=verdicts,
        )

    return None


def display_reactions(letter: Letter, response: Response):
    """Display reader reactions to published response."""
    time.sleep(0.5)
    console.print("\n[bold]READER REACTIONS[/bold]\n")

    reactions = generate_reactions(letter, response, count=3)
    for reaction in reactions:
        style = {
            "supportive": "green",
            "critical": "red",
            "mixed": "yellow",
            "humorous": "cyan",
        }.get(reaction.tone, "white")
        console.print(f"[{style}]{reaction.text}[/{style}]")
        time.sleep(0.3)

    stats = generate_engagement_stats()
    console.print(
        f"\n[dim]Engagement: {stats['comments']} comments, {stats['shares']} shares[/dim]"
    )


def display_session_end(game_state: GameState, telemetry: TelemetryLogger):
    """Display end of session summary."""
    console.print("\n" + "═" * 60)
    console.print(
        Panel(
            f"[bold]END OF SESSION[/bold]\n\n"
            f"Letters answered: {game_state.letters_answered}\n"
            f"Session ID: {game_state.session_id}\n\n"
            f"[dim]Telemetry saved to: {telemetry.filepath}[/dim]",
            border_style="blue",
        )
    )

    stats = telemetry.compute_statistics()
    if stats["total_trials"] > 0:
        console.print(f"\n[dim]Analysis: {stats['total_trials']} trials recorded[/dim]")


@cli.command()
@click.option("--letter-id", required=True, help="Letter ID to display")
def preview(letter_id: str):
    """Preview a specific letter."""
    bank = create_default_letter_bank()
    letter = bank.get(letter_id)

    if not letter:
        console.print(f"[red]Letter not found: {letter_id}[/red]")
        console.print(f"\nAvailable letters: {', '.join(bank.list_ids()[:10])}...")
        return

    game_state = GameState()
    display_letter(letter, game_state)


@cli.command()
@click.option(
    "--protocol", type=click.Choice([p.value for p in Protocol]), help="Filter by protocol"
)
def list_letters(protocol: str | None):
    """List available letters."""
    bank = create_default_letter_bank()

    table = Table(title="Available Letters")
    table.add_column("ID", style="cyan")
    table.add_column("Protocol", style="green")
    table.add_column("Subject", style="white")

    proto_filter = Protocol(protocol) if protocol else None
    for letter_id in bank.list_ids(proto_filter):
        letter = bank.get(letter_id)
        if letter:
            table.add_row(
                letter.letter_id,
                letter.protocol.value,
                letter.subject[:40] + "..." if len(letter.subject) > 40 else letter.subject,
            )

    console.print(table)
    console.print(f"\nTotal: {bank.count(proto_filter)} letters")


@cli.command()
@click.argument("session_file", type=click.Path(exists=True))
def analyze(session_file: str):
    """Analyze telemetry from a session file."""
    from dear_ethicist.telemetry import compute_bond_index, load_session

    records = load_session(Path(session_file))

    console.print("\n[bold]Session Analysis[/bold]")
    console.print(f"Records: {len(records)}")

    # By protocol
    by_protocol: dict[str, int] = {}
    correct = 0
    total_verdicts = 0

    for record in records:
        proto = record.protocol.value
        by_protocol[proto] = by_protocol.get(proto, 0) + 1

        for verdict in record.verdicts:
            total_verdicts += 1
            if verdict.is_correct:
                correct += 1

    console.print("\nBy Protocol:")
    for proto, count in by_protocol.items():
        console.print(f"  {proto}: {count}")

    if total_verdicts > 0:
        console.print(f"\nAccuracy: {correct}/{total_verdicts} ({100*correct/total_verdicts:.1f}%)")

    bond_index = compute_bond_index(records)
    console.print(f"Bond Index: {bond_index:.3f}")


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
