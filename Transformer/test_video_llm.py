"""
================================================================================
 test_video_llm.py  --  Interactive tester for the Video Idea LLM
================================================================================

This script loads a pre-trained video idea generation model and lets you
test it interactively without having to retrain.

Usage:
    1. First train the model:     python video_idea_llm.py
    2. Then test it anytime:      python test_video_llm.py

Commands in interactive mode:
    - Type any prompt to generate a video idea (e.g., "Coding a", "How I built")
    - 'temp X'     - set temperature (e.g., 'temp 1.2' for more creativity)
    - 'tokens X'   - set max tokens (e.g., 'tokens 15')
    - 'batch N'    - generate N ideas with current prompt (e.g., 'batch 5')
    - 'demo'       - show example prompts with different temperatures
    - 'stats'      - show model information
    - 'help'       - show this help
    - 'quit'/'exit'- exit the program

================================================================================
"""

import os
import sys
from video_idea_llm import load_model, generate_video_idea


def print_help():
    """Display help information."""
    print()
    print("  COMMANDS:")
    print("    <prompt>       Generate a video idea (e.g., 'Coding a', 'How I built')")
    print("    temp X         Set temperature (0.1-2.0, default 0.9)")
    print("    tokens X       Set max tokens (1-20, default 12)")
    print("    batch N        Generate N ideas with the last prompt")
    print("    demo           Show example prompts")
    print("    stats          Show model information")
    print("    help           Show this help")
    print("    quit/exit      Exit the program")
    print()


def show_demo(model, temperature, max_tokens):
    """Show some demo generations."""
    print()
    print("  DEMO PROMPTS:")
    print("  " + "-" * 66)

    demo_prompts = [
        "Coding a",
        "How I built",
        "Can you beat",
        "I learned",
        "Explaining",
        "Building a",
        "I survived",
    ]

    for prompt in demo_prompts:
        idea = generate_video_idea(model, prompt, max_tokens=max_tokens,
                                  temperature=temperature)
        print(f"    {prompt:18s} -> {idea}")
    print()


def show_stats(model):
    """Display model statistics."""
    cfg = model["config"]
    vocab_size = len(model["stoi"])

    print()
    print("  MODEL STATISTICS:")
    print("  " + "-" * 66)
    print(f"    Vocabulary size:  {vocab_size} tokens")
    print(f"    Model dimension:  {cfg['d_model']}")
    print(f"    Feed-forward dim: {cfg['d_ff']}")
    print(f"    Context window:   {cfg['block_size']} tokens")
    print(f"    Training epochs:  {cfg.get('n_epochs', 'N/A')}")

    # Show some vocabulary samples
    words = [model["itos"][i] for i in range(4, min(14, len(model["itos"])))]
    print(f"    Sample vocab:     {', '.join(words)}...")
    print()


def batch_generate(model, prompt, num_ideas, temperature, max_tokens):
    """Generate multiple ideas with the same prompt."""
    print()
    print(f"  Generating {num_ideas} ideas for '{prompt}':")
    print("  " + "-" * 66)

    for i in range(num_ideas):
        idea = generate_video_idea(model, prompt, max_tokens=max_tokens,
                                  temperature=temperature)
        print(f"    {i+1:2d}. {idea}")
    print()


def main():
    """Main interactive loop."""
    # Find the model file
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "saved_models", "video_idea_model.pkl")

    if not os.path.exists(model_path):
        print()
        print("=" * 70)
        print("  ERROR: No trained model found!")
        print("=" * 70)
        print()
        print("  Please train the model first by running:")
        print("    python video_idea_llm.py")
        print()
        print(f"  Expected model location: {model_path}")
        print()
        sys.exit(1)

    # Load the model
    print()
    print("=" * 70)
    print("  VIDEO IDEA LLM - Interactive Tester")
    print("=" * 70)
    print()
    model = load_model(model_path)

    # Settings
    temperature = 0.90
    max_tokens = 12
    last_prompt = ""

    print()
    print("  Current settings:")
    print(f"    Temperature: {temperature:.2f} (lower=safe, higher=creative)")
    print(f"    Max tokens:  {max_tokens}")
    print()
    print("  Type 'help' for commands, or enter a prompt to start!")
    print("=" * 70)
    print()

    while True:
        try:
            user_input = input("  > ").strip()

            if not user_input:
                continue

            # Quit command
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("\n  Goodbye!\n")
                break

            # Help command
            if user_input.lower() == 'help':
                print_help()
                continue

            # Stats command
            if user_input.lower() == 'stats':
                show_stats(model)
                continue

            # Demo command
            if user_input.lower() == 'demo':
                show_demo(model, temperature, max_tokens)
                continue

            # Temperature command
            if user_input.lower().startswith('temp '):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"  Temperature set to {temperature:.2f}\n")
                    else:
                        print("  Temperature should be between 0.1 and 2.0\n")
                except (ValueError, IndexError):
                    print("  Usage: temp 0.9\n")
                continue

            # Max tokens command
            if user_input.lower().startswith('tokens '):
                try:
                    new_tokens = int(user_input.split()[1])
                    if 1 <= new_tokens <= 20:
                        max_tokens = new_tokens
                        print(f"  Max tokens set to {max_tokens}\n")
                    else:
                        print("  Max tokens should be between 1 and 20\n")
                except (ValueError, IndexError):
                    print("  Usage: tokens 12\n")
                continue

            # Batch generation command
            if user_input.lower().startswith('batch '):
                try:
                    num_ideas = int(user_input.split()[1])
                    if not last_prompt:
                        print("  Please enter a prompt first before using batch\n")
                        continue
                    if 1 <= num_ideas <= 20:
                        batch_generate(model, last_prompt, num_ideas,
                                     temperature, max_tokens)
                    else:
                        print("  Number of ideas should be between 1 and 20\n")
                except (ValueError, IndexError):
                    print("  Usage: batch 5\n")
                continue

            # Generate video idea
            last_prompt = user_input
            print(f"  [temp={temperature:.2f}] ", end="")
            idea = generate_video_idea(model, user_input,
                                      max_tokens=max_tokens,
                                      temperature=temperature)
            print(f"{idea}")
            print()

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Goodbye!\n")
            break
        except EOFError:
            print("\n\n  Goodbye!\n")
            break
        except Exception as e:
            print(f"\n  Error: {e}\n")


if __name__ == "__main__":
    main()
