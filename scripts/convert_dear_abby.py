"""
Convert Dear Abby archive CSV to YAML format for Dear Ethicist.

The raw letters don't have Hohfeldian metadata, so we:
1. Assign CORRELATIVE protocol (most common for advice columns)
2. Extract a signoff from the title
3. Leave expected_state as null (no ground truth - player classifications are captured)
"""

import csv
import re
import yaml
from pathlib import Path


def sanitize_id(title: str, letter_id: str) -> str:
    """Create a valid letter ID from title."""
    # Take first few words of title, lowercase, replace spaces with underscores
    words = re.sub(r'[^a-zA-Z\s]', '', title.lower()).split()[:4]
    base = '_'.join(words) if words else 'letter'
    return f"da_{letter_id}_{base}"


def extract_signoff(title: str) -> str:
    """Extract a signoff from the title."""
    # Clean up the title to use as signoff
    title = title.strip().upper()
    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."
    return title


def parse_parties_from_text(text: str) -> list[dict]:
    """
    Try to extract party information from letter text.
    Returns basic writer + other party structure.
    """
    parties = [
        {
            "name": "You",
            "role": "letter_writer",
            "is_writer": True,
            "expected_state": None
        }
    ]

    # Look for common relationship patterns
    text_lower = text.lower()

    other_party = None
    if 'husband' in text_lower or 'wife' in text_lower:
        other_party = {"name": "Spouse", "role": "spouse"}
    elif 'mother' in text_lower or 'mom' in text_lower:
        other_party = {"name": "Mother", "role": "parent"}
    elif 'father' in text_lower or 'dad' in text_lower:
        other_party = {"name": "Father", "role": "parent"}
    elif 'sister' in text_lower or 'brother' in text_lower:
        other_party = {"name": "Sibling", "role": "sibling"}
    elif 'friend' in text_lower:
        other_party = {"name": "Friend", "role": "friend"}
    elif 'neighbor' in text_lower:
        other_party = {"name": "Neighbor", "role": "neighbor"}
    elif 'boss' in text_lower or 'employer' in text_lower:
        other_party = {"name": "Boss", "role": "employer"}
    elif 'coworker' in text_lower or 'colleague' in text_lower:
        other_party = {"name": "Coworker", "role": "colleague"}
    else:
        other_party = {"name": "Other party", "role": "other"}

    other_party["is_writer"] = False
    other_party["expected_state"] = None
    parties.append(other_party)

    return parties


def detect_protocol(text: str) -> str:
    """
    Detect likely protocol based on letter content.
    Default to CORRELATIVE since most advice questions involve rights/obligations.
    """
    text_lower = text.lower()

    # Gate detection: commitment/promise language
    if any(word in text_lower for word in ['promised', 'committed', 'agreed to', 'said i would']):
        return "GATE_DETECTION"

    # Path dependence: historical framing
    if any(phrase in text_lower for phrase in ['for years', 'always have', 'used to', 'changed']):
        return "PATH_DEPENDENCE"

    # Default to correlative (rights/obligations questions)
    return "CORRELATIVE"


def clean_body(text: str) -> str:
    """Clean and format the letter body."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Add "Dear Ethicist" opener if not present
    if not text.lower().startswith('dear'):
        text = f"Dear Ethicist,\n\n{text}"

    return text


def convert_letter(row: dict) -> dict:
    """Convert a CSV row to letter format."""
    letter_id = sanitize_id(row['title'], row['letterId'])
    body = clean_body(row['question_only'])
    signoff = extract_signoff(row['title'])
    protocol = detect_protocol(body)
    parties = parse_parties_from_text(body)

    return {
        "letter_id": letter_id,
        "protocol": protocol,
        "protocol_params": {"source": "dear_abby", "year": int(float(row['year']))},
        "signoff": signoff,
        "subject": row['title'].strip(),
        "voice": "genuinely_stuck",
        "body": body,
        "parties": parties,
        "questions": []  # Raw letters don't have structured questions
    }


def main():
    csv_path = Path("dear_abby_data/raw_da_qs.csv")
    output_dir = Path("src/dear_ethicist/letters/archive")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    letters = []
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if row.get('question_only') and len(row['question_only'].strip()) > 50:
                    letter = convert_letter(row)
                    letters.append(letter)
            except Exception as e:
                print(f"Skipping row {row.get('letterId', '?')}: {e}")

    print(f"Converted {len(letters)} letters")

    # Split into chunks of 500 letters per file
    chunk_size = 500
    for i in range(0, len(letters), chunk_size):
        chunk = letters[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        output_path = output_dir / f"dear_abby_{chunk_num:02d}.yaml"

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(chunk, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"Wrote {len(chunk)} letters to {output_path}")

    print(f"\nTotal: {len(letters)} letters in {(len(letters) + chunk_size - 1) // chunk_size} files")


if __name__ == "__main__":
    main()
