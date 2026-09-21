# SPDX-License-Identifier: Apache-2.0
# Supply distinct statement and request cases for native model admission checks.
# Inputs: case and variant indices. Outputs: exact source and expected pairs. Raises on invalid indices.
CASE_COUNT = 23


def source_case(index, variant=0):
    if not 0 <= index < CASE_COUNT or variant < 0:
        raise ValueError("invalid source case")
    name = f"Person{variant}" if variant else "Mira Vale"
    other = f"Partner{variant}" if variant else "Jonah Reed"
    place = f" in room {variant}" if variant else ""
    sentence = "statement"
    request = "request"
    cases = [
        [(f"{name} will cook lentils tonight.", sentence), (f"Their peer is {other}.", sentence)],
        [(f"Correction: {name} will not cook lentils tonight.", sentence), ("The earlier cooking plan changed.", sentence)],
        [(f"{name} may bake bread tomorrow.", sentence), ("This is uncertain.", sentence)],
        [(f"Will {name} cook lentils tonight?", request)],
        [(f"Console {variant} keeps its active state in device memory.", sentence), ("Its disk file is a delayed copy.", sentence)],
        [(f"{name} is a researcher.", sentence), (f"Their colleague is {other}.", sentence)],
        [(f"Please state the plan{place}.", request), ("Do not add a memory item.", request)],
        [(f"The door in room {variant} may be closed.", sentence), (f"The window in room {variant} is open.", sentence)],
        [(f"Should the fire door{place} remain shut?", request)],
        [(f"If rain starts, the loading yard{place} may flood.", sentence), ("If readings differ, report the gap.", request)],
        [(f"I would like you to list the spare parts{place}.", request), ("I want to read tonight.", sentence)],
        [(f"The team asked me to count parts{place} yesterday.", sentence), ("Count the parts now.", request)],
        [(f"Find the inspection record{place}.", request), ("Use three bullets.", request), ("About fifty words, please.", request)],
        [(f"Memory{place} uses a disk file.", sentence), ("Use the disk file.", request)],
        [(f"The guard must lock the gate{place}.", sentence), ("Lock the gate.", request)],
        [(f"The cargo{place} may arrive tomorrow.", sentence), ("Its arrival date is uncertain.", sentence)],
        [(f"The archive{place} holds the inspection report.", sentence), ("From that report, identify the inspector.", request),
         ("If the name is missing, say so.", request), ("Around seventy words, please.", request)],
        [(f"Before answering, check the table{place}.", request), ("Please make the reply brief.", request),
         ("Do not save this request as a fact.", request)],
        [(f"The manual{place} says to close the valve.", sentence), ("Could you close the valve?", request)],
        [(f"Please tell me whether the crane{place} is ready.", request), ("The crane is ready.", sentence)],
        [(f"For the next reply{place}, use full names.", request), ("When a record is absent, state that it is absent.", request)],
        [(f'The report{place} contains the words "list the parts".', sentence), ("List the parts from the report.", request)],
        [(f"Room {variant} has two exits." if variant else "This room has two exits.", sentence),
         ("Show both exits.", request), ("The north exit is locked.", sentence),
         ("If it is locked, use the south exit.", request)],
    ]
    pairs = cases[index]
    if index < 8:
        pairs = pairs + [("Reply with exactly one word: noted.", request)]
    return " ".join(quote for quote, _ in pairs), [list(pair) for pair in pairs]
