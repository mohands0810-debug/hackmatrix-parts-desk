# Pixal Pirates - Workshop Parts Desk (HackMatrix 1.0, Agentic AI)

An LLM-assisted Python controller for the HackMatrix 1.0 Agentic AI task. It reads parts requests written in everyday language, uses a limited budget of tool calls (inspect, verify, clarify), and either dispatches a compatible in-stock part or escalates the request to a person.

## What it does
1. **Picks the current document.** Requests and catalogue entries arrive as several revisions (drafts, amendments, late retries). The controller selects the effective approved revision in code.
2. **Reads the text with two readers.** A language model (via our own HTTPS gateway) extracts voltage, connector and wattage. An independent rule-based reader does the same. If they disagree, a second model call adjudicates, and any value that is still unclear stays unknown. Values must appear in the source text.
3. **Spends the tool budget carefully.** A small expectimax planner and a budget allocator choose which checks are worth their cost and keep spending at 13 units or fewer.
4. **Decides safely.** Plain code makes every dispatch decision, and only dispatches parts that are certainly compatible. Anything uncertain is escalated, because one incompatible dispatch zeroes the whole batch.

## Post-twist mode
After the organizers' twist, the controller also handles wattage ranges (minimum and maximum), shared stock with quantities (a matching step avoids double-booking), `issued_at` document precedence, stock documents checked against the warehouse revision, and a cap on model-call size.

## Files
- `solution.py` - the controller. It exposes `decide(observation, state)` and uses only the Python standard library.
- `gateway_example.py` - the organizers' example gateway. Provider keys are read from environment variables and are never stored in this repository.

## Running it
The controller needs the organizers' local runner and data, which are not included here. With those files in place:

    python local_runner.py solution.py practice.json --gateway-url https://YOUR-GATEWAY/gateway

## Results on the public sets
About 99 to 100 on training and practice with zero incompatible dispatches. Post-twist behaviour was tested on worlds we generated from the written rules and the released operating records.

## Team 
Team: Pixal Pirates. 
(Built during HackMatrix 1.0)
