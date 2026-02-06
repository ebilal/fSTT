# Shokudo Phone Orders (Synthetic)

This dataset contains **synthetic** phone conversations for an Asian restaurant (Shokudo), covering pickup and delivery orders.

## Data format

A single CSV file: `shokudo_conversations_100.csv`

Columns:
- `dialog_id` (int): conversation id
- `utterance_id` (int): turn index within the conversation (starts at 0)
- `speaker` (str): `customer` or `agent`
- `text` (str): utterance text

## Example

dialog_id,utterance_id,speaker,text
0,0,agent,"Hi, thanks for calling Shokudo. How can I help you today?"
0,1,customer,"Hi! I'd like to place an order for pickup."
0,2,customer,"Can I get 1 tonkotsu ramen?"
0,3,agent,"Got it: 1 x tonkotsu ramen."

## Notes

- Conversations are generated from the provided menu JSON (items, prices, and ordering instructions).
- The `agent` asks required clarification questions for items that include `ASK:` instructions (e.g., pork vs chicken, soup vs salad, beef doneness).
- Totals are **subtotals before tax** and exclude delivery fees (if any).
