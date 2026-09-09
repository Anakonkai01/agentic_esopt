import re
import torch


class QwenSudokuAgent:
    def __init__(
        self,
        model,
        tokenizer,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_new_tokens=32,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    def board_to_text(self, board):
        lines = []

        for row in board:
            line = " ".join(
                "." if value == 0 else str(value)
                for value in row
            )
            lines.append(line)

        return "\n".join(lines)

    def build_prompt(self, observation):
        board_text = self.board_to_text(observation)

        size = len(observation)

        return f"""
You are playing a {size}x{size} Sudoku puzzle.

Current board:

{board_text}

"." means an empty cell.

Choose exactly ONE valid move.

Rows and columns are zero-indexed.

Respond using exactly this format:

ACTION: row,col,value

Example:

ACTION: 1,2,3

Do not output anything else.
""".strip()

    def parse_action(self, text):
        match = re.search(
            r"ACTION:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
            text,
            re.IGNORECASE,
        )

        if match is None:
            return None

        row = int(match.group(1))
        col = int(match.group(2))
        value = int(match.group(3))

        return row, col, value

    @torch.inference_mode()
    def act(self, observation):
        prompt = self.build_prompt(observation)

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        generated_tokens = outputs[
            0,
            inputs["input_ids"].shape[1]:
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        action = self.parse_action(response)

        return action, response