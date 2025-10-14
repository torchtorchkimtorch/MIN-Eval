class Verify:
    def __init__(self, config, boxed_prompt, gold, answer):
        self.config = config
        self.boxed_prompt = boxed_prompt
        self.gold = gold
        self.answer = answer

    def tf_verify(self):
        if self.config['type'] == "math" and self.boxed_prompt:
            from math_verify import verify, parse
            gold = parse(self.gold)
            answer = parse(self.answer)
            return verify(gold, answer)
            