from transformers import StoppingCriteria


class StopTokenCriteria(StoppingCriteria):
    """Stop generation after a separator/end token or a local token budget."""

    def __init__(self, tokenizer, max_new_tokens=50, end_Instruct="<|im_end|>"):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.start_length = None
        stop_tokens = [
            "<SEP>",
            "<EOQ>",
            "<EOE>",
            "<EOS>",
            "<EOP>",
            "<EOT>",
            "<EOR>",
            "<EOA>",
            end_Instruct,
        ]
        self.stop_token_ids = {
            token_id
            for token_id in tokenizer.convert_tokens_to_ids(stop_tokens)
            if token_id is not None and token_id != tokenizer.unk_token_id
        }

    def __call__(self, input_ids, scores, **kwargs):
        if self.start_length is None:
            self.start_length = input_ids.shape[-1]

        if input_ids.shape[-1] - self.start_length >= self.max_new_tokens:
            return True

        return input_ids[0, -1].item() in self.stop_token_ids
