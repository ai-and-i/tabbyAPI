import traceback
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.generator.filters import ExLlamaV2Filter, ExLlamaV2PrefixFilter
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.exllamav2 import (
    ExLlamaV2TokenEnforcerFilter,
    build_token_enforcer_tokenizer_data,
)
from loguru import logger
from typing import List
from functools import lru_cache


class OutlinesTokenizerWrapper:
    """Wrapper for Outlines tokenizer"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary = {
            self.tokenizer.tokenizer_model.id_to_piece(idx): idx
            for idx in range(self.tokenizer.get_vocab_size())
        }
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token
        self.special_tokens = self.special_tokens = set(
            self.tokenizer.extended_id_to_piece.values()
        )
        self.id_to_piece = self.tokenizer.get_id_to_piece_list()

    def convert_token_to_string(self, token):
        return self.id_to_piece[self.vocabulary[token]]

    def decode(self, tokens):
        s = ""
        for t in tokens:
            s += self.id_to_piece[t]
        return s


class ExLlamaV2OutlinesFilter(ExLlamaV2Filter):
    """Filter class for outlines-based FSM"""

    def __init__(self, model, tokenizer, guide, state=0):
        super().__init__(model, tokenizer)

        self.guide = guide
        self.state = state

    def begin(self, prefix_str=""):
        self.state = 0

    def feed(self, token):
        self.state = self.guide.get_next_state(self.state, token.item())

    def next(self):
        return self.guide.get_next_instruction(self.state).tokens, set()

    def clone(self, c=None):
        if c is None:
            return ExLlamaV2OutlinesFilter(
                self.model, self.tokenizer, self.guide, self.state
            )
        else:
            c.model = self.model
            c.tokenizer = self.tokenizer
            c.guide = self.guide
            c.state = self.state
            return c


@lru_cache(10)
def _get_lmfe_tokenizer_data(tokenizer: ExLlamaV2Tokenizer):
    return build_token_enforcer_tokenizer_data(tokenizer)


@lru_cache
def _get_regex_guide(pattern, tokenizer):
    from outlines.fsm.guide import RegexGuide

    return RegexGuide(pattern, OutlinesTokenizerWrapper(tokenizer))


@lru_cache
def _get_cfg_guire(ebnf_string, tokenizer):
    from outlines.fsm.guide import CFGGuide

    return CFGGuide(ebnf_string, OutlinesTokenizerWrapper(tokenizer))


def clear_grammar_func_cache():
    """Flush tokenizer_data cache to avoid holding references to
    tokenizers after unloading a model"""

    _get_lmfe_tokenizer_data.cache_clear()
    _get_regex_guide.cache_clear()
    _get_cfg_guire.cache_clear()


class ExLlamaV2Grammar:
    """ExLlamaV2 class for various grammar filters/parsers."""

    filters: List[ExLlamaV2Filter]

    def __init__(self):
        self.filters = []

    def add_json_schema_filter(
        self,
        json_schema: dict,
        model: ExLlamaV2,
        tokenizer: ExLlamaV2Tokenizer,
    ):
        """Adds an ExllamaV2 filter based on a JSON schema."""

        # Create the parser
        try:
            schema_parser = JsonSchemaParser(json_schema)
        except Exception:
            traceback.print_exc()
            logger.error(
                "Skipping because the JSON schema couldn't be parsed. "
                "Please read the above error for more information."
            )

            return

        # Allow JSON objects or JSON arrays at the top level
        json_prefixes = ["[", "{"]

        lmfilter = ExLlamaV2TokenEnforcerFilter(
            schema_parser, _get_lmfe_tokenizer_data(tokenizer)
        )
        prefix_filter = ExLlamaV2PrefixFilter(model, tokenizer, json_prefixes)

        # Append the filters
        self.filters.extend([lmfilter, prefix_filter])

    def add_regex_filter(
        self,
        pattern: str,
        model: ExLlamaV2,
        tokenizer: ExLlamaV2Tokenizer,
    ):
        """Adds an ExllamaV2 filter based on regular expressions."""

        try:
            guide = _get_regex_guide(pattern, tokenizer)
            regex_filter = ExLlamaV2OutlinesFilter(model, tokenizer, guide)
        except ImportError:
            logger.error(
                "Skipping regex parsing because Outlines is not installed.\n"
                "Please run the following command in your environment "
                "to install extra packages:\n"
                "pip install -U .[extras]"
            )

            return

        self.filters.append(regex_filter)

    def add_ebnf_filter(
        self,
        ebnf_string: str,
        model: ExLlamaV2,
        tokenizer: ExLlamaV2Tokenizer,
    ):
        """
        Add an EBNF grammar filter.
        """

        try:
            guide = _get_cfg_guire(ebnf_string, tokenizer)
            ebnf_filter = ExLlamaV2OutlinesFilter(model, tokenizer, guide)
        except ImportError:
            logger.error(
                "Skipping EBNF parsing because Outlines is not installed.\n"
                "Please run the following command in your environment "
                "to install extra packages:\n"
                "pip install -U .[extras]"
            )

            return

        self.filters.append(ebnf_filter)
