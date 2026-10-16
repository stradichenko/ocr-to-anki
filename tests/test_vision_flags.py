"""Tests for the vision command line the CLI backend builds.

The image token budget is the only encoder-side control the runtime exposes,
and it is easy to pass unconditionally and then mistake for a working knob on
models that ignore it. These tests pin when the flags are emitted and when they
are not.
"""

from pathlib import Path

import pytest

from backends.mtmd_cli import (
    DEFAULT_VISION_SYSTEM_PROMPT,
    LlamaMtmdCli,
    strip_reasoning_channel,
)


def _cli(**overrides):
    """Build a LlamaMtmdCli without touching the filesystem or the GPU.

    The real constructor requires the model files to exist and probes the
    hardware. Neither is relevant to how the argument list is assembled, so the
    instance is made directly and only the attributes _build_cmd reads are set.
    """
    cli = LlamaMtmdCli.__new__(LlamaMtmdCli)
    defaults = dict(
        binary_path=Path("/bin/true"),
        model_path=Path("/models/model.gguf"),
        mmproj_path=Path("/models/mmproj.gguf"),
        ctx_size=4096,
        threads=2,
        temp=0.1,
        top_k=40,
        top_p=0.9,
        n_gpu_layers=-1,
        max_tokens=512,
        mmproj_offload=True,
        _is_opencl=False,
        image_min_tokens=None,
        image_max_tokens=None,
        system_prompt=DEFAULT_VISION_SYSTEM_PROMPT,
    )
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(cli, key, value)
    return cli


def _cmd(**overrides):
    return _cli(**overrides)._build_cmd("/tmp/image.png", "extract the text")


class TestImageTokenFlags:
    def test_absent_by_default(self):
        """Left unset, the projector's own default applies."""
        cmd = _cmd()
        assert "--image-max-tokens" not in cmd
        assert "--image-min-tokens" not in cmd

    def test_max_tokens_is_passed_when_set(self):
        cmd = _cmd(image_max_tokens=1120)
        assert "--image-max-tokens" in cmd
        assert cmd[cmd.index("--image-max-tokens") + 1] == "1120"

    def test_min_tokens_is_passed_when_set(self):
        cmd = _cmd(image_min_tokens=70)
        assert "--image-min-tokens" in cmd
        assert cmd[cmd.index("--image-min-tokens") + 1] == "70"

    def test_both_can_be_set_together(self):
        cmd = _cmd(image_min_tokens=70, image_max_tokens=280)
        assert "70" in cmd and "280" in cmd

    def test_zero_is_passed_rather_than_treated_as_unset(self):
        """Zero is a meaningful budget, so the None check must not be falsy."""
        cmd = _cmd(image_max_tokens=0)
        assert "--image-max-tokens" in cmd

    def test_frame_skips_the_flags(self):
        """The sweep's control case: same command, no budget applied."""
        cmd = _cmd()
        for flag in ("--image-min-tokens", "--image-max-tokens"):
            assert flag not in cmd


class TestCommandShape:
    def test_essentials_are_present(self):
        cmd = _cmd()
        assert "-m" in cmd and "--mmproj" in cmd and "--image" in cmd
        assert "-p" in cmd and "--jinja" in cmd

    def test_mmproj_offload_flag_only_when_disabled(self):
        assert "--no-mmproj-offload" not in _cmd()
        assert "--no-mmproj-offload" in _cmd(mmproj_offload=False)

    def test_flash_attention_disabled_on_opencl(self):
        # The OpenCL backend has no flash attention kernel for the vision
        # encoder, so FA is turned off there.
        assert "-fa" in _cmd(_is_opencl=True)

    def test_token_flags_do_not_disturb_the_rest(self):
        plain = _cmd()
        budgeted = _cmd(image_max_tokens=280)
        removed = budgeted[:budgeted.index("--image-max-tokens")]
        removed += budgeted[budgeted.index("--image-max-tokens") + 2:]
        assert removed == plain

class TestSystemPrompt:
    """The reasoning-channel guard.

    Gemma 4 otherwise answers inside a reasoning channel. On an image that ate
    the entire output budget: 512 tokens of thinking and no words after it, so
    the word list came back empty and the narration was all there was to parse.
    """

    def test_system_prompt_is_passed_by_default(self):
        cmd = _cmd()
        assert "-sys" in cmd
        assert cmd[cmd.index("-sys") + 1] == DEFAULT_VISION_SYSTEM_PROMPT

    def test_default_prompt_forbids_narration(self):
        lowered = DEFAULT_VISION_SYSTEM_PROMPT.lower()
        assert "ocr" in lowered
        # The wording that actually suppressed the channel in testing.
        assert "do not think out loud" in lowered or "do not explain" in lowered

    def test_a_custom_prompt_is_honoured(self):
        cmd = _cmd(system_prompt="Only the words.")
        assert cmd[cmd.index("-sys") + 1] == "Only the words."

    def test_it_can_be_disabled(self):
        assert "-sys" not in _cmd(system_prompt=None)

    def test_an_empty_prompt_is_not_sent(self):
        """An empty -sys would be pointless noise on the command line."""
        assert "-sys" not in _cmd(system_prompt="")


class TestStripReasoningChannel:
    """The deterministic half of the reasoning-channel fix.

    The system prompt suppresses the channel on simple images but does not
    eliminate it: on a dense handwritten page the model still narrated for
    several hundred tokens before answering. The caller parses the return value
    as a word list, so the narration has to be removed rather than relied upon
    not to appear.
    """

    def test_plain_text_is_untouched(self):
        assert strip_reasoning_channel("BONJOUR\nCASA") == "BONJOUR\nCASA"

    def test_reasoning_before_the_answer_is_dropped(self):
        raw = "<|channel>thought\nLet me look at this...\n<channel|>BONJOUR\nCASA"
        assert strip_reasoning_channel(raw) == "BONJOUR\nCASA"

    def test_an_unclosed_channel_yields_nothing(self):
        """No answer was produced, so there is no word list to return."""
        assert strip_reasoning_channel("<|channel>thought\nstill thinking") == ""

    def test_narration_never_leaks_through(self):
        raw = "<|channel>thought\n1. Analyze\n2. Transcribe<channel|>fenetre"
        out = strip_reasoning_channel(raw)
        assert "Analyze" not in out
        assert out == "fenetre"

    def test_handles_empty_and_none(self):
        assert strip_reasoning_channel("") == ""
        assert strip_reasoning_channel(None) == ""

    def test_keeps_a_multi_line_answer_intact(self):
        raw = "<|channel>thought\nx<channel|>cota\nl'amont, l'aval,\nbouleaux"
        assert strip_reasoning_channel(raw).splitlines() == [
            "cota", "l'amont, l'aval,", "bouleaux",
        ]
