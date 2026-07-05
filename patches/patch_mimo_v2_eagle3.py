from pathlib import Path


path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py"
)
source = path.read_text()

source = source.replace(
    "from .interfaces import MixtureOfExperts, SupportsPP",
    "from .interfaces import EagleModelMixin, MixtureOfExperts, SupportsEagle3, SupportsPP",
)
source = source.replace(
    "class MiMoV2Model(nn.Module):",
    "class MiMoV2Model(nn.Module, EagleModelMixin):",
)
source = source.replace(
    "class MiMoV2FlashForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):",
    "class MiMoV2FlashForCausalLM(nn.Module, SupportsPP, SupportsEagle3, MixtureOfExperts):",
)

old_forward = """        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states
"""

new_forward = """        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states
"""

if old_forward not in source and new_forward not in source:
    raise SystemExit("MiMoV2 forward block not found")

if old_forward in source:
    source = source.replace(old_forward, new_forward, 1)

path.write_text(source)
