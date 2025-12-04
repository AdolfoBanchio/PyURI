import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from fiuri import FIURIModule, FiuriDenseConn, FiuriSparseGJConn, FIURIModuleV2


class TWC (nn.Module):
        """  
        When creating the module a proper input decoder to the 4 sensory neurons must be provided
        if you plan to pass raw observations. Decoder must accept keyword args
        n_inputs and device (compatible with utils.twc_io_wrapper.default_obs_encoder).
     
        """
        def __init__(self, in_layer: FIURIModuleV2,
                    hid_layer: FIURIModuleV2,
                    out_layer: FIURIModuleV2,
                    in2hid_IN: FiuriDenseConn,
                    in2hid_GJ: FiuriSparseGJConn,
                    hid_IN: FiuriDenseConn,
                    hid_EX: FiuriDenseConn,
                    hid2out_EX: FiuriDenseConn,
                    obs_encoder: Callable,
                    action_decoder: Callable,
                    log_stats: bool = True,
                    **kwargs):
            super().__init__(**kwargs)
            # neuron layers
            self.in_layer = in_layer
            self.hid_layer = hid_layer
            self.out_layer = out_layer

            # connections
            self.in2hid_IN = in2hid_IN
            self.in2hid_GJ = in2hid_GJ

            self.hid_IN = hid_IN
            self.hid_EX = hid_EX

            self.hid2out = hid2out_EX

            # I/O
            self.obs_encoder = obs_encoder
            self.action_decoder = action_decoder

            # MONITOR
            self.log = log_stats
            self.monitor = {
                "in": [],
                "hid": [],
                "out": [],
            }
            self._state = None

        def _init_layer_state(self, layer: FIURIModuleV2, batch_size: int, device, dtype):
            return layer.init_state(batch_size=batch_size, device=device, dtype=dtype)

        def _make_state(self, batch_size: int, device, dtype):
            return {
                "in": self._init_layer_state(self.in_layer, batch_size, device, dtype),
                "hid": self._init_layer_state(self.hid_layer, batch_size, device, dtype),
                "out": self._init_layer_state(self.out_layer, batch_size, device, dtype),
            }

        def _ensure_state(self, batch_size: int, device, dtype):
            if self._state is None:
                self._state = self._make_state(batch_size, device, dtype)
                return self._state

            sample_E, _ = self._state["in"]
            if sample_E.shape[0] != batch_size or sample_E.device != device or sample_E.dtype != dtype:
                self._state = self._make_state(batch_size, device, dtype)
            return self._state

        def get_initial_state(self, batch_size: int, device, dtype=torch.float32):
            """Returns a batched, zeroed state dict for starting a BPTT unroll."""
            return {
                "in": self._init_layer_state(self.in_layer, batch_size, device, dtype),
                "hid": self._init_layer_state(self.hid_layer, batch_size, device, dtype),
                "out": self._init_layer_state(self.out_layer, batch_size, device, dtype),
            }

        def _recurrent_step(self, 
                            x: torch.Tensor, 
                            state_in: dict[str, tuple[torch.Tensor, torch.Tensor]]
                            ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:   
            device = next(self.parameters()).device
            assert x.device == device, "Pasa x.to(device) antes de llamar al TWC"

            ex_in, in_in = self.obs_encoder(x, n_inputs=4, device=device)
            B = ex_in.size(0)
            
            current_state = state_in 
            # Create a new state dict, don't modify the old one inplace
            state_out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

            # For input layer: directly set states from encoder values (matching ariel feedNN behavior)
            # We do the same: E = encoded_value, O = encoded_value
            # The encoder returns (ex_in, in_in) where:
            #   ex_in: [PVD_EX=0, PLM_EX, AVM_EX=0, ALM_EX] - excitatory channel
            #   in_in: [PVD_IN, PLM_IN=0, AVM_IN, ALM_IN=0] - inhibitory channel
            # Input neuron order: [PVD, PLM, AVM, ALM]
            input_values = ex_in + in_in
            in_E = input_values
            in_O = input_values
            in_state = (in_E, in_O)
            
            # Ariel's two-phase update:
            # Phase 1: computeVnext() for all neurons
            #   - For input neurons: feedNN() just set states, so getOutputState() returns NEW states
            #   - For other neurons: getOutputState() returns OLD states (before commit)
            in_out_new, in_state_new = self.in_layer.neuron_step(in_state)
            in_out_for_connections = in_O
            state_out["in"] = in_state_new

            # Hidden Layer
            hid_E_old, hid_O_old = current_state["hid"]
            hid_state_old = (hid_E_old, hid_O_old)   # sin clone            
            
            in2hid_influence = self.in2hid_IN(in_out_for_connections)
            in2hid_gj_bundle = self.in2hid_GJ(in_out_for_connections)
                        
            hid_ex_influence = self.hid_EX(hid_O_old)
            hid_in_influence = self.hid_IN(hid_O_old)
            hid2hid_influence = hid_ex_influence + hid_in_influence
            
            hid_out_new, hid_state_new = self.hid_layer(
                in2hid_influence + hid2hid_influence,
                state=hid_state_old,
                gj_bundle=in2hid_gj_bundle,
                o_pre=in_out_for_connections,
            )
            state_out["hid"] = hid_state_new

            hid_out_old_final = hid_O_old
            hid2out_ex_influence = self.hid2out(hid_out_old_final)
            

            out_E_old, out_O_old = current_state["out"]
            out_state_old = (out_E_old, out_O_old)
            out_output, out_layer_state = self.out_layer(
                hid2out_ex_influence,
                state=out_state_old
            )
            state_out["out"] = out_layer_state
            
            # Matching ariel's getFeedBackNN() which uses getInternalState()
            out_internal_states = out_layer_state[0]
            action = self.action_decoder(out_internal_states)
            # Return the action and the new state
            return action, state_out
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            This is the STATEFUL method for ROLLOUT/EVALUATION.
            It uses and updates its internal self._state.
            """
            device = next(self.parameters()).device
            if x.device != device:
                x = x.to(device)
                
            B = x.shape[0] if x.dim() > 1 else 1
            dtype = x.dtype
            
            # 1. Get the current internal state
            current_state = self._ensure_state(B, device, dtype)
            
            # 2. Detach it (we never want gradients flowing during rollouts)
            current_state_detached = {
                name: (state_pair[0].detach(), state_pair[1].detach())
                for name, state_pair in current_state.items()
            }

            # 3. Call the pure recurrent step
            action, new_state = self._recurrent_step(x, current_state_detached)
            
            # 4. Update the internal state for the next call
            self._state = new_state
            
            if self.log:
                self.log_monitor(self._state)

            # 5. Return just the action
            return action
        
        # --- NEW: This is the STATELESS "training" function (Job 2) ---
        def forward_bptt(self, x: torch.Tensor, state_in: dict) -> tuple[torch.Tensor, dict]:
            """
            This is the STATELESS method for BPTT/TRAINING.
            It's just a public wrapper for the pure _recurrent_step.
            This is what 'update_step_bptt' should call.
            """
            return self._recurrent_step(x, state_in)

        def forward_sequence(self,
                             x_seq: torch.Tensor,
                             state_in: dict | None = None
                             ) -> tuple[torch.Tensor, dict]:
            """
            Procesa una secuencia completa de observaciones (B, T, 2) en una pasada.
            Devuelve (acciones (B,T,A), estado_final).

            - Mantiene la misma dinámica que _recurrent_step (usa valores "old" en conexiones).
            - Optimiza cómputo pre-calculando pesos efectivos y el bundle de GJ una sola vez.
            """
            device = next(self.parameters()).device
            if x_seq.device != device:
                x_seq = x_seq.to(device)

            assert x_seq.dim() == 3 and x_seq.size(-1) == 2, "x_seq debe ser (B, T, 2)"
            B, T, _ = x_seq.shape
            dtype = x_seq.dtype

            # Estado inicial
            if state_in is None:
                state = self.get_initial_state(B, device, dtype)
            else:
                state = state_in

            # Pre-encode inputs en bloque
            x_flat = x_seq.reshape(B * T, -1)
            ex_flat, in_flat = self.obs_encoder(x_flat, n_inputs=4, device=device)
            inp_flat = ex_flat + in_flat  # (BT, 4)
            inp_seq = inp_flat.view(B, T, -1)  # (B, T, 4)

            # Precompute effective weights (una sola vez, con gradientes)
            def eff_w_dense(conn):
                import torch.nn.functional as F
                w_pos = F.softplus(conn.w)
                return w_pos * conn.w_mask

            w_in2hid_in = eff_w_dense(self.in2hid_IN)   # (n_in, n_hid)
            w_hid_ex = eff_w_dense(self.hid_EX)         # (n_hid, n_hid)
            w_hid_in = eff_w_dense(self.hid_IN)         # (n_hid, n_hid)
            w_hid2out = eff_w_dense(self.hid2out)       # (n_hid, n_out)

            # Precompute GJ bundle (índices + pesos positivos)
            gj_src_idx = self.in2hid_GJ.gj_idx[0]
            gj_dst_idx = self.in2hid_GJ.gj_idx[1]
            gj_w_pos = F.softplus(self.in2hid_GJ.gj_w)
            gj_bundle = (gj_src_idx, gj_dst_idx, gj_w_pos)

            actions = []

            # Desempaquetar estado actual
            in_state = state["in"]
            hid_state = state["hid"]
            out_state = state["out"]

            for t in range(T):
                # 1) Capa de entrada: seteo directo (E=O=input_values)
                input_values = inp_seq[:, t, :]  # (B, 4)
                in_E = input_values
                in_O = input_values
                in_state_tmp = (in_E, in_O)
                # Actualizar dinámica interna para logging/consistencia
                _, in_state = self.in_layer.neuron_step(in_state_tmp)

                # 2) Conexiones a Hidden usando outputs "old" de input (in_O)
                in_out_for_conn = in_O
                # IN: signo negativo
                in2hid_influence = - in_out_for_conn.matmul(w_in2hid_in)  # (B, n_hid)

                # 3) Hidden recurrente con EX/IN densos + GJ de entrada
                hid_E_old, hid_O_old = hid_state
                hid_ex_infl = hid_O_old.matmul(w_hid_ex)
                hid_in_infl = - hid_O_old.matmul(w_hid_in)
                hid2hid_infl = hid_ex_infl + hid_in_infl

                hid_chem = in2hid_influence + hid2hid_infl
                hid_out_new, hid_state_new = self.hid_layer(
                    hid_chem,
                    state=hid_state,
                    gj_bundle=gj_bundle,
                    o_pre=in_out_for_conn,
                )

                # 4) Proyección a salida usando hid_O_old (fase dos etapas)
                hid2out_ex_infl = hid_O_old.matmul(w_hid2out)

                out_out, out_state_new = self.out_layer(
                    hid2out_ex_infl,
                    state=out_state,
                )

                # 5) Decodificar acción desde estado interno de salida
                out_internal_states = out_state_new[0]
                a_t = self.action_decoder(out_internal_states)  # (B, A)
                actions.append(a_t)

                # Avanzar estado
                hid_state = hid_state_new
                out_state = out_state_new

            a_seq = torch.stack(actions, dim=1)  # (B, T, A)
            final_state = {"in": in_state, "hid": hid_state, "out": out_state}
            return a_seq, final_state

        def reset(self):
            """Resets the internal state variables for each layer."""
            self._state = None
        
        def reset_internal_only(self):
            """
            Reset only internal states (E) to their initial value while preserving
            output states (O), to mirror Ariel's Reset behavior.
            """
            if self._state is None:
                return
            # For each layer, set E := init_E and keep O unchanged
            for layer_key, layer in (("in", self.in_layer), ("hid", self.hid_layer), ("out", self.out_layer)):
                E, O = self._state[layer_key]
                new_E = torch.full_like(E, layer._init_E)
                self._state[layer_key] = (new_E, O)
        
        def detach(self):
            if self._state is not None:
                self._state = {
                    name: (state_pair[0].detach(), state_pair[1].detach())
                    for name, state_pair in self._state.items()
                }

        def log_monitor(self, state):
            """  
            In each layer list logs a dictonary like this
            {
            "in_state":self.in_state,
            "out_state": self.out_state,
            "threshold": self.threshold,
            "decay_factor": self.decay,
            }
            """
            def _pack(layer, state_pair):
                return {
                    "in_state": state_pair[0].detach().cpu(),
                    "out_state": state_pair[1].detach().cpu(),
                    "threshold": layer.threshold.detach().cpu(),
                    "decay_factor": layer.decay.detach().cpu(),
                }

            self.monitor["in"].append(_pack(self.in_layer, state["in"]))
            self.monitor["hid"].append(_pack(self.hid_layer, state["hid"]))
            self.monitor["out"].append(_pack(self.out_layer, state["out"]))

        def _set_all_weights_to_one(self):
            """Force every learnable connection weight to produce 1 after softplus."""
            softplus_inv_one = math.log(math.e - 1.0)
            with torch.no_grad():
                dense_conns = (
                    self.in2hid_IN,
                    self.hid_IN,
                    self.hid_EX,
                    self.hid2out,
                )
                for conn in dense_conns:
                    conn.w.fill_(softplus_inv_one)
                self.in2hid_GJ.gj_w.fill_(softplus_inv_one)
