def inj_forward(self, hidden_states, context=None, mask=None):

    is_dict_format = True
    if context is not None:
        try:
            context_tensor = context["CONTEXT_TENSOR"]
        except:
            context_tensor = context
            is_dict_format = False

    else:
        context_tensor = hidden_states

    batch_size, sequence_length, _ = hidden_states.shape

    query = self.to_q(hidden_states)

    key = self.to_k(context_tensor)
    value = self.to_v(context_tensor)

    dim = query.shape[-1]

    query = self.reshape_heads_to_batch_dim(query)
    key = self.reshape_heads_to_batch_dim(key)
    value = self.reshape_heads_to_batch_dim(value)

    attention_scores = torch.matmul(query, key.transpose(-1, -2))

    attention_size_of_img = attention_scores.shape[-2]
    if context is not None:
        if is_dict_format:
            f: Callable = context["WEIGHT_FUNCTION"]
            w = context[f"CROSS_ATTENTION_WEIGHT_{attention_size_of_img}"]
            sigma = context["SIGMA"]

            cross_attention_weight = f(w, sigma, attention_scores)
        else:
            cross_attention_weight = 0.0
    else:
        cross_attention_weight = 0.0

    attention_scores = (attention_scores + cross_attention_weight) * self.scale

    attention_probs = attention_scores.softmax(dim=-1)

    hidden_states = torch.matmul(attention_probs, value)

    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    return hidden_states
