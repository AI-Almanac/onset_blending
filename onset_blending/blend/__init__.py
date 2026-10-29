"""Blending methods: each turns the shared stages' features into onset
probabilities over the week bins.

logit: the multinomial logistic blend of formula features.
random_forest: the CRPS random forest over each model's daily rainfall and
the climatology (evaluation.random_forest), giving daily probabilities too.
"""
