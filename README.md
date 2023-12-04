## automatic_detect

code for automatic detecting attacks in ETH protocols

## Simple DRL

To reproduce bribe attack in HTLC using simple DRL

```shell
python DRL/Simple_DRL/btc.py
```

## Hierarchical DRL

***training***

pre-training:

```shell
python DRL/hDRL/training/pre-training/B_1/_start_the_game.py
```
```shell
python DRL/hDRL/training/pre-training/B_2/_start_the_game.py
```

fine-tuning:

```shell
python DRL/hDRL/training/option/B_option/_start_the_game.py
```

***testing***

```shell
python DRL/hDRL/test/1.compare_DRL_with_policy/_start_the_game.py
```

**Traversal**

```shell
python Traversal/explore_mad/search.py
```
