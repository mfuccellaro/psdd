# psdd

Repo is the implementation of the PSDD algorithm

Sample usage

```python
rules, data_dict_train = PSDD_train(train, y_train)
data_dict_test = PSDD_test(test_, rules, data_dict_train)
drift = compute_stat(data_dict_train, data_dict_test, alpha=alpha, beta=beta)

```
