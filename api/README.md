# A few thoughts about an API

To run the script 
```commandline
python -m v0.mnist --root <dataset-path>
```

Main ideas 
* Having `Trainer` and `Evaluator` based on a `Driver` embedding an `ignite.Engine`
* Having some handlers and helpers for reducing code and the complexity
* Chaining the handlers
* Having a collective API : no more `if idist.get_rank() == 0`
* Don't have a simplistic API and don't hide the concepts of `ignite`