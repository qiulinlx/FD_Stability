# Chapter 1 

Using Double Debiased Machine Learning to explore the effects of Diversity on Ecosystem Functioning 

## Tracking to do 

* 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

```bash
git clone  https://github.com/qiulinlx/FD_Stability.git 

```

2. Install dependencies (recommended to use a virtual environment):

```bash

pip install -r requirements.txt

```

### Usage

```python
import numpy as np
from functional_diversity_utils import functional_richness, functional_evenness

traits = np.array([[1,2],[2,3],[3,4]])
abundances = [0.5, 0.3, 0.2]

FRic = functional_richness(traits)
FEve = functional_evenness(traits, abundances)

print("FRic:", FRic)
print("FEve:", FEve)
```

## Examples


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License

