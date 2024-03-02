# Data Splits and Neural Network Performance Evaluation

Data Splits:

* Training Split: used to minimize the Loss Function
* Validation Split: used to choose best hyperparameters, such as the learning rate, number of layers, etc.
* Test Split: the neural network never observes this set. The developer never uses this set in the design process

Dataset $\sim$ 10000 :
| Split | Percentage |
|---|---|
| Training | 60% |
| Validation | 20% |
| Testing | 20% |

Dataset $\sim$ 1000000 :
| Split | Percentage |
|---|---|
| Training | 98% |
| Validation | 1% |
| Testing | 1% |

## Behavior of Split Specific Loss Functions

|  | **Training (6000) $$J(\bm{\theta})_{\text{train}}$$** | **Validation (2000) $$J(\bm{\theta})_{\text{val}}$$** | **Testing (2000) $$J(\bm{\theta})_{\text{test}}$$** | **$\sim{}10000$ $$J(\bm{\theta})_{\text{Minimum}}$$** |
|---|---|---|---|---|
| Good Estimator | 0.21 | 0.25 | 0.30 | 0.18 |
| Underfitting | 1.90 | 1.90 | 2.10 |  |
| Overfitting | 0.21 | 2.05 | 2.10 |  |

> The goal is always try to minimize $J(\bm{\theta})_{\text{test}}$ which is computed by data **not seeing** by the NN.</br>
The gap between $J(\bm{\theta})_{\text{train}}$ and $J(\bm{\theta})_{\text{val}}$ is called generalization gap

## Reducing the Effect of Underfitting/Overfitting

* Underfitting: (Training loss is high)
  * Train longer
  * More layers or more parameters per layer
  * Change architecture
* Overfitting: (Generalization gap is large)
  * More training data
  * Regularization
  * Change architecture
