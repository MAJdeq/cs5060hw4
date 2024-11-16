# cs5060hw4

## **Collaborators**
* Ethan Ford
* Nathan Freestone

## *Prequisites*
* There is a videos folder inside the project that has example videos from each part of our project
* Instead of ```gym```, we ended up having to use ```gymnasium```, as ```gym``` had been
deprecated from ```stable_baselines3```

## Part 1

### *Baseline Average*
* 419

## Part 2 | Modifying the Reward Function
* New Reward: 498.15
* Total Train time: 31s
* Changing the reward function drastically changes how the cartpole corrects itself. In our code, we decided to
implement a threshold so that our cart will not allow itself or the pole to cross it. When accounting for that threshold,
the cart would drift towards the threshold to the right or left while carefully balancing the pole; when getting to close
to the threshold, we subtracted 10 off of its reward, and it slowly drifted back the other way. However, to achieve this 
behavior, we had to give the model more time to train, changing the original 10000 timestep to 50000.

## Part 3 | Modifying the Model
* New Reward: 498.47
* Total Train time: 30s
* Changing the model itself will have varying complexity, depending on how many nodes and hidden layers we have. The current
configuration of our custom model seems to take a little less time as opposed to modifying the reward function. The average
reward is a tiny bit better as well. In order to allow the model to learn more, we increased the timestep to 50000, and 
our loss dropped significantly after about 25s.
