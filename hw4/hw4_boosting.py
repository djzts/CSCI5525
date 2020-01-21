from test_score import score
import numpy as np
import matplotlib.pyplot as plt

#read
true_values = np.genfromtxt('true_values.csv', delimiter=',')
weak_tensor = np.expand_dims(true_values, axis=1)

#get threshold
def threshold_gen(n):

    s = 0
    for i in range(n):
        np.random.shuffle(weak_tensor) #random
        s += score(weak_tensor) #applies the func
    return s/n


threshold_bound = threshold_gen(50)

percentage = np.sum(true_values == 0) / true_values.shape[0]

#get aggregated learner
def find_weak_learner(learners, index):

    if index == 0:
        return np.copy(learners[0])
    final = np.copy(learners[0])
    for id_i in range(1, index):
        final += learners[id_i]
    final /= index
    final[final < np.percentile(final, percentage*100)] = 0
    return final

num_weak_learner = 1000  # number of weak learners
score_average = [0] * num_weak_learner
weak_learner = [0] * num_weak_learner
i = 0
while i < num_weak_learner:
    np.random.shuffle(weak_tensor)
    if score(weak_tensor) < threshold_bound:
        weak_learner[i] = np.copy(weak_tensor)
        agg_l = find_weak_learner(weak_learner, i)
        score_average[i] = score(agg_l)
        i += 1

plt.plot(score_average)
plt.xlabel('Numbers of weak learners')
plt.ylabel('Scores')
plt.show()
