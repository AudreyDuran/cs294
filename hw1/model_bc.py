"""
Reseau de neurone qui va faire le behavorial cloning
Prend en entree un ensemble d'observations et actions associees issues de l'expert

Example usage:
    python3 model_bc.py expert_data/Hopper-v2_20rollouts.pkl Hopper-v2 --render --num_rollout 20
"""

import os
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--dagger', action='store_true')

    args = parser.parse_args()

    d = load_data(args)
    model = build_model(d, args)

    if args.dagger:
        dag(model, d, args)
    else:
        b_c(model, d, args)

def b_c(model, data, args):
    print('Start Behavorial Cloning')
    num_epochs_list = [2] #[1, 5, 10, 20, 50, 80, 150]
    res = []
    for num_epochs in num_epochs_list:
        Hist = train(model, data, args, num_epochs)
        plot_loss(Hist, args, num_epochs)
        r = test(model, args)[0]
        res.append(np.mean(r))

    print(res)
    fig2, ax2 = plt.subplots()
    ax2.plot(num_epochs_list, res)
    ax2.set(xlabel='epochs',
            ylabel='returns',
            title='Evolution of returns depending on training epochs nb %s' % args.envname)
    ax2.grid()
    plt.show()
    plt.close()


def dag(model, data, args):
    print("Let's go for DAgger!")
    num_epochs = 20
    dagger_iteration = 20

    expert_policy_file = 'experts/%s.pkl' % args.envname
    print('loading and building expert policy %s'%expert_policy_file)
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')

    with tf.Session():


        import gym
        env = gym.make(args.envname)
        max_steps = env.spec.timestep_limit  # args.max_timesteps or

        res =[]

        for i in range(dagger_iteration):
            print('DAgger iteration%f'%i)
            Hist = train(model, data, args, num_epochs)
            #plot_loss(Hist, args, num_epochs)

            returns = []
            observations = []
            actions_expert = []
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = model.predict(obs[None, :])  # chgt par rapport a run_expert
                    action_expert = policy_fn(obs[None,:])
                    observations.append(obs)
                    actions_expert.append(action_expert)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()  # montre l'envir si --render dans l'appel
                    if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            d = {'observations': np.array(observations),
                 'actions': np.array(actions_expert)} #actions de l'expert a ajouter !! pas celles de notre modele

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))

            np.append(data['observations'], d['observations'], axis=0)
            d['actions'] = d['actions'].squeeze()
            np.append(data['actions'], d['actions'], axis=0)
            res.append(np.mean(returns))

    print(res)
    fig2, ax2 = plt.subplots()
    ax2.plot(range(dagger_iteration), res)
    ax2.set(xlabel='DAgger iteration',
            ylabel='returns',
            title='Evolution of returns with DAgger iteration %s' % args.envname)
    ax2.grid()
    plt.show()
    plt.close()

def load_data(args):
    print('loading expert data')
    with open(args.expert_data_file, 'rb') as f:
        data = pickle.loads(f.read())  # dictionnaire cles 'observations', 'actions'
        # data['observations'].shape = (20000, 376) #47 obs * 8
        # data['actions'].shape) = (20000, 1, 17) #17 actions possibles pr humanoid
    return(data)

def build_model(data, args):
    print('building and compiling the model')

    obs_shape = data['observations'].shape[1]
    num_data = data['observations'].shape[0]
    # num_train = round(3/4*num_data) #pas besoin de separer les data en test et train, tt pdre pour train puis test dans
    #num_train = num_data
    num_actions = data['actions'].shape[2]
    data['actions'] = data['actions'].squeeze()  # enleve la dimension 1

    task = args.envname
    # modele pour Hopper
    if task == 'Hopper-v2':
        model = keras.Sequential([# data deja flat, pas besoin de flatten
            keras.layers.Dense(64, activation=tf.nn.tanh, name='hopdense1', input_shape=(obs_shape,)),
            keras.layers.Dense(32, activation=tf.nn.tanh, name='hopdense2'),
            keras.layers.Dense(num_actions, name='hopdense3')])  # activation = tf.nn.softmax,activation
        print('Hopper model built')

    # modele Humanoid
    else:
        model = keras.Sequential([
            keras.layers.Dense(128, activation = tf.nn.relu, name = 'humdense1', input_shape=(obs_shape,)),
            keras.layers.Dense(64, activation = tf.nn.leaky_relu, name = 'humdense2'),
            keras.layers.Dense(32, activation=tf.nn.leaky_relu, name='humdense3'),
            keras.layers.Dense(num_actions, name = 'humdense4' )])
        print('Humanoid model built')

    # compiler le modele
    model.compile(optimizer='adam',
                  loss='mean_squared_error', #'sparse_categorical_crossentropy' ne fonctionne pas!!
                  metrics=['accuracy'])

    return(model)

def train(model, data, args, num_epochs):
    #indexes = np.random.choice(np.arange(num_data), num_train, replace=False)
    #train_data = data['observations'][indexes]
    #train_labels = data['actions'][indexes]
    train_data = data['observations']
    train_labels = data['actions']

    # os.makedirs("./logs/" + "run_%s/"%args.envname)
    tb_callback = TensorBoard(log_dir="./logs/" + "run_%s/" % args.envname)

    # entrainer le modele
    History = model.fit(train_data,
                        train_labels,
                        epochs=num_epochs,
                        batch_size=32, # pq plus le batch est petit, plus c'est long et meilleurs sont les resultats ?
        #car update plus souvent le gradient
                        callbacks=[tb_callback])

    # evaluer l'accuracy
    # test_data = data['observations'][-indexes]
    # test_labels = data['actions'][-indexes]
    # test_loss, test_acc = model.evaluate(test_data, test_labels)
    # print('test loss :', test_loss, ', test acc:', test_acc)

    return(History)


def test(model, args):
    # l'environnement gym avec la policy apprise

    # with tf.Session(): #mais pas besoin ?
    # tf_util.initialize() #ne pas mettre!!!! car reinitialise ttes les var

    import gym
    env = gym.make(args.envname)
    max_steps = env.spec.timestep_limit  # args.max_timesteps or

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model.predict(obs[None, :])  # chgt par rapport a run_expert
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()  # montre l'envir si --render dans l'appel
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    return(returns, expert_data)

def plot_loss(History, args, num_epochs):
    d = History.history
    loss = d['loss']

    # Plot de l'evol de la loss dans le tps (dans les epochs)
    x = np.arange(1, num_epochs + 1, 1)
    fig, ax = plt.subplots()
    ax.plot(x, loss)
    ax.set(xlabel='epochs',
           ylabel='loss',
           title='Evolution of loss during training for behavioral cloning for %s' % args.envname)
    ax.grid()
    plt.show()
    fig.savefig('warmup_bc_%s.png' % args.envname)
    plt.close()

def dagger():
    pass

if __name__ == '__main__':
    main()