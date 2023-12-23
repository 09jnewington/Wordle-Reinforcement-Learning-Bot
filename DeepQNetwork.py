import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

replay_memory = deque(maxlen=10000)  # Adjust size as needed


class WordleEnvironment:
    def __init__(self, word_list):
        self.word_list = word_list
        self.target_word = None
        self.state = None
        self.letters_guessed = set()
        self.max_guesses = 6  # Define the maximum number of guesses allowed

    def initialize_state(self):
        # Existing logic for initializing guess feedback
        max_guesses = 6
        guess_feedback = [[0] * len(self.target_word) for _ in range(max_guesses)]

        # Initialize the alphabet status array
        alphabet_status = [0] * 26  # One element for each letter

        # Combine both components to form the state
        self.state = {'guess_feedback': guess_feedback, 'alphabet_status': alphabet_status}
        return self.state

    def reset(self):
        self.target_word = random.choice(self.word_list)
        self.state = self.initialize_state()
        return self.state

    def step(self, action_index):
        guess = self.word_list[action_index]
        self.letters_guessed.update(set(guess))

        done = guess == self.target_word
        reward = calculate_reward(guess, self.target_word, len(self.state['guess_feedback']), self.max_guesses)

        self.update_state(guess)
        return self.state, reward, done

    def evaluate_guess(self, guess):
        # Compare the guess with the target word
        reward = calculate_reward(guess, self.target_word, self.letters_guessed)
        done = guess == self.target_word
        return reward, done

    def update_state(self, guess):
        # Create feedback for the current guess
        feedback = [0] * len(guess)
        for i in range(len(guess)):
            if guess[i] == self.target_word[i]:
                feedback[i] = 2  # Correct letter in the correct position
            elif guess[i] in self.target_word:
                feedback[i] = 1  # Correct letter in the wrong position

        # Update the guess feedback in the state
        for row in self.state['guess_feedback']:
            if sum(row) == 0:
                for i in range(len(feedback)):
                    row[i] = feedback[i]
                break

        # Update the alphabet status in the state
        for i, letter in enumerate(guess):
            letter_index = ord(letter.lower()) - ord('a')  # Index for the letter in the alphabet
            if feedback[i] == 2:
                self.state['alphabet_status'][letter_index] = 2  # Correct position
            elif feedback[i] == 1:
                if self.state['alphabet_status'][letter_index] != 2:
                    self.state['alphabet_status'][letter_index] = 1  # Correct letter, wrong position
            else:
                if self.state['alphabet_status'][letter_index] == 0:
                    self.state['alphabet_status'][letter_index] = -1  # Letter not in word


def calculate_reward(guess, target_word, turn_count, max_turns):
    if guess == target_word:
        return 10  # Reward for guessing the right word
    elif turn_count >= max_turns:
        return -10  # Penalty for not guessing the word within the allowed turns
    else:
        return 0  # No reward or penalty during intermediate steps

def load_word_list(file_path):
    with open(file_path, 'r') as file:
        words = [line.strip() for line in file.readlines()]
    return words

# Number of guesses allowed
num_guesses = 6

# Number of letters in each guess (Wordle uses 5-letter words)
num_letters = 5

# Calculate state size
state_size = num_guesses * num_letters
state_size = state_size + 26

# Initialize environment
word_list = load_word_list(r'wordle_words.txt')  # List of valid 5-letter words
import random

# Assuming word_list is your list of words
word_list = random.sample(word_list, 10)
env = WordleEnvironment(word_list)

# Define your state_size based on how you represent the state in your environment
#state_size = [state_size]

# Define action_size based on the word list
action_size = len(word_list)  # Number of possible actions


batch_size = 32  # This is an example value; adjust based on your needs and resources

# Define the model (for a Deep Q-Network)
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(state_size,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(action_size, activation='linear')
])


learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 1  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 20  # Total number of episodes to train

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')

max_steps_per_episode = 6  # Corresponds to the number of guesses allowed in one game of Wordle

loss_history = []  # List to store loss values

for episode in range(num_episodes):
    state = env.reset()
    # Before feeding the state into the model
    guess_feedback = np.array(state['guess_feedback']).flatten()
    alphabet_status = np.array(state['alphabet_status'])
    combined_state = np.concatenate([guess_feedback, alphabet_status])
    combined_state = np.reshape(combined_state, [1, state_size])
    print(f"combined_state {combined_state}")


    for step in range(max_steps_per_episode):
        print(f"Episode: {episode}")
        print(f"step: {step}")
        # Epsilon-greedy action selection
        if np.random.rand() <= epsilon:
            action = np.random.choice(action_size)
        else:
            q_values = model.predict(combined_state, verbose=0)
            print(f"q_values: {q_values}")
            action = np.argmax(q_values[0])

        guess = word_list[action]
        print(f"Guess: {range(len(guess))}")

        next_state, reward, done = env.step(action)
        next_state_copy = next_state
         # Process next_state for storing in replay memory and training
        next_state_guess_feedback = np.array(next_state['guess_feedback']).flatten()
        next_state_alphabet_status = np.array(next_state['alphabet_status'])
        next_combined_state = np.concatenate([next_state_guess_feedback, next_state_alphabet_status])
        next_combined_state = np.reshape(next_combined_state, [1, state_size])
        print(f"next combined_state {next_combined_state}")


        # Store the combined state in replay memory instead of the raw state
        replay_memory.append((combined_state, action, reward, next_combined_state, done))


        # Training from replay memory
        if len(replay_memory) > batch_size:
            minibatch = random.sample(replay_memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                # Process next_state for training
                next_state_guess_feedback = np.array(next_state_copy['guess_feedback']).flatten()
                next_state_alphabet_status = np.array(next_state_copy['alphabet_status'])
                next_combined_state = np.concatenate([next_state_guess_feedback, next_state_alphabet_status])
                next_combined_state = np.reshape(next_combined_state, [1, state_size])

                #R(s,a)
                target = reward
                if not done:
                    #The right hand side of bellmann equation, np.amax() represents maxQ(s',a') the max Q value for the next state
                    target = reward + gamma * np.amax(model.predict(next_state, verbose=0)[0])
                #This gets the current Q-values for all actions at the current state.
                target_f = model.predict(state)
                #Updates Q value for current state
                target_f[0][action] = target
                #backpropogation using gradient descent to adjust parameters to 
                history = model.fit(state, target_f, epochs=1, verbose=0)
                loss = history.history['loss'][0]
                loss_history.append(loss)

        if done:
            break

    # Reduce epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print(f"loss history {loss_history}")
model.save(r"C:\Users\joshi\OneDrive\Desktop\ML Projects\wordle_model")
plt.plot(loss_history)
plt.title('MSE Loss Over Episodes')
plt.xlabel('Training Steps')
plt.ylabel('MSE Loss')
plt.show()

##TESTING THE MODL

num_test_episodes = 20  # Number of episodes to test the model
success_count = 0

for episode in range(num_test_episodes):
    state = env.reset()
   
    # Process the initial state for the model
    guess_feedback = np.array(state['guess_feedback']).flatten()
    alphabet_status = np.array(state['alphabet_status'])
    combined_state = np.concatenate([guess_feedback, alphabet_status])
    combined_state = np.reshape(combined_state, [1, state_size])

    done = False

    print(f"Episode {episode + 1}")
    print(f"Target Word: {env.target_word}")  # Print the target word

    for step in range(max_steps_per_episode):
        action = np.argmax(model.predict(combined_state)[0])  # Choose action based on model's prediction
        guess = word_list[action]
        next_state, reward, done = env.step(action)
        print(f"Guess {step + 1}: {guess}")  # Print each guess


         # Process next_state for the next iteration
        next_state_guess_feedback = np.array(next_state['guess_feedback']).flatten()
        next_state_alphabet_status = np.array(next_state['alphabet_status'])
        next_combined_state = np.concatenate([next_state_guess_feedback, next_state_alphabet_status])
        next_combined_state = np.reshape(next_combined_state, [1, state_size])

        combined_state = next_combined_state


        if done:
            if reward > 0:  # Assuming a positive reward indicates successful guessing
                success_count += 1
            break

success_rate = success_count / num_test_episodes
print(f"Success Rate: {success_rate * 100}%")
