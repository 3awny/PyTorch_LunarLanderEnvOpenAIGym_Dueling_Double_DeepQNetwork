# PyTorch_LunarLanderEnvOpenAIGym_Dueling_Double_DeepQNetwork
The implementation was organised by classes for different components of the agent. Namely, classes for buffer memory, defining the Deep Q-Network (DQN), an agent class to perform respective operations mainly with-in several methods acting as a high-level access to the agent’s “brain” for the main.py file. Agent class contains a learn function involving the mathematical implementation behind a specific DQN approach and consequently, action choice. Here Dueling Double Deep Q-learning was applied achieving highest score of 239 after some hyperparameter tweaks.

1. install python 3.8.9
2. `python3 venv -m venv` -- create virtual environment named venv
3. `source venv/bin/activate` -- activate virtual environment
4. `pip install -r requirments.txt` -- install packages
5. `python3 main.py`-- begin model training
