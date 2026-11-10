# Attempt-At-Neural-Network
This was an attempt by myself (a year 12 student) to make a neural network in C++ (Albeit it ended up failing due to it converging to a single set of values near the mean). If anybody could give advice as to how to fix this issue I would very much appreciate it.

Python - Copies the california dataset into a txt file for it to be equated to a matrix

C++ - Utilises LeakyRelu (chooses the max of the number and 0, returns itself if number > 0 and multiplies by 0.01 if not)
      Set up with 2 hidden layers and an output layer: Layer 1 - 8 * 32 (32 neurons); Layer 2 - 32 * 16 (16 neurons); Layer 3 - 16 * 1 (One Neuron)

      Split into two sections: Training and testing
      Defines every mathematical function used within c++

Trained on the 20,176 california house dataset
