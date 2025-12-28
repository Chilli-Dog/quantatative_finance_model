# quantatative_finance_model
The READMe was made by Gemini AI. This should be used with care and do your own research before using the model with real money


This project implements a Neuro-evolutionary algorithm to predict stock market prices (specifically AAPL) using financial time-series data. Instead of standard backpropagation, it uses competitive evolution to optimize neural network topologies and weights.

Key Features
Custom Data Pipeline: Uses a custom DataBase module to ingest CSV files and a Scaler to normalize OHLC (Open, High, Low, Close) data, ensuring the model processes values on a consistent scale.

Time-Series Windowing: Implements a sliding window strategy (5-day windows) to provide the model with temporal context while strictly avoiding data leakage.

Neuro-evolutionary Training:

Population-Based Learning: Utilizes a Population of genomes that undergo mutation and selection.

Ancestor Bootstrapping: Loads a pre-existing "ancestor" genome from JSON to seed the initial population, allowing for iterative refinement.

Competitive Selection: Implements a generational cycle where genomes are evaluated based on Mean Squared Error (MSE), and the least fit are deleted and replaced by mutated survivors.

Inference & De-normalization: The final "Best Genome" predicts the price of the latest window, which is then transformed back from scaled values to actual USD prices for human-readable results.

Technical Stack
Language: Rust

Math/Arrays: ndarray for matrix operations.

Domain Logic: neuro_evo (Custom library for evolutionary logic).

Data Handling: CSV and Serde for genome serialization.
