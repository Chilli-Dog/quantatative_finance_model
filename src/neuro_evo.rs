use ndarray::{Array2, Array1, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use serde::{Serialize, Deserialize};
use rand::prelude::*;
use std::fs;
use crate::{DataBase, Scaler};
// use these if you use the json functions in the neural network impl
use std::fs::File;
use std::io::Write;


#[derive(Deserialize)]
struct JsonModel {
    layers: Vec<JsonLayer>,
    lr: f64,
}

#[derive(Deserialize)]
struct JsonLayer {
    weights: JsonData,
    biases: JsonData,
}

#[derive(Deserialize)]
struct JsonData {
    dim: Vec<usize>,
    data: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Genome {
    pub phenotype: NeuralNetwork,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
    pub _epoch: i32,
    pub lr: f64,
    pub fitness: f64,
}

impl Genome {
    // use the load_genome_from_json func as if it were the new() func.
    // then mutate the model loaded to explore different paths
    // could make the genome also have a activation function to be able to find the best
    // function for the use case
    pub fn new(phenotype: NeuralNetwork, weights: Vec<f64>, biases: Vec<f64>,
    _epoch: i32, fitness: f64, lr: f64) -> Self {
        Genome {
            phenotype,
            weights,
            biases,
            _epoch,
            lr,
            fitness,
        }
    }
    
    pub fn train(&mut self, x: Array2<f64>, y: Array2<f64>) {
        self.phenotype.train(x, y);
    }
    
    pub fn load_genome_from_json(path: String, _epoch: i32, fitness: f64) -> Result<Genome, Box<dyn std::error::Error>> {
        let file_content = fs::read_to_string(path)?;
        let raw: JsonModel = serde_json::from_str(&file_content)?;

        let mut flat_weights = Vec::new();
        let mut flat_biases = Vec::new();
        let mut network_layers = Vec::new();

        for raw_layer in raw.layers {
            flat_weights.extend(raw_layer.weights.data.clone());
            flat_biases.extend(raw_layer.biases.data.clone());

            let layer_weights = Array2::from_shape_vec(
                (raw_layer.weights.dim[0], raw_layer.weights.dim[1]),
                raw_layer.weights.data
            )?;
            let layer_biases = Array1::from_vec(raw_layer.biases.data);

            network_layers.push(Layer {
                weights: layer_weights,
                biases: layer_biases,
            });
        }

        let phenotype = NeuralNetwork { layers: network_layers, lr: raw.lr, };

        Ok(Genome {
            phenotype, // the actual model for prediction
            weights: flat_weights, // the flat_weights for mutation
            biases: flat_biases,   // the flat_biases for mutation
            _epoch,
            lr: raw.lr,
            fitness,
        })
    }

    pub fn mse(nn: &NeuralNetwork, input_data: Vec<f64>, actual_price: f64) -> f64 {
        // 1. Ensure input_data matches the first layer of the network (topology[0])
        // If your first layer is 30, input_data.len() must be 30.
        let input_matrix = Array2::from_shape_vec((1, input_data.len()), input_data)
            .expect("Input data size does not match NN input layer");

        let scaled_output = nn.predict(input_matrix);
        let predicted_scaled = scaled_output[[0, 0]];

        let error = actual_price - predicted_scaled;
        let mse_value = error * error;

        mse_value
    }

    pub fn mutate(
        &mut self,
        topology: Vec<usize>,
        batch_x: Vec<Vec<f64>>,
        batch_y: Vec<Vec<f64>>,
    ) -> Genome {
        let mut rng = rand::rng();
        let mutation_rate: f64 = rng.random_range(0.001..0.05);
        let mutation_step: f64 = rng.random_range(0.01..0.1);

        let mut new_weights = self.weights.clone();
        for w in new_weights.iter_mut() {
            if rng.random_bool(mutation_rate) {
                let nudge = rng.random_range(-mutation_step..mutation_step);
                *w = (*w + nudge).clamp(-5.0, 5.0);
            }
        }

        let mut new_biases = self.biases.clone();
        for b in new_biases.iter_mut() {
            if rng.random_bool(mutation_rate) {
                let nudge = rng.random_range(-mutation_step..mutation_step);
                *b = (*b + nudge).clamp(-2.0, 2.0);
            }
        }

        let nn = NeuralNetwork::load_from_params(topology, new_weights.clone(), new_biases.clone(), self.lr);

        let fitness = Genome::batch_mse(&nn, &batch_x, &batch_y);

        Self {
            phenotype: nn,
            weights: new_weights,
            biases: new_biases,
            _epoch: self._epoch,
            lr: self.lr,
            fitness,
        }
    }

    pub fn batch_mse(nn: &NeuralNetwork, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) -> f64 {
        let mut total_error = 0.0;
        let num_samples = targets.len() as f64;

        for (x, y_vec) in inputs.iter().zip(targets.iter()) {
            let input_matrix = Array2::from_shape_vec((1, x.len()), x.clone()).unwrap();
            let pred_matrix = nn.predict(input_matrix);
            let mut sequence_error = 0.0;
            for (i, &target_val) in y_vec.iter().enumerate() {
                let pred_val = pred_matrix[[0, i]];
                sequence_error += (target_val - pred_val).powi(2);
            }

            total_error += sequence_error / y_vec.len() as f64;
        }

        total_error / num_samples
    }
}

pub struct Population {
    pub population: Vec<Genome>,
    pub deletion_stand: f64,
}

impl Population {
    // could make multiple populations with random deletion_stand's?
    // could make it so it can compare the mse's to other children in the current
    // generation. If it is the top 3% then it produces more children to dominate more
    // of the populations minority
    // could make the first generations be trained so that the last "perfect" neural network
    // can come faster

    pub fn new(del_stand: f64) -> Self {
        Population {
            population: Vec::new(),
            deletion_stand: del_stand,
        }
    }

    pub fn reproduce(topology: Vec<usize>, deletion_stand: f64, db: DataBase, parent_1: NeuralNetwork, parent_2: NeuralNetwork,
                     x: Vec<f64>, real_price: f64, _epoch: i32) -> Self {

        let mse_0 = Genome::mse(&parent_1, x.clone(), real_price);
        let mse_1 = Genome::mse(&parent_2, x.clone(), real_price);

        // determine the better parent
        let (best_parent, other_parent) = if mse_0 < mse_1 {
            (&parent_1, &parent_2)
        } else {
            (&parent_2, &parent_1)
        };

        let mut child_weights = Vec::new();
        let mut child_biases = Vec::new();

        for (l_idx, best_layer) in best_parent.layers.iter().enumerate() {
            let other_layer = &other_parent.layers[l_idx];

            let mut w_best: Vec<f64> = best_layer.weights.iter().cloned().collect();
            let w_other: Vec<f64> = other_layer.weights.iter().cloned().collect();

            let split_idx = (w_best.len() as f64 * 0.7) as usize;
            for i in split_idx..w_best.len() {
                w_best[i] = w_other[i];
            }
            child_weights.extend(w_best);

            let mut b_best: Vec<f64> = best_layer.biases.iter().cloned().collect();
            let b_other: Vec<f64> = other_layer.biases.iter().cloned().collect();
            let b_split = (b_best.len() as f64 * 0.7) as usize;

            for i in b_split..b_best.len() {
                b_best[i] = b_other[i];
            }
            child_biases.extend(b_best);
        }

        // creates child's phenotype
        let lr = best_parent.lr;
        let phenotype = NeuralNetwork::load_from_params(topology, child_weights.clone(), child_biases.clone(), lr);
        let fitness = Genome::mse(&phenotype, x.clone(), real_price);

        let child = Genome::new(phenotype, child_weights, child_biases, _epoch, fitness, lr);

        Population {
            population: vec![child],
            deletion_stand,
        }
    }

    pub fn filter_underperforming(&mut self) {
        if self.population.is_empty() { return; }

        self.population.sort_by(|a, b| a.fitness.total_cmp(&b.fitness));

        let keep_count = ((self.population.len() as f64 * self.deletion_stand).ceil() as usize).max(2);

        if keep_count < self.population.len() {
            self.population.truncate(keep_count);
        }
    }

    pub fn start_generation(
        &mut self,
        topology: Vec<usize>,
        pop_size: usize,
        batch_y: Vec<Vec<f64>>,
        batch_x: Vec<Vec<f64>>,
    ) {
        // 1. Sort by fitness and remove the bottom performers
        // Lower MSE is better, so the best (lowest) are at the front
        self.filter_underperforming();

        if self.population.is_empty() {
            println!("Your population is extinct! Check deletion_standard and fitness logic.");
            return;
        }

        let mut rng = rand::rng();
        let mut new_generation: Vec<Genome> = Vec::with_capacity(pop_size);

        // 2. Elitism: Keep the single best genome exactly as it is
        if let Some(best) = self.population.first() {
            new_generation.push((*best).clone());
        }

        // 3. Fill the rest of the population with mutated offspring
        while new_generation.len() < pop_size {
            // Selection: Pick two parents randomly from the survivors
            let p1_idx = rng.random_range(0..self.population.len());
            let p2_idx = rng.random_range(0..self.population.len());

            let parent_1 = &self.population[p1_idx];
            let parent_2 = &self.population[p2_idx];

            // Reproduction: This creates a small temporary population (usually 2 children)
            // using your crossover logic
            let mut child_batch = Population::reproduce(
                topology.clone(),
                self.deletion_stand,
                crate::DataBase { records: vec![] },
                parent_1.phenotype.clone(),
                parent_2.phenotype.clone(),
                batch_x[0].clone(),
                batch_y[0][0],
                parent_1._epoch
            );

            for mut child in child_batch.population.drain(..) {
                if new_generation.len() < pop_size {
                    let mutated_child = child.mutate(
                        topology.clone(),
                        batch_x.clone(),
                        batch_y.clone(),
                    );
                    new_generation.push(mutated_child);
                }
            }
        }

        self.population = new_generation;
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Layer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
}

impl Layer {
    fn new(out_dim: usize, in_dim: usize) -> Self {
        let std_dev = (2.0 / (in_dim + out_dim) as f64).sqrt();
        let weights = Array2::<f64>::random((out_dim, in_dim), StandardNormal) * std_dev;
        let biases = Array1::<f64>::zeros(out_dim);
        Layer {
            weights,
            biases
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub lr: f64,
}

impl NeuralNetwork {
    pub fn new(topology: &[usize], lr: f64) -> NeuralNetwork {
        let mut layers = Vec::new();
        for dims in topology.windows(2) {
            layers.push(Layer::new(dims[1], dims[0]));
        }
        NeuralNetwork {
            layers,
            lr
        }
    }

    pub fn load_from_params(topology: Vec<usize>, weights: Vec<f64>, biases: Vec<f64>, lr: f64) -> Self {
        let mut layers = Vec::new();
        let mut weight_ptr = 0;
        let mut bias_ptr = 0;

        // iterate through layer pairs
        for dims in topology.windows(2) {
            let in_dim = dims[0];
            let out_dim = dims[1];

            let w_size = in_dim * out_dim;
            let mut layer_weights_vec = Vec::with_capacity(w_size);

            for i in 0..w_size {
                let current_idx = weight_ptr + i;
                if current_idx < weights.len() {
                    layer_weights_vec.push(weights[current_idx]);
                }
                else {
                    let fallback = weights.last().copied().unwrap_or(0.0);
                    layer_weights_vec.push(fallback);
                }
            }

            let layer_weights = Array2::from_shape_vec((out_dim, in_dim), layer_weights_vec)
                .expect("Weight vector size mismatch for topology");
            weight_ptr += w_size;

            let b_size = out_dim;
            let mut layer_biases_vec = Vec::with_capacity(b_size);

            for i in 0..b_size {
                let current_idx = bias_ptr + i;
                if current_idx < biases.len() {
                    layer_biases_vec.push(biases[current_idx]);
                }
                else {
                    let fallback = biases.last().copied().unwrap_or(0.0);
                    layer_biases_vec.push(fallback);
                }
            }
            let layer_biases = Array1::from_vec(layer_biases_vec);
            bias_ptr += b_size;

            layers.push(Layer {
                weights: layer_weights,
                biases: layer_biases, });
        }

        NeuralNetwork { layers, lr }
    }

    pub fn relu(z: Array2<f64>) -> Array2<f64> {
        z.mapv(|x| x.max(0.0))
    }

    pub fn relu_deriv(z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    pub fn forward_pass(&self, input: Array2<f64>) -> (Array2<f64>, Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut current_a = input.clone();
        let mut activations = vec![input];
        let mut linears = Vec::new();

        let num_layers = self.layers.len();

        for (i, layer) in self.layers.iter().enumerate() {
            let z = current_a.dot(&layer.weights.t()) + &layer.biases;
            linears.push(z.clone());

            current_a = if i == num_layers - 1 {
                z
            } else {
                Self::relu(z)
            };
            activations.push(current_a.clone());
        }
        (current_a, linears, activations)
    }

    pub fn train(&mut self, x: Array2<f64>, y: Array2<f64>) {
        let batch_size = x.nrows() as f64;
        let (output, linears, activations) = self.forward_pass(x);

        // loss derivative for mse
        let mut error = output - y;

        for i in (0..self.layers.len()).rev() {
            let layer_input = &activations[i];

            // avg grads over the entire batch
            let d_weights = error.t().dot(layer_input) / batch_size;
            let d_biases = error.sum_axis(Axis(0)) / batch_size;

            if i > 0 {
                let z_prev = &linears[i - 1];
                error = error.dot(&self.layers[i].weights) * Self::relu_deriv(z_prev);
            }

            self.layers[i].weights -= &(d_weights * self.lr);
            self.layers[i].biases -= &(d_biases * self.lr);
        }
        println!("Successfully trained the model!");
    }

    pub fn predict(&self, x: Array2<f64>) -> Array2<f64> {
        self.forward_pass(x).0
    }

    pub fn model_to_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self)?;
        File::create(path)?.write_all(json.as_bytes())
    }

    pub fn load_from_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }
}