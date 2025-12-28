use lib::{DataBase, Scaler, neuro_evo::*, actuary::*};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = r"THE_PATH_TO_YOUR_CSV_FILE".to_string();
    let path_pred = r"THE_PATH_TO_YOUR_CSV_FILE_(PREDICTION)".to_string();

    // 1. Load Databases
    let db = DataBase::load_from_csv(path.clone())?;
    let db_pred = DataBase::load_from_csv(path_pred.clone())?;
    println!("Loaded {} records for training and {} for prediction", db.records.len(), db_pred.records.len());

    // 2. Setup Scalers based on training data
    let closes: Vec<f64> = db.records.iter().map(|r| r.close).collect();
    let s_close = Scaler::new(&closes);
    let s_open = Scaler::new(&db.records.iter().map(|r| r.open).collect::<Vec<_>>());
    let s_high = Scaler::new(&db.records.iter().map(|r| r.high).collect::<Vec<_>>());
    let s_low = Scaler::new(&db.records.iter().map(|r| r.low).collect::<Vec<_>>());

    let total_records = db_pred.records.len();
    let window_size = 5;
    let look_ahead = 60; // predicts 60 minutes into the future
    let sequence_len = 15;
    let batch_size = 20; // Number of different time windows to train on

    if total_records <= (window_size + look_ahead + sequence_len + batch_size) {
        panic!("Not enough data records to create a batch of windows.");
    }

    let max_possible_samples = total_records - (window_size + look_ahead + sequence_len) - 1;

    let training_size = 300.min(max_possible_samples);

    let mut batch_inputs = Vec::with_capacity(batch_size);
    let mut batch_targets = Vec::with_capacity(batch_size);

    for i in 0..training_size {
        // We slide the 'end point' back one record at a time
        let target_end = total_records - i;
        let target_start = target_end - sequence_len;
        let window_end = target_start - look_ahead;
        let window_start = window_end - window_size;

        // Collect target sequence
        let target_sequence: Vec<f64> = db_pred.records[target_start..target_end]
            .iter()
            .map(|r| s_close.transform(r.close))
            .collect();

        // Collect features
        let mut window_features = Vec::with_capacity(window_size * 4);
        for record in &db_pred.records[window_start..window_end] {
            window_features.push(s_open.transform(record.open));
            window_features.push(s_high.transform(record.high));
            window_features.push(s_low.transform(record.low));
            window_features.push(s_close.transform(record.close));
        }

        batch_inputs.push(window_features);
        batch_targets.push(target_sequence);
    }

    let deletion_standard = 0.25;
    let mut pop = Population::new(deletion_standard);

    let load_from_json = false;
    let topology = vec![20, 64, 32, 16, 8, 4, sequence_len];
    let pop_size = 50;
    let generations = 100;

    if load_from_json == false {
        let mut ancestor_nn: NeuralNetwork = NeuralNetwork::new(&topology, 0.001);
        let training_size = batch_inputs.len(); // Use the actual length of the harvested data
        let input_dim = 20; // window_size * 4
        let output_dim = sequence_len;

        // 1. Flatten and check inputs
        let flat_inputs: Vec<f64> = batch_inputs.clone().into_iter().flatten().collect();
        let x = Array2::from_shape_vec((training_size, input_dim), flat_inputs)?;

        // 2. Flatten and check targets
        let flat_targets: Vec<f64> = batch_targets.clone().into_iter().flatten().collect();
        let y = Array2::from_shape_vec((training_size, output_dim), flat_targets)?;

        println!("Training shapes: X={:?}, Y={:?}", x.dim(), y.dim());

        ancestor_nn.train(x, y);
        ancestor_nn.model_to_json(r"THE_PATH_TO_YOUR_CSV_FILE")?;
    }

    let ancestor = Genome::load_genome_from_json(
        r"THE_PATH_TO_YOUR_CSV_FILE".into(),
        0,
        0.0
    ).expect("Failed to load starting ancestor");

    for _ in 0..pop_size {
        let mut ancestor_clone = ancestor.clone();
        let mutated_ancestor = ancestor_clone.mutate(
            topology.clone(),
            batch_inputs.clone(),
            batch_targets.clone(),
        );
        pop.population.push(mutated_ancestor);
    }

    println!("Starting Competitive Evolution for {} generations...", generations);
    println!("Latest Target price: ${:.2}", db_pred.records[total_records - 1].close);

    for gen in 0..generations {
        pop.start_generation(
            topology.clone(),
            pop_size,
            batch_targets.clone(),
            batch_inputs.clone()
        );

        if let Some(best) = pop.population.first() {
            println!("Gen {}: Best Batch MSE: {:.8}", gen, best.fitness);
            // continue;
        }
    }

    if let Some(best_genome) = pop.population.first() {
        let latest_input = &batch_inputs[0];
        let input_matrix = Array2::from_shape_vec((1, 20), latest_input.clone())?;
        let scaled_output = best_genome.phenotype.predict(input_matrix);
        let mut output: Vec<f64> = Vec::new();

        println!("\n--- 15-Step Sequence Forecast (Starting {} mins from now) ---", look_ahead);
        for i in 0..sequence_len {
            let actual = s_close.reverse(batch_targets[0][i]);
            let predicted = s_close.reverse(scaled_output[[0, i]]);
            output.push(predicted);
            println!("T+{:02}: Actual: ${:.2} | Pred: ${:.2} | Diff: ${:.2}", look_ahead + i, actual, predicted, (actual - predicted).abs());
        }
        best_genome.phenotype.model_to_json(r"THE_PATH_TO_YOUR_CSV_FILE")?;
        let buy_or_sell: bool = decide_purchase(output);
    }

    Ok(())
}