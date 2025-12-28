// pub mod multi_layer_nn;
pub mod neuro_evo;
pub mod actuary;

use csv;
use serde::Deserialize;
use std::error::Error;
use ndarray::Array2;

#[derive(Clone)]
pub struct DataBase {
    pub records: Vec<Record>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Record {
    // pub index: i32,
    pub symbol: String,
    pub timestamp: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub trade_count: f64,
    pub vwap: f64,
}

impl Record {
    pub fn to_ndarray(&self) -> Array2<f64> {
        // creates a 1d array of the 7 features
        let data = vec![
            self.open as f64,
            self.close as f64,
            self.high as f64,
            self.low as f64,
            self.vwap as f64,
            self.trade_count as f64,
            self.volume as f64,
        ];

        // shape to (7, 1) for the 7 inputs and 1 output
        Array2::from_shape_vec((7, 1), data).expect("Failed to create ndarray from Record")
    }
}

impl DataBase {
    pub fn load_from_csv(path: String) -> Result<Self, Box<dyn Error>> {
        println!("Attempting to read {}", path);
        let mut reader = csv::Reader::from_path(path)?;
        let mut records = Vec::new();
        let mut error_count = 0;

        // Use reader.deserialize() but don't use the '?' operator inside the loop
        for (line_num, result) in reader.deserialize().enumerate() {
            match result {
                Ok(record) => {
                    records.push(record);
                }
                Err(e) => {
                    error_count += 1;
                    // We print the error but DO NOT return, so the loop continues
                    println!("Row {} skipped: Error parsing data - {}", line_num + 1, e);
                }
            }
        }

        if error_count > 0 {
            println!("Finished loading with {} errors skipped.", error_count);
        }

        Ok(DataBase { records })
    }

    pub fn stat_avg(&self) {
        if self.records.is_empty() {
            println!("Database is empty.");
            return;
        }

        let len_f = self.records.len() as f64;
        //let len_i = self.records.len() as i64;

        println!("Statistics for {} Records:", self.records.len());
        println!("You are trading in {}", self.records[0].symbol);

        let avg_open: f64  = self.records.iter().map(|r| r.open).sum::<f64>() / len_f;
        let avg_high: f64  = self.records.iter().map(|r| r.high).sum::<f64>() / len_f;
        let avg_low: f64   = self.records.iter().map(|r| r.low).sum::<f64>() / len_f;
        let avg_close: f64 = self.records.iter().map(|r| r.close).sum::<f64>() / len_f;
        let avg_vwap: f64  = self.records.iter().map(|r| r.vwap).sum::<f64>() / len_f;

        println!("Avg Open:       {:.2}", avg_open);
        println!("Avg High:       {:.2}", avg_high);
        println!("Avg Low:        {:.2}", avg_low);
        println!("Avg Close:      {:.2}", avg_close);
        println!("Avg VWAP:       {:.4}", avg_vwap);

        let total_vol: i64 = self.records.iter().map(|r| r.volume as i64).sum();
        let total_tc: i64  = self.records.iter().map(|r| r.trade_count as i64).sum();

        println!("Avg Volume:     {}", total_vol / self.records.len() as i64);
        println!("Avg Trade Cnt:  {}", total_tc / self.records.len() as i64);
    }

    pub fn head(&self) {
        for i in 0..5 {
            println!("{:?}", self.records[i]);
        }
    }

    pub fn tail(&self) {
        for i in self.records.len() - 5 .. self.records.len() {
            println!("{:?}", self.records[i]);
        }
    }
}

#[derive(Clone)]
pub struct Scaler {
    pub min: f64,
    pub max: f64,
}

impl Scaler {
    pub fn new(data: &[f64]) -> Self {
        let min = data.iter().filter(|x| !x.is_nan()).copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().filter(|x| !x.is_nan()).copied().fold(f64::NEG_INFINITY, f64::max);
        Scaler { min, max }
    }

    pub fn transform(&self, value: f64) -> f64 {
        if self.max == self.min { return 0.5; }
        (value - self.min) / (self.max - self.min)
    }

    pub fn reverse(&self, scaled_value: f64) -> f64 {
        scaled_value * (self.max - self.min) + self.min
    }
}
