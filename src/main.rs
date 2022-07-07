use arrow::array::Float32Array;
use arrow::compute::sum;
use parquet::file::reader::SerializedFileReader;
use parquet::arrow::{ParquetFileArrowReader, ArrowReader};
use std::sync::Arc;
use std::fs::File;
use std::time::Instant;

use tantivy::Index;
use tantivy::fastfield::FastFieldReader;

fn parquet_test() {
    let start = Instant::now();
    let file = File::open("./data/noaa-1m.parquet").expect("Parquet data no found");
    let file_reader = SerializedFileReader::new(file).expect("Can not open reader for Parquet data");
    let mut arrow_reader = ParquetFileArrowReader::new(Arc::new(file_reader));

    let record_batch_reader = arrow_reader.get_record_reader(1000000).unwrap();

    println!("Parquet Setup {:?}", start.elapsed());

    for maybe_record_batch in record_batch_reader {
        let record_batch = maybe_record_batch.unwrap();
        let column = record_batch.column(2).as_ref();

        let column = column.as_any()
            .downcast_ref::<Float32Array>()
            .expect("Failed to downcast");

        let start = Instant::now();
        let mut total = 0.0;
        for i in 0..column.len() {
            total = total + column.value(i);
        }

        println!("Parquet / Arrow Loop Sum: {:?}", start.elapsed());

        let start = Instant::now();
        let total2 = sum(column).unwrap() as f64;

        println!("Parquet / Arrow Compute Kernel sum(): {:?}", start.elapsed());
        // println!("Totals: {} vs {}", total2, total);
    }
}

fn tantivy_test() {
    let start = Instant::now();
    let index = Index::open_in_dir("./data/noaa-1m").expect("Could not open Tantivy index.");
    let schema = index.schema();
    let temperature = schema.get_field("temperature_c").expect("Error accessing temperature_c field");

    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    let readers = searcher.segment_readers();

    let fast_fields = readers[0].fast_fields();
    println!("Tantivy Setup {:?}", start.elapsed());

    let fast_field = fast_fields.f64(temperature).unwrap();

    let start = Instant::now();
    let mut total = 0.0;
    for i in 0..readers[0].num_docs() {
        total += fast_field.get(i);
    }

    println!("Tantivy Fast Field Loop Sum {:?}", start.elapsed());
    // println!("Total: {}", total);
    let segments = index.searchable_segments().unwrap();
    assert_eq!(segments.len(), 1);
}


fn main() {
    println!("First Pass - Startup Results");
    println!("==================");
    parquet_test();
    println!("------------------");
    tantivy_test();

    println!("");
    println!("Second Pass - Warmed Results");
    println!("==========================");
    parquet_test();
    println!("------------------");
    tantivy_test();
}
