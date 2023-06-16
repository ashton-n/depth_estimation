use std::env;
use std::fmt::Error;
use std::path::PathBuf;

// reads in parameters passed to programme
pub fn read_term_args() -> Result<(PathBuf, PathBuf), Box<dyn std::error::Error>> {
    
    // get environment arguments
    let args: Vec<String> = env::args().collect();
    
    
    
    match args.len() {
        2 => {
            let command = std::env::args().nth(1).unwrap();
            if command == "eval" {
                let generated_dm_dir =  std::env::args().nth(2).unwrap();
                let ground_truth_dm_dir =  std::env::args().nth(2).unwrap();
            } else {
                // wrong no. of args
            }
        }
    }


    // check that the correct number of arguments have been passed
    if args.len() != 5 {
        println!("Invalid input: input must be: cargo run -- [left_image path] [right_image path] [ground_truth_image path] [depthmap filename]");
        return Err(Box::new(Error));
    } else {
        
        // get input directory string and convert to PathBuf
        let left_img_dir =          std::env::args().nth(1).expect("No input directory provided");
        let right_img_dir =         std::env::args().nth(2).expect("No input directory provided");
        let ground_truth_img_dir =  std::env::args().nth(3).expect("No input directory provided");
        let depthmap_dir =          std::env::args().nth(4).expect("No input directory provided");


        let left_img_dir = PathBuf::from(&input_dir);
        let right_img_dir = PathBuf::from(&input_dir);
        let ground_truth_img_dir = PathBuf::from(&input_dir);
        let left_img_dir = PathBuf::from(&input_dir);
        
        // get output filename add extention and convert to PathBuf
        let output_filename = std::env::args().nth(2).expect("No output filename provided");
        let mut output_filename = PathBuf::from(output_filename);
        output_filename.set_extension("dat");

        // return input directory and output filename
        Ok((input_dir, output_filename))
    } 
}
