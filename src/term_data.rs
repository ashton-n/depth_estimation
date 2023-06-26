use std::env;
use std::fmt::Error;
use std::path::PathBuf;

// reads in parameters passed to programme
/*pub fn read_term_args() -> Result<(String, [PathBuf;2], Option<String>), Box<dyn std::error::Error>> {
    
    // get environment arguments
    let args: Vec<String> = env::args().collect();
    println!("args: {:?}", args);
    match args.len() {
        // check for -> "eval <generated_depthmap_dir> <ground_truth_depthmap_path>" 
        4 => {
            let command = std::env::args().nth(1).unwrap();
            println!("I get to eval");
            if command == "eval" {
                let generated_depthmap_path =  env::args().nth(2).unwrap();
                let ground_truth_depthmap_path =  env::args().nth(3).unwrap();

                let generated_depthmap_path = PathBuf::from(&generated_depthmap_path);
                let ground_truth_depthmap_path = PathBuf::from(&ground_truth_depthmap_path);

                return Ok((command, [generated_depthmap_path, ground_truth_depthmap_path], None));
            } else {
                return Err(Box::new(Error));
            }
        }
        // check for -> "depthmap <left_img_path> <right_img_path> <output_fn>"
        5 => {
            let command = std::env::args().nth(1).unwrap();
            //println!("I get to depthmap");
            if command == "depthmap" {
                let left_img_path =  env::args().nth(2).unwrap();
                let right_img_path =  env::args().nth(3).unwrap();
                let output_fn =  env::args().nth(4).unwrap();

                let left_img_path = PathBuf::from(&left_img_path);
                let right_img_path = PathBuf::from(&right_img_path);
                

                return Ok((command, [left_img_path, right_img_path], Some(output_fn)));
            } else {
                return Err(Box::new(Error));
            }
        }
        _ => return Err(Box::new(Error)),
    }
}*/

pub fn read_term_args() -> Result<(String, PathBuf, PathBuf), Box<dyn std::error::Error>> {
    
    // get environment arguments
    let args: Vec<String> = env::args().collect();
    println!("args: {:?}", args);
    match args.len() {
        4 => {
            let command = std::env::args().nth(1).unwrap();
            if command != "eval" || command != "depthmap"  {

                let data_dir =  env::args().nth(2).unwrap();
                let generated_depthmap_path =  env::args().nth(3).unwrap();

                let data_dir_path = PathBuf::from(&data_dir);
                let generated_depthmap_path = PathBuf::from(&generated_depthmap_path);
                return Ok((command, data_dir_path, generated_depthmap_path));
            } else {
                return Err(Box::new(Error));
            }
        }
        _ => return Err(Box::new(Error)),
    }
}