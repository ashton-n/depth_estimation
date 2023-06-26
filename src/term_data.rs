use std::env;
use std::fmt::Error;
use std::path::PathBuf;

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