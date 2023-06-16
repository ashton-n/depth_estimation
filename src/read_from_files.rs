use std::ffi::OsStr;
use std::io;
use std::fs::{read_dir};
use std::path::{Path, PathBuf};

// filters out irrelevant files and groups the green and red files together according
// to the sample name
pub fn get_file_pathes(input_dir: &Path) -> io::Result<Vec<(PathBuf, PathBuf)>> {    
    // read in all files in the input directory
    let entries = read_dir(input_dir).expect("read_dir call failed");

    // unwrap paths from entries into a vector of PathBufs
    let paths = entries.map(|x| x.unwrap().path()).collect::<Vec<PathBuf>>();

    // filter out all files that are not relevant
    let mut data_paths = paths.into_iter()
                              .filter(|path| path.extension() == Some(OsStr::new("dat")))
                              .filter(|path| path.extension() != None)
                              .filter(|path| path
                                .file_stem()
                                .and_then(|stem| stem.to_str())
                                .and_then(|str| str.chars().last())
                                .map(|last_char| last_char == 'n' || last_char == 'd').unwrap())
                              .collect::<Vec<PathBuf>>();

    // sort the paths so that the green and red files are next to each other
    data_paths.sort();

    // split the paths into two vectors, one for green and one for red
    let (green, red): (Vec<PathBuf>, Vec<PathBuf>) = data_paths.into_iter()
                                                   .partition(|path| path
                                                        .file_stem()
                                                        .and_then(|stem| stem.to_str())
                                                        .and_then(|str| str.chars().last())
                                                        .map(|last_char| last_char == 'n')
                                                        .unwrap_or(false)
                                                    );                                                

    // zip the two vectors together to get a vector of tuples
    let sample_pair_paths = green.into_iter()
                                 .zip(red.into_iter())
                                 .collect::<Vec<(_, _)>>();

    // return the vector of tuples
    Ok(sample_pair_paths)
}

// split the pathes of the files to be read, into chunks
// so that each process can read in its own chunk of data.
pub fn portion_input_data (mut sample_pair_paths: Vec<(PathBuf, PathBuf)>, num_procs: usize) 
   -> (Vec<Vec<(PathBuf, PathBuf)>>, Option<(PathBuf, PathBuf)>){
  
     // check if the number of files is divisible by the number of processes
     if sample_pair_paths.len() % num_procs != 0 {
        
        // if not store the remainder data in a separate variable
        // and remove it from the vector of file paths
        let remainder = sample_pair_paths.remove(sample_pair_paths.len()-1);
        
        // find the right chunk size for the number of processes
        let chunk_size = sample_pair_paths.len() / num_procs;
        
        // split the vector of file paths into chunks of size chunk_size
        let chunks = sample_pair_paths.chunks(chunk_size)
                                      .map(|chunk| chunk.to_vec())
                                      .collect::<Vec<_>>();
        // return the chunks and the remainder
        (chunks, Some(remainder))
     } else {
          // find the right chunk size for the number of processes
          let chunk_size = sample_pair_paths.len() / num_procs;

          // split the vector of file paths into chunks of size chunk_size
          let chunks = sample_pair_paths.chunks(chunk_size)
                                        .map(|chunk| chunk.to_vec())        
                                        .collect::<Vec<_>>();
          // return the chunks and no remainder
          (chunks, None)
      }
}

#[test]
fn test_get_file_pathes() {
    let input_dir = PathBuf::from("unit_test_data");
    let file_pathes = get_file_pathes(&input_dir).unwrap();
    assert_eq!(file_pathes.len(), 4);
}