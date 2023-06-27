use std::io::{BufReader, BufRead, Error, ErrorKind, Result};
use std::fs::File;
use std::path::PathBuf;

pub fn get_data_paths(data_dir: &PathBuf) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf, PathBuf)> {
    let mut left_img_path = data_dir.clone();
    left_img_path.push("view1.png");

    let mut right_img_path = data_dir.clone();
    right_img_path.push("view5.png");

    let mut dmin_path = data_dir.clone();
    dmin_path.push("dmin.txt");

    let mut disp1_path = data_dir.clone();
    disp1_path.push("disp1.png");

    let mut disp5_path = data_dir.clone();
    disp5_path.push("disp5.png");

    //let mut full_size_path = data_dir.clone();
    //disp5_path.push("disp5.png");

    Ok((left_img_path, right_img_path, dmin_path, disp1_path, disp5_path))
} 

pub fn get_dmin(dmin_path: &PathBuf) -> std::io::Result<u32> {
    let file = File::open(dmin_path)?;
    let reader = BufReader::new(file);

    let no_lines = &reader.lines().count();
    match no_lines {
        1 => {
                let file = File::open(dmin_path)?;
                let reader = BufReader::new(file);
                let dmin = reader.lines().next().unwrap().unwrap().parse::<u32>().unwrap(); 
                return Ok(dmin)
             },
        _ => return Err(Error::new(ErrorKind::Other, "Unexpected number of lines")),
    }
    
}

pub fn get_original_dims(data_dir: &PathBuf) -> (u32, u32) {
    let filename = data_dir.file_name().expect("Could not get file name for full-size version");

    let new_filename = format!("{}-2views", filename.to_string_lossy());

    let full_size_path = data_dir
        .parent()
        .and_then(|parent| parent.parent())
        .and_then(|parent| parent.parent())
        .map(|parent| parent.join("FullSize"))
        .unwrap_or_else(|| PathBuf::from("FullSize"))
        .join(&new_filename)
        .join(filename)
        .join("view1.png");

    let (width, height) = image::image_dimensions(full_size_path).unwrap();

    (width, height)
}