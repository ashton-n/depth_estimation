use std::ffi::OsStr;
use std::io::{self, BufReader, BufRead, Error, ErrorKind};
use std::fs::{read_dir, File};
use std::path::{Path, PathBuf};

// for Middlebury 2014
pub fn get_data_paths(data_dir: &PathBuf) -> io::Result<(PathBuf, PathBuf, PathBuf, PathBuf, PathBuf)> {
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

// for Middlebury 2014


// for Middlebury 2014
pub fn get_calib_info(calibration_info_path: &PathBuf) -> std::io::Result<()> {
    
    let file = File::open(calibration_info_path).expect("file not found!");
    let buf_reader = BufReader::new(file);
    
    buf_reader.lines().enumerate().for_each(|(line_no, line)|{
        let line = line.unwrap();
        println!("line_no: {:?} {:?}", line_no, &line[0..5]);
    
        if line_no == 0 {
            let cam0 = &line[line.find("=").unwrap()..line.len() - 1];
                                              //.expect("Cannot Parse Calib.txt: No vector collection found");
        } else if line_no == 1 {
            let cam1 = &line[line.find("=").unwrap()..line.len() - 1];
        } else {
            println!("End of Conditional");
        }

        //println!("{:?}", cam0);
            
    });
    Ok(())
       
}


/*
// for Middlebury 2014
pub fn get_data_paths(data_dir: &PathBuf) -> io::Result<(PathBuf, PathBuf, PathBuf)> {
    let mut left_img_path = data_dir.clone();
    left_img_path.push("im0.png");

    let mut right_img_path = data_dir.clone();
    right_img_path.push("im0.png");

    let mut calibration_info_path = data_dir.clone();
    calibration_info_path.push("calib.txt");

    Ok((left_img_path, right_img_path, calibration_info_path))
} 


// for Middlebury 2014
pub fn load_variables_from_file(calibration_info_path: &PathBuf) 
    -> std::io::Result<(f32, f32, f32, usize, usize, usize, usize)> 
    {
    let file = File::open(calibration_info_path)?;
    let reader = BufReader::new(file);

    let mut variables = Vec::new();

    for line in reader.lines() {
        if let Ok(line) = line {
            let parts: Vec<&str> = line.split('=').map(|s| s.trim()).collect();
            if parts.len() == 2 {
                let variable_name = parts[0].to_string();
                let variable_value = parts[1].trim_matches(|c| c == '[' || c == ']').to_string();
                variables.push((variable_name, variable_value));
            }
        }
    }

    //let cam0 = &variables[0].1;
    //println!("cam0: {:?}", cam0);
    

    let cam0 = &variables[0].1.split_whitespace().collect::<Vec<&str>>();
    //println!("cam0: {:?}", &cam0);
    let focal_length = &cam0[0].parse::<f32>().expect("Problem reading focal legnth");
    //println!("focal_legnth: {:?}", &focal_legnth);

    //let cam1 = &variables[1].1;
    //println!("cam1: {:?}", cam1);
    let doffs = &variables[2].1.parse::<f32>().expect("Problem reading focal legnth");
    //println!("doffs: {:?}", doffs);
    let baseline = &variables[3].1.parse::<f32>().expect("Problem reading focal legnth");
    //println!("baseline: {:?}", baseline);
    let width = &variables[4].1.parse::<usize>().expect("Problem reading focal legnth");
    //println!("width: {:?}", width);
    let height = &variables[5].1.parse::<usize>().expect("Problem reading focal legnth");
    //println!("height: {:?}", height);
    let ndisp = &variables[6].1.parse::<usize>().expect("Problem reading focal legnth");
    //println!("ndisp: {:?}", ndisp);
    let isint = &variables[7].1.parse::<usize>().expect("Problem reading focal legnth");
    //println!("isint: {:?}", isint);
    Ok((*focal_length, *doffs, *baseline, *width, *height, *ndisp, *isint))
    
}

// for Middlebury 2014
pub fn get_calib_info(calibration_info_path: &PathBuf) -> std::io::Result<()> {
    
    let file = File::open(calibration_info_path).expect("file not found!");
    let buf_reader = BufReader::new(file);
    
    buf_reader.lines().enumerate().for_each(|(line_no, line)|{
        let line = line.unwrap();
        println!("line_no: {:?} {:?}", line_no, &line[0..5]);
    
        if line_no == 0 {
            let cam0 = &line[line.find("=").unwrap()..line.len() - 1];
                                              //.expect("Cannot Parse Calib.txt: No vector collection found");
        } else if line_no == 1 {
            let cam1 = &line[line.find("=").unwrap()..line.len() - 1];
        } else {
            println!("End of Conditional");
        }

        //println!("{:?}", cam0);
            
    });
    Ok(())
       
}*/
