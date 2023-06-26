extern crate image;

use image::{GenericImageView, ImageBuffer, Pixel, ImageFormat, Luma, DynamicImage};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::{fs::File, time::Instant};
mod term_data;
mod image_ops;
mod read_from_files;

fn main() {
    //command, data_dir_path, generated_depthmap_path
    let (cmd, data_dir_path, output_fn_or_gen_dm_path) = term_data::read_term_args()
        .expect("\n\nInvalid input: input must be: ./target/release/depth_estimation [command] [data directory] [output filename of depthmap]\n\n");

    let (left_img_path, 
         right_img_path, 
         dmin_path, 
         disp1_path, 
         _disp5_path) = read_from_files::get_data_paths(&data_dir_path).unwrap();

    match &cmd[..] {
        "depthmap" => {
            println!("Do depthmap Calcs");
            image_ops::get_depthmap(&left_img_path, &right_img_path, &dmin_path);
        },
        "eval" => {
            println!("Do eval Calcs");
            //let eval: String = evaluate(generated_depthmap_path, ground_truth_depthmap_path)
        },
        _ => println!("Invalid Command: must be 'eval' or 'depthmap'")
    }

    //let file = "data/center.ppm";
    //let img = image::open(file).unwrap();
   

    //let (w, h) = img.dimensions();
    //let x = img.get_pixel(0, 0); 
    

    //println!("width: {}, height: {}", w, h);
    //let mut output = ImageBuffer::new(w, h); // create a new buffer for our output


    //let fout = &mut File::create(&format!("{}.png", file)).unwrap();
    //for (x, y, pixel) in img.pixels() {
    //    output.put_pixel(x, y, 
            // pixel.map will iterate over the r, g, b, a values of the pixel
    //        pixel.map(|p| p.saturating_sub(100))
    //    );
    //}

    //output.save("depth_output_test.png").unwrap();

}
