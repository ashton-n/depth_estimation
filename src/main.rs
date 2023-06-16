extern crate image;

use image::{GenericImageView, ImageBuffer, Pixel, ImageFormat};
use std::fs::File;
mod term_data;

fn main() {
    
    //let (img_dir, depthmap_name) = term_data::read_term_args()
    //    .expect("Invalid input: input must be: cargo run -- [images directory] [output filename of depthmap]");

    let file = "data/center.ppm";
    let img = image::open(file).unwrap();
   

    let (w, h) = img.dimensions();
    let x = img.get_pixel(0, 0); 
    

    println!("width: {}, height: {}", w, h);
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
