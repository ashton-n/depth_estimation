extern crate image;

use std::{path::PathBuf, io::Cursor, time::Instant};

use image::{GenericImageView, ImageBuffer, Pixel, ImageFormat, DynamicImage, Rgb, GenericImage, Luma, RgbImage, ImageResult, Rgb32FImage};
use rayon::prelude::*;

use crate::read_from_files::{self, get_dmin};
// use image2::io::pfm::{save_pfm, PfmData};
// NB!! Requires "sudo apt install libopenimageio-dev"
// NB!! Requires +1
// NB!! Requires +2

//use opencv::core::{Mat, CV_32FC1};
//use opencv::imgcodecs::imwrite;
// NB!! Requires "apt install libopencv-dev clang libclang-dev"


//pub fn load_img()

pub fn pad_img(img: &ImageBuffer<Luma<u8>, Vec<u8>>, border_depth: &u32) -> ImageBuffer<Luma<u8>, Vec<u8>>{

    let (img_w, img_h) = img.dimensions(); //returns the width then the height of the image
    let img_w = img_w + border_depth*2;
    let img_h = img_h + border_depth*2;
    // NB! this is zero padding
    let mut image_buf: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_pixel(img_w, img_h, Luma([0]));
    image_buf.copy_from(img, *border_depth, *border_depth).expect("Could not copy");
    image_buf
}

pub fn get_pxl_block_idxs(img: &ImageBuffer<Luma<u8>, Vec<u8>>, kernal_size: &u32) -> Vec<Vec<Vec<Vec<u32>>>> {
    //img should be unpadded
    let pxl_block_idxs = img.enumerate_pixels().map(|(x, y, _ )| {
        let x_vec = (x..(x + kernal_size)).collect::<Vec<u32>>();
        let y_vec = (y..(y + kernal_size)).collect::<Vec<u32>>();
        vec![x_vec, y_vec]
    }).collect::<Vec<Vec<Vec<u32>>>>();

    pxl_block_idxs.chunks(img.width() as usize)
                  .map(|x| x.to_vec())
                  .collect::<Vec<Vec<Vec<Vec<u32>>>>>()
    
}

pub fn get_pxl_blocks_vec(img: &ImageBuffer<Luma<u8>, Vec<u8>>, kernal_size: &u32) -> Vec<Vec<Vec<u8>>> {
    // this fetches a vector of sub image blocks, where each block is kernel_size x kernel_size in size
    // NOTE! img gets padded with zeros

    let border_depth = kernal_size/2;
    
    let padded_img = pad_img(img, &border_depth);

    let pxl_idxs = get_pxl_block_idxs(img, kernal_size);
    
    let vec_blocks = img.enumerate_pixels().map(|(cols, rows, _ )| {

        // fetch pixel indices for the block around the current pixel 
        let yx_idx_vecs = &pxl_idxs[rows as usize][cols as usize];
        
        //let block = vec_yx_idxs[0].iter().zip(vec_yx_idxs[1].iter()).map(|(y, x)| {
        let block = yx_idx_vecs[0].par_iter().map(|&y| {
            let block_row = yx_idx_vecs[1].par_iter().map(|&x| {
                let val = padded_img.get_pixel_checked(y, x)
                                    .unwrap_or_else(|| {
                                        println!("Could not get pixel: invalid index x: {:?} y: {:?}", x, y);
                                        panic!();
                                    });
                let x = val[0];
                x
            }).collect::<Vec<u8>>();
            block_row
        }).collect::<Vec<Vec<u8>>>();

        block

    }).collect::<Vec<Vec<Vec<u8>>>>();
    vec_blocks
}

pub fn validate_block_size(block_size: &u32) -> bool {
    //println!("{:?}", kernal_size);
    //println!("{:?}", *kernal_size % 2_u32 != 0_u32);
    //println!("{:?}", *kernal_size > 3_u32);

    assert!(*block_size % 2_u32 != 0_u32, "kernal_size/block_size must be odd");
    assert!(*block_size >= 3_u32, "kernal_size/block_size must be an u32 that is 3 or greater");
    true
}


// this function splits a DynamicImage into its RGB channels
pub fn get_channels(img: &DynamicImage) -> 
    Vec<ImageBuffer<Luma<u8>, Vec<u8>>>
    {
    
    let img = img.to_rgb8();
    
    let mut red = ImageBuffer::from_pixel(img.width(), img.height(), Luma([0]));
    let mut green = ImageBuffer::from_pixel(img.width(), img.height(), Luma([0]));
    let mut blue = ImageBuffer::from_pixel(img.width(), img.height(), Luma([0]));

    img.enumerate_pixels().for_each(|(x, y, pxl )| {
        let pxl = pxl.to_rgb();
        red.put_pixel(x, y, Luma([pxl[0]]));
        green.put_pixel(x, y, Luma([pxl[1]]));
        blue.put_pixel(x, y, Luma([pxl[2]]));
    });

    vec![red, green, blue]

}

pub fn get_similarity_value(left_block: &Vec<u8>, right_block: &Vec<u8>) -> i32 {
    // this function returns the similarity value between two blocks
    // the similarity value is the sum of the absolute difference between each pixel in the block
    // the similarity value is a u8, so it is between 0 and 255
    assert_eq!(left_block.len(), right_block.len(), "left and right blocks must be the same size");
    let similarity_value = left_block.into_par_iter()
                                     .zip(right_block.into_par_iter())
                                     .map(|(left_pxl, right_pxl)| {
                                            (*left_pxl as i32 - *right_pxl as i32).abs()
                                        })
                                     .sum::<i32>();
    similarity_value
}

pub fn print_img_buf_luma_u8(img: &ImageBuffer<Luma<u8>, Vec<u8>>) {
    println!("");
    img.rows().for_each(|row|{
        row.into_iter().for_each(|pxl| {
            print!("{:?} ", pxl[0]);
        });
        println!("");
    });
    println!("");
}

pub fn compute_disparity_map(left_img: &ImageBuffer<Luma<u8>, Vec<u8>>, 
    right_img: &ImageBuffer<Luma<u8>, Vec<u8>>, 
    block_size: &u32) 
    -> Vec<Vec<u32>> {

    assert!(validate_block_size(block_size), "in Correspondence Matching");
    assert_eq!(left_img.dimensions(), right_img.dimensions(), "Correspondence Matching: Left and Right images must be the same dimensions");

    let (width, height) = left_img.dimensions();

    let border_depth = block_size/2;
    let left_padded = pad_img(left_img, &border_depth);
    let right_padded = pad_img(right_img, &border_depth);

    //print_img_buf_luma_u8(&left_padded);
    //print_img_buf_luma_u8(&right_padded);

    // x and y reference the reference image pixel
    let disparity_map = (0..height).map(|ref_y| {
        //let find_row_disp_per_pxl = Instant::now();
        let disparity_row = (0..width).into_par_iter().map(|ref_x|{
            // x and y are the ref_block coordinates
            //println!("ref_y: {:?} ref_x: {:?}", ref_y, ref_x);
            //let find_row_disp_per_pxl = Instant::now();
            
            let lowest_sad_block_idx = (0..width).into_par_iter().map(|cmp_block_x|{ 
                //println!("{:?}", cmp_block_x);
                
                let left_block_vals = (ref_y..ref_y + block_size).flat_map(|padded_y|{
                    let left_val_row = (ref_x..ref_x + block_size).map(|padded_x|{
                        let left_val = left_padded.get_pixel(padded_x, padded_y)[0];
                                //.expect("Invalid pixel co-ords")[0];
                        left_val    
                    }).collect::<Vec<u8>>();
                    left_val_row
                }).collect::<Vec<u8>>();

                let right_block_vals = (ref_y..ref_y + block_size).map(|padded_y|{
                    let right_val_row = (cmp_block_x..cmp_block_x + block_size).map(|padded_x|{
                        let right_val = right_padded.get_pixel(padded_x, padded_y)[0];
                                //.expect("Invalid pixel co-ords")[0];
                        right_val    
                    }).collect::<Vec<u8>>();
                    right_val_row
                }).flatten().collect::<Vec<u8>>();
                
                let sad_block_score: usize = left_block_vals.iter()
                                                     .zip(right_block_vals.iter())
                                                     .map(|(&y, &x)| {
                                                        y.abs_diff(x) as usize
                                                     }).sum();  
                //println!("left_block_vals: {:?}", left_block_vals);
                //println!("right_block_vals: {:?}", right_block_vals);
                //println!("BLOCK_SAD_SCORE: {:?}", sad_block_score);

                (cmp_block_x, sad_block_score)
            }).min_by_key(|(_cmp_block_x, sad_score_block)| sad_score_block.clone()).unwrap().0;
            
            let disparity = ref_x.abs_diff(lowest_sad_block_idx); 
            disparity

        }).collect::<Vec<u32>>();//pixel disparity (keep u32 -?> dependant on image width size)
        //println!("[find_row_disp_per_pxl] Completed in: {:?}", find_row_disp_per_pxl.elapsed());
        disparity_row

    }).collect::<Vec<Vec<u32>>>();
    disparity_map

}

/*pub fn compute_disparity_map(
    left_img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    right_img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    block_size: &u32,
) -> Vec<Vec<u32>> {
    assert!(validate_block_size(block_size), "in Correspondence Matching");
    assert_eq!(
        left_img.dimensions(),
        right_img.dimensions(),
        "Correspondence Matching: Left and Right images must be the same dimensions"
    );

    let (width, height) = left_img.dimensions();

    let border_depth = block_size / 2;
    let left_padded = pad_img(left_img, &border_depth);
    let right_padded = pad_img(right_img, &border_depth);

    let disparity_map = (0..height)
        .map(|ref_y| {
            (0..width)
                .map(|ref_x| {
                    let lowest_sad_block_idx = (0..width)
                        .into_par_iter()
                        .map(|cmp_block_x| {
                            let sad_block_score: usize = (0..*block_size)
                                .into_par_iter()
                                .flat_map(|padded_y| {
                                    left_padded
                                        .enumerate_pixels()
                                        .skip((padded_y + ref_y) as usize * width as usize + ref_x as usize)
                                        .take(*block_size as usize)
                                        .map(|(_, _, p)| p[0])
                                        .zip(
                                            right_padded
                                                .enumerate_pixels()
                                                .skip((padded_y + ref_y) as usize * width as usize + cmp_block_x as usize)
                                                .take(*block_size as usize)
                                                .map(|(_, _, p)| p[0]),
                                        )
                                        .map(|(y, x)| y.abs_diff(x) as usize)
                                        .collect::<Vec<usize>>()
                                })
                                .sum();

                            (cmp_block_x, sad_block_score)
                        })
                        .min_by_key(|(_, sad_score_block)| *sad_score_block)
                        .unwrap()
                        .0;

                    ref_x.abs_diff(lowest_sad_block_idx)
                })
                .collect::<Vec<u32>>()
        })
        .collect::<Vec<Vec<u32>>>();

    disparity_map
}*/

pub fn calc_depthmap(disparity_map: Vec<Vec<u32>>, 
                     focal_length: &u32, 
                     baseline: &u32, 
                     dmin: &u32) 
    -> Vec<Vec<f32>> {
    
    // for Middlebury 2014
    //let (focal_length, 
    //     doffs, 
    //     baseline, 
    //     width, 
    //     height, 
    //     ndisp, 
    //     isint) = read_from_files::load_variables_from_file(&calibration_info_path).unwrap();

    // for Middlebury 2006 (see https://vision.middlebury.edu/stereo/data/scenes2006/)


    let height = disparity_map.len() as u32;
    let width  = disparity_map[0].len() as u32;

    let depth_map = (0..height).map(|y| {
        (0..width).map(|x| {
            let disparity = disparity_map[y as usize][x as usize];
            let depth = (baseline * focal_length) as f32 / (dmin + disparity) as f32;
            //let depth_value = (depth / 10.0) as f32; // Scale the depth for visualization
            depth
        }).collect::<Vec<f32>>()
    }).collect::<Vec<Vec<f32>>>();
    depth_map
}


/*pub fn calc_depthmap(disparity_map: Vec<Vec<Luma<u32>>>, 
                     baseline: &f32, 
                     focal_length: &f32, 
                     doffs: &f32) 
                     -> Vec<Vec<f32>> {

    let height = disparity_map.len() as u32;
    let width  = disparity_map[0].len() as u32;

    let depth_map = (0..height).map(|y| {
        (0..width).map(|x| {
            let disparity = disparity_map[y as usize][x as usize][0] as f32;
            let depth = baseline * focal_length / (doffs + disparity);
            let depth_value = (depth / 10.0) as f32; // Scale the depth for visualization
            depth_value
        }).collect::<Vec<f32>>()
    }).collect::<Vec<Vec<f32>>>();
    depth_map
}*/

fn vec2image(red_pixels: &Vec<Vec<u32>>, green_pixels: &Vec<Vec<u32>>, blue_pixels: &Vec<Vec<u32>>) -> DynamicImage {
    let height = red_pixels.len();
    let width = red_pixels[0].len();
    println!("{width} x {height}");

    let mut image_buffer: ImageBuffer<Rgb<f32>, Vec<f32>> =
        ImageBuffer::new(width as u32, height as u32);

    (0..height).for_each(|y|{
        (0..width).for_each(|x|{
            image_buffer.put_pixel(x as u32,    
                                   y as u32, 
                                   Rgb([red_pixels[y][x] as f32,
                                        green_pixels[y][x] as f32, 
                                        blue_pixels[y][x] as f32]
                                      )
                                  );
        })
    });

    let dynamic_image: DynamicImage = DynamicImage::ImageRgb32F(image_buffer);
    dynamic_image
}

pub fn scale_u32_to_u8(x: &u32, min: &u32, max: &u32) -> u8 {
    let val = ((x-min) * 255_u32) / (max-min)  ;
    let x_scaled_u8 = val.to_be_bytes();
    let ans = x_scaled_u8[3];
    return ans
}

pub fn scale_u32_to_u16(x: &u32, min: &u32, max: &u32) -> u16 {
    let val = ((x-min) * 65536_u32) / (max-min)  ;
    let x_scaled_u8 = val.to_be_bytes();
    let ans = (x_scaled_u8[2] as u16) << 8 | x_scaled_u8[3] as u16;
    return ans
}

fn vec_to_luma16(pixels: &Vec<Vec<u32>>) -> DynamicImage {
    let height = pixels.len();
    let width = pixels[0].len();
    println!("{width} x {height}");

    let mut image_buffer: ImageBuffer<Luma<u16>, Vec<u16>> =
        ImageBuffer::new(width as u32, height as u32);

    let min = pixels.iter().flatten().min().unwrap();
    let max = pixels.iter().flatten().max().unwrap();

    image_buffer.enumerate_pixels_mut().for_each(|(x, y, pxl)| {
        let val = pixels[y as usize][x as usize];
        let val = scale_u32_to_u16(&val, &min, &max);
        //println!("{:?}", val);
        *pxl = Luma([val]);
    });
   
    let dynamic_image: DynamicImage = DynamicImage::ImageLuma16(image_buffer);
    dynamic_image
}

fn vec_to_luma8(pixels: &Vec<Vec<u32>>) -> DynamicImage {
    let height = pixels.len();
    let width = pixels[0].len();
    println!("{width} x {height}");

    let mut image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::new(width as u32, height as u32);

    let min = pixels.iter().flatten().min().unwrap();
    let max = pixels.iter().flatten().max().unwrap();

    image_buffer.enumerate_pixels_mut().for_each(|(x, y, pxl)| {
        let val = pixels[y as usize][x as usize];
        let val = scale_u32_to_u8(&val, &min, &max);
        *pxl = Luma([val]);
    });
   
    let dynamic_image: DynamicImage = DynamicImage::ImageLuma8(image_buffer);
    dynamic_image
}

// Average out f32 channel depthmaps to get the final depth map estimation
// Convert depthmap to u32 and store in Luma<u32> 
pub fn average_ch_depth(red_depths: Vec<Vec<f32>>, 
                        green_depths: Vec<Vec<f32>>, 
                        blue_depths: Vec<Vec<f32>>) 
                        {
    let r_height = red_depths.len();
    let r_width = red_depths[0].len();

    let g_height = green_depths.len();
    let g_width = green_depths[0].len();

    let b_height = blue_depths.len();
    let b_width = blue_depths[0].len();
    
    assert_eq!(r_height, g_height);
    assert_eq!(b_height, g_height);
    
    assert_eq!(r_width, g_width);
    assert_eq!(g_width, b_width);

    let width  = b_width;
    let height = b_height;

    let mut image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::new(width as u32, height as u32);

    let avgs = (0..height).map(|y|{
        (0..width).map(|x|{
            let avg: f32 = (red_depths[y][x] + green_depths[y][x] + blue_depths[y][x]) / 3.0;
            avg
        }).collect::<Vec<f32>>()
    }).collect::<Vec<Vec<f32>>>();


    /*avgs.iter().for_each(|row| {
        println!("MAX: {:?}", row.iter().max_by(|&a, &b| a.partial_cmp(b).unwrap()) );
        println!("MIN: {:?}", row.iter().min_by(|&a, &b| a.partial_cmp(b).unwrap()) );
    });*/

}



//pub fn contrast_stretch(img: Vec<Vec<Luma<u32>>>) {
//
//}

// pub fn filter(img: DynamicImage, kernal: Vec<u8>)
 
pub fn get_depthmap(left_img_path: &PathBuf, right_img_path: &PathBuf, dmin_path: &PathBuf) {
    let left_img = image::open(left_img_path).expect("Path does not exist, make sure to include extension");
    let right_img = image::open(right_img_path).expect("Path does not exist, make sure to include extension");

    // Split image into rgb channels
    let left_rgb_channels = get_channels(&left_img);
    let right_rgb_channels = get_channels(&right_img);
    
    // find disparity between L and R images for each channels (RGB)
    // sub steps
    // choose comparison block size
    let block_size = 5_u32; // this is the size of the block that we will be using to compare pixels 

    let start_corr_matchs = Instant::now();
    let red_disparities = compute_disparity_map(&left_rgb_channels[0], &right_rgb_channels[0], &block_size); 
    let green_disparities = compute_disparity_map(&left_rgb_channels[1], &right_rgb_channels[1], &block_size);
    let blue_disparities = compute_disparity_map(&left_rgb_channels[2], &right_rgb_channels[2], &block_size);
    println!("Completed in: {:?}", start_corr_matchs.elapsed());

    let focal_length = 3740_u32; // in pixels
    let baseline     = 160_u32;  // in mm
    let dmin         = get_dmin(dmin_path).unwrap();

    //let red_depth = calc_depthmap(red_disparities, &focal_length, &baseline, &dmin);
    //let green_depth = calc_depthmap(green_disparities, &focal_length, &baseline, &dmin);
    //let blue_depth = calc_depthmap(red_disparities, &focal_length, &baseline, &dmin);

    //let avgerage_depths = average_ch_depth(red_depth, green_depth, blue_depth);

    //red_disparities.iter().for_each(|row| {
    //    println!("max: {:?}", row.iter().max().unwrap());
    //    println!("min: {:?}\n", row.iter().min().unwrap());
    //});
    

    let saving_image = Instant::now();
    vec_to_luma8(&red_disparities).save("Avg_into_rgb_2006_u8.png").expect("WTF!");
    vec_to_luma16(&red_disparities).save("Avg_into_rgb_2006_u16.tif").expect("WTF!");
    //vec2image(&red_disparities, &green_disparities, &blue_disparities).save("Avg_into_rgb_2014.exr").expect("WTF!");
    //vec2image(&green_disparities).save("green_disparities.exr").expect("WTF!");
    //vec2image(&blue_disparities).save("blue_disparities.exr").expect("WTF!");
    println!("[saving_image] Completed in: {:?}", saving_image.elapsed());

    

    // Compute depth maps for each RGB channel
    // -> here we need the following: baseline, focal legnth and doffs
    /*let baseline = 1_f32;
    let focal_legnth = 2.0_f32;
    let doffs = 0.12_f32;
    let red_depths = calc_depthmap(red_disparities, &baseline, &focal_legnth, &doffs);
    let green_depths = calc_depthmap(green_disparities, &baseline, &focal_legnth, &doffs);
    let blue_depths = calc_depthmap(blue_disparities, &baseline, &focal_legnth, &doffs);

    // average depth maps to create a final f32 depthmap estimation
    let averaged_channel_depths = red_depths.iter()
                                            .zip(green_depths.iter())
                                            .zip(blue_depths.iter())
                                            .map(|((red_row, green_row), blue_row)| {
                                                red_row.iter()
                                                       .zip(green_row.iter())
                                                       .zip(blue_row.iter())
                                                       .map(|((red, green), blue)| {
                                                            (red + green + blue) / 3.0
                                                        }).collect::<Vec<f32>>()
                                            }).collect::<Vec<Vec<f32>>>();*/
    //

    //averaged_channel_depths.iter().for_each(|row|{
    //    println!("{:?}", row);
    //})
    //let width = red_depths[0].len();
    //let height = red_depths.len();
    //let image_buffer: ImageBuffer<Rgb<f32>, Vec<f32>> = ImageBuffer::from_raw(width as u32, height as u32, red_depths.concat()).expect("Failed to create image buffer");

    //let image: RgbImage = vec2image(&averaged_channel_depths).expect("Failed to create image from data");
    //let dynamic_image: DynamicImage = DynamicImage::ImageRgb8(image);
                                            
    //let save_result = dynamic_image.save("float_depth_map.png"); // add time stamp
    //match save_result {
    //    Ok(_) => println!("Image saved successfully."),
    //    Err(err) => eprintln!("Failed to save image: {}", err),
    //}

    
    // zip left and right rgb channel pairs
    //let rgb_channels = left_rgb_channels.iter().zip(right_rgb_channels.iter()).collect::<Vec<_>>();
    
    //disparity_img.save("depth_output_test.png").unwrap();

}

//#[test]
//fn test_scale_u32_to_u8() {
//    let test_vars = vec![];
//    scale_u32_to_u8(x: &u32, min: &u32, max: &u32)
//}

#[test]
fn test_compute_disparity_map() {
    let mut left_img = ImageBuffer::new(5, 5);
    let mut right_img = ImageBuffer::new(5, 5);

    //Sets the pixel intensities
    left_img.enumerate_pixels_mut().for_each(|(x, y, pxl)| {
        let val = 1_u8 + ((y*5) + x) as u8;
        *pxl = Luma([val]);
        print!("{:?} ", val);
    });
    println!("");
    right_img.enumerate_pixels_mut().for_each(|(x, y, pxl)| {
        let val = 26_u8 - (1 + ((y*5) + x)) as u8;
        *pxl = Luma([val]);
        print!("{:?} ", val);
    });
    println!("");


    let block_size = 3_u32;

    let disparities = compute_disparity_map(&left_img, &right_img, &block_size);

    let check = vec![vec![Luma([0_u32]);5];5];

    disparities.iter().for_each(|row|{
        println!("{:?}", row);
    });

    //println!("disparities: {:?}", disparities);
    //println!("check: {:?}", check);

    //assert_eq!(disparities, check);

}
    
 
#[test]
fn test_pad_img() {
    let img : ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(4, 5);
    let border_depth = 3_u32;
    let img_pad = pad_img(&img, &border_depth);
    let x = &img_pad.clone().into_vec();
    //x.iter().for_each(|x| {
    //    print!("{x}");
    //});
    //print!("{:?}", img_pad.dimensions());
    assert_eq!((img.width() + &border_depth*2, img.height() + &border_depth*2), 
               (img_pad.width(), img_pad.height())
              );
    
}

/*fn test_compute_disparity_map() {

    let left_img = ImageBuffer::new(5, 5);
    let right_img = ImageBuffer::new(5, 5);

    let disparity = compute_disparity_map(&left_img, &right_img, &3_u32);

    disparity.iter().for_each(|row|{
        println!("{:?}", row);
    });
        
}*/
/*
#[test]
fn test_get_pxl_block_idxs() {
    let img = ImageBuffer::new(5, 5);
    let kernal_size = 3_u32;
    let pxl_block_idxs = get_pxl_block_idxs(&img, &kernal_size);

    assert_eq!(pxl_block_idxs.len(), img.height() as usize);
    
    pxl_block_idxs.iter().enumerate().for_each(|(row_id, rows)| {
        //println!("{:?}", rows);
        
        assert_eq!(rows[0][0], vec![0, 1, 2]);
        assert_eq!(rows[1][0], vec![1, 2, 3]);
        assert_eq!(rows[2][0], vec![2, 3, 4]);
        assert_eq!(rows[3][0], vec![3, 4, 5]);
        assert_eq!(rows[4][0], vec![4, 5, 6]);
        
        match row_id {
            0 => {
                assert_eq!(rows[0][1], vec![0, 1, 2]);
                assert_eq!(rows[1][1], vec![0, 1, 2]);
                assert_eq!(rows[2][1], vec![0, 1, 2]);
                assert_eq!(rows[3][1], vec![0, 1, 2]);
                assert_eq!(rows[4][1], vec![0, 1, 2]);
            },
            1 => {
                assert_eq!(rows[0][1], vec![1, 2, 3]);
                assert_eq!(rows[1][1], vec![1, 2, 3]);
                assert_eq!(rows[2][1], vec![1, 2, 3]);
                assert_eq!(rows[3][1], vec![1, 2, 3]);
                assert_eq!(rows[4][1], vec![1, 2, 3]);
            },
            2 => {
                assert_eq!(rows[0][1], vec![2, 3, 4]);
                assert_eq!(rows[1][1], vec![2, 3, 4]);
                assert_eq!(rows[2][1], vec![2, 3, 4]);
                assert_eq!(rows[3][1], vec![2, 3, 4]);
                assert_eq!(rows[4][1], vec![2, 3, 4]);
            },
            3 => {
                assert_eq!(rows[0][1], vec![3, 4, 5]);
                assert_eq!(rows[1][1], vec![3, 4, 5]);
                assert_eq!(rows[2][1], vec![3, 4, 5]);
                assert_eq!(rows[3][1], vec![3, 4, 5]);
                assert_eq!(rows[4][1], vec![3, 4, 5]);
            },
            4 => {
                assert_eq!(rows[0][1], vec![4, 5, 6]);
                assert_eq!(rows[1][1], vec![4, 5, 6]);
                assert_eq!(rows[2][1], vec![4, 5, 6]);
                assert_eq!(rows[3][1], vec![4, 5, 6]);
                assert_eq!(rows[4][1], vec![4, 5, 6]);
            },
            _ => panic!("row out of bounds")
        }
    });
}

#[test]
fn test_get_pxl_blocks_vec() {

    let mut img = ImageBuffer::new(5, 5);

    //Sets the pixel intensities
    img.enumerate_pixels_mut().for_each(|(x, y, pxl)| {
        *pxl = Luma([((y*5) + x) as u8]);
    });

    //img.pixels().for_each(|pxl| {
    //    println!("{:?}", pxl);
    //});

    let kernal_size = 3_u32;

    let vec_blocks = get_pxl_blocks_vec(&img, &kernal_size);
    
    assert_eq!(vec_blocks.len(), 25);

    vec_blocks.iter().enumerate().for_each(|(idx, block)| {
        assert_eq!(block.len(), 9);
        match idx { 
            0 => assert_eq!(*block, vec![0, 0, 0, 0, 0, 1, 0, 5, 6]),
            1 => assert_eq!(*block, vec![0, 0, 1, 0, 5, 6, 0, 10, 11]),
            2 => assert_eq!(*block, vec![0, 5, 6, 0, 10, 11, 0, 15, 16]), 
            3 => assert_eq!(*block, vec![0, 10, 11, 0, 15, 16, 0, 20, 21]), 
            4 => assert_eq!(*block, vec![0, 15, 16, 0, 20, 21, 0, 0, 0]), 
            5 => assert_eq!(*block, vec![0, 0, 0, 0, 1, 2, 5, 6, 7]), 
            6 => assert_eq!(*block, vec![0, 1, 2, 5, 6, 7, 10, 11, 12]), 
            7 => assert_eq!(*block, vec![5, 6, 7, 10, 11, 12, 15, 16, 17]), 
            8 => assert_eq!(*block, vec![10, 11, 12, 15, 16, 17, 20, 21, 22]), 
            9 => assert_eq!(*block, vec![15, 16, 17, 20, 21, 22, 0, 0, 0]), 
            10 => assert_eq!(*block, vec![0, 0, 0, 1, 2, 3, 6, 7, 8]), 
            11 => assert_eq!(*block, vec![1, 2, 3, 6, 7, 8, 11, 12, 13]), 
            12 => assert_eq!(*block, vec![6, 7, 8, 11, 12, 13, 16, 17, 18]), 
            13 => assert_eq!(*block, vec![11, 12, 13, 16, 17, 18, 21, 22, 23]), 
            14 => assert_eq!(*block, vec![16, 17, 18, 21, 22, 23, 0, 0, 0]), 
            15 => assert_eq!(*block, vec![0, 0, 0, 2, 3, 4, 7, 8, 9]), 
            16 => assert_eq!(*block, vec![2, 3, 4, 7, 8, 9, 12, 13, 14]), 
            17 => assert_eq!(*block, vec![7, 8, 9, 12, 13, 14, 17, 18, 19]), 
            18 => assert_eq!(*block, vec![12, 13, 14, 17, 18, 19, 22, 23, 24]), 
            19 => assert_eq!(*block, vec![17, 18, 19, 22, 23, 24, 0, 0, 0]), 
            20 => assert_eq!(*block, vec![0, 0, 0, 3, 4, 0, 8, 9, 0]), 
            21 => assert_eq!(*block, vec![3, 4, 0, 8, 9, 0, 13, 14, 0]), 
            22 => assert_eq!(*block, vec![8, 9, 0, 13, 14, 0, 18, 19, 0]), 
            23 => assert_eq!(*block, vec![13, 14, 0, 18, 19, 0, 23, 24, 0]), 
            24 => assert_eq!(*block, vec![18, 19, 0, 23, 24, 0, 0, 0, 0]),
            _ => panic!("idx out of bounds"),
        }
    });
}

#[test]
fn test_get_channels() {
    let mut img = ImageBuffer::new(3, 3);
    img.enumerate_pixels_mut().for_each(|(x, y, pix)|{
        //let intensity = (x + y * 3) as u8;
        *pix = Rgb::<u8>([5, 10 , 15]);
    });

    //let img = img.into_raw();
    let img = DynamicImage::ImageRgb8(img);

    //println!("{:?}", img);
    let channels = get_channels(&img);

    assert_eq!(channels[0].as_ref(), vec![5, 5, 5, 5, 5, 5, 5, 5, 5]);
    assert_eq!(channels[1].as_ref(), vec![10, 10, 10, 10, 10, 10, 10, 10, 10]);
    assert_eq!(channels[2].as_ref(), vec![15, 15, 15, 15, 15, 15, 15, 15, 15]);
}

#[test]
fn test_get_similarity_value() {
    let ref_block = (0..9).collect::<Vec<u8>>();
    let block1 = (9..18).collect::<Vec<u8>>();
    let block2 = (0..9).collect::<Vec<u8>>();
    let block3 = (18..27).collect::<Vec<u8>>();

    let sim1 = get_similarity_value(&ref_block, &block1);
    let sim2 = get_similarity_value(&ref_block, &block2);
    let sim3 = get_similarity_value(&ref_block, &block3);

    assert_eq!(sim1, 81);
    assert_eq!(sim2, 0);
    assert_eq!(sim3, 162);

}*/