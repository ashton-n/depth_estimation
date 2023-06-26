extern crate image;
mod term_data;
mod image_ops;
mod read_from_files;

fn main() {
    //command, data_dir_path, generated_depthmap_path
    let (cmd, data_dir_path, output_fn_or_gen_dm_path) = term_data::read_term_args()
        .expect("\n\nInvalid input: input must be: ./target/release/depth_estimation [command] [data directory] [output filename of depthmap]\n\n");

    let (fullsize_width, _) = read_from_files::get_original_dims(&data_dir_path);

    let (left_img_path, 
         right_img_path, 
         dmin_path, 
         disp1_path, 
         _disp5_path) = read_from_files::get_data_paths(&data_dir_path).unwrap();

    match &cmd[..] {
        "depthmap" => {
            image_ops::get_depthmap(&left_img_path, 
                                    &right_img_path, 
                                    &dmin_path,
                                    &disp1_path,
                                    &output_fn_or_gen_dm_path,
                                    &fullsize_width);
            
        },
        "eval" => {
            println!("Do eval Calcs");
            //let eval: String = evaluate(generated_depthmap_path, ground_truth_depthmap_path)
        },
        _ => println!("Invalid Command: must be 'eval' or 'depthmap'")
    }
}
