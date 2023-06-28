use anyhow::Result; // Automatically handle the error types
use indicatif::ProgressBar;
use terminal_size::{terminal_size, Width, Height};

use std::io::{stdout, Write};
use std::thread::sleep;
use std::time::Duration;

use clap::Parser;

use crossterm::{
    execute, queue,
    style::{Print},
    cursor::{Show, Hide, MoveTo},
};

use opencv::{
    prelude::*,
    videoio,
    core::{Size, Vec3b},
    imgproc::{resize, INTER_LINEAR}, // optflow::ST_BILINEAR
};

const SKIP_EVERY: usize = 0; // 0 for no skipping, 2 for skipping every other frame, etc.


fn get_video_frames(path: &str) -> Result<Vec<Mat>, opencv::Error> {
    let mut video_capture = videoio::VideoCapture::from_file(path, videoio::CAP_ANY)?;

    if !video_capture.is_opened()? {
        return Err(opencv::Error::new(opencv::core::StsError, "Failed to open video file"));
    } 

    let mut frames = Vec::new();
    let mut frame = Mat::default();

    while video_capture.read(&mut frame)? {
        frames.push(frame.clone());
    }

    Ok(frames)
}

fn skip_frames(frames: &Vec<Mat>, skip_every: usize) -> Result<Vec<Mat>, opencv::Error> {
    if skip_every == 0 {
        return Ok(frames.clone()); // if skip_every is 0, skip no frames
    }

    let mut new_frames = Vec::new();

    let mut i = 0;
    while i < frames.len() {
        if i % skip_every != 0 {
            new_frames.push(frames[i].clone());
        }
        i += 1;
    }

    Ok(new_frames)
}

fn resize_frames(frames: Vec<Mat>, size_x: i32, size_y: i32) -> Result<Vec<Mat>, opencv::Error> {
    let mut resized_frames = Vec::new();
    let target_size = Size::new(size_x, size_y);


    let mut i = 0;
    while i < frames.len() {
        let mut resized_frame = Mat::default();

        resize(&frames[i], &mut resized_frame, target_size, 0.0, 0.0, INTER_LINEAR)
            .expect("Error while resizing frame");

        resized_frames.push(resized_frame);
        i += 1;
    }

    Ok(resized_frames)
}

fn frame_to_text(frame: &Mat, size_x: i32, size_y: i32) -> Result<String, opencv::Error> {
    let mut frame_text = String::new();

    let mut y = 0;
    while y < size_y {
        let mut x = 0;
        while x < size_x {
            let pixel = frame.at_2d::<Vec3b>(y, x).unwrap();
            let color = Color {b: pixel[0], g: pixel[1], r: pixel[2]};

            let closest_char = color_to_character(color)
                .expect("Error while converting color to character");

            frame_text.push_str(&closest_char);
            x += 1;
        }
    
        frame_text.push_str("\n");
        y += 1;
    }


    Ok(frame_text)
}

fn color_to_character(color: Color) -> Result<String, opencv::Error> {
    let shades = vec![
        Shade { symbol: "█", color: Color { r: 0, g: 0, b: 0 } },
        Shade { symbol: "▓", color: Color { r: 51, g: 51, b: 51 } },
        Shade { symbol: "▒", color: Color { r: 153, g: 153, b: 153 } },
        Shade { symbol: "░", color: Color { r: 204, g: 204, b: 204 } },
        Shade { symbol: " ", color: Color { r: 255, g: 255, b: 255 } },
    ];

    // let shades = vec![
    //     Shade { symbol: "O", color: Color { r: 0, g: 0, b: 0 } },
    //     Shade { symbol: " ", color: Color { r: 255, g: 255, b: 255 } },
    // ];

    // let shades = vec![
    //     Shade { symbol: "█", color: Color { r: 0, g: 0, b: 0 } },
    //     Shade { symbol: " ", color: Color { r: 255, g: 255, b: 255 } },
    // ];

    // match the color to the closest color in the shades vector
    let mut i = 0;
    let mut closest_distance: u32 = 195075;
    let mut character = String::new();

    while i < shades.len() {
        let shade = &shades[i];

        let distance: u32 = (color.r.saturating_sub(shade.color.r) as u32).pow(2) +
            (color.g.saturating_sub(shade.color.g) as u32).pow(2) +
            (color.b.saturating_sub(shade.color.b) as u32).pow(2);

        if distance < closest_distance {
            closest_distance = distance;
            character = String::from(shade.symbol);
        }

        i += 1;
    }

    Ok(character)
}

fn print_frames(frames_text: &[String], frame_delay: u64) -> crossterm::Result<()> {
    let mut stdout = stdout();

    execute!(stdout, Hide)?;

    for frame_text in frames_text {
        queue!(
            stdout,
            // Clear(ClearType::All),
            MoveTo(0, 0),
            Print(frame_text),
        )?;

        stdout.flush()?;
        sleep(Duration::from_millis(frame_delay));
    }

    execute!(stdout, Show)?;

    Ok(())
}

fn same_line_print(text: &str) {
    print!("{}", text);
    std::io::stdout().flush().unwrap();
}


struct Color {
    r: u8,
    b: u8,
    g: u8
}
struct Shade {
    symbol: &'static str,
    color: Color,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, required = true, name = "video-path")]
    path: String,

    #[arg(short, long, default_value_t = 30.0)]
    fps: f64,

    #[arg(short, long, default_value_t = false)]
    autosize: bool,

    #[arg(long, default_value_t = 120)]
    size_x: i32,

    #[arg(long, default_value_t = 40)]
    size_y: i32,
}


fn main() {
    let mut frames: Vec<Mat> = Vec::new();

    let args = Args::parse();
    let video_path = &args.path;

    let frame_delay: u64 = (1.0 / args.fps * 1000.0) as u64; // 60 fps

    let mut size_x = args.size_x;
    let mut size_y = args.size_y;


    if args.autosize {
        if let Some((Width(width), Height(height))) = terminal_size() {
            // Calculate size_y based on width and the 4:1 aspect ratio
            size_y = (width as f64 / 4.0) as i32;

            // Adjust size_x to maintain the 4:1 aspect ratio
            size_x = size_y * 4;

            // Check if the calculated size exceeds the available height
            if size_y > height as i32 {
                // Adjust size_y to fit the available height
                size_y = height as i32;
                size_x = size_y * 4;
            }
        }
    }


    match get_video_frames(&video_path) {
        Ok(result_frames) => {
            let skipped_frames = skip_frames(&result_frames, SKIP_EVERY)
                .expect("Error while skipping frames");
    
            let resized_frames = resize_frames(skipped_frames, size_x, size_y)
                .expect("Error while resizing frames");

            frames = resized_frames;        
        },
        Err(e) => {
            println!("Error: {}", e);
        }
    }


    let mut frames_text: Vec<String> = Vec::new();

    let progress_bar = ProgressBar::new(frames.len() as u64);
    let now = std::time::Instant::now();

    same_line_print("Converting frames to text: ");
    for frame in frames.iter() {
        let frame_text = frame_to_text(frame, size_x, size_y)
            .expect("Failed to convert frame");

        frames_text.push(frame_text);
        progress_bar.inc(1);
    }

    progress_bar.finish();


    println!("\nTime taken to get frames: {}ms", now.elapsed().as_millis());
    same_line_print("Press enter to start animation ");

    std::io::stdin().read_line(&mut String::new()) // wait for input
        .expect("Failed to read line");

    
    // for frame_text in frames_text.iter() {
    //     print!("\x1B[2J\x1B[1;1H");
    //     println!("{}", frame_text);

    //     std::thread::sleep(std::time::Duration::from_millis(FRAME_DELAY));
    // }
    print_frames(&frames_text, frame_delay)
        .expect("Error while printing frames");
}
