# aaaa.sh/fishes

[This is the website, if you haven't seen it already.](https://aaaa.sh/fishes)

It's a music video done in text on a webpage! This repository contains all of the source code.

The main part of this is a **Rust program**, which takes in a png file and produces a JSON file that approximates that png using colored text. The Rust program uses [rusttype](https://crates.io/crates/rusttype) & [vulkano](https://vulkano.rs/) for drawing, and a genetic algorithm. This prorgam is written in the same way you would write a throwaway Python script. There is no configuration: in order to change the parameters for better results on specific images, you need to change the script itself. The image & the JSON file are written to every 10 generations, so you can see the image as it's generating (you might want a live-updating photo preview, like MacOS Finder Gallery view or `feh` on Linux). The program also "picks up where it left off" via the `best.json` file, so you can kill it with Ctrl+C and continue it later.

In the `html` directory, you can see the code for the actual website, and the scroll-to-audio code. This is split into two sections, `template.html`, which has the scroll code & the page layout, and `words.html`, which has the complete set of `div`s containing words. The `words.html` was generated mostly by a large Python script, and then hand edited later.

Some notes:
- For testing, or if you just want to rip through the website faster, type `window.speedMultiplier = Infinity` to uncap the speed.
- If you want to use this script yourself, clone the repo and run `cargo r --release`.
- Usually, it takes about a minute to get a pretty good image. For this project, a couple larger images were left in the oven longer, the longest was ~3 hours. Of course, since it's a genetic algorithm, you're going to run into diminishing returns after a while.
- Because placing your fingers onto the trackpad slows the song down temporarily, the playback when scrolling is slower overall than the original song. To compensate for this, the first half of the song is sped up by 9.7%.
- The words that aren't word clouds (like the 'guitar's and the drum hits) were just placed by hand. Once you do one measure, you can repeat it a bunch, so that saved time.
- The 'opening eyes' (appearing after the second "your eyes") were done as an animation, where the Rust program was configured to control the speed of each word, and was tuned against a image containing multiple frames of an eye opening. The result is that the eyelid covers the pupil for a time, until it is revealed as you scroll further. Pretty cool! Some friends of mine said it was creepy, though.
- This website took me about 2 weeks total to create. I microblogged the process on [Bluesky](https://bsky.app/profile/aaaashley.bsky.social/post/3lc7pkyezcc2k).
