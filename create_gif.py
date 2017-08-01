from moviepy.editor import *

clip = VideoFileClip("gif1.mp4")
clip.write_gif("gif1.gif")

clip = VideoFileClip("gif2_sub.mp4")
clip.write_gif("gif2.gif")

clip = VideoFileClip("gif3_sub.mp4")
clip.write_gif("gif3.gif")

# clip = (VideoFileClip("test_videos_output_DEBUG/challenge.mp4").subclip(1,3.6).resize(0.8))
# clip.write_gif("showcase4.gif")