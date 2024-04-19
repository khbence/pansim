ffmpeg -framerate 5 -pattern_type glob -i '*.png'  -c:v libx264 -pix_fmt yuv420p \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -r 24 \
    _Video.mp4
