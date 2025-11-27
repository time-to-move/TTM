# Cut and Drag GUI

We provide a GUI to generate cut-and-drag examples that will later be used for video generation given this input signal using **Time to Move**!  
Given an input frame, you can cut and drag polygons from the initial image, transform their colors, and also add external images that can be dragged into the initial scene.

## ✨ General Guide
- Select an initial image.
- Draw polygons in the image and drag them in several segments.
- During segments you can rotate, scale, and change the polygon colors!
- You can also add an external image into the scene and move it (or polygons cut from it) across segments. Transparency is preserved.
- Write a text prompt that will be used to generate the video afterwards.
- You can preview the motion signal in an in-app demo.
- In the end, all the inputs needed for **Time-to-Move** are saved automatically in a selected output directory.

## 🧰 Requirements
Install dependencies:
```bash
pip install PySide6 opencv-python numpy imageio imageio-ffmpeg
```

## 🚀 Run
Just run the python script:
```bash
python cut_and_drag.py
```

## 🖱️ How to Use
* Select Image — Click 🖼️ Select Image and choose an image.
    * Choose Center Crop / Center Pad at the top of the toolbar if needed.
* Add a Polygon “cutting” the part of the image by clicking Add Polygon.
    * Left-click to add points.
    * After finishing drawing the polygon, press ✅ Finish Polygon Selection.
* Drag to move the polygon
    * During segments you’ll see corner circles and a top dot which can be used for scaling and rotating during the segments; in the video the shape is interpolated between the initial frame status and the final segment one.
    * Also, color transformation can be applied (using hue transformation) in the segments to change polygon colors.
    * Click 🎯 End Segment to capture the segment annotated.
    * The movement trajectory can be constructed from multiple segments: repeat move → 🎯 End Segment → move → 🎯 End Segment…
* External Image
    * Another option is to add an external image to the scene.
    * Click 🖼️➕ Add External Image, pick a new image (transparent PNGs are supported).
    * Position/scale/rotate it for its initial pose, then click ✅ Place External Image or right-click on the canvas to lock its starting pose.
    * Now animate it like before: you can move the external image itself, or cut a polygon from it and move it.
* Prompt
    * Type any text prompt you want associated with this example; it will be used later for video generation with our method.
* Preview and Save
    * Preview using ▶️ Play Demo.
    * Click 💾 Save, choose an output folder and then enter a subfolder name.
    * Click 🆕 New to start a new project.

## Output Files
* first_frame.png — the initial frame for video generation
* motion_signal.mp4 — the reference warped video
* mask.mp4 — grayscale mask of the motion
* prompt.txt — your prompt text


## 🧾 License / Credits
Built with PySide6, OpenCV, and NumPy.
You own the images and exports you create with this tool.
Motivation for creating an easy-to-use tool from [Go-With-The-Flow](https://github.com/GoWithTheFlowPaper/gowiththeflowpaper.github.io).
