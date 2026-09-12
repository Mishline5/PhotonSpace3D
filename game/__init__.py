"""Game runtime: turns a scene built in the editor into something you can
walk around in. Uses `engine/` for rendering, the scene authored via
`editing/` for the environment, and this package for the game code.

For now there is no specific game - launching from the editor drops you into
a default FPS character (mouse-look + ZQSD/WASD, gravity, mesh-accurate
collisions, no fly). A real game would add its own logic here on top of the
same character/collision building blocks.
"""
