"""Blender-like interactive editing tool: a second consumer of `engine/`,
exactly as design_report.md's original architecture section anticipated
("un futur outil 'Editing' ... comme un second consommateur du même package
engine/, sans avoir à dupliquer ou déplacer quoi que ce soit").

Nothing in `engine/` imports from this package - the dependency runs one
way. `editing/` adds selection, transform/material gizmos, and the graphics
settings panel on top of the engine; `engine/` itself stays pure rendering
and scene management.
"""
