import base64
import os
import tempfile
import trimesh
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import html

def render_glb_bytes(glb_bytes: bytes, title: str, height: int = 650):
    b64 = base64.b64encode(glb_bytes).decode()
    st.subheader(title)
    viewer = f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <model-viewer src="data:model/gltf-binary;base64,{b64}" alt="{title}" auto-rotate camera-controls shadow-intensity="1" style="width:100%; height:{height-50}px; background:#111;"></model-viewer>
    """
    html(viewer, height=height)

def load_trimesh_from_glb_bytes(glb_bytes: bytes) -> trimesh.Trimesh:
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp.write(glb_bytes)
        tmp_path = tmp.name
    try:
        scene = trimesh.load(tmp_path, file_type="glb")
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes: raise ValueError("No mesh geometry found in GLB.")
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene
        mesh = mesh.copy()
        mesh.remove_unreferenced_vertices()
        return mesh
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def plot_meshes(rough: trimesh.Trimesh, cut: trimesh.Trimesh):
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=rough.vertices[:, 0], y=rough.vertices[:, 1], z=rough.vertices[:, 2], i=rough.faces[:, 0], j=rough.faces[:, 1], k=rough.faces[:, 2], opacity=0.20, color="blue", name="Rough Gem"))
    fig.add_trace(go.Mesh3d(x=cut.vertices[:, 0], y=cut.vertices[:, 1], z=cut.vertices[:, 2], i=cut.faces[:, 0], j=cut.faces[:, 1], k=cut.faces[:, 2], opacity=0.88, color="red", name="Optimized Cut"))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode="data"), showlegend=True)
    return fig