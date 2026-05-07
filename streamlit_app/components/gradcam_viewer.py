"""
Grad-CAM visualization component.
"""

import streamlit as st
from PIL import Image
from pathlib import Path


def render_gradcam(gradcam_path: str):
    """Render Grad-CAM visualization with controls."""
    if not gradcam_path or not Path(gradcam_path).exists():
        return

    st.markdown("""
    <div class="section-header">
        <h2>🔥 Grad-CAM Explainability</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="padding: 12px;">
        <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
            Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions 
            of the image that most influenced the model's prediction. Warmer colors (red/yellow) 
            indicate areas of higher importance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Display the Grad-CAM visualization
    gradcam_image = Image.open(gradcam_path)
    st.image(
        gradcam_image,
        caption="Original | Grad-CAM Heatmap | Overlay",
        use_container_width=True,
    )
