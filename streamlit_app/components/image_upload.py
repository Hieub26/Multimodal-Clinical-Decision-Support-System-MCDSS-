"""
Image upload component for medical images.
"""

import streamlit as st
from PIL import Image


def render_image_upload():
    """Render the medical image upload section."""

    st.markdown("""
    <div class="section-header">
        <h2>🩻 Medical Image</h2>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a medical image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        key="image_upload",
        help="Supported formats: PNG, JPEG, BMP, TIFF",
    )

    if uploaded_file:
        # Show preview
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image, caption="Uploaded Image", width="stretch")

        with col2:
            st.markdown(f"""
            <div class="glass-card" style="padding: 16px;">
                <div style="font-size: 0.8rem; color: #64748b; text-transform: uppercase;
                            letter-spacing: 0.05em; margin-bottom: 12px;">Image Details</div>
                <div style="margin-bottom: 8px;">
                    <span style="color: #94a3b8; font-size: 0.85rem;">Filename:</span><br>
                    <span style="color: #f0f4f8; font-weight: 500;">{uploaded_file.name}</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <span style="color: #94a3b8; font-size: 0.85rem;">Size:</span><br>
                    <span style="color: #f0f4f8; font-weight: 500;">{image.size[0]} × {image.size[1]} px</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <span style="color: #94a3b8; font-size: 0.85rem;">Mode:</span><br>
                    <span style="color: #f0f4f8; font-weight: 500;">{image.mode}</span>
                </div>
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem;">File size:</span><br>
                    <span style="color: #f0f4f8; font-weight: 500;">{uploaded_file.size / 1024:.1f} KB</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Reset file position for later reading
        uploaded_file.seek(0)

    return uploaded_file
