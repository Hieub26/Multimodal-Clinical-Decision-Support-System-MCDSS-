"""
Sidebar component for Streamlit UI.
"""

import streamlit as st


def render_sidebar():
    """Render the sidebar navigation and settings."""
    with st.sidebar:
        # Logo / title area
        st.markdown("""
        <div style="text-align: center; padding: 16px 0 24px 0;">
            <div style="font-size: 2.5rem; margin-bottom: 4px;">🏥</div>
            <div style="font-size: 1.1rem; font-weight: 700; 
                        background: linear-gradient(135deg, #14b8a6, #3b82f6);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Clinical DSS
            </div>
            <div style="font-size: 0.75rem; color: #64748b;">v1.0.0</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Navigation
        st.markdown("##### 📋 Navigation")
        page = st.radio(
            "Go to",
            ["🔬 Diagnosis", "📊 History", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown("##### Analysis Mode")
        mode = st.radio(
            "Mode",
            ["Multimodal", "Text Only", "Image Only"],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        # Disclaimer
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.2);
                    border-radius: 8px; padding: 10px; font-size: 0.75rem; color: #f59e0b;">
            ⚕️ <b>Disclaimer:</b> For educational purposes only. Not for real clinical use.
        </div>
        """, unsafe_allow_html=True)

    return page, mode, 0.8
