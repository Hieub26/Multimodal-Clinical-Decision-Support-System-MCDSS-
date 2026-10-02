"""
Case history viewer component.
"""

import os
from html import escape

import streamlit as st
import httpx


API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8008/api")


def render_history():
    """Render the case history page."""

    st.markdown("""
    <div class="hero-header">
        <h1>📊 Case History</h1>
        <p>Review past diagnosis cases and reports</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        response = httpx.get(f"{API_BASE}/reports/history", timeout=10)
        if response.status_code == 200:
            data = response.json()
            cases = data.get("cases", [])
            total = data.get("total", 0)

            st.markdown(f"**Total cases:** {total}")

            if not cases:
                st.info("No diagnosis cases found yet. Run your first diagnosis!")
                return

            # Display cases as cards
            for case in cases:
                safety_status = case.get("safety_status", "unknown")
                badge_class = "badge-safe" if safety_status == "approved" else "badge-warning"
                confidence = case.get("confidence", 0)
                case_id = case.get("case_id", "")

                st.markdown(f"""
                <div class="glass-card" style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div style="font-weight: 600; color: #f0f4f8; font-size: 1.05rem;">
                                {escape(str(case.get("primary_diagnosis", "N/A")))}
                            </div>
                            <div style="color: #64748b; font-size: 0.8rem; margin-top: 4px;">
                                {escape(str(case.get("created_at", "")))} · {escape(str(case.get("input_type", "")).upper())}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <span class="badge {badge_class}">{escape(str(safety_status).upper())}</span>
                            <div style="color: #14b8a6; font-weight: 700; font-size: 1.1rem; margin-top: 8px;">
                                {confidence:.0%}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Details are fetched only for cases the user opens. An
                # expander would run its body (one request per case) on
                # every rerun, whether or not it is expanded.
                if st.toggle(f"View details — {case_id}", key=f"details_{case_id}"):
                    try:
                        detail_resp = httpx.get(
                            f"{API_BASE}/reports/history/{case_id}",
                            timeout=10,
                        )
                        if detail_resp.status_code == 200:
                            detail = detail_resp.json()
                            st.json(detail)
                        else:
                            st.warning("Could not load case details")
                    except Exception:
                        st.warning("Could not load case details")

        else:
            st.error(f"Failed to fetch history: {response.status_code}")

    except httpx.ConnectError:
        st.error(
            "⚠️ Cannot connect to the API backend. "
            "Make sure FastAPI is running on port 8008."
        )
    except Exception as e:
        st.error(f"Error loading history: {e}")
