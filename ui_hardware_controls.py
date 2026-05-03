import streamlit as st
from hardware_serial import (
    get_ports,
    connect_serial,
    disconnect_serial,
    send_command,
    read_serial_messages
)

def render_hardware_page():
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🔌 Device Connection")

        ports = get_ports()
        selected_port = st.selectbox(
            "COM Port",
            ports if ports else ["No ports found"]
        )

        c1, c2, c3 = st.columns(3)

        if c1.button("🔄 Refresh Ports"):
            st.rerun()

        if c2.button("⚡ Connect"):
            if not ports or selected_port == "No ports found":
                st.error("No COM ports detected.")
            else:
                ok, err = connect_serial(selected_port)
                if ok:
                    st.success(f"Connected to {selected_port}")
                else:
                    st.error(err)

        if c3.button("❌ Disconnect"):
            disconnect_serial()
            st.warning("Disconnected.")

        st.markdown(
            '<hr><p style="color:#94a3b8;">Use refresh to read the latest serial message.</p>',
            unsafe_allow_html=True
        )

        if st.button("📡 Refresh Hardware Status"):
            read_serial_messages()
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Device Control")

        a1, a2 = st.columns(2)
        b1, b2 = st.columns(2)

        if a1.button("💡 LED ON"):
            ok, err = send_command("LED_ON")
            if not ok:
                st.error(err)

        if a2.button("🌑 LED OFF"):
            ok, err = send_command("LED_OFF")
            if not ok:
                st.error(err)

        if b1.button("🔄 START ROTATE"):
            ok, err = send_command("START")
            if not ok:
                st.error(err)

        if b2.button("🛑 STOP ROTATE"):
            ok, err = send_command("STOP")
            if not ok:
                st.error(err)

        st.markdown(
            "<br>"
            "<span class='mini-tag'>DIY Device</span>"
            "<span class='mini-tag'>Polarized Sheets</span>"
            "<span class='mini-tag'>LED Base</span>"
            "<span class='mini-tag'>Stepper Motor</span>"
            "</div>",
            unsafe_allow_html=True
        )