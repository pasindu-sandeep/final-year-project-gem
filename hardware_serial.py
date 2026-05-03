import time
import serial
import serial.tools.list_ports
import streamlit as st

def get_ports():
    return [p.device for p in serial.tools.list_ports.comports()]

def connect_serial(port):
    try:
        ser = serial.Serial(port, 115200, timeout=0.2)
        time.sleep(1)
        ser.reset_input_buffer()
        st.session_state.ser = ser
        st.session_state.connected_port = port
        st.session_state.hardware_status = f"Connected to Hardware ({port})"
        return True, None
    except Exception as e:
        st.session_state.ser = None
        st.session_state.connected_port = None
        return False, str(e)

def disconnect_serial():
    try:
        if st.session_state.ser:
            st.session_state.ser.close()
    except Exception:
        pass
    st.session_state.ser = None
    st.session_state.connected_port = None
    st.session_state.hardware_status = "Disconnected"

def send_command(cmd):
    if not st.session_state.ser:
        return False, "Please connect to a COM port first."
    try:
        st.session_state.ser.write((cmd + "\n").encode())
        st.session_state.hardware_status = f"Command sent: {cmd}"
        return True, None
    except Exception as e:
        return False, str(e)

def read_serial_messages():
    if not st.session_state.ser:
        return
    try:
        messages = []
        while st.session_state.ser.in_waiting > 0:
            line = st.session_state.ser.readline().decode(errors="ignore").strip()
            if line:
                messages.append(line)

        if messages:
            last = messages[-1]
            if last.startswith("ANGLE="):
                try:
                    ang = int(last.split("=", 1)[1])
                    st.session_state.hardware_status = f"Hardware rotated to {ang}°"
                except Exception:
                    st.session_state.hardware_status = f"Hardware: {last}"
            else:
                st.session_state.hardware_status = f"Hardware: {last}"
    except Exception:
        pass

def wait_for_angle(timeout=20):
    """
    Wait until Arduino sends ANGLE=<value>.
    Returns: angle, error
    """
    if not st.session_state.ser:
        return None, "Please connect to a COM port first."

    start = time.time()

    while time.time() - start < timeout:
        try:
            while st.session_state.ser.in_waiting > 0:
                line = st.session_state.ser.readline().decode(errors="ignore").strip()

                if not line:
                    continue

                if line.startswith("ANGLE="):
                    try:
                        angle = int(line.split("=", 1)[1])
                        st.session_state.hardware_status = f"Hardware rotated to {angle}°"
                        return angle, None
                    except Exception:
                        return None, f"Invalid angle message: {line}"

                st.session_state.hardware_status = f"Hardware: {line}"

        except Exception as e:
            return None, str(e)

        time.sleep(0.05)

    return None, "Timeout waiting for motor angle message."