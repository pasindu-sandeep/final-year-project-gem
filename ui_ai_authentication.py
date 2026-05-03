import streamlit as st
from pathlib import Path
from datetime import datetime
import time

from hardware_serial import send_command, wait_for_angle
from api_client import run_ai_request, run_ai_request_from_paths
from camera_capture import capture_angle_image

def render_ai_auth_page():
    left, right = st.columns([1.45, 1])

    # =====================================================
    # LEFT SIDE: RESULT + AUTO CAPTURE PREVIEW
    # =====================================================
    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🧠 AI Authentication Result")

        if st.session_state.last_ai_result is None:
            st.info("No result yet. Upload 4 images manually or use Auto Capture & Verify.")
        else:
            result = st.session_state.last_ai_result
            
            verdict = str(result.get("final_verdict", "UNKNOWN")).strip().upper()
            patterns = result.get("natural_pattern_detections", 0)

            if verdict == "AUTHENTIC":
                st.markdown(
                    f'<div class="result-good">✅ AUTHENTIC | Patterns in {patterns}/4 angles</div>',
                    unsafe_allow_html=True
                )
            elif verdict in ["NOT_A_GEM", "NOT_GEM", "NON_GEM", "NO_GEM_DETECTED", "INVALID", "UNKNOWN"]:
                st.markdown(
                    '<div class="result-warning">⚠️ NO GEM DETECTED (Please upload a clear image of a gemstone)</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-bad">❌ FAKE OR GLASS (No natural pattern detected)</div>',
                    unsafe_allow_html=True
                )

            with st.expander("View raw JSON"):
                st.json(result)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📷 Last Auto Captured Images")

        if not st.session_state.auto_captured_paths:
            st.info("No auto captured images yet.")
        else:
            if st.session_state.last_auto_run_dir:
                st.caption(f"Saved folder: {st.session_state.last_auto_run_dir}")

            preview_cols = st.columns(4)

            for i, path in enumerate(st.session_state.auto_captured_paths):
                img_path = Path(path)
                with preview_cols[i]:
                    if img_path.exists():
                        st.image(str(img_path), caption=img_path.name, use_container_width=True)
                    else:
                        st.warning(f"Missing: {img_path.name}")

        with st.expander("Auto Capture Logs"):
            if st.session_state.auto_capture_log:
                for log in st.session_state.auto_capture_log:
                    st.write(log)
            else:
                st.write("No logs yet.")

        clear_auto_col1, clear_auto_col2 = st.columns(2)

        if clear_auto_col1.button("🧹 Clear Auto Preview"):
            st.session_state.auto_captured_paths = []
            st.session_state.auto_capture_log = []
            st.session_state.last_auto_run_dir = None
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # RIGHT SIDE: MANUAL UPLOAD + AUTO CAPTURE
    # =====================================================
    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📸 Manual Upload Angles")

        img0 = st.file_uploader("Angle 0°", type=["jpg", "jpeg", "png"], key="img0")
        img90 = st.file_uploader("Angle 90°", type=["jpg", "jpeg", "png"], key="img90")
        img180 = st.file_uploader("Angle 180°", type=["jpg", "jpeg", "png"], key="img180")
        img270 = st.file_uploader("Angle 270°", type=["jpg", "jpeg", "png"], key="img270")

        c1, c2 = st.columns(2)

        if c1.button("🧹 Clear Result"):
            st.session_state.last_ai_result = None
            st.rerun()

        if c2.button("🚀 VERIFY WITH AI"):
            files = [img0, img90, img180, img270]
            if any(f is None for f in files):
                st.error("Please upload all 4 images.")
            else:
                with st.spinner("Analyzing manually uploaded images with AI..."):
                    ok, res = run_ai_request(files)
                if ok:
                    st.session_state.last_ai_result = res
                    st.success("Manual AI verification completed.")
                    st.rerun()
                else:
                    st.error(res)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("🤖 Auto Capture & Upload")

        camera_index = st.number_input("USB Webcam Index", min_value=0, max_value=5, value=1, step=1)
        st.caption("First use TEST AUTO CAPTURE ONLY. If images preview correctly, then use AUTO CAPTURE & VERIFY.")

        if st.button("📷 TEST AUTO CAPTURE ONLY"):
            st.session_state.auto_capture_log = []
            st.session_state.auto_captured_paths = []
            st.session_state.last_auto_run_dir = None

            try:
                run_dir = Path("captures") / datetime.now().strftime("%Y%m%d_%H%M%S_test")
                captured_paths = []
                status_box = st.empty()
                progress = st.progress(0)

                st.session_state.auto_capture_log.append("Starting camera-only test capture...")
                status_box.info("Starting camera-only test capture...")

                test_angles = [0, 90, 180, 270]

                for i, angle in enumerate(test_angles):
                    status_box.info(f"Capturing test image at {angle}°...")
                    st.session_state.auto_capture_log.append(f"Capturing test image at {angle}°...")
                    img_path = capture_angle_image(angle, camera_index=int(camera_index), save_dir=run_dir)
                    captured_paths.append(img_path)
                    st.session_state.auto_capture_log.append(f"Saved: {img_path}")
                    progress.progress(int(((i + 1) / len(test_angles)) * 100))

                st.session_state.auto_captured_paths = captured_paths
                st.session_state.last_auto_run_dir = str(run_dir)
                status_box.success("Test auto capture completed. Check previews on the left side.")
                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.session_state.auto_capture_log.append(f"Camera test capture failed: {e}")
                st.error(f"Camera test capture failed: {e}")

        if st.button("🚀 AUTO CAPTURE & VERIFY"):
            st.session_state.auto_capture_log = []
            st.session_state.auto_captured_paths = []
            st.session_state.last_auto_run_dir = None

            if not st.session_state.ser:
                st.error("Please connect Arduino COM port first.")
            else:
                try:
                    run_dir = Path("captures") / datetime.now().strftime("%Y%m%d_%H%M%S")
                    captured_paths = []
                    status_box = st.empty()
                    progress = st.progress(0)

                    status_box.info("Stopping motor...")
                    st.session_state.auto_capture_log.append("Stopping motor...")
                    ok, err = send_command("STOP")
                    if not ok:
                        st.error(err); st.stop()
                    time.sleep(0.5); progress.progress(5)

                    status_box.info("Turning LED ON...")
                    ok, err = send_command("LED_ON")
                    if not ok:
                        st.error(err); st.stop()
                    time.sleep(1); progress.progress(10)

                    status_box.info("Resetting current position as 0°...")
                    ok, err = send_command("RESET_ANGLE")
                    if not ok:
                        st.error(err); st.stop()
                    time.sleep(0.7)

                    try:
                        st.session_state.ser.reset_input_buffer()
                    except:
                        pass
                    progress.progress(15)

                    status_box.info("Capturing image at 0°...")
                    img0_path = capture_angle_image(0, camera_index=int(camera_index), save_dir=run_dir)
                    captured_paths.append(img0_path)
                    progress.progress(30)

                    target_angles = [90, 180, 270]
                    progress_values = {90: 50, 180: 70, 270: 90}

                    for expected_angle in target_angles:
                        status_box.info(f"Rotating motor to {expected_angle}°...")
                        try:
                            st.session_state.ser.reset_input_buffer()
                        except:
                            pass
                        
                        ok, err = send_command("STEP")
                        if not ok:
                            st.error(err); st.stop()

                        angle, err = wait_for_angle(timeout=30)
                        if err:
                            st.error(err); st.stop()

                        status_box.info(f"Motor stopped at {angle}°. Capturing image...")
                        img_path = capture_angle_image(angle, camera_index=int(camera_index), save_dir=run_dir)
                        captured_paths.append(img_path)
                        progress.progress(progress_values.get(expected_angle, 90))

                    st.session_state.auto_captured_paths = captured_paths
                    st.session_state.last_auto_run_dir = str(run_dir)

                    status_box.info("Uploading 4 captured images to AI model...")
                    ok, res = run_ai_request_from_paths(captured_paths)

                    if ok:
                        st.session_state.last_ai_result = res
                        progress.progress(100)
                        status_box.success("Auto capture and AI verification completed successfully.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(res)

                except Exception as e:
                    st.error(f"Auto capture failed: {e}")

        st.markdown("</div>", unsafe_allow_html=True)