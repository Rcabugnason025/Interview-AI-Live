# Live Interview AI Copilot

This is a web-based AI assistant for live interviews. It listens to audio, transcribes it in real-time, and provides AI-generated suggestions based on your resume and job description.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    *   Open `.env` and add your `OPENAI_API_KEY`.
    *   Alternatively, you can enter the key in the UI sidebar.

## Running the App

Run the following command in your terminal:

```bash
python -m streamlit run app.py
```

## How to Use

1.  **Upload Context**:
    *   Upload your **Resume** (PDF/DOCX).
    *   Paste the **Job Description**.
    *   (Optional) Paste a **Script** or expected questions.
    *   Click **Load Context**.

2.  **Audio Setup**:
    *   Click "Start" in the Audio Stream section.
    *   **Crucial**: To capture the *Interviewer's* voice, you must ensure your input device captures system audio.
        *   **Windows**: Enable "Stereo Mix" in Sound Settings and select it as the microphone in the browser popup.
        *   **Mac/Linux**: Use a virtual audio cable (e.g., BlackHole, VB-Cable).
        *   **Simple Method**: Use your speakers and microphone. The mic will pick up the sound from the speakers (might cause echo, but works for simple setups).

3.  **Live Assistance**:
    *   The app will transcribe audio in chunks (every ~3-5 seconds).
    *   AI suggestions will appear on the right side.
