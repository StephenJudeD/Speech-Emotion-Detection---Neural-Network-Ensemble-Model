<!DOCTYPE html>
<html>
<head>
    <title>Emotion Recognition</title>
</head>
<body>
    <h1>Speak Your Emotion</h1>
    <button id="recordButton">Record</button>
    <audio id="audioPlayback" controls></audio>
    <div id="result"></div>

    <script>
        const recordButton = document.getElementById("recordButton");
        const audioPlayback = document.getElementById("audioPlayback");
        let mediaRecorder;
        let audioChunks = [];

        recordButton.onclick = async () => {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.start();
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks);
                const audioUrl = URL.createObjectURL(audioBlob);
                audioPlayback.src = audioUrl;
                audioChunks = [];

                // Send audioBlob to your server for processing
                const formData = new FormData();
                formData.append('audio', audioBlob, 'user_audio.wav');
                await fetch('/predict-emotion', { method: 'POST', body: formData });
            };

            recordButton.innerText = "Stop Recording";
            recordButton.onclick = () => {
                mediaRecorder.stop();
                recordButton.innerText = "Record";
            };
        };
    </script>
</body>
</html>
