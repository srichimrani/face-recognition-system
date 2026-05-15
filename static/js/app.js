/**
 * Face Recognition Web App - Webcam & Upload with real-time prediction
 */
(function () {
  const video = document.getElementById("webcam");
  const preview = document.getElementById("preview");
  const canvas = document.getElementById("captureCanvas");
  const ctx = canvas.getContext("2d");
  const fileInput = document.getElementById("fileInput");
  const modelSelect = document.getElementById("modelSelect");
  const resultPanel = document.getElementById("resultPanel");
  const resultLabel = document.getElementById("resultLabel");
  const resultConfidence = document.getElementById("resultConfidence");
  const resultModel = document.getElementById("resultModel");
  const compareResults = document.getElementById("compareResults");
  const annotatedImage = document.getElementById("annotatedImage");

  let stream = null;
  let inputMode = "webcam"; // webcam | upload

  document.getElementById("btnWebcam").addEventListener("click", () => setInputMode("webcam"));
  document.getElementById("btnUpload").addEventListener("click", () => {
    setInputMode("upload");
    fileInput.click();
  });
  document.getElementById("btnStartCam").addEventListener("click", startWebcam);
  document.getElementById("btnCapture").addEventListener("click", captureAndPredict);
  fileInput.addEventListener("change", onFileSelected);

  function setInputMode(mode) {
    inputMode = mode;
    document.getElementById("btnWebcam").classList.toggle("active", mode === "webcam");
    document.getElementById("btnUpload").classList.toggle("active", mode === "upload");
    video.hidden = mode !== "webcam";
    preview.hidden = mode !== "upload";
  }

  async function startWebcam() {
    try {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      video.srcObject = stream;
      setInputMode("webcam");
      document.getElementById("btnCapture").disabled = false;
    } catch (err) {
      alert("Webcam access denied or unavailable: " + err.message);
    }
  }

  function onFileSelected(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      preview.src = reader.result;
      preview.hidden = false;
      video.hidden = true;
      setInputMode("upload");
      predictFromImageData(reader.result);
    };
    reader.readAsDataURL(file);
  }

  function captureAndPredict() {
    if (inputMode === "webcam" && video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
      predictFromImageData(dataUrl);
    }
  }

  async function predictFromImageData(dataUrl) {
    const model = modelSelect.value;
    showLoading();

    const payload = { image_base64: dataUrl };

    try {
      let data;
      if (model === "both") {
        const res = await fetch("/api/predict/both", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        data = await res.json();
        showCompareResults(data);
      } else {
        const res = await fetch(`/api/predict?model=${model}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        data = await res.json();
        showSingleResult(data);
      }
    } catch (err) {
      showError(err.message);
    }
  }

  function showLoading() {
    resultPanel.className = "result-panel neutral";
    resultLabel.textContent = "Processing...";
    resultConfidence.textContent = "Confidence: —";
    compareResults.hidden = true;
  }

  function showSingleResult(data) {
    compareResults.hidden = true;
    if (!data.success) {
      showError(data.error || "Prediction failed");
      return;
    }
    updatePanel(data);
    if (data.image_base64) {
      annotatedImage.src = "data:image/jpeg;base64," + data.image_base64;
      annotatedImage.hidden = false;
    }
  }

  function showCompareResults(data) {
    annotatedImage.hidden = true;
    compareResults.hidden = false;
    compareResults.innerHTML = "";

    ["ml", "tm"].forEach((key) => {
      const r = data[key];
      const div = document.createElement("div");
      div.className = "compare-item";
      if (!r || !r.success) {
        div.textContent = `${key.toUpperCase()}: ${r?.error || "Unavailable"}`;
      } else {
        const isMe = r.is_me;
        div.classList.add(isMe ? "me" : "not-me");
        div.innerHTML = `<strong>${key === "ml" ? "Classical ML" : "Teachable Machine"}</strong><br>
          ${r.label} — ${r.confidence}% confidence`;
        if (key === "ml" && r.success) updatePanel(r);
      }
      compareResults.appendChild(div);
    });
  }

  function updatePanel(data) {
    const isMe = data.is_me;
    resultPanel.className = "result-panel " + (isMe ? "me" : "not-me");
    resultLabel.textContent = data.label;
    resultConfidence.textContent = `Confidence: ${data.confidence}%`;
    resultModel.textContent = `Model: ${data.model}`;
  }

  function showError(msg) {
    resultPanel.className = "result-panel neutral";
    resultLabel.textContent = "Error";
    resultConfidence.textContent = msg;
    resultModel.textContent = "";
  }
})();
