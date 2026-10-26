import { useCallback, useEffect, useRef, useState } from "react";
import { detectObjects, fetchClasses, fetchHealth } from "../services/api";

const INITIAL_CONFIDENCE = Number(
  import.meta.env.VITE_DEFAULT_CONFIDENCE || 0.35
);
const INITIAL_INTERVAL_MS = Number(
  import.meta.env.VITE_CAPTURE_INTERVAL_MS || 250
);
const MAX_UPLOAD_WIDTH = Number(import.meta.env.VITE_MAX_UPLOAD_WIDTH || 640);

function formatCameraError(err) {
  if (!err) {
    return "Unable to access the camera.";
  }
  const name = err.name || "";
  if (
    name === "NotAllowedError" ||
    name === "PermissionDeniedError" ||
    name === "SecurityError"
  ) {
    return "Camera permission was denied. Please grant camera access in your browser site settings (and macOS System Settings > Privacy & Security > Camera) and try again.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No camera device was found on this system. Please connect a webcam and try again.";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "Camera is already in use by another application or hardware error occurred. Close other video apps and retry.";
  }
  if (name === "OverconstrainedError") {
    return "The requested camera resolution is not supported by your webcam.";
  }
  return err.message || "An unexpected error occurred while accessing the webcam.";
}

function canvasToJpegBlob(canvas, quality = 0.82) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error("Failed to encode video frame as JPEG."));
        }
      },
      "image/jpeg",
      quality
    );
  });
}

export function useWebcamDetection() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const inFlightRef = useRef(false);
  const abortControllerRef = useRef(null);
  const mountedRef = useRef(true);

  // Camera state
  const [cameraStatus, setCameraStatus] = useState("idle"); // 'idle' | 'starting' | 'active' | 'error'
  const [cameraError, setCameraError] = useState(null);
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });

  // Detection loop state
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionStatus, setDetectionStatus] = useState("idle"); // 'idle' | 'running' | 'paused' | 'error'
  const [detectionError, setDetectionError] = useState(null);

  // Configuration state
  const [confidence, setConfidence] = useState(INITIAL_CONFIDENCE);
  const [captureIntervalMs, setCaptureIntervalMs] = useState(INITIAL_INTERVAL_MS);
  const confidenceRef = useRef(confidence);

  // Detection results state
  const [detections, setDetections] = useState([]);
  const [detectionCount, setDetectionCount] = useState(0);
  const [inferenceTimeMs, setInferenceTimeMs] = useState(null);
  const [roundTripTimeMs, setRoundTripTimeMs] = useState(null);
  const [sourceFrameSize, setSourceFrameSize] = useState({
    width: 640,
    height: 480,
  });
  const [lastUpdatedAt, setLastUpdatedAt] = useState(null);

  // Backend connection & model readiness state
  const [backendStatus, setBackendStatus] = useState("connecting"); // 'connecting' | 'ready' | 'loading' | 'disconnected'
  const [healthInfo, setHealthInfo] = useState(null);
  const [supportedClasses, setSupportedClasses] = useState([]);

  // Keep confidenceRef synchronized so slider updates apply on the very next frame
  useEffect(() => {
    confidenceRef.current = confidence;
  }, [confidence]);

  // Check backend health and supported COCO classes
  const checkBackendHealth = useCallback(async () => {
    try {
      const health = await fetchHealth();
      if (!mountedRef.current) return;
      setHealthInfo(health);
      if (health.model_loaded && health.model_ready !== false) {
        setBackendStatus("ready");
      } else {
        setBackendStatus("loading");
      }
    } catch (err) {
      if (!mountedRef.current) return;
      setBackendStatus("disconnected");
      setHealthInfo((prev) => ({
        ...(prev || {}),
        error: err.message,
      }));
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    checkBackendHealth();
    const healthTimer = setInterval(checkBackendHealth, 5000);
    return () => {
      mountedRef.current = false;
      clearInterval(healthTimer);
    };
  }, [checkBackendHealth]);

  useEffect(() => {
    if (backendStatus === "ready" && supportedClasses.length === 0) {
      fetchClasses()
        .then((res) => {
          if (mountedRef.current && Array.isArray(res.class_names)) {
            setSupportedClasses(res.class_names);
          }
        })
        .catch(() => {
          // Non-fatal; classes will be retried when needed
        });
    }
  }, [backendStatus, supportedClasses.length]);

  // Stop webcam tracks cleanly
  const stopCamera = useCallback(() => {
    setIsDetecting(false);
    setDetectionStatus("idle");
    setDetectionError(null);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    inFlightRef.current = false;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // Ignore track stop errors
        }
      });
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraStatus("idle");
    setVideoDimensions({ width: 0, height: 0 });
    setDetections([]);
    setDetectionCount(0);
    setInferenceTimeMs(null);
    setRoundTripTimeMs(null);
  }, []);

  // Ensure tracks are stopped on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  // Start webcam and automatically begin detection
  const startCamera = useCallback(async () => {
    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices ||
      typeof navigator.mediaDevices.getUserMedia !== "function"
    ) {
      setCameraStatus("error");
      setCameraError(
        "Your browser does not support webcam access via the MediaDevices API. Ensure you are running on localhost or HTTPS."
      );
      return;
    }

    setCameraError(null);
    setDetectionError(null);
    setCameraStatus("starting");

    try {
      // Stop any existing stream before requesting a new one
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        const vw = videoRef.current.videoWidth || 640;
        const vh = videoRef.current.videoHeight || 480;
        setVideoDimensions({ width: vw, height: vh });
      }

      setCameraStatus("active");
      setIsDetecting(true);
      setDetectionStatus("running");
    } catch (err) {
      if (!mountedRef.current) return;
      setCameraStatus("error");
      setCameraError(formatCameraError(err));
      setIsDetecting(false);
      setDetectionStatus("idle");
    }
  }, []);

  // Pause detection without stopping the camera
  const pauseDetection = useCallback(() => {
    if (cameraStatus !== "active") return;
    setIsDetecting(false);
    setDetectionStatus("paused");
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    inFlightRef.current = false;
  }, [cameraStatus]);

  // Resume detection on the active camera stream
  const resumeDetection = useCallback(() => {
    if (cameraStatus !== "active") return;
    setDetectionError(null);
    setIsDetecting(true);
    setDetectionStatus("running");
  }, [cameraStatus]);

  const toggleDetection = useCallback(() => {
    if (isDetecting) {
      pauseDetection();
    } else {
      resumeDetection();
    }
  }, [isDetecting, pauseDetection, resumeDetection]);

  // Single frame capture and inference request (non-overlapping)
  const captureAndDetectFrame = useCallback(async () => {
    if (!mountedRef.current) return;
    if (inFlightRef.current) {
      // Prevent overlapping requests when inference or network is slower than interval
      return;
    }

    const videoEl = videoRef.current;
    if (
      !videoEl ||
      videoEl.readyState < 2 ||
      videoEl.videoWidth <= 0 ||
      videoEl.videoHeight <= 0
    ) {
      return;
    }

    inFlightRef.current = true;
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const rawWidth = videoEl.videoWidth;
      const rawHeight = videoEl.videoHeight;

      // Resize frame before upload if wider than MAX_UPLOAD_WIDTH while preserving aspect ratio
      const scale =
        rawWidth > MAX_UPLOAD_WIDTH ? MAX_UPLOAD_WIDTH / rawWidth : 1.0;
      const targetWidth = Math.max(1, Math.round(rawWidth * scale));
      const targetHeight = Math.max(1, Math.round(rawHeight * scale));

      if (!captureCanvasRef.current) {
        captureCanvasRef.current = document.createElement("canvas");
      }
      const canvas = captureCanvasRef.current;
      if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
      }

      const ctx = canvas.getContext("2d", { willReadFrequently: false });
      if (!ctx) {
        return;
      }
      ctx.drawImage(videoEl, 0, 0, targetWidth, targetHeight);

      const frameBlob = await canvasToJpegBlob(canvas, 0.82);
      const requestStart = performance.now();

      const result = await detectObjects(
        frameBlob,
        confidenceRef.current,
        controller.signal
      );

      const rttMs = Math.round(performance.now() - requestStart);

      if (!mountedRef.current || !isDetecting) return;

      const uploadedWidth =
        result?.image_dimensions?.width || result?.image_width || targetWidth;
      const uploadedHeight =
        result?.image_dimensions?.height || result?.image_height || targetHeight;

      setSourceFrameSize({
        width: uploadedWidth,
        height: uploadedHeight,
      });
      setDetections(Array.isArray(result.detections) ? result.detections : []);
      setDetectionCount(
        typeof result.detection_count === "number"
          ? result.detection_count
          : (result.detections || []).length
      );
      setInferenceTimeMs(
        typeof result.inference_time_ms === "number"
          ? result.inference_time_ms
          : null
      );
      setRoundTripTimeMs(rttMs);
      setLastUpdatedAt(new Date());
      setDetectionError(null);
      setDetectionStatus("running");
      setBackendStatus("ready");
    } catch (err) {
      if (err.name === "AbortError" || !mountedRef.current) {
        return;
      }
      setDetectionError(err.message || "Detection request failed.");
      setDetectionStatus("error");
      if (err.status === 0) {
        setBackendStatus("disconnected");
      } else if (err.status === 503) {
        setBackendStatus("loading");
      }
    } finally {
      inFlightRef.current = false;
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  }, [isDetecting]);

  // Controlled detection interval loop
  useEffect(() => {
    if (cameraStatus !== "active" || !isDetecting) {
      return undefined;
    }

    // Trigger immediate frame check, then schedule at controlled interval
    captureAndDetectFrame();
    const intervalId = setInterval(captureAndDetectFrame, captureIntervalMs);

    return () => {
      clearInterval(intervalId);
    };
  }, [cameraStatus, isDetecting, captureIntervalMs, captureAndDetectFrame]);

  const handleVideoMetadataLoaded = useCallback(() => {
    if (videoRef.current) {
      setVideoDimensions({
        width: videoRef.current.videoWidth || 640,
        height: videoRef.current.videoHeight || 480,
      });
    }
  }, []);

  return {
    videoRef,
    cameraStatus,
    cameraError,
    videoDimensions,
    isDetecting,
    detectionStatus,
    detectionError,
    confidence,
    setConfidence,
    captureIntervalMs,
    setCaptureIntervalMs,
    detections,
    detectionCount,
    inferenceTimeMs,
    roundTripTimeMs,
    sourceFrameSize,
    lastUpdatedAt,
    backendStatus,
    healthInfo,
    supportedClasses,
    startCamera,
    stopCamera,
    pauseDetection,
    resumeDetection,
    toggleDetection,
    checkBackendHealth,
    handleVideoMetadataLoaded,
  };
}
