// Web Audio API synthesized alert sound (zero external asset dependencies)
let audioCtx = null;

function getAudioContext() {
  if (typeof window === "undefined") return null;
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === "suspended") {
    audioCtx.resume().catch(() => {});
  }
  return audioCtx;
}

/**
 * Play a warning alert chime when an item goes missing.
 */
export function playMissingAlertSound() {
  try {
    const ctx = getAudioContext();
    if (!ctx) return;

    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "sine";
    // Double-tone alert: 520Hz -> 390Hz
    osc.frequency.setValueAtTime(520, now);
    osc.frequency.setValueAtTime(390, now + 0.12);

    gain.gain.setValueAtTime(0.2, now);
    gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + 0.36);
  } catch {
    // Non-fatal if audio context is blocked by browser policy
  }
}

/**
 * Play a gentle success chime when a missing item is restored.
 */
export function playRestoredAlertSound() {
  try {
    const ctx = getAudioContext();
    if (!ctx) return;

    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "sine";
    // Uplifting chime: 440Hz -> 660Hz
    osc.frequency.setValueAtTime(440, now);
    osc.frequency.setValueAtTime(660, now + 0.1);

    gain.gain.setValueAtTime(0.18, now);
    gain.gain.exponentialRampToValueAtTime(0.01, now + 0.3);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + 0.31);
  } catch {
    // Non-fatal
  }
}

// Siren alarm tracking
let sirenInterval = null;
let sirenTimeout = null;

/**
 * Start continuous European/police-style siren alarm using Web Audio oscillators.
 * Automatically shuts off after maxDurationSeconds (default: 30s) as a safety measure.
 */
export function startSirenAlert(maxDurationSeconds = 30) {
  if (sirenInterval) return; // already active

  try {
    const ctx = getAudioContext();
    if (!ctx) return;

    let isHigh = false;

    const playTone = () => {
      try {
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.type = "sawtooth";
        const freq = isHigh ? 960 : 720;
        isHigh = !isHigh;

        osc.frequency.setValueAtTime(freq, now);
        gain.gain.setValueAtTime(0.28, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.28);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start(now);
        osc.stop(now + 0.29);
      } catch {
        // ignore
      }
    };

    // Play immediately and repeat every 300ms
    playTone();
    sirenInterval = setInterval(playTone, 300);

    // Auto-silence safety timer
    sirenTimeout = setTimeout(() => {
      stopSirenAlert();
    }, maxDurationSeconds * 1000);
  } catch {
    // Non-fatal
  }
}

/**
 * Stop active siren alarm immediately.
 */
export function stopSirenAlert() {
  if (sirenInterval) {
    clearInterval(sirenInterval);
    sirenInterval = null;
  }
  if (sirenTimeout) {
    clearTimeout(sirenTimeout);
    sirenTimeout = null;
  }
}

/**
 * Check if siren is currently sounding.
 */
export function isSirenActive() {
  return sirenInterval !== null;
}


