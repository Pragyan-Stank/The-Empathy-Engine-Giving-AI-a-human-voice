'use strict';

document.addEventListener('DOMContentLoaded', () => {
    const form         = document.getElementById('synth-form');
    const textInput    = document.getElementById('text-input');
    const charCount    = document.getElementById('char-count');
    const emotionSel   = document.getElementById('emotion-override');
    const intensitySlider = document.getElementById('intensity-slider');
    const intensityVal = document.getElementById('intensity-val');
    const rateSlider   = document.getElementById('rate-slider');
    const pitchSlider  = document.getElementById('pitch-slider');
    const volumeSlider = document.getElementById('volume-slider');
    const rateVal      = document.getElementById('rate-val');
    const pitchVal     = document.getElementById('pitch-val');
    const volumeVal    = document.getElementById('volume-val');
    const resetBtn     = document.getElementById('reset-controls');
    const generateBtn  = document.getElementById('generate-btn');
    const errorBox     = document.getElementById('error-box');
    const resultPanel  = document.getElementById('result-panel');

    // Result elements
    const resEmotion   = document.getElementById('res-emotion');
    const resBadge     = document.getElementById('sentiment-badge');
    const resConf      = document.getElementById('res-confidence');
    const resIntensity = document.getElementById('res-intensity');
    const chipRate     = document.getElementById('chip-rate');
    const chipPitch    = document.getElementById('chip-pitch');
    const chipVolume   = document.getElementById('chip-volume');
    const audioEl      = document.getElementById('audio-output');
    const resSSML      = document.getElementById('res-ssml');

    // ── Character counter ────────────────────────────────────────────
    textInput.addEventListener('input', () => {
        charCount.textContent = textInput.value.length;
    });

    // ── Intensity slider ─────────────────────────────────────────────
    intensitySlider.addEventListener('input', () => {
        intensityVal.textContent = parseFloat(intensitySlider.value).toFixed(2);
    });

    // ── Voice control sliders ────────────────────────────────────────
    function formatRate(v)   { return v === 0 ? 'auto' : (v > 0 ? `+${v}%` : `${v}%`); }
    function formatPitch(v)  { return v === 0 ? 'auto' : (v > 0 ? `+${v}st` : `${v}st`); }
    function formatVolume(v) { return v === 0 ? 'auto' : (v > 0 ? `+${v}dB` : `${v}dB`); }

    rateSlider.addEventListener('input', () => {
        const v = parseFloat(rateSlider.value);
        rateSlider.dataset.auto = (v === 0) ? 'true' : 'false';
        rateVal.textContent = formatRate(v);
    });

    pitchSlider.addEventListener('input', () => {
        const v = parseFloat(pitchSlider.value);
        pitchSlider.dataset.auto = (v === 0) ? 'true' : 'false';
        pitchVal.textContent = formatPitch(v);
    });

    volumeSlider.addEventListener('input', () => {
        const v = parseFloat(volumeSlider.value);
        volumeSlider.dataset.auto = (v === 0) ? 'true' : 'false';
        volumeVal.textContent = formatVolume(v);
    });

    // ── Reset sliders ────────────────────────────────────────────────
    function resetSliders() {
        rateSlider.value   = 0;  rateSlider.dataset.auto   = 'true'; rateVal.textContent   = 'auto';
        pitchSlider.value  = 0;  pitchSlider.dataset.auto  = 'true'; pitchVal.textContent  = 'auto';
        volumeSlider.value = 0;  volumeSlider.dataset.auto = 'true'; volumeVal.textContent = 'auto';
    }

    resetBtn.addEventListener('click', resetSliders);

    // ── Set sliders to detected prosody (after response) ──────────────
    function applyProsodyToSliders(prosody) {
        // Only update sliders that haven't been manually moved
        if (rateSlider.dataset.auto === 'true') {
            const rv = parsePct(prosody.rate);
            rateSlider.value = rv;
            rateVal.textContent = formatRate(rv);
        }
        if (pitchSlider.dataset.auto === 'true') {
            const pv = parseSt(prosody.pitch);
            pitchSlider.value = pv;
            pitchVal.textContent = formatPitch(pv);
        }
        if (volumeSlider.dataset.auto === 'true') {
            const vv = parseDb(prosody.volume);
            volumeSlider.value = vv;
            volumeVal.textContent = formatVolume(vv);
        }
    }

    function parsePct(s)  { if (!s || s === 'default') return 0; return parseInt(s.replace('%','').replace('+','')) || 0; }
    function parseSt(s)   { if (!s || s === 'default') return 0; return parseFloat(s.replace('st','').replace('+','')) || 0; }
    function parseDb(s)   { if (!s || s === 'default') return 0; return parseFloat(s.replace('dB','').replace('+','')) || 0; }

    // ── Build payload ────────────────────────────────────────────────
    function buildPayload() {
        const rateN   = parseFloat(rateSlider.value);
        const pitchN  = parseFloat(pitchSlider.value);
        const volumeN = parseFloat(volumeSlider.value);

        const payload = {
            text:             textInput.value.trim(),
            emotion_override: emotionSel.value || null,
            intensity:        parseFloat(intensitySlider.value),
        };

        if (rateSlider.dataset.auto !== 'true' && rateN !== 0)
            payload.rate_override = formatRate(rateN);
        if (pitchSlider.dataset.auto !== 'true' && pitchN !== 0)
            payload.pitch_override = pitchN > 0 ? `+${pitchN}st` : `${pitchN}st`;
        if (volumeSlider.dataset.auto !== 'true' && volumeN !== 0)
            payload.volume_override = volumeN > 0 ? `+${volumeN}dB` : `${volumeN}dB`;

        return payload;
    }

    // ── Form submit ──────────────────────────────────────────────────
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!textInput.value.trim()) return;

        errorBox.classList.add('hidden');
        resultPanel.classList.add('hidden');
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating…';

        try {
            const res = await fetch('/api/v1/synthesize', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify(buildPayload()),
            });

            const data = await res.json();

            if (!res.ok || !data.success) {
                throw new Error(data.error || data.detail || 'Synthesis failed.');
            }

            // Populate results
            resEmotion.textContent   = data.detected_emotion;
            resBadge.textContent     = data.sentiment;
            resConf.textContent      = Math.round(data.confidence * 100) + '%';
            resIntensity.textContent = data.intensity.toFixed(2);

            chipRate.textContent   = `Rate: ${data.prosody.rate}`;
            chipPitch.textContent  = `Pitch: ${data.prosody.pitch}`;
            chipVolume.textContent = `Vol: ${data.prosody.volume}`;

            resSSML.textContent = data.ssml_preview || '';

            // Audio
            audioEl.src = data.audio_url + '?t=' + Date.now();
            audioEl.load();

            resultPanel.classList.remove('hidden');
            resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

            // Update sliders to reflect what was actually used
            applyProsodyToSliders(data.prosody);

            // Auto-play (may be blocked by browser policy)
            audioEl.play().catch(() => {});

        } catch (err) {
            errorBox.textContent = err.message;
            errorBox.classList.remove('hidden');
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Speech';
        }
    });
});
