document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('synth-form');
    const textInput = document.getElementById('text-input');
    const emotionOverride = document.getElementById('emotion-override');
    const intensitySlider = document.getElementById('intensity-slider');
    const intensityVal = document.getElementById('intensity-val');
    const generateBtn = document.getElementById('generate-btn');
    const errorBox = document.getElementById('error-box');
    const resultPanel = document.getElementById('result-panel');

    const uiCards = {
        emotion: document.getElementById('res-emotion'),
        sentimentBadge: document.getElementById('sentiment-badge'),
        confidence: document.getElementById('res-confidence'),
        rate: document.getElementById('res-rate'),
        pitch: document.getElementById('res-pitch'),
        volume: document.getElementById('res-volume'),
        ssml: document.getElementById('res-ssml'),
        audio: document.getElementById('audio-output')
    };

    intensitySlider.addEventListener('input', (e) => {
        intensityVal.textContent = parseFloat(e.target.value).toFixed(1);
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const text = textInput.value.trim();
        if (!text) return;

        // Reset UI state
        errorBox.classList.add('hidden');
        resultPanel.classList.add('hidden');
        generateBtn.disabled = true;
        generateBtn.textContent = 'Processing...';

        const payload = {
            text: text,
            emotion_override: emotionOverride.value || null,
            intensity: parseFloat(intensitySlider.value)
        };

        try {
            const response = await fetch('/api/v1/synthesize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Failed to synthesize text');
            }

            // Populate mapping
            uiCards.emotion.textContent = data.detected_emotion;
            uiCards.sentimentBadge.textContent = data.sentiment;
            uiCards.confidence.textContent = (data.confidence * 100).toFixed(0) + '%';
            
            uiCards.rate.textContent = data.prosody.rate;
            uiCards.pitch.textContent = data.prosody.pitch;
            uiCards.volume.textContent = data.prosody.volume;
            
            uiCards.ssml.textContent = data.ssml_preview;
            
            // Handle audio
            uiCards.audio.src = data.audio_url + '?t=' + new Date().getTime(); // bust cache on playback just in case
            uiCards.audio.load();

            // Reveal result
            resultPanel.classList.remove('hidden');
            
            // Auto-play attempt
            const playPromise = uiCards.audio.play();
            if (playPromise !== undefined) {
                playPromise.catch(_ => {
                    // Autoplay prevented
                });
            }

        } catch (err) {
            errorBox.textContent = err.message;
            errorBox.classList.remove('hidden');
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Speech';
        }
    });
});
