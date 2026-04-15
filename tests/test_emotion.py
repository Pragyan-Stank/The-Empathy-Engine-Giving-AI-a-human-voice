import pytest
from app.services.emotion.mapping import get_prosody_base, get_base_emotion
from app.services.emotion.intensity import calculate_prosody
from app.services.tts.ssml_builder import SSMLBuilder

def test_emotion_mapping():
    assert get_base_emotion("happy") == "joy"
    assert get_base_emotion("frustrated") == "anger"
    assert get_base_emotion("unknown") == "neutral"

def test_prosody_calculation():
    # Joy base is +20% rate, +2st pitch, +2dB vol
    prosody_max = calculate_prosody("joy", 1.0)
    assert prosody_max["rate"] == "+20%"
    assert prosody_max["pitch"] == "+2.0st"
    
    prosody_half = calculate_prosody("joy", 0.5)
    assert prosody_half["rate"] == "+10%"
    assert prosody_half["pitch"] == "+1.0st"

def test_ssml_builder():
    builder = SSMLBuilder()
    prosody = {"rate": "+10%", "pitch": "+1st", "volume": "+1dB"}
    text = "Hello world! This is important."
    
    ssml = builder.build_ssml(text, prosody)
    
    # Check it wrapped properly
    assert "<speak>" in ssml
    assert "<prosody rate=\"+10%\" pitch=\"+1st\" volume=\"+1dB\">" in ssml
    assert "<emphasis level=\"strong\">important.</emphasis>" in ssml

def test_ssml_builder_neutral():
    builder = SSMLBuilder()
    prosody = {"rate": "default", "pitch": "default", "volume": "default"}
    text = "Just normal."
    ssml = builder.build_ssml(text, prosody)
    
    assert "<speak>" in ssml
    assert "<prosody" not in ssml # Should omit prosody tag if all defaults
