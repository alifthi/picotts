#include <dutch_ipa.h>

// espeak initialization.
int init_dutch_ipa(){

    if (espeak_Initialize(AUDIO_OUTPUT_RETRIEVAL, 0, NULL, 0) == -1) {
        fprintf(stderr, "Failed to initialize eSpeak-NG\n");
        return 1;
    }

    if (espeak_SetVoiceByName("nl") != EE_OK) {
        fprintf(stderr, "Failed to set Dutch voice\n");
        return 1;
    }

    return 0;
}

// Convert input text to phonems
int text_to_phonemes(const char *text, const void **textptr, char * ipas){
    const void *textptr = (const void *)text;
    ipas = espeak_TextToPhonemes(&textptr, TEXT_MODE, espeakPHONEMES_IPA);
    if (!ipas) {
        fprintf(stderr, "Failed to convert text to phonemes\n");
        return 1;
    }
    return 0;
}