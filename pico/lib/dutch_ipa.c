#include <dutch_ipa.h>

// Global variable definitions
int TEXT_MODE = 0;                  
int PHONEM_MODE = espeakPHONEMES_IPA;

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
char* text_to_phonemes(const char *text){
    const void *textptr = (const void *)text;
    char * ipas = espeak_TextToPhonemes(&textptr, TEXT_MODE, espeakPHONEMES_IPA);
    
    return ipas;
}