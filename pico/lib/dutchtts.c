#include <dutchtts.h>

// Session initialization
int init_session(DutchTTSContext * ctx, const char * text2mel_path,
                const char * vocoder_path, const char * config_path){
    
    ctx->ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    ctx->env = NULL;
    ctx->sess_opts = NULL;
    ctx->meminfo = NULL;

    ctx->ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "tts", &env);
    ctx->ort->CreateSessionOptions(&sess_opts);
    ctx->ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &meminfo);
    
    ctx->text2mel_session = NULL;
    ctx->vocoder_session = NULL;

    ort->CreateSession(ctx->env, text2mel_path, ctx->sess_opts, &ctx->text2mel_session);
    if (!ctx->text2mel_session) { 
        fprintf(stderr, "Failed to create text2mel session\n"); 
        ort->ReleaseMemoryInfo(ctx->meminfo);
        ort->ReleaseSessionOptions(ctx->sess_opts);
        ort->ReleaseEnv(ctx->env);
        return 1; 
    }
    ort->CreateSession(ctx->env, vocoder_path, ctx->sess_opts, &ctx->vocoder_session);

    if (!ctx->vocoder_session) {
        fprintf(stderr, "Failed to create vocoder session\n");
        ort->ReleaseSession(ctx->text2mel_session);
        ort->ReleaseMemoryInfo(ctx->meminfo);
        ort->ReleaseSessionOptions(ctx->sess_opts);
        ort->ReleaseEnv(ctx->env);
        return 1;
    }
    return 0;
}

int run_mel2text_session(DutchTTSContext * ctx, IntVec ids,
                        OrtValue* mel_tensor){

}

int renove_zeros(DutchTTSContext * ctx, OrtValue* mel_tensor,
                float* mel_data){

}

int run_vocoder_session(DutchTTSContext * ctx, float* mel_data,
                        float* audio_data){

}