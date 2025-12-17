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

    ctx->ort->CreateSession(ctx->env, text2mel_path, ctx->sess_opts, &ctx->text2mel_session);
    if (!ctx->text2mel_session) { 
        fprintf(stderr, "Failed to create text2mel session\n"); 
        ctx->ort->ReleaseMemoryInfo(ctx->meminfo);
        ctx->ort->ReleaseSessionOptions(ctx->sess_opts);
        ctx->ort->ReleaseEnv(ctx->env);
        return 1; 
    }
    ctx->ort->CreateSession(ctx->env, vocoder_path, ctx->sess_opts, &ctx->vocoder_session);

    if (!ctx->vocoder_session) {
        fprintf(stderr, "Failed to create vocoder session\n");
        ctx->ort->ReleaseSession(ctx->text2mel_session);
        ctx->ort->ReleaseMemoryInfo(ctx->meminfo);
        ctx->ort->ReleaseSessionOptions(ctx->sess_opts);
        ctx->ort->ReleaseEnv(ctx->env);
        return 1;
    }
    return 0;
}

// Inference from mel2text model.
int run_mel2text_session(DutchTTSContext * ctx, IntVec ids,
                        OrtValue* mel_tensor){
    
    int64_t input_ids[ids.len];       
    int64_t input_ids_shape[] = {1, ids.len};
    int64_t input_len[] = {ids.len};
    int64_t input_len_shape[] = {1};

    for (size_t i = 0; i < ids.len; ++i) {
        input_ids[i] = (int64_t)ids.data[i];
    }


    OrtValue* ids_tensor = NULL;
    OrtValue* len_tensor = NULL;

    ctx->ort->CreateTensorWithDataAsOrtValue(meminfo, input_ids, sizeof(input_ids),
                                         input_ids_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &ids_tensor);
    
    ctx->ort->CreateTensorWithDataAsOrtValue(meminfo, input_len, sizeof(input_len),
                                         input_len_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &len_tensor);

    const char* glow_in_names[] = {"input1", "input2"};
    const OrtValue* glow_in_vals[] = {ids_tensor, len_tensor};
    const char* glow_out_names[] = {"output"};
    mel_tensor = NULL;

    ctx->ort->Run(glow_sess, NULL, glow_in_names, glow_in_vals, 2, glow_out_names, 1, &mel_tensor);
    if (!mel_tensor) { 
        fprintf(stderr, "GlowTTS Run failed or returned NULL mel\n"); 
        ctx->ort->ReleaseSession(ctx->text2mel_session);
        ctx->ort->ReleaseSession(ctx->vocoder_session);
        ctx->ort->ReleaseMemoryInfo(ctx->meminfo);
        ctx->ort->ReleaseSessionOptions(ctx->sess_opts);
        ctx->ort->ReleaseEnv(ctx->env);
        ctx->ort->ReleaseValue(ids_tensor);
        ctx->ort->ReleaseValue(len_tensor);
        ctx->ort->ReleaseValue(glow_in_vals);
        free(glow_in_names);
        free(glow_out_names);

        return 1; 
    }
    ctx->ort->ReleaseValue(ids_tensor);
    ctx->ort->ReleaseValue(len_tensor);
    ctx->ort->ReleaseValue(glow_in_vals);
    free(glow_in_names);
    free(glow_out_names);
    return 0;

}

int remove_zeros(DutchTTSContext * ctx, OrtValue* mel_tensor,
                float* mel_data){

}

int run_vocoder_session(DutchTTSContext * ctx, float* mel_data,
                        float* audio_data){

}