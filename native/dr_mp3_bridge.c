/* GenMelodies memory-only dr_mp3 bridge. Original code, MIT licensed. */
#define DR_MP3_IMPLEMENTATION
#define DR_MP3_NO_STDIO
#include "../vendor/dr_mp3.h"
#include <stdint.h>

#if defined(_WIN32)
#define GM_EXPORT __declspec(dllexport)
#else
#define GM_EXPORT __attribute__((visibility("default")))
#endif

/* Ownership: a successful call returns dr_mp3-allocated PCM. Always release
   it with gm_free_pcm, including when the Python side rejects the metadata. */
GM_EXPORT float *gm_decode_mp3(const unsigned char *data, size_t size,
                              unsigned int *channels, unsigned int *rate,
                              uint64_t *frames)
{
    drmp3_config config;
    drmp3_uint64 frame_count = 0;
    float *pcm;
    if (!data || !size || !channels || !rate || !frames) return NULL;
    *channels = 0;
    *rate = 0;
    *frames = 0;
    pcm = drmp3_open_memory_and_read_pcm_frames_f32(data, size, &config,
                                                 &frame_count, NULL);
    if (!pcm) return NULL;
    if (!frame_count || config.channels < 1 || config.channels > 2 ||
        config.sampleRate < 8000 || config.sampleRate > 192000 ||
        frame_count > SIZE_MAX / (sizeof(float) * config.channels)) {
        drmp3_free(pcm, NULL);
        return NULL;
    }
    *channels = config.channels;
    *rate = config.sampleRate;
    *frames = frame_count;
    return pcm;
}

GM_EXPORT void gm_free_pcm(float *pcm)
{
    drmp3_free(pcm, NULL);
}
