/**************************************************************
 * @file    realtime.c
 * @ingroup src
 * @brief   Process input data in realtime
 * @author  Adam Miller
 *
 * 9/15/12
 *************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <portaudio.h>
#include <Accelerate/Accelerate.h>
#include "audio_tools.h"


/* Setup basic options */
#define SAMPLE_RATE         44100
#define FRAMES_PER_BUFFER   32 /* smaller buffer for realtime */
#define NUM_SECONDS         20
#define NUM_CHANNELS        2   // Stereo

/* Setup the sample format. */
#define PA_SAMPLE_TYPE paFloat32
typedef float SAMPLE;
#define SAMPLE_SILENCE      0.0f
#define SAMPLE_FORMAT       "%.8f"

/* Constants for devices */
#define MICROPHONE      0
#define BUILTIN_IN      1
#define BUILTIN_OUT     2

#define ABS(a) ( a < 0 ? a = -a : a = a )
#define FREQUENCY .0005 // Frequency of sin to modulate with

/* Encapsulate information needed to access and transfer
 * data into one struct. This way the struct can be
 * passed around instead of a bunch of different variables. */
typedef struct
{
    int     frameIndex;
    int     maxFrameIndex;
    SAMPLE  *samples;
    SAMPLE  *sin_table;
} paData;

static int realtimeModulateCallback(  const void *inputBuffer,
                            void *outputBuffer,
                            unsigned long framesPerBuffer,
                            const PaStreamCallbackTimeInfo *timeInfo,
                            PaStreamCallbackFlags statusFlags,
                            void *userData);
static int playCallback(  const void *inputBuffer,
                            void *outputBuffer,
                            unsigned long framesPerBuffer,
                            const PaStreamCallbackTimeInfo *timeInfo,
                            PaStreamCallbackFlags statusFlags,
                            void *userData);
void setupInputParameters( PaStreamParameters * );
void setupOutputParameters( PaStreamParameters *);
void processData( paData *);



/*****************************************************************/
int main()
{
    PaStreamParameters  inputParameters,
                        outputParameters;
    PaStream            *stream;
    PaError             err = paNoError;
    paData              data;
    int                 i, totalFrames, numSamples, numBytes;
    SAMPLE              max, val;
    double              average;

    /* Call Pa_Initialize() and recorded error. This call is
     * necessary to set up everything properly with PortAudio */
    err = Pa_Initialize();
    if (err != paNoError) {
        pa_fatal("Error during intialization", err);
    }

    /* Setup parameters */
    setupInputParameters(&inputParameters);
    setupOutputParameters(&outputParameters);

    /* Determine sizes of data buffer */
    data.maxFrameIndex = totalFrames = NUM_SECONDS * SAMPLE_RATE;
    data.frameIndex = 0;
    numSamples = totalFrames * NUM_CHANNELS;
    numBytes = numSamples * sizeof(SAMPLE);

    /* Setup buffer to hold data */
    data.samples = (SAMPLE *) malloc(numBytes);
    if (data.samples == NULL) fatal("allocating buffer");
    for (i=0; i<numSamples; i++) data.samples[i] = 0; // clear

    /* Setup sin function table */
    numBytes = totalFrames * sizeof(SAMPLE);
    data.sin_table = (SAMPLE *)malloc(numBytes);
    if (data.sin_table == NULL) fatal("Allocating sin_table");
    for (i=0; i<totalFrames; i++) data.sin_table[i] = sin(FREQUENCY*i);
    

    /* Look at device info */
    printf("in: %d, out: %d\n", Pa_GetDefaultInputDevice(), Pa_GetDefaultOutputDevice());
    printf("Number of devices: %d\n", Pa_GetDeviceCount());
    for (i = 0; i < Pa_GetDeviceCount(); i++) {
        printf("%d: %s\n",i, Pa_GetDeviceInfo(i)->name);
    }


    /* Begin recording audio */
    err = Pa_OpenStream(
            &stream,
            &inputParameters,
            &outputParameters,
            SAMPLE_RATE,
            FRAMES_PER_BUFFER,
            paClipOff,          /* Don't worry about clipping */
            realtimeModulateCallback,
            (void *) (&data) );
    if (err!=paNoError) pa_fatal("Pa_OpenStream", err);
    err = Pa_StartStream(stream);
    if (err != paNoError) pa_fatal("Pa_StartStream", err);
    printf("\n--- Now recording into microphone ---\n");
    fflush(stdout);

    /* Check to see if the stream is still recording.
     * Pa__IsStreamActive will return 1 if the callback function
     * returns paContinue. Otherwise it will return 0 */
    while((err = Pa_IsStreamActive(stream)) == 1) {
        Pa_Sleep(1000);
        printf("frameindex = %d\n", data.frameIndex); fflush(stdout);
    }
    if (err != paNoError) pa_fatal("Pa_IsStreamActive", err);

    /* Close the open stream */
    err = Pa_CloseStream(stream);
    if (err != paNoError) pa_fatal("Pa_CloseStream", err);

    /* Terminate PortAudio and free memory */
    Pa_Terminate();
    if (data.samples) free(data.samples);

    return err;
}

/*  realtimeModulateCallback
 *    Takes input data from inputBuffer and multiplies is by a
 *    sin function. It then sends to the output buffer
 */
static int realtimeModulateCallback(    const void *inputBuffer,
                                void *outputBuffer,
                                unsigned long framesPerBuffer,
                                const PaStreamCallbackTimeInfo* timeInfo,
                                PaStreamCallbackFlags statusFlags,
                                void *userData)
{
    /* Check for NULL buffers */
    if (inputBuffer == NULL)
        fatal_terminate("recordCallback: NULL inputBuffer");
    if (outputBuffer == NULL)
        fatal_terminate("recordCallback: NULL outputBuffer");

    /* Setup tracking pointers/variables */
    paData *data = (paData *)userData;
    const SAMPLE *in_ptr = (const SAMPLE *)inputBuffer;
    SAMPLE *out_ptr = (SAMPLE *)outputBuffer;
    SAMPLE *data_ptr = &data->samples[data->frameIndex * NUM_CHANNELS];
    SAMPLE *sin_ptr = &data->sin_table[data->frameIndex];

    int finished;
    unsigned int framesLeft = data->maxFrameIndex - data->frameIndex;
    
    /* See if there is more than buffer's worth of data 
     * still to be read */
    if (framesLeft < framesPerBuffer) {
        finished = paComplete;
    } else {
        framesLeft = framesPerBuffer; // limit to size of buffer
        finished = paContinue;
    }

    /* Transfer from input into output and data buffers */
    for (int i = 0; i < framesLeft; i++) {
        for (int j=0; j < NUM_CHANNELS; j++) {
            SAMPLE modified = *in_ptr++ * *sin_ptr;
            *out_ptr++ = modified;
            *data_ptr++ = modified;
        }
        sin_ptr++;
    }
    
    /* Update the information contained in the paData struct */
    data->frameIndex += framesLeft;
    return finished;
}



/* setupInputParameters
 * Takes in a pointer to a PaStreamParameters struct. This 
 * function will initailize all the members of the passed
 * in struct. */
void setupInputParameters( PaStreamParameters *in_pars )
{
    /* Check if microcone available. If so, use */
    if (Pa_GetDeviceCount() > 3) {
        if (strcmp(Pa_GetDeviceInfo(3)->name, 
            "Microcone USB 2.0 Audio In") == 0) {
            in_pars->device = 3;
            printf("Using microcone\n");
        } 
    } else {
        in_pars->device = Pa_GetDefaultInputDevice();
        printf("Unable to use microcone\n");
    }

    /* Check to see if device choice failed */
    if (in_pars->device == paNoDevice) 
        fatal_terminate("No default input device");
    /* Setup remaining parameters */
    //in_pars->channelCount = 
    //    Pa_GetDeviceInfo(in_pars->device)->maxInputChannels;
    in_pars->channelCount = NUM_CHANNELS;
    printf("Using %d channels", in_pars->channelCount);
    in_pars->sampleFormat = PA_SAMPLE_TYPE;
    in_pars->suggestedLatency = 
       Pa_GetDeviceInfo(in_pars->device)->defaultLowInputLatency;
    in_pars->hostApiSpecificStreamInfo = NULL;
}




/* setupOutputParameters
 * Takes in a pointer to a PaStreamParameters struct. This 
 * function will initailize all the members of the passed
 * in struct. */
void setupOutputParameters( PaStreamParameters *out_pars )
{
    out_pars->device = Pa_GetDefaultOutputDevice();
    if (out_pars->device == paNoDevice)
        fatal_terminate("No default output device");
    out_pars->channelCount = NUM_CHANNELS;
    out_pars->sampleFormat = PA_SAMPLE_TYPE;
    out_pars->suggestedLatency = 
        Pa_GetDeviceInfo(out_pars->device)->defaultLowOutputLatency;
    out_pars->hostApiSpecificStreamInfo = NULL;
}

/* processData
 * Runs the samples in the input data struct through a sin
 * wave and then stores the new data back into the struct. */
void processData( paData *data )
{
    const float FREQ = 0.0002;
    unsigned long i;

    SAMPLE *curr = data->samples;

    for (i = 0; i < data->maxFrameIndex; i++) {
        *curr++ *= sin(FREQ*(float)i);
        if (NUM_CHANNELS == 2) *curr++ *= sin(FREQ*(float)i);
    }
}

