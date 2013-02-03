/**************************************************************
 * @file    record.c
 * @ingroup src
 * @brief   Record input to an array and then playback
 * @author  Adam Miller
 *
 * 9/14/12
 *************************************************************/

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <portaudio.h>
 #include "audio_tools.h"

/* Setup basic options */
#define SAMPLE_RATE         44100
#define FRAMES_PER_BUFFER   512
#define NUM_SECONDS         5
#define NUM_CHANNELS        2   // Stereo

/* Setup the sample format. */
#define PA_SAMPLE_TYPE paFloat32
typedef float SAMPLE;
#define SAMPLE_SILENCE      0.0f
#define SAMPLE_FORMAT       "%.8f"

#define ABS(a) ( a < 0 ? a = -a : a = a )



/* Encapsulate information needed to access and transfer
 * data into one struct. This way the struct can be
 * passed around instead of a bunch of different variables. */
typedef struct
{
    int     frameIndex;
    int     maxFrameIndex;
    SAMPLE  *samples;
} paData;

/* Function declarations. See bottom for implementations */
static int recordCallback(  const void *inputBuffer,
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

    /* Determine variable values */
    data.maxFrameIndex = totalFrames = NUM_SECONDS * SAMPLE_RATE;
    data.frameIndex = 0;
    numSamples = totalFrames * NUM_CHANNELS;
    numBytes = numSamples * sizeof(SAMPLE);

    /* Setup buffer to hold data */
    data.samples = (SAMPLE *)malloc(numBytes);
    if (data.samples == NULL) fatal("allocating buffer");
    for (i=0; i<numSamples; i++) data.samples[i] = 0;
    
    /* Call Pa_Initialize() and recorded error. This call is
     * necessary to set up everything properly with PortAudio */
    err = Pa_Initialize();
    if (err != paNoError) {
        pa_fatal("Pa_Initialize", err);
    }

    /* Setup input parameters */
    setupInputParameters(&inputParameters);

    /* Begin recording audio */
    err = Pa_OpenStream(
            &stream,
            &inputParameters,
            NULL,                /* &outputParameters */
            SAMPLE_RATE,
            FRAMES_PER_BUFFER,
            paClipOff,          /* Don't worry about clipping */
            recordCallback,
            (void *) (&data) );
    if (err!=paNoError) pa_fatal("Pa_OpenStream - recording ", err);
    err = Pa_StartStream(stream);
    if (err != paNoError) pa_fatal("Pa_StartStream - recording", err);
    printf("\n--- Now recording into microphone ---\n");
    fflush(stdout);

    /* Check to see if the stream is still recording.
     * Pa__IsStreamActive will return 1 if the callback function
     * returns paContinue. Otherwise it will return 0 */
    while((err = Pa_IsStreamActive(stream)) == 1) {
        Pa_Sleep(1000);
        //printf("frameindex = %d\n", data.frameIndex); fflush(stdout);
    }
    if (err != paNoError) pa_fatal("Pa_IsStreamActive - recording", err);

    /* Close the open stream */
    err = Pa_CloseStream(stream);
    if (err != paNoError) pa_fatal("Pa_CloseStream",err);

    /* Process audio */
    processData(&data);

    /********** Now playback recording ***********/
    data.frameIndex = 0; // Set back to beginning of data
    setupOutputParameters(&outputParameters);

    printf("\n--- Now playing back recorded audio---\n"); 
    fflush(stdout);
    err = Pa_OpenStream(
            &stream,
            NULL,       /* &inputParameters */
            &outputParameters,
            SAMPLE_RATE,
            FRAMES_PER_BUFFER,
            paClipOff,
            playCallback,
            (void *)&data );
    if (!stream) fatal_terminate("NULL stream before playback");
    if (err!=paNoError) pa_fatal("Pa_OpenStream - playback", err);
    if ((err = Pa_StartStream(stream)) != paNoError) 
		pa_fatal("Pa_StartStream - playback", err);
    printf("Waiting for playback to finish.\n"); fflush(stdout);
    while ((err = Pa_IsStreamActive(stream)) == 1) Pa_Sleep(1000);
    if (err != paNoError) pa_fatal("Pa_IsStreaActive - playback", err);
    
    /* Close stream now that done with playback */
    if ( (err = Pa_CloseStream(stream)) != paNoError) 
		pa_fatal("Pa_CloseStream - playback", err);
    printf("Playback completed.\n"); fflush(stdout);

    /* Terminate PortAudio and free memory */
    Pa_Terminate();
    if (data.samples) free(data.samples);

    return err;
}




/* recordCallback
 * This function serves as a callback function that will be 
 * called by the PortAudio engine when it is recording. The
 * function name should be passed in when the stream is
 * created, which will ensure that when it is opened the call-
 * back function is called. */
static int recordCallback(  const void *inputBuffer,
                            void *outputBuffer,
                            unsigned long framesPerBuffer,
                            const PaStreamCallbackTimeInfo* timeInfo,
                            PaStreamCallbackFlags statusFlags,
                            void *userData)
{
    /* Interpret the userData input as a parameter of type paData.
     * Then define a SAMPLE * ptr to point at the input buffer, so
     * we can avoid casting and such later on.
     * After this we set up the starting point of the output pointer
     * so that we can keep track of where we need to beginning
     * writing. Note that the index is multiplied by NUM_CHANNELS
     * since each row will have both channels, so you must increment
     * by two slots if it is stereo to get to the next frame.
     * We will read from the input buffer and write into the data
     * struct passed in from the stream. */
    if (inputBuffer == NULL)
        fatal_terminate("recordCallback: NULL inputBuffer");
    paData *data = (paData *)userData;
    const SAMPLE *in_ptr = (const SAMPLE *)inputBuffer;
    SAMPLE *out_ptr = &data->samples[data->frameIndex * NUM_CHANNELS];

    int finished;
    unsigned int framesLeft = data->maxFrameIndex - data->frameIndex;

    /* Check to see if the amount of data left is less than one
     * buffer's worth of data. If so, we should return paComplete,
     * indicating that the stream should stop invoking the callback
     * as soon as the remaining samples have been processed. If
     * there is more than one buffer's worth of data, we should
     * return paContinue, indicating that the stream should continue
     * invoking the callback function. */
    if (framesLeft < framesPerBuffer) {
        finished = paComplete;
    } else {
        framesLeft = framesPerBuffer; // limit to size of buffer
        finished = paContinue;
    }

    /* Check if an input source is actually present. If not simply
     * write "silent" samples into the output paData struct */
    if (in_ptr == NULL) { // No input
        for (int i = 0; i < framesLeft; i++) {
            *out_ptr++ = SAMPLE_SILENCE; // left 
            if (NUM_CHANNELS == 2) *out_ptr++ = SAMPLE_SILENCE;
        }
    } else { // input present
        for (int i = 0; i < framesLeft; i++) {
            *out_ptr++ = *in_ptr++; // left
            if (NUM_CHANNELS == 2) *out_ptr++ = *in_ptr++;
        }
    }
    
    /* Update the information contained in the paData struct */
    data->frameIndex += framesLeft;
    return finished;
}


/* playCallback
 * This callback function will be invoked by the system when
 * its associated stream is started. It will take data from
 * the passed in data struct (userData parameter) and write
 * the data into the provided outputbuffer. */
static int playCallback(    const void *inputBuffer,
                            void *outputBuffer,
                            unsigned long framesPerBuffer,
                            const PaStreamCallbackTimeInfo *timeInfo,
                            PaStreamCallbackFlags statusFlags,
                            void *userData )
{
    /* Setup data struct and ptrs for tracking in buffer */
    paData *data = (paData *)userData;
    SAMPLE *in_ptr = &data->samples[data->frameIndex*NUM_CHANNELS];
    SAMPLE *out_ptr = (SAMPLE *)outputBuffer;
    int finished;
    unsigned int i;

    /* Check how much data is left and whether it can all be 
     * contained in one bufer */
    unsigned int framesLeft = data->maxFrameIndex - data->frameIndex;
    if (framesLeft < framesPerBuffer) { // Less than buffer of data
        finished = paComplete;
    } else { // More than buffer's worth of data left
        framesLeft = framesPerBuffer;
        finished = paContinue;
    }

    if ( outputBuffer == NULL) 
        exit(1);
        //fatal_terminate("playCallback: NULL outputBuffer");
    for (i=0; i<framesLeft; i++) {
        *out_ptr++ = *in_ptr++;
        if (NUM_CHANNELS == 2) *out_ptr++ = *in_ptr++;
    }
 //   for ( ; i<framesPerBuffer; i++) { // fill rest w/ silence
 //       *out_ptr++ = SAMPLE_SILENCE;
 //       if (NUM_CHANNELS == 2) *out_ptr++ = SAMPLE_SILENCE;
 //   }

    data->frameIndex += framesLeft;
    return finished;
}




/* setupInputParameters
 * Takes in a pointer to a PaStreamParameters struct. This 
 * function will initailize all the members of the passed
 * in struct. */
void setupInputParameters( PaStreamParameters *in_pars )
{
    in_pars->device = Pa_GetDefaultInputDevice();
    if (in_pars->device == paNoDevice) 
        fatal_terminate("No default input device");
    in_pars->channelCount = NUM_CHANNELS;
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
    out_pars->device = 2;
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
