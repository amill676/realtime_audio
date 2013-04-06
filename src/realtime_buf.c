/**************************************************************
 * @file    realtime_sin.c
 * @ingroup src
 * @brief   Process input data in realtime and puts through
 *          a sin function before outputting.
 * @author  Adam Miller
 *
 * 9/15/12
 *************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <portaudio.h>
#include "audio_tools.h"
#include <portaudio/pa_util.h>
#include <portaudio/pa_ringbuffer.h>

/* Setup basic options */
#define SAMPLE_RATE         44100
#define FRAMES_PER_BUFFER   256 /* smaller buffer for realtime */
#define NUM_SECONDS         2
#define RING_BUFFER_SECONDS	0.5
#define NUM_CHANNELS        1   // Stereo

/* Setup the sample format. */
#define PA_SAMPLE_TYPE paFloat32
typedef float SAMPLE;
#define SAMPLE_SILENCE      0.0f
#define SAMPLE_FORMAT       "%.8f"

/* Constants for devices */
#define MICROPHONE      0
#define BUILTIN_IN      1
#define BUILTIN_OUT     2

/* debug flag */
#define DEBUG	0 

#define ABS(a) ( a < 0 ? (-1 * a) : a )
#define FREQUENCY 50 // Frequency of sin to modulate with

/* Encapsulate information needed to access and transfer
 * data into one struct. This way the struct can be
 * passed around instead of a bunch of different variables. */
typedef struct
{
	/* Ring buffer for raw input samples */
    PaUtilRingBuffer	in_rb;
	SAMPLE				*in_rb_data;

	/* Ring buffer for samples after putting through sin fcn */
	PaUtilRingBuffer	sin_rb;
	SAMPLE				*sin_rb_data;
} paData;

/* Function declarations */
static int sinCallback(  const void *inputBuffer,
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
void setupInputParametersWithCone( PaStreamParameters * );
void setupOutputParameters( PaStreamParameters *);
void processData( SAMPLE *, int, int);



/*****************************************************************/
int main()
{
    PaStreamParameters  inputParameters,
                        outputParameters;
    PaStream            *in_stream, *out_stream;
    PaError             err = paNoError;
    paData              data;
    int                 i, totalFrames, numSamples, numBytes;
    SAMPLE              max, val;
	SAMPLE				*calc_buffer;
    double              average;


	/* Determine size of ring buffers (both same size) */
	unsigned long num_samples = RING_BUFFER_SECONDS * SAMPLE_RATE;
	/* Round num_samples up to next power of 2 */
	int max_ind = 0;
	for (i=0; i < 8*sizeof(num_samples); i++) {
		if (num_samples & 0x1) max_ind = i;
		num_samples >>= 1;
	}
	num_samples = 0x1 << (max_ind + 1);
	if (DEBUG) printf("Size of ring buffer will be %lu\n", num_samples);
	const unsigned long num_bytes = num_samples * NUM_CHANNELS * sizeof(SAMPLE);

	/* Call Pa_Initialize() and recorded error. This call is
	 * necessary to set up everything properly with PortAudio */
	err = Pa_Initialize();
	if (err != paNoError) {
        pa_fatal("Error during intialization", err);
    }

    
	if (DEBUG) printf("Setting up input parameters... ");
    setupInputParametersWithCone(&inputParameters);
	if (DEBUG) printf("Succeeded using device: %s\n",
							Pa_GetDeviceInfo(inputParameters.device)->name);
	if (DEBUG) printf("Setting up output parameters... ");
    setupOutputParameters(&outputParameters);
	if (DEBUG) printf("Succeeded using device: %s\n",
							Pa_GetDeviceInfo(outputParameters.device)->name);

	/* Allocate space for ring buffers */
	if (DEBUG) printf("Allocating memory for ring buffers... ");
	if ((data.in_rb_data = (SAMPLE *)malloc(num_bytes)) == NULL)
		fatal("PaUtil_AllocateMemory: in_rb");
	if ((data.sin_rb_data = (SAMPLE *)malloc(num_bytes)) == NULL)
		fatal("PaUtil_AllocateMemory: sin_rb");
	if (DEBUG) printf("Succeeded\n");

	/* Allocate space for temporary calculation buffers */
	if (DEBUG) printf("Allocating buffer for temporary calculations... ");
	if ((calc_buffer = malloc(num_bytes)) == NULL)
		fatal("malloc: calc_buffer");
	if (DEBUG) printf("Succeeded\n");

	/* Initialize the ring buffers */
	if (DEBUG) printf("Initializing ring buffers... ");
	err = PaUtil_InitializeRingBuffer(&data.in_rb, 
			sizeof(SAMPLE) * NUM_CHANNELS, num_samples, data.in_rb_data);
	if (err != paNoError)
		pa_fatal("Initializing ring in ring buffer", err);
	err = PaUtil_InitializeRingBuffer(&data.sin_rb, 
			sizeof(SAMPLE) * NUM_CHANNELS, num_samples, data.sin_rb_data);
	if (err != paNoError)
		pa_fatal("Initializing ring in ring buffer", err);
	PaUtil_FlushRingBuffer(&data.in_rb);
	PaUtil_FlushRingBuffer(&data.sin_rb);
	if (DEBUG) printf("Succeeded\n");



    /* Setup streams */
    err = Pa_OpenStream( &in_stream, &inputParameters, &outputParameters, 
						SAMPLE_RATE, FRAMES_PER_BUFFER, paClipOff, 
						sinCallback, &data );
    if (err!=paNoError) pa_fatal("Pa_OpenStream - Input", err);

	/* Start streams */
	err = Pa_StartStream(in_stream);
	if (err != paNoError) pa_fatal("Pa_StartStream-in_stream", err);

	/* Compute while stream is active */
	ring_buffer_size_t read_num, write_num, num_complete, num_written,
						success;
	SAMPLE* data_ptr;
	while( (err = Pa_IsStreamActive(in_stream)) == 1) {
		/* Check if data is available */
		if ((read_num = PaUtil_GetRingBufferReadAvailable(&data.in_rb)) != 0) {
			if (DEBUG) printf("Data available in input ring buffer.\n");
			if (DEBUG) printf("Reading from input buffer... ");
			if (read_num > FRAMES_PER_BUFFER) read_num = FRAMES_PER_BUFFER;
			num_complete = PaUtil_ReadRingBuffer(&data.in_rb, calc_buffer, 
													read_num);
			if (DEBUG) printf("read %d samples.\n", num_complete);
			
			/* Perform calculations on samples */
			processData(calc_buffer, num_complete, NUM_CHANNELS);

			/* Check how much data can be written */
			write_num = PaUtil_GetRingBufferWriteAvailable(&data.sin_rb);
			if (write_num != 0) {
				if (DEBUG) printf("Room for data in output ring buffer.\n");
				/* Only write the number of frames we read in */
				if (write_num > num_complete)
					write_num = num_complete;
				num_written = 0;
				data_ptr = calc_buffer;
				if (DEBUG) printf("Writing data to output ring buffer... ");
				while (num_written < write_num) {
					success = 
						PaUtil_WriteRingBuffer(&data.sin_rb, data_ptr,
													write_num-num_written);
					num_written += success;
					data_ptr += success;
				} // end write while
				if (DEBUG) printf("Succeeded.\n");
			} // end write_num != 0 if
		} // end read_num = ... if

		Pa_Sleep(5); // Try to prevent fans from going crazy 
	} // end while
	if (err < 0) pa_fatal("Pa_IsStreamActive", err);

	/* Free memory associated with ring buffers */
	if (DEBUG) printf("Freeing ring buffer memory... ");
	if (data.in_rb_data)
		free(data.in_rb_data);
	if (data.sin_rb_data)
		free(data.sin_rb_data);
	if (DEBUG) printf("Succeeded.\n");

    /* Terminate PortAudio and free memory */
    Pa_Terminate();
    if (calc_buffer) free(calc_buffer);

    return err;
}




static int sinCallback(    const void *inputBuffer,
                                void *outputBuffer,
                                unsigned long framesPerBuffer,
                                const PaStreamCallbackTimeInfo* timeInfo,
                                PaStreamCallbackFlags statusFlags,
                                void *userData)
{
    /* Check for NULL buffers */
    if (inputBuffer == NULL)
        fatal_terminate("sinCallback: NULL inputBuffer");
	if (outputBuffer == NULL)
		fatal_terminate("sinCallback: NULL outputBuffer");

    /* Setup tracking pointers/variables */
    paData *data = (paData *)userData;
    const SAMPLE *in_ptr = (const SAMPLE *)inputBuffer;
    SAMPLE *out_ptr = (SAMPLE *)outputBuffer; 
	ring_buffer_size_t write_num, read_num;
        

	/* Write from inputBuffer to 'in' ring buffer */
	if ((write_num = PaUtil_GetRingBufferWriteAvailable(&data->in_rb)) != 0) {
		if (write_num >= framesPerBuffer) write_num = framesPerBuffer;
		PaUtil_WriteRingBuffer(&data->in_rb, in_ptr, write_num);
	}

//	/* Sanity check - just output sound */
//	int i;
//	for (i=0; i < framesPerBuffer; i++) {
//		*out_ptr++ = *in_ptr++;
//		*out_ptr++ = *in_ptr++;
//	}

	/* Write from 'sin' ring buffer to output Buffer */
	if ((read_num = PaUtil_GetRingBufferReadAvailable(&data->sin_rb)) != 0) {
		if (read_num >= framesPerBuffer) read_num = framesPerBuffer;
		PaUtil_ReadRingBuffer(&data->sin_rb, out_ptr, read_num);
	}

	return paContinue;
}

void setupInputParametersWithCone( PaStreamParameters *in_pars )
{
    /* Check if microcone available. If so, use */
    int i;
    int found = 0;
    for (i = 0; i < Pa_GetDeviceCount(); i++) {
        if (strcmp(Pa_GetDeviceInfo(i)->name, 
            "Microcone USB 2.0 Audio In") == 0) {
            in_pars->device = i;
            printf("Using microcone\n");
            found = 1;
            break;
        } 
    } 
    if (!found) {
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
//static int playCallback(    const void *inputBuffer,
//                            void *outputBuffer,
//                            unsigned long framesPerBuffer,
//                            const PaStreamCallbackTimeInfo *timeInfo,
//                            PaStreamCallbackFlags statusFlags,
//                            void *userData )
//{
//
//    /* Check for NULL output buffer */
//    if (outputBuffer == NULL)
//        fatal_terminate("recordCallback: NULL inputBuffer");
//
//    /* Setup tracking pointers/variables */
//    paData *data = (paData *)userData;
//    const SAMPLE *data_ptr = 
//        &data->samples[data->frameIndex * NUM_CHANNELS];
//    SAMPLE *out_ptr = (SAMPLE *)outputBuffer;
//
//
//    unsigned int framesLeft = FRAMES_PER_BUFFER;
//    if (framesLeft > FRAMES_PER_BUFFER)
//        framesLeft = FRAMES_PER_BUFFER;
//
//    /* Loop through and write data to output buffer */
//    for (int i = 0; i < framesLeft; i++) {
//        for (int j = 0; j < NUM_CHANNELS; j++) {
//            *out_ptr++ = *data_ptr++;
//        }
//    }
//
//    /* Update the frameIndex in the data struct now that 
//     * it has been played back */
//    data->frameIndex += framesLeft;
//
//    return paComplete;
//}
//    
//
//
//
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
    out_pars->device = Pa_GetDefaultOutputDevice();
    if (out_pars->device == paNoDevice)
        fatal_terminate("No default output device");
    out_pars->channelCount = NUM_CHANNELS;
    out_pars->sampleFormat = PA_SAMPLE_TYPE;
    out_pars->suggestedLatency = 
        Pa_GetDeviceInfo(out_pars->device)->defaultLowOutputLatency;
    out_pars->hostApiSpecificStreamInfo = NULL;
}


/**
 * Processes the data in a given buffer
 *
 * @param buf buffer holding data to be processed/manipulated
 * @param samples 	number of samples in buffer to process
 * @param num_channels number of channels for audio samples. This
 *   affects the layout of data in the buffer.
 */
void processData( SAMPLE *buf, int samples, int num_channels )
{
	/* Loop through available data and put through sin fcn */
	if (DEBUG) printf("Processing data in calculation buffer... ");
	for ( SAMPLE *data_ptr = buf; 
			data_ptr < buf + (num_channels * samples); data_ptr++) {
		/* Filter out one side of audio */
		//if ((unsigned long)data_ptr & 0x1) *data_ptr = 0;
		*data_ptr++ *= sin(FREQUENCY * *data_ptr);
	}
	if (DEBUG) printf("Finished.\n");
}
	
	
