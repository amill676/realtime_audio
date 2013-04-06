/**********************************************************//**
 * @file    realtime_dft.c
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
#include <unistd.h>
#include <Accelerate/Accelerate.h>
#include <vDSP.h>
#include <pa_util.h>
#include <pa_ringbuffer.h>
#include <pthread.h>
#include <libkern/OSAtomic.h>
#include "audio_tools.h"
#include "realtimestft.h"

/* Setup basic options */
#define SAMPLE_RATE         16000
#define FRAMES_PER_BUFFER   2048 /* smaller buffer for realtime */
#define NUM_SECONDS         2
#define FRAMES_PER_RING_BUFFER 4 * FRAMES_PER_BUFFER
#define NUM_CHANNELS        2
#define NUM_PLOTS			1

/* Setup the sample format. */
#define PA_SAMPLE_TYPE paFloat32
typedef float SAMPLE;
#define SAMPLE_SILENCE      0.0f
#define SAMPLE_FORMAT       paFloat32

/* Constants for devices */
#define MICROPHONE_STR      "Built_in Microph"
#define BUILTIN_IN_STR      "Built-in Input"
#define BUILTIN_OUT_STR     "Built-in Output"
#define AIRPLAY_STR			"AirPlay"

/* Constants for DFT options */
#define DFT_LOG_LEN		11 // For length 2^n
#define DFT_LEN			(1 << DFT_LOG_LEN)
#define DFT_RADIX	kFFTRadix2
#define WINDOW_LOG_LEN  11
#define WINDOW_SIZE 	(1 << WINDOW_LOG_LEN)
#define HOP_LOG_N		(WINDOW_LOG_LEN - 1)
#define HOP_SIZE		(1 << HOP_LOG_N)

#define FRAMES_IN_SPEC  (44)
#define PLOT_WAIT_USEC  (120000)
#define EPS (.00001)
#define DAT_FILE_NAME_PREFIX	"output"

/* debug flag */
#define DEBUG	1 

#define ABS(a) ( a < 0 ? (-1 * a) : a )
#define FREQUENCY .5 // Frequency of sin to modulate with

static bool do_plot = false;
static bool user_did_quit = false;
#define QUIT_CHAR 'q'
static OSSpinLock write_lock = OS_SPINLOCK_INIT;
static OSSpinLock spec_lock = OS_SPINLOCK_INIT;

/* Encapsulate information needed to access and transfer
 * data into one struct. This way the struct can be
 * passed around instead of a bunch of different variables. */
typedef struct
{
	/* Ring buffer for raw input samples */
    PaUtilRingBuffer	in_rb;
	SAMPLE				*in_rb_data;

	/* Ring buffer for samples after putting through sin fcn */
	PaUtilRingBuffer	out_rb;
	SAMPLE				*out_rb_data;
} paData;

/* Static variable declarations */
static FILE *plots[NUM_PLOTS];
static int curr_x = -1;
static int num_written = 0;

/**
 * Writes the given number of elements from the data_buffer into
 * the ringBuffer.
 * @param ringBuffer	Ring buffer to write to
 * @param data_buffer	Data buffer to write from
 * @param n_elements	Number of elements to write
 * @return	number of elements successfully written
 */
static int writeOutputRingBuffer( 	PaUtilRingBuffer *ringBuffer,
									const void *data_buffer,	
									int n_elements )
{
	int write_num = PaUtil_GetRingBufferWriteAvailable(ringBuffer);
	if (write_num != 0) {
		if (DEBUG) printf("Room for data in output ring buffer.\n");
		/* Only write the number of frames we read in */
		if (write_num > n_elements)
			write_num = n_elements;
		int success, num_written = 0;
		const SAMPLE *data_ptr = (const SAMPLE *)data_buffer;
		if (DEBUG) printf("Writing data to output ring buffer... ");
		/* Keep trying until all elements written */
		while (num_written < write_num) {
			success = 
				PaUtil_WriteRingBuffer(ringBuffer, data_ptr,
											write_num-num_written);
			num_written += success;
			data_ptr += success;
		} // end write while
		if (DEBUG) printf("Succeeded.\n");
	} // end write_num != 0 if
}

/**
 * Callback function that moves data from the inputBuffer
 * passed to a given input ring buffer and from a given
 * output ring buffer to the parameter outputBuffer
 */
static int ringBufferCallback(  const void *inputBuffer,
                                void *outputBuffer,
                                unsigned long framesPerBuffer,
                                const PaStreamCallbackTimeInfo* timeInfo,
                                PaStreamCallbackFlags statusFlags,
                                void *userData)
{
    /* Check for NULL buffers */
    if (inputBuffer == NULL)
        fatal_terminate("ringBufferCallback: NULL inputBuffer");
	if (outputBuffer == NULL)
		fatal_terminate("ringBufferCallback: NULL outputBuffer");

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
	if ((read_num = PaUtil_GetRingBufferReadAvailable(&data->out_rb)) != 0) {
		if (read_num >= framesPerBuffer) read_num = framesPerBuffer;
		PaUtil_ReadRingBuffer(&data->out_rb, out_ptr, read_num);
	}

	return paContinue;
}







void high_pass_filter( DSPSplitComplex *dft, float frac, int len)
{
	int max_ind = (int)((float)frac*len);
	int i;
	/* Zero out DC, but not nyquist */
	if (max_ind > 0) dft->realp[0] = 0; // DC component
	for (i=1; i < max_ind; i++) {
		dft->realp[i] = 0;
		dft->imagp[i] = 0;
	}
}
	
	
void low_pass_filter( DSPSplitComplex *dft, float frac, int len)
{
	int min_ind = (int)((float)frac*len);
	int i;
	/* Zero out Nyquist component, but no DC */
	if (min_ind < len) dft->imagp[0] = 0; // Nyquist component

	for (i=min_ind; i < len; i++) {
		dft->realp[i] = 0;
		dft->imagp[i] = 0;
	}
}

void *checkForQuit( void *quit_flag )
{
	while (true) {
		//if (DEBUG) cout << "beginning of checkForQuit loop" << endl;
		int c = getc(stdin);
		if (c != -1)
			if (DEBUG) printf("Char has been read in checkForQuit: %c\n",c);
		if (c == QUIT_CHAR) {
			//OSSpinLockLock(&user_did_quit_lock);
			//if (DEBUG) cout << "checkForQuit got lock" << endl;
			*((bool *)quit_flag) = true;
			//OSSpinLockUnlock(&user_did_quit_lock);
			pthread_exit(NULL);
		}
		usleep(40000);
	}
	if (DEBUG) printf("Returning for checkForQuit\n");
	return quit_flag;
}

void *plot_dft( void *args )
{
	const char* f_name = (const char *)((void **)args)[0];
	FILE *pipe = (FILE *)((void **)args)[1];
	while (true) {
		fprintf(pipe, "plot '%s' with lines\n", f_name);
		fflush(pipe);
		usleep(40000);
	}
}

void printAvailableDevices()
{
	int i;
	for (i = 0; i < Pa_GetDeviceCount(); i++) {
		printf("%d: %s", i, Pa_GetDeviceInfo(i)->name);
		printf(" -- Max channels: %d\n", Pa_GetDeviceInfo(i)->maxInputChannels);
	}

}

static void writeDFTToFile( DSPSplitComplex *dft, FILE *dat_file )
{
	int i;
	rewind(dat_file);
	fprintf(dat_file, "#k\t\tX[k]\n");
	for (i = 0; i <= DFT_LEN/2; i++) {
		if (i == 0) { // DC component
			fprintf(dat_file, "%d\t\t%20f\n", i, 
					fabs(dft->realp[0]));
		} else if (i== (DFT_LEN/2)) {
			fprintf(dat_file, "%d\t\t%20f\n", i, 
					fabs(dft->imagp[0]));
		} else {
			fprintf(dat_file, "%d\t\t%20f\n", i, 
						sqrt(pow(dft->realp[i],2) +
							 pow(dft->imagp[i],2)));
		}
		fflush(dat_file);
	}
}

static void writeBufferedSTFTToFile(float *stft,
									int start_frame,
									FILE *dat_file )
{
	int i, j, frame_ind;
	rewind(dat_file);
	for (i = 0; i < FRAMES_IN_SPEC; i++) {
		for (j = 0; j <= DFT_LEN/2; j++) {
			frame_ind = (i+start_frame) % FRAMES_IN_SPEC;
			fprintf(dat_file, "%20d\t%20d\t%20f\n", i, j, 
							stft[frame_ind*(DFT_LEN/2+1) + j]);
		}
	}
	fflush(dat_file);
}


static void writeSTFTToFile( 	DSPSplitComplex *dft,
							  	int start_frame,
							  	FILE *dat_file)
{
	int temp_x = curr_x;
	temp_x++;
	int i;
	OSSpinLockLock(&write_lock);
	for (i = 0; i <= DFT_LEN/2; i++) {
		float temp;
		if (i == 0) { // DC component
			temp = (fabs(dft->realp[0])); //+ EPS);
		} else if (i== (DFT_LEN/2)) {
			temp = (fabs(dft->imagp[0]));// + EPS);
		} else {
			temp = (sqrt(pow(dft->realp[i],2) +
				 pow(dft->imagp[i],2)));// + EPS);
		}
		fprintf(dat_file, "%20d %20d %20f\n", temp_x, i, temp);
		fflush(dat_file);
	}
	fprintf(dat_file, "\n"); // delimiter for data_blocks in gnuplot
	fflush(dat_file);
	OSSpinLockUnlock(&write_lock);
	curr_x++; // not this value contains valid data set
	if (++num_written >= FRAMES_IN_SPEC) {
		//rewind(dat_file);
		num_written = 0;
	}

}


static void setupPlotPipeForDFT( FILE **pipe )
{
	*pipe = popen("gnuplot", "w");
	if (*pipe == NULL) fatal("Opening gnuplot pipe");
	fprintf(*pipe, "set xrange [0:%d]\n", DFT_LEN/2 - 1);
	fprintf(*pipe, "set yrange [0:100]\n");
}
	
static void *update_stft( void *args )
{
	OSSpinLockLock(&spec_lock);
	DSPSplitComplex *dft = (DSPSplitComplex *)((void **)args)[0];
	float *stft = (float *)((void **)args)[1];
	int *curr_spec_frame = (int *)((void **)args)[2];
	FILE *dat_file = (FILE *)((void **)args)[3];
	int i;
		
	for (i = 0; i <= DFT_LEN/2; i++) {
		if (i == 0) { // DC component
			stft[*curr_spec_frame * (DFT_LEN/2+1) + i] =
			log10(fabs(dft->realp[0]) + EPS);
		} else if (i== (DFT_LEN/2)) {
			stft[*curr_spec_frame * (DFT_LEN/2+1) + i] = 
			log10(fabs(dft->imagp[0]) + EPS);
		} else {
			stft[*curr_spec_frame * (DFT_LEN/2+1) + i] = 
			log10(sqrt(pow(dft->realp[i],2) +
				 pow(dft->imagp[i],2)) + EPS);
		}
	}
	*curr_spec_frame = (*curr_spec_frame + 1) % FRAMES_IN_SPEC;

	OSSpinLockLock(&write_lock);
	writeBufferedSTFTToFile( stft, *curr_spec_frame, dat_file);
	OSSpinLockUnlock(&write_lock);
	OSSpinLockUnlock(&spec_lock);
	return NULL;
}

/*****************************************************************/
int main()
{
    PaStreamParameters  inputParameters,
                        outputParameters;
    PaStream            *in_stream, *out_stream;
    PaError             err = paNoError;
    paData              data;
    int                 i, j, totalFrames, numSamples, numBytes;
    SAMPLE              max, val;
	SAMPLE				*calc_buffer;
    double              average;
	int					curr_spec_frame = 0;
	char file_names[NUM_PLOTS][30];
	FILE *dat_files[NUM_PLOTS];

	/* Basic setup for plotting */
	if (do_plot) {
		/* Ensure number of plots is valid */
		if (NUM_CHANNELS < NUM_PLOTS || NUM_PLOTS <= 0) {
			printf("Number of plots must be non-zero and <= number \
												of audio channels\n");
			exit(1);
		}


		/* Variables for output data file */
		for (i = 0; i < NUM_PLOTS; i++) {
			strcpy(file_names[i], DAT_FILE_NAME_PREFIX);
			char num[2] = {(const char)(i+'0'), '\0'};
			strcat(file_names[i], num);
			strcat(file_names[i], ".dat");
			dat_files[i] = fopen(file_names[i], "w");
			if (dat_files[i] == NULL) fatal("Opening output data file");
			setupPlotPipeForDFT(&plots[i]);
			printf("%s\n", file_names[i]);
		}
	}

	/* Determine size of ring buffers (both same size) */
	unsigned long num_frames = FRAMES_PER_RING_BUFFER;
	const unsigned long num_bytes = num_frames*NUM_CHANNELS*sizeof(SAMPLE);

	/* Call Pa_Initialize() and recorded error. This call is
	 * necessary to set up everything properly with PortAudio */
	err = Pa_Initialize();
	if (err != paNoError) {
        pa_fatal("Error during intialization", err);
    }

	/* Allocate space for ring buffers */
	if (DEBUG) printf("Allocating memory for ring buffers... ");
	if ((data.in_rb_data = (float *)malloc(num_bytes)) == NULL)
		fatal("PaUtil_AllocateMemory: in_rb");
	if ((data.out_rb_data = (float *)malloc(num_bytes)) == NULL)
		fatal("PaUtil_AllocateMemory: out_rb");
	if (DEBUG) printf("Succeeded\n");

	/* Allocate space for temporary calculation buffers */
	if ((calc_buffer = malloc(2* WINDOW_SIZE * NUM_CHANNELS
									* sizeof(float))) == NULL)
		fatal("malloc: calc_buffer");


	/* Initialize the ring buffers */
	if (DEBUG) printf("Initializing ring buffers... ");
	err = PaUtil_InitializeRingBuffer(&data.in_rb, 
			sizeof(float) * NUM_CHANNELS, num_frames, data.in_rb_data);
	if (err != paNoError)
		pa_fatal("Initializing ring in ring buffer", err);
	err = PaUtil_InitializeRingBuffer(&data.out_rb, 
			sizeof(float) * NUM_CHANNELS, num_frames, data.out_rb_data);
	if (err != paNoError)
		pa_fatal("Initializing ring in ring buffer", err);
	PaUtil_FlushRingBuffer(&data.in_rb);
	PaUtil_FlushRingBuffer(&data.out_rb);
	if (DEBUG) printf("Succeeded\n");
    
	/* Setup input and output parameters */
	int device_num = -1;
	if (DEBUG) printf("Setting up input parameters... ");
	printf("Enter the number of the desired device from the following:\n");
	printAvailableDevices();
	scanf("%d", &device_num);
	setupInputParametersWithDeviceNumber(&inputParameters, 
											device_num, SAMPLE_FORMAT);
	if (DEBUG) printf("Succeeded using device: %s\n",
						Pa_GetDeviceInfo(inputParameters.device)->name);
	if (DEBUG) printf("Setting up output parameters... ");
    setupDefaultOutputParameters(&outputParameters, SAMPLE_FORMAT);
	if (DEBUG) printf("Succeeded using device: %s\n",
						Pa_GetDeviceInfo(outputParameters.device)->name);


	/* Setup thread to check for quitting */
	pthread_t check_tID;
	err = 0;
	err = pthread_create(&check_tID, NULL, checkForQuit, &user_did_quit);
	if (err != 0) fatal_terminate("Error creating check_thread");

	if (do_plot) {
		/* Setup thread for plotting DFT */
		pthread_t plot_tIDs[NUM_PLOTS];
		void *args[2*NUM_PLOTS]; 
		for (i = 0; i < NUM_PLOTS; i++) {
			err = 0;
			args[0 + 2*i] = &file_names[i];
			args[1 + 2*i] = plots[i];
			err = pthread_create(&plot_tIDs[i], NULL, plot_dft, &args[2*i]);
			if (err != 0) fatal_terminate("Error creating plot_thread");
		}
	}
	
    /* Setup streams */
    err = Pa_OpenStream( &in_stream, &inputParameters, &outputParameters, 
						SAMPLE_RATE, FRAMES_PER_BUFFER, paClipOff, 
						ringBufferCallback, &data );
    if (err!=paNoError) pa_fatal("Pa_OpenStream - Input", err);

	/* Start streams */
	err = Pa_StartStream(in_stream);
	if (err != paNoError) pa_fatal("Pa_StartStream-in_stream", err);

	/* Setup stft object */
	realtimeSTFT stft;
    int use_window_fcn = 1;
	err = createRealtimeSTFT(&stft, DFT_LOG_LEN, 
							WINDOW_LOG_LEN, HOP_LOG_N, NUM_CHANNELS, 
                            use_window_fcn, sizeof(SAMPLE));
    char err_msg_buf[ERR_MSG_BUF_LEN];
	if (err != STFT_OK) {
        getErrorMsg(err_msg_buf);
        fatal_terminate(err_msg_buf);
    }

	/* Setup variables for main loop */
	ring_buffer_size_t read_num, wrie_num, num_complete, num_written,
						success;
	SAMPLE *data_ptr, *data_in_ptr = calc_buffer;
	int plot_count = 0;
	while( (err = Pa_IsStreamActive(in_stream)) == 1 && !user_did_quit) {
		/* Check if data is available */
		if ((read_num = PaUtil_GetRingBufferReadAvailable(&data.in_rb))
													>= WINDOW_SIZE	) {
			if (DEBUG) printf("Reading from input buffer... ");
			if (read_num > WINDOW_SIZE) read_num = WINDOW_SIZE;
			/* IMPLEMENT SMARTER READ TO ENSURE PROPER NUMBER OF SMAPLES */
			num_complete = PaUtil_ReadRingBuffer(&data.in_rb, calc_buffer, 
													read_num);
			if (DEBUG) printf("read %d samples.\n", num_complete);

			/* Do stft business */
			performSTFT(&stft, calc_buffer);

			/* Filter */
			int i;
			//for (j = 0; j < stft.num_channels; j++) {
			//	for (i = 0; i < stft.num_dfts; i++) {
			//		//low_pass_filter(&stft.dfts[i+stft.num_channels], 0.15, DFT_LEN/2);
			//		//high_pass_filter(&stft.dfts[i], 0.025, DFT_LEN/2);
            //        int k;
            //        printf("DFT #%d:\n", j * stft.num_dfts + i);
            //        DSPSplitComplex * dft = 
            //            (DSPSplitComplex *)&(stft.dfts[j*stft.num_dfts + i]);
            //        for (k=0; k < 12; k++) {
            //            printf("%f\t", dft->realp[k]);
            //        }
            //        printf("\n");
			//	}
            //}

			if (do_plot) {
				/* Write newest DFT to file so it can be plotted */
				for (i = 0; i < NUM_PLOTS; i++) {
					writeDFTToFile(&stft.dfts[0+i*NUM_PLOTS], 
														dat_files[i]);
				}
			}
			/* Perform the istft */
			performISTFT(&stft, calc_buffer);
			/* Write to output ring buffer */
			writeOutputRingBuffer(&data.out_rb, calc_buffer, WINDOW_SIZE);

		} // end read_num = ... if

		Pa_Sleep(3); // Try to prevent fans from going crazy 

	} // end while
	if (err < 0) pa_fatal("Pa_IsStreamActive", err);

	/* Try to output all zeros to avoid loud sounds all closing */
	if (Pa_IsStreamActive(in_stream) == 1) {
		memset(calc_buffer, 0, WINDOW_SIZE);
		writeOutputRingBuffer(&data.out_rb, calc_buffer, WINDOW_SIZE);
		Pa_StopStream(in_stream);
	}
		
    /* Terminate PortAudio and free memory */
	if (DEBUG) printf("Terminating port audio... ");
    Pa_Terminate();
	if (DEBUG) printf("Succeeded.\n");

	/* Free memory associated with ring buffers */
	if (DEBUG) printf("Freeing ring buffer memory... ");
	if (data.in_rb_data)
		free(data.in_rb_data);
	if (data.out_rb_data)
		free(data.out_rb_data);
	if (DEBUG) printf("Succeeded.\n");

	/* Free memory associated with STFT object */
	if (DEBUG) printf("Freeing stft object... ");
	err = destroyRealtimeSTFT( &stft );
	if (err != STFT_OK) fatal_terminate("destroyRealtimeSTFT");
	if (DEBUG) printf("Succeeded\n");

	/* Free temporary buffer */
	if (DEBUG) printf("Freeing calc_buffer... ");
    if (calc_buffer) free(calc_buffer);
	if (DEBUG) printf("Succeeded.\n");

	/* Close output file */
	if (DEBUG) printf("Closing file streams... ");
	for (i=0; i < NUM_PLOTS; i++) if (dat_files[i]) fclose(dat_files[i]);
	for (i=0; i < NUM_PLOTS; i++) if (plots[i]) pclose(plots[i]);
	if (DEBUG) printf("Succeeded.\n");

	printf("Program exited successfully.\n");
    return err;
}
