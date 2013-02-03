/***************************************************************
 * @file    audio_tools.c
 * @group   src
 * @brief   Implements functions delcared in audio_tools.h
 * @author  Adam Miller
 *
 * 9/15/12
 **************************************************************/

#include "audio_tools.h"
#include <string.h>
#include <stdbool.h>

 /**
 * Takes in a string containing a custom error message.
 * Prints out the error message and then exits with code 1
 * @param s custom error message
 */
void fatal(char s[]) {
    fprintf(stderr, "ERROR: %s\n", s);
    exit(1);
}


/**
 * Same as fatal, but calls Pa_Terminate() first. This is
 * required if a call to Pa_Initialize() has been made
 * @param s custom error message
 */
void fatal_terminate(char s[])
{
    Pa_Terminate();
    fatal(s);
}

/**
 * Takes in a PaError and will call Pa_Terminate() and then
 * print out the error code and message contained in the
 * PaError argument
 * @param s custom error message
 * @param errr portaudio error code
*/
void pa_fatal(char s[], PaError err)
{
    Pa_Terminate(); // MUST CALL TO AVOID AUDIO DEVICE PROBLEMS
    fprintf(stderr, "ERROR: %s\n", s);
    fprintf(stderr, "CODE: %d\t MESSAGE: %s\n", 
                    err, Pa_GetErrorText(err));
    exit(1);
}


/** 
 * Initializes the output parameters for playback
 *
 * Function will initailize all the members of the passed
 * in PaStreamParameters struct.
 * @param out_pars a pointer to a PaStreamParameters struct. 
 * @param format the sample format for the output device to use
 */
void setupDefaultOutputParameters( PaStreamParameters *out_pars,
									PaSampleFormat format)
{
    out_pars->device = Pa_GetDefaultOutputDevice();
    if (out_pars->device == paNoDevice)
        fatal_terminate("No default output device");
    out_pars->channelCount = 
			Pa_GetDeviceInfo(out_pars->device)-> maxOutputChannels;
    out_pars->sampleFormat = format;
    out_pars->suggestedLatency = 
        Pa_GetDeviceInfo(out_pars->device)->defaultLowOutputLatency;
    out_pars->hostApiSpecificStreamInfo = NULL;
}


/**
 * Sets up the input device with name given by the string
 * passed in as a parameter
 * @param in_pars	Poiner to a PaStreamParameters struct that will be
 * 					modified and setup properly
 * @param device	String with the desired device name 
 * @param format the sample format for the input device to use
 */
void setupInputParametersWithDeviceName( PaStreamParameters *in_pars, 
										const char *device, 
										PaSampleFormat format)
{
	int i;
	int device_found = false;
	for (i = 0; i < Pa_GetDeviceCount(); i++) {
		if(strcmp(Pa_GetDeviceInfo(i)->name, device) == 0) {
			in_pars->device = i;
			printf("Using %s\n", device);
			device_found = true;
			break;
		}
	}

	/* If device isn't found, use the default device */
	if (!device_found) {
		printf("Requested device not found. Using default\n");
		in_pars->device = Pa_GetDefaultInputDevice();
	}
    if (in_pars->device == paNoDevice) 
        fatal_terminate("No default input device");
    in_pars->channelCount = 
    in_pars->sampleFormat = format;
    in_pars->suggestedLatency = 
		Pa_GetDeviceInfo(in_pars->device)->maxInputChannels;
       Pa_GetDeviceInfo(in_pars->device)->defaultLowInputLatency;
    in_pars->hostApiSpecificStreamInfo = NULL;
}


/**
 * Setups the input parameters to use the input device
 * specified by the given device number
 * @param in_pars input parameters object to modify and setup
 * @param device_num the device number of the device to use
 * @param format the sample format for the input device to use
 */
void setupInputParametersWithDeviceNumber( PaStreamParameters *in_pars,
											int device_num, 
											PaSampleFormat format)
{
	/* Ensure device number is valid */
	bool device_found = false;
	if (0 <= device_num && device_num < Pa_GetDeviceCount()) {
		device_found = true;
		in_pars->device = device_num;
	}

	/* If device isn't found, use the default device */
	if (!device_found) {
		printf("Requested device (%d) not found. Using default\n",
														device_num );
		in_pars->device = Pa_GetDefaultInputDevice();
	}
    if (in_pars->device == paNoDevice) 
        fatal_terminate("No default input device");
    in_pars->channelCount = Pa_GetDeviceInfo(in_pars->device)->maxInputChannels;
    in_pars->sampleFormat = format;
    in_pars->suggestedLatency = 
       Pa_GetDeviceInfo(in_pars->device)->defaultLowInputLatency;
    in_pars->hostApiSpecificStreamInfo = NULL;
}
