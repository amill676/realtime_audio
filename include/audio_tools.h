/*****************************************************************
 * @file    audio_tools.h
 * @group   include
 * @brief   Contains function definitions for audio utilities
 * @author  Adam Miller
 *
 * 9/15/12
 ****************************************************************/

#include <portaudio.h> 
#include <stdio.h>
#include <stdlib.h>


/* fatal
 * Takes in a string containing a custom error message.
 * Prints out the error message and then exits with code 1 */
void fatal( char[] );


/* fatal_terminate
 * Same as fatal, but calls Pa_Terminate() first. This is
 * required if a call to Pa_Initialize() has been made */
void fatal_terminate( char[] );


/* pa_fatal
 * Takes in a PaError and will call Pa_Terminate() and then
 * print out the error code and message contained in the
 * PaError argument. */
void pa_fatal( char[], PaError );

