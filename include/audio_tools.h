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

void fatal( char[] );
void fatal_terminate( char[] );
void pa_fatal( char[], PaError );
void setupDefaultOutputParameters( PaStreamParameters*, PaSampleFormat );
void setupInputParametersWithDeviceName( 
				PaStreamParameters*, const char *, PaSampleFormat);
void setupInputParametersWithDeviceNumber( 
				PaStreamParameters*, int, PaSampleFormat);

