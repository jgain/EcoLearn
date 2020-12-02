/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  J.E. Gain (jgain@cs.uct.ac.za)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/

#ifndef TimerC
#define TimerC
/* file: timer.h
   author: (c) James Gain, 2006
   notes: fairly accurate timing routines
*/

#include <sys/time.h>

class Timer
{

private:
    struct timeval tbegin, tend;
    struct timezone zone;

public:

    /// Start timer with call to timeofday
    void start();

    /// Stop timer with call to timeofday
    void stop();

    /// Get the current elapsed time between the latest calls to start and stop
    float peek();
};

#endif
