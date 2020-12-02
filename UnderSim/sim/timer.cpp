/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems (Undergrowth simulator)
 * Copyright (C) 2020  J.E. Gain  (jgain@cs.uct.ac.za)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

void Timer::start()
{
    gettimeofday(&tbegin, &zone); // gettimeofday used for accuracy
}

void Timer::stop()
{
    gettimeofday(&tend, &zone);
}

float Timer::peek()
{
    float total_time;
    long time_in_sec, time_in_ms;

    time_in_sec = tend.tv_sec - tbegin.tv_sec;
    time_in_ms = tend.tv_usec - tbegin.tv_usec;
    total_time = ((float) time_in_ms)/1000000.0;
    total_time += ((float) time_in_sec);
    return total_time;
}
