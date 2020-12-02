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

// ipc.cpp: class for controlling interprocess communication with TensorFlow sketching process using ZeroMQ
// author: James Gain
// date: 7 January 2019

#ifndef IPC_H
#define IPC_H

#include "typemap.h"
#include "terrain.h"
#include "grass.h"
#include <zmq.hpp>

const float mtoft = 3.28084f;

class IPC{

private:

    zmq::context_t * context;
    zmq::socket_t * socket;

    /**
     * @brief composeTransmission   composeTransmission Set a pixel in the data transmission packet for sending
     * @param tmap  Current sketch image
     * @param cmap  Buffer of transmission data
     * @param sktch Paint value from sketch map
     * @param hght  Height value from terrain
     * @param x     x-coordinate in tmap
     * @param y     y-coordinate in tmap
     */
    void composeTransmission(TypeMap * tmap, unsigned char * cmap, unsigned short sktch, unsigned short hght, int x, int y, int scale);

    /**
     * @brief decomposeTransmission Extract a pixel from the data transmission packet received
     * @param tmap  Footprint image
     * @param cmap  Buffer of received data
     * @param x     x-coordinate in downsampled tmap
     * @param y     y-coordinate in downsampled tmap
     * @return      extracted pixel value (0 or 1)
     */
    unsigned short decomposeTransmission(MapFloat * fmap, unsigned char * cmap, int x, int y, int upsample_factor);

public:

    /// IPC: Create ZeroMQ context
    IPC();

    ~IPC()
    {
    } // this may actually be more complicated. Need to consult zeroMQ documentation

    /**
     * @brief send  Bundle and send two-type sketch image to TensorFlow process
     * @param tmap  Current sketch image
     * @param ter   Current terrain
     */
    void send(TypeMap * tmap, Terrain * ter, int scale_down);

    /**
     * @brief receive Receive and decode tree coverage map from TensorFlow process
     * @param tmap  Current tree coverage image
     */
    void receive(MapFloat * fmap, int scale);
    void receive_only(MapFloat *fmap);
};


#endif // IPC_H
