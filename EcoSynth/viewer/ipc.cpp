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

#include <string>
#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include "ipc.h"

using namespace std;

IPC::IPC()
{
    //  Prepare context and socket
    context = new zmq::context_t(1);
    socket = new zmq::socket_t((* context), ZMQ_REQ);
    //socket->connect("tcp://137.158.59.252:5555");
    //socket->connect("tcp://localhost:5555");
    //socket->connect("tcp://192.168.101.251:5555");
    socket->connect("tcp://127.0.1.1:5555");		// for the local machine

    //socket->connect("tcp://137.158.60.235:5555");  // for the other machine
    cerr << "IPC: finished connection" << endl;
}

static int s_interrupted = 0;
static void s_signal_handler (int signal_value)
{
    s_interrupted = 1;
}

static void s_catch_signals (void)
{
    struct sigaction action;
    action.sa_handler = s_signal_handler;
    action.sa_flags = 0;
    sigemptyset (&action.sa_mask);
    sigaction (SIGINT, &action, NULL);
    sigaction (SIGTERM, &action, NULL);
}

void IPC::composeTransmission(TypeMap * tmap, unsigned char * cmap, unsigned short sktch, unsigned short hght, int x, int y, int scale)
{
    // transmission structure
    // r,g,b layers are interleaved
    // msb is first then lsb second
    // r-channel holds terrain heights
    // g-channel holds paint map
    int yoffset = tmap->height()/scale;
    int pos = (x * yoffset + y) * 2 * 3;

    // sep into upper and lower byte
    unsigned char upper, lower;

    // pack terrain height
    lower = (hght&0xFF); //extract first byte
    upper = ((hght>>8)&0xFF); //extract second byte
    cmap[pos] = lower;
    cmap[pos+1] = upper;

    // pack sketch map
    lower = (sktch&0xFF); //extract first byte
    upper = ((sktch>>8)&0xFF); //extract second byte
    cmap[pos+2] = lower;
    cmap[pos+3] = upper;

    cmap[pos+4] = 0;
    cmap[pos+5] = 0;
}

unsigned short IPC::decomposeTransmission(MapFloat * fmap, unsigned char * cmap, int x, int y, int upsample_factor)
{
    // only one channel (green) needed, because of greyscale.

    // int loffset = ((tmap->width() * tmap->height()) / 16) * 2 * 3;
    int yoffset = fmap->height()/upsample_factor;
    int loc = (x * yoffset + y) * 2 * 3;
    unsigned short tval = (unsigned short) cmap[loc+3]; // green bits
    tval = (tval << 8);
    tval += (unsigned short) cmap[loc+2];
    return tval;

    // return (int) cmap[loc];
}

void IPC::send(TypeMap * tmap, Terrain * ter, int scale_down)
{
    // data transmission format width * height * 3 * uint16 at 4m per cell sampling
    // potential issues with data formats

    // downsample tmap to scale_down (was 4meters) per cell resolution
    int csize = ((tmap->width() * tmap->height()) / (scale_down * scale_down)) * 3 * 2;
    unsigned char * cmap = new unsigned char[csize];

    // downsample tmap according to majority type in 4x4 area
    int ecnt, scnt, dcnt;
    for(int x = 0; x < tmap->width(); x+=scale_down)
        for(int y = 0; y < tmap->height(); y+=scale_down)		// this increment value was originally 4...I assumed that it was the scaling down factor?
        {
            ecnt = 0; scnt = 0; dcnt = 0;
            for(int i = 0; i < scale_down; i++)
                for(int j = 0; j < scale_down; j++)
                {
                   int paint = tmap->get(y+j, x+i);
                   switch(paint)
                   {
                   case 0:
                       ecnt++;
                       break;
                   case 1:
                       scnt++;
                       break;
                   case 2:
                       dcnt++;
                       break;
                   default:
                       ecnt++;
                       break;
                   }
                }

            // ?? order of x, y in getHeight
            unsigned int hght = (unsigned int) (ter->getHeight(x, y) / (3000.0f / mtoft) * 65535.0f);
            // cerr << " " << ter->getHeight(x, y) << " " << hght;
            // set according to highest count
            if((ecnt > scnt) && (ecnt > dcnt))
            {
                composeTransmission(tmap, cmap, 0, hght, x/scale_down, y/scale_down, scale_down);
            }
            else
            {
                if(scnt > dcnt)
                     composeTransmission(tmap, cmap, 32768, hght, x/scale_down, y/scale_down, scale_down);
                else
                     composeTransmission(tmap, cmap, 65535, hght, x/scale_down, y/scale_down, scale_down);
            }
        }

    // send data using zeroMQ
    zmq::message_t outward (csize);
    memcpy (outward.data (), cmap, csize);

    try
    {
        socket->send (outward, 0);
    }
    catch(zmq::error_t& e)
    {
        cerr << "send failed: " << e.what() << endl;
    }
    if (s_interrupted)
    {
        cerr << "interrupt received" << endl;
    }
    delete cmap;
}

void IPC::receive(MapFloat * fmap, int scale)
{
    zmq::message_t request;

    // wait for image from TensorFlow server
    socket->recv(&request);

    cerr << "msg decode phase" << endl;
    // decode and repack image into tmap
    int csize = ((fmap->width() * fmap->height()) / (scale * scale)) * 3 * 2;
    unsigned char * cmap = (unsigned char *) request.data();
    // report message size
    // cerr << "message size " << (int) request.size() << " expected " << csize << endl;

    fmap->fill(0.0f);
    // set block of <scale> pixels to the same value since TensorFlow output is downsampled
    for(int x = 0; x < fmap->width() / scale; x++)
        for(int y = 0; y < fmap->height() / scale; y++)
        {
            unsigned short tval = decomposeTransmission(fmap, cmap, x, y, scale);
            // if(tval != 0)
            //      cerr << " " << tval;

            //if(tval > 65000)
                for(int i = x*scale; i < x*scale+scale; i++) // upsample to scale x scale block of pixels
                    for(int j = y*scale; j < y*scale+scale; j++)
                        //fmap->set(i, j, 35.0f); // tall tree value at the moment, connect to heights later
                        //fmap->set(i, j, i);	// just to test the display of CHM (remove ASAP)
                        fmap->set(i, j, tval);
        }
}

void IPC::receive_only(MapFloat *fmap)
{
    zmq::message_t request;

    // wait for image from TensorFlow server
    socket->recv(&request);

    cerr << "msg decode phase" << endl;
    cerr << "msg width, height (elements): " << fmap->width() << ", " << fmap->height() << std::endl;
    // decode and repack image into tmap
    int csize = (fmap->width() * fmap->height()) * 3 * 2;
    unsigned char * cmap = (unsigned char *) request.data();
    // report message size
    // cerr << "message size " << (int) request.size() << " expected " << csize << endl;

    fmap->fill(0.0f);
    for(int x = 0; x < fmap->width(); x++)
        for(int y = 0; y < fmap->height(); y++)
        {
            unsigned short tval = decomposeTransmission(fmap, cmap, x, y, 1);
            fmap->set(x, y, tval);
        }
}

/*
void IPC::receive_only(MapFloat * fmap)
{
    zmq::message_t request;

    // wait for image from TensorFlow server
    socket->recv(&request);

    cerr << "msg decode phase" << endl;
    // decode and repack image into tmap
    int csize = ((fmap->width() * fmap->height())) * 3 * 2;
    unsigned char * cmap = (unsigned char *) request.data();
    // report message size
    // cerr << "message size " << (int) request.size() << " expected " << csize << endl;

    fmap->fill(0.0f);
    // set block of 4 pixels to the same value since TensorFlow output is downsampled
    for(int x = 0; x < fmap->width(); x++)
        for(int y = 0; y < fmap->height(); y++)
        {
            unsigned short tval = decomposeTransmission(fmap, cmap, x, y);
            // if(tval != 0)
            //      cerr << " " << tval;

            //if(tval > 65000)
                for(int i = x*1; i < x*1+1; i++) // upsample to 1x1 block of pixels
                    for(int j = y*1; j < y*1+1; j++)
                        //fmap->set(i, j, 35.0f); // tall tree value at the moment, connect to heights later
                        //fmap->set(i, j, i);	// just to test the display of CHM (remove ASAP)
                        fmap->set(i, j, tval);
        }
}
*/
