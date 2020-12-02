/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  K.P. Kapp  (konrad.p.kapp@gmail.com)
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



#ifndef BASIC_TYPES_H
#define BASIC_TYPES_H

#include <vector>
#include <map>
#include <cstring>
#include <iostream>
#include <cmath>
#include <algorithm>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

//#include <cuda_runtime_api.h>
//#include <cuda.h>


template<typename T>
struct xy
{
    CUDA_CALLABLE_MEMBER
    xy(T x, T y) : x(x), y(y)
    {}

    CUDA_CALLABLE_MEMBER
    xy() : x(0), y(0)
    {}

    CUDA_CALLABLE_MEMBER
    bool operator ==(const xy<int> &other)
    {
        return other.x == this->x && other.y == this->y;
    }

    T x, y;

    void trim(int minx, int maxx, int miny, int maxy)
    {
        if (minx > maxx) std::swap(minx, maxx);
        if (miny > maxy) std::swap(miny, maxy);
        if (x < minx) x = minx;
        if (x > maxx) x = maxx;
        if (y < miny) y = miny;
        if (y > maxy) y = maxy;
    }
};

enum all_species
{
    SPECIES1,
    SPECIES2,
    SPECIES3,
    SPECIES4,
    SPECIES5,
    SPECIES6,
    SPECIES7,
    SPECIES8,
    SPECIES9,
    SPECIES10,
    SPECIES_END
};


enum veg_class_name  // a class that defines the probability of each species occurring in a stand
{

};

struct data_struct
{
    float *data;
    int width, height;
    bool owner;

    // Note the 'owner' arg, if true, will cause this structure to take ownership of the data in the 'data' arg.
    // if 'owner' is false, data will simply point to the memory
    data_struct(float *data, int width, int height, bool owner)
        : data(owner ? new float[width * height] : data), width(width), height(height), owner(owner)
    {
        if (owner)
        {
            std::memcpy(this->data, data, width * height * sizeof(float));
        }
        //std::cout << "reference ctor called" << std::endl;
    }

    data_struct(int width, int height, float value)
        : data(new float[width * height]), owner(true), width(width), height(height)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                (*this)(x, y) = value;
            }
        }
    }

    data_struct()
        : data_struct(NULL, 0, 0, false)
    {}

    data_struct(const data_struct &other)
    {
        width = other.width;
        height = other.height;
        owner = other.owner;
        if (owner)	// if we copy from an owner, the current object must also become an owner. We therefore copy the other's data
        {
            data = new float[width * height];
            memcpy(data, other.data, sizeof(float) * width * height);
        }
        else
        {
            data = other.data;
        }
    }

    data_struct& operator=(const data_struct &other)
    {
        if (owner)	// if the current object is an owner, we need to free the data it holds
        {
            delete []data;
        }
        width = other.width;
        height = other.height;
        owner = other.owner;
        if (owner)	// if we copy from an owner, the current object must also become an owner. We therefore copy the other's data
        {
            data = new float[width * height];
            memcpy(data, other.data, sizeof(float) * width * height);
        }
        else
        {
            data = other.data;
        }
    }

    data_struct(data_struct &&other)
    {
        data = other.data;
        width = other.width;
        height = other.height;
        owner = other.owner;
        other.owner = false;	// when we move, ownership gets transferred
    }

    data_struct& operator=(data_struct &&other)
    {
        if (owner)	// if the current object is an owner, we need to free the data it holds
        {
            delete []data;
        }
        width = other.width;
        height = other.height;
        owner = other.owner;
        if (owner)	// if we copy from an owner, the current object must also become an owner. We therefore copy the other's data
        {
            data = new float[width * height];
            memcpy(data, other.data, sizeof(float) * width * height);
        }
        else
        {
            data = other.data;
        }
    }

    ~data_struct()
    {
        if (owner)
            delete [] data;
    }

    float &operator() (int x, int y)
    {
        return data[y * width + x];
    }

    float &operator[] (int idx)
    {
        return data[idx];
    }

    inline int size() { return width * height; }
};

struct species_probs
{
    void add_prob(int species, float prob)
    {
        probs[species] = prob;
    }

    float get(int species)
    {
        return probs[species];
    }

    std::map<int, float> probs;
};

struct veg_class_set
{
    void add_class(veg_class_name name, species_probs probs)
    {
        classes[name] = probs;
    }

    species_probs get(veg_class_name name)
    {
        return classes[name];
    }

    std::map<veg_class_name, species_probs> classes;
};

struct basic_tree
{
    CUDA_CALLABLE_MEMBER
    basic_tree(float x, float y, float radius, float height)
        : x(x), y(y), radius(radius), height(height)
    {}

    CUDA_CALLABLE_MEMBER
    basic_tree()
        : x(-1), y(-1), radius(-1), height(-1)
    {}

    bool operator == (const basic_tree &other) const
    {
        float dx = fabs(other.x - this->x);
        float dy = fabs(other.y - this->y);
        float dr = fabs(other.radius - this->radius);
        float dh = fabs(other.height - this->height);
        bool speq = other.species == this->species;
        return dx < 1e-5f && dy < 1e-5f && dr < 1e-5f && dh < 1e-5f && speq;
    }

    bool operator != (const basic_tree &other) const
    {
        return !(*this == other);
    }

    float distsq(const basic_tree &other) const
    {
        return (x - other.x) * (x - other.x) + (y - other.y) * (y - other.y);
    }

    float discdist(const basic_tree &other) const
    {
        float dist = sqrt(distsq(other));
        return dist - other.radius - radius;
    }

    float x, y;
    float radius;
    float height;
    float energy;
    int species;
    //std::vector<float> velocity = std::vector<float> (2, 0.0f);
    int r, g, b, a;
};

struct output_tree : basic_tree
{
    output_tree(float x, float y, float z, float radius, float height)
        : basic_tree(x, y, radius, height), z(z)
    {}

    int z;
};

struct grid_tree : basic_tree
{
    CUDA_CALLABLE_MEMBER
    grid_tree()
        : basic_tree()
    {}

    CUDA_CALLABLE_MEMBER
    grid_tree(float x, float y, float radius, float height)
        : basic_tree(x, y, radius, height), valid(true)
    {}
    bool valid;
};

struct mosaic_tree : grid_tree
{
    CUDA_CALLABLE_MEMBER
    mosaic_tree()
        : grid_tree()
    {}

    CUDA_CALLABLE_MEMBER
    mosaic_tree(float x, float y, float radius, float height, bool local_max)
        : grid_tree(x, y, radius, height)
    {
        this->local_max = local_max;
    }

    bool local_max;

};

struct species_params
{
    species_params(std::string name, float a, float b)
        : species_params(name, a, b, {}, {}, 0.0f)
    {
    }

    species_params(std::string name, float a, float b, std::vector<float> location, std::vector<float> width_scales)
        : species_params(name, a, b, location, width_scales, 0.0f)
    {
    }

    species_params(std::string name, float a, float b, std::vector<float> location, std::vector<float> width_scales, float percentage)
        : a(a), b(b), locs(location), width_scales(width_scales), percentage(percentage), name(name)
    {
    }

    std::string to_string(bool after_specassign)
    {
        std::string str = "";
        str += std::to_string(a) + " " + std::to_string(b) + " " + std::to_string(percentage);

        if (after_specassign)
        {
            for (int i = 0; i < locs.size() && i < width_scales.size(); i++)
            {
                str += " ";
                str += std::to_string(locs[i]) + " " + std::to_string(width_scales[i]);
            }
        }

        return str;
    }

    std::string name;
    float a;
    float b;
    std::vector<float> locs;
    std::vector<float> width_scales;
    float percentage;
};

struct MinimalPlant
{
    float x; //< x-position in m
    float y; //< y-position in m
    float h;	// height in m
    float r; //< radius in m		(note: all these were cm, not meters, in ecolearn code originally)
    bool s; //< shaded status for individual plant
    int species;

        void print() { std::cerr << "x: " << x << ", y: " << y << ", r: " << r << ", s: " << s << std::endl; }

        bool operator==(const MinimalPlant &other) { return x == other.x && y == other.y && r == other.r && s == other.s; }
        bool operator!=(const MinimalPlant &other) { return !(*this == other); }
};


namespace basic_types
{
    class MapFloat
    {
    private:
        int gx, gy;                     //< grid dimensions
        std::vector<float> fmap;        //< grid of floating point values


    public:

        MapFloat(){ gx = 0; gy = 0; initMap(); }

        ~MapFloat(){ delMap(); }

        /// return the row-major linearized value of a grid position
        inline int flatten(int dx, int dy) const { return dy * gx + dx; }

        inline void idx_to_xy(int idx, int &x, int &y) const
        {
            x = idx % gx;
            y = idx / gx;
        }

        /// getter for grid dimensions
        void getDim(int &dx, int &dy) const
        { dx = gx; dy = gy; }

        int height() const { return gy; }
        int width() const{ return gx; }

        /// setter for grid dimensions
        void setDim(int dx, int dy){ gx = dx; gy = dy; initMap(); }
        template<typename T>
        void setDim(const T &other)
        {
            int w, h;
            other.getDim(w, h);
            this->setDim(w, h);
        }

        template<typename T>
        void clone(const T &other)
        {
            setDim(other);
            int w, h;
            other.getDim(w, h);
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    set(x, y, other.get(x, y));
                }
            }
        }

        /// clear the contents of the grid to empty
        void initMap(){ fmap.clear(); fmap.resize(gx*gy); }

        /// completely delete map
        void delMap(){ fmap.clear(); }

        /// set grass heights to a uniform value
        void fill(float h){ fmap.clear(); fmap.resize(gx*gy, h); }

        /// getter and setter for map elements
        float get(int x, int y) const { return fmap[flatten(x, y)]; }
        float get(int idx) const{ return fmap[idx]; }
        void set(int x, int y, float val){ fmap[flatten(x, y)] = val; }

        /**
         * @brief read  read a floating point data grid from file
         * @param filename  name of file to be read
         * @return true if the file is found and has the correct format, false otherwise
         */
        bool read(std::string filename);

        float *data() { return fmap.data(); }

        std::vector<float>::iterator begin()
        {
            return fmap.begin();
        }

        std::vector<float>::iterator end()
        {
            return fmap.end();
        }

    };
}

template<typename T>
class ValueMap
{
protected:
    int gx, gy;                     //< grid dimensions
    std::vector<T> fmap;        //< grid of floating point values


public:

    ValueMap(){ gx = 0; gy = 0; initMap(); }

    ValueMap(int gx, int gy)
    : gx(gx), gy(gy)
    {
        initMap();
    }

    /*
    ValueMap(const ValueMap<T> &other)
    {
        clone(other);
    }
    */

    ~ValueMap(){ delMap(); }

    /// return the row-major linearized value of a grid position
    inline int flatten(int dx, int dy) const
    { return dy * gx + dx; }

    template<typename U>
    inline void clone(const ValueMap<U> &other)
    {
        int w, h;
        other.getDim(w, h);
        if (gx != w || gy != h)
        {
            setDim(w, h);
        }
        memcpy(fmap.data(), other.data(), sizeof(float) * w * h);
    }

    inline void idx_to_xy(int idx, int &x, int &y) const
    {
        x = idx % gx;
        y = idx / gx;
    }

    template<typename U>
    bool eqdim(const ValueMap<U> &other)
    {
        int gw, gh;
        other.getDim(gw, gh);
        return gw == gx && gh == gy;
    }

    /// getter for grid dimensions
    template<typename U>
    void getDim(U &dx, U &dy) const
    { dx = static_cast<U>(gx); dy = static_cast<U>(gy); }

    int height() const { return gy; }
    int width() const { return gx; }

    /// setter for grid dimensions
    void setDim(int dx, int dy){ gx = dx; gy = dy; initMap(); }

    template<typename U>
    void setDim(const U &other)
    {
        int w, h;
        other.getDim(w, h);
        this->setDim(w, h);
    }

    /// clear the contents of the grid to empty
    void initMap(){ fmap.clear(); fmap.resize(gx*gy); }

    /// completely delete map
    void delMap(){ fmap.clear(); }

    /// set grass heights to a uniform value
    void fill(T h){ fmap.clear(); fmap.resize(gx*gy, h); }
    void fill(T *ptr) {fmap.clear(); fmap.resize(gx*gy); memcpy(fmap.data(), ptr, sizeof(T) * gx * gy);}

    /// getter and setter for map elements
    T get(int x, int y) const
    { return fmap.at(flatten(x, y)); }

    T get(int idx) const
    { return fmap.at(idx); }

    T &operator ()(int x, int y)
    {
        return fmap[flatten(x, y)];
    }
    T &operator ()(int idx)
    {
        return fmap[idx];
    }

    const T &operator ()(int x, int y) const
    {
        return fmap[flatten(x, y)];
    }
    const T &operator ()(int idx) const
    {
        return fmap[idx];
    }

    void set(int x, int y, T val){ fmap[flatten(x, y)] = val; }
    void set(int idx, T val) { fmap[idx] = val; }


    /**
     * @brief read  read a floating point data grid from file
     * @param filename  name of file to be read
     * @return true if the file is found and has the correct format, false otherwise
     */
    bool read(std::string filename);

    const T *data() const { return fmap.data(); }
    T *data() { return fmap.data(); }

    typename std::vector<T>::iterator begin()
    {
        return fmap.begin();
    }

    typename std::vector<T>::iterator end()
    {
        return fmap.end();
    }

    int nelements() const
    {
        return gx * gy;
    }

    T calcmax() const
    {
        return *std::max_element(fmap.begin(), fmap.end());
    }

    T calcmin() const
    {
        return *std::min_element(fmap.begin(), fmap.end());
    }
};


template<typename T>
class ValueGridMap : public ValueMap<T>
{
protected:
    using ValueMap<T>::gx;
    using ValueMap<T>::gy;
    using ValueMap<T>::fmap;
public:
    using ValueMap<T>::get;
    using ValueMap<T>::initMap;
protected:
    float rx = 0.0f, ry = 0.0f;
    xy<float> togrid_conv, toreal_conv;
public:
    ValueGridMap(int gw, int gh, float rw, float rh)
        : rx(rw), ry(rh)
    {
        gx = gw;
        gy = gh;
        setDim(gx, gy);
        //reset_convs();
    }

    ValueGridMap()
    {
        setDim(0, 0);
        initMap();
    }

    ValueGridMap(const ValueGridMap<T> &other)
        : ValueMap<T>(other.gx, other.gy), rx(other.rx), ry(other.ry)
    {
        setDim(gx, gy);
        setDimReal(other);
        fmap = other.fmap;
    }

    static ValueGridMap<T> fromValueMap(float rw, float rh, const ValueMap<T> &obj)
    {
        int gx, gy;
        obj.getDim(gx, gy);
        ValueGridMap<T> newmap(gx, gy, rw, rh);
        memcpy(newmap.data(), obj.data(), sizeof(T) * gx * gy);

        return newmap;
    }

    template<typename U>
    bool eqdim(const ValueGridMap<U> &other)
    {
        float rw, rh;
        other.getDimReal(rw, rh);
        return ValueMap<T>::eqdim(other) && fabs(this->rx - rw) < 1e-5f && fabs(this->ry - rh) < 1e-5f;
    }

    void reportdims(std::ostream &ofs) const
    {
        ofs << "Grid width, height: " << gx << ", " << gy << std::endl;
        ofs << "Real width, height: " << rx << ", " << ry << std::endl;
    }

    void reset_convs()
    {
        if (fabs(rx * ry * gx * gy) < 1e-5)
        {
            togrid_conv.x = togrid_conv.y = 0.0f;
            toreal_conv.x = toreal_conv.y = 0.0f;
        }
        else
        {
            togrid_conv = xy<float>(gx / (rx + 1e-4), gy / (ry + 1e-4));
            toreal_conv = xy<float>(rx / (gx + 1e-4), ry / (gy + 1e-4));
        }
    }

    bool in_landscape(float x, float y)
    {
        return x >= 0.0f && x < rx - 1e-5f && y >= 0.0f && y < ry - 1e-5f;
    }

    bool in_grid(int x, int y)
    {
        return x >= 0 && x < gx && y >= 0 && y < gy;
    }

    void setDim(int x, int y)
    {
        ValueMap<T>::setDim(x, y);
        reset_convs();
    }

    void setDimReal(float rx, float ry)
    {
        this->rx = rx;
        this->ry = ry;
        reset_convs();
    }

    template<typename U>
    void setDim(const U &other)
    {
        int w, h;
        other.getDim(w, h);
        this->setDim(w, h);
    }

    template<typename U>
    void setDimReal(const U &other)
    {
        float rx, ry;
        other.getDimReal(rx, ry);
        setDimReal(rx, ry);
    }

    xy<int> togrid(float x, float y) const
    {
        return xy<int>(floor(togrid_conv.x * x), floor(togrid_conv.y * y));
    }

    xy<int> togrid_safe(float x, float y) const
    {
        auto val = xy<int>(floor(togrid_conv.x * x), floor(togrid_conv.y * y));
        if (val.x >= gx) val.x = gx - 1;
        if (val.y >= gy) val.y = gy - 1;
        if (val.x < 0) val.x = 0;
        if (val.y < 0) val.y = 0;
        return val;
    }

    xy<float> toreal(int x, int y) const
    {
        return xy<float>(toreal_conv.x * x, toreal_conv.y * y);
    }

    xy<float> get_togrid_conv() const
    {
        return togrid_conv;
    }

    xy<float> get_toreal_conv() const
    {
        return toreal_conv;
    }

    T get_fromreal(float real_x, float real_y) const
    {
        xy<int> coords = togrid(real_x, real_y);
        if (coords.x < 0 || coords.x >= gx || coords.y < 0 || coords.y >= gy)
        {
            throw std::runtime_error("Grid coordinates outside range");
        }
        return get(coords.x, coords.y);
    }

    T set_fromreal(float real_x, float real_y, T val)
    {
        xy<int> coords = togrid(real_x, real_y);
        if (real_x > rx || real_y > ry || real_x < 0 || real_y < 0)
        {
            std::string msg = "real coordinates (" + std::to_string(real_x) + ", " + std::to_string(real_y) + ") out of range in ValueGridMap::set_fromreal";
            throw std::out_of_range(msg.c_str());
        }
        if (coords.x >= gx || coords.y >= gy || coords.x < 0 || coords.y < 0)
        {
            std::string msg = "grid coordinates (" + std::to_string(coords.x) + ", " + std::to_string(coords.y) + ") out of range in ValueGridMap::set_fromreal (although real coordinates were valid)";
            throw std::out_of_range(msg);
        }
        this->set(coords.x, coords.y, val);
    }

    void getDimReal(float &rw, float &rh) const
    {
        rw = this->rx;
        rh = this->ry;
    }
};


class MapInt
{
private:
    int gx, gy;                     //< grid dimensions
    std::vector<int> imap;        //< grid of integer values


public:

    MapInt(){ gx = 0; gy = 0; initMap(); }

    ~MapInt(){ delMap(); }

    /// return the row-major linearized value of a grid position
    inline int flatten(int dx, int dy){ return dx * gy + dy; }

    /// getter for grid dimensions
    void getDim(int &dx, int &dy){ dx = gx; dy = gy; }

    int height(){ return gy; }
    int width(){ return gx; }

    /// setter for grid dimensions
    void setDim(int dx, int dy){ gx = dx; gy = dy; initMap(); }
    template<typename T>
    void setDim(const T &other)
    {
        int w, h;
        other.getDim(w, h);
        this->setDim(w, h);
    }

    /// clear the contents of the grid to empty
    void initMap(){ imap.clear(); imap.resize(gx*gy); }

    /// completely delete map
    void delMap(){ imap.clear(); }

    /// set map to a uniform value
    void fill(int h){ imap.clear(); imap.resize(gx*gy, h); }

    /// getter and setter for map elements
    int get(int x, int y){ return imap[flatten(x, y)]; }
    int get(int idx){ return imap[idx]; }
    void set(int x, int y, int val){ imap[flatten(x, y)] = val; }

    /**
     * @brief read  read an integer point data grid from file
     * @param filename  name of file to be read
     * @return true if the file is found and has the correct format, false otherwise
     */
    bool read(std::string filename);

    int *data() { return imap.data(); }

    std::vector<int>::iterator begin()
    {
        return imap.begin();
    }

    std::vector<int>::iterator end()
    {
        return imap.end();
    }

};

struct sim_info
{
    sim_info(std::vector<float> &rfall, float slopethresh, float slopemax, float evap,
             float runofflim, float soilsat, float riverlevel)
        : rainfall(rfall),
          slopethresh(slopethresh),
          slopemax(slopemax),
          evap(evap),
          runofflim(runofflim),
          soilsat(soilsat),
          riverlevel(riverlevel)
    {}

    sim_info() {}

    std::vector<float> rainfall;
    float slopethresh;
    float slopemax;
    float evap;
    float runofflim;
    float soilsat;
    float riverlevel;
};

#endif // BASIC_TYPES_H
