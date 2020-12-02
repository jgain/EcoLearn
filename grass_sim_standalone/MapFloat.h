#ifndef MAPFLOAT_H
#define MAPFLOAT_H

#include <vector>
#include <string>

// Original code (MapFloat class) by James Gain
// Adapted by K.P. Kapp (April/May 2019)

// these classes can easily be a single template, but I am keeping it like this for compatibility

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

    /// getter for grid dimensions
    void getDim(int &dx, int &dy) const
    { dx = gx; dy = gy; }

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
    void initMap(){ fmap.clear(); fmap.resize(gx*gy); }

    /// completely delete map
    void delMap(){ fmap.clear(); }

    /// set grass heights to a uniform value
    void fill(float h){ fmap.clear(); fmap.resize(gx*gy, h); }

    /// getter and setter for map elements
    float get(int x, int y) const { return fmap[flatten(x, y)]; }
    float get(int idx) const { return fmap[idx]; }
    void set(int x, int y, float val){ fmap[flatten(x, y)] = val; }

    //**
    // * @brief read  read a floating point data grid from file
    // * @param filename  name of file to be read
    // * @return true if the file is found and has the correct format, false otherwise
    // *
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
/*
template<typename T>
class ValueMap
{
private:
    int gx, gy;                     //< grid dimensions
    std::vector<T> fmap;        //< grid of floating point values


public:

    ValueMap(){ gx = 0; gy = 0; initMap(); }

    ~ValueMap(){ delMap(); }

    /// return the row-major linearized value of a grid position
    inline int flatten(int dx, int dy){ return dy * gx + dx; }

    /// getter for grid dimensions
    void getDim(int &dx, int &dy) const
    { dx = gx; dy = gy; }

    int height(){ return gy; }
    int width(){ return gx; }

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
    void fill(float h){ fmap.clear(); fmap.resize(gx*gy, h); }

    /// getter and setter for map elements
    T get(int x, int y){ return fmap[flatten(x, y)]; }
    T get(int idx){ return fmap[idx]; }
    void set(int x, int y, T val){ fmap[flatten(x, y)] = val; }

    //**
    // * @brief read  read a floating point data grid from file
    // * @param filename  name of file to be read
    // * @return true if the file is found and has the correct format, false otherwise
    // *
    bool read(std::string filename);

    T *data() { return fmap.data(); }

    typename std::vector<T>::iterator begin()
    {
        return fmap.begin();
    }

    typename std::vector<T>::iterator end()
    {
        return fmap.end();
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

    //**
    // * @brief read  read an integer point data grid from file
    // * @param filename  name of file to be read
    // * @return true if the file is found and has the correct format, false otherwise
    // *
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

*/
#endif // MAPFLOAT_H
