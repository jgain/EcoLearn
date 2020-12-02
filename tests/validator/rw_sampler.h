#include <limits>
#include <random>
#include <algorithm>
#include <stdexcept>

class rw_sampler
{
public:
        rw_sampler(const std::vector<float> &ranges)
                : ranges(ranges)
        {
            std::sort(this->ranges.begin(), this->ranges.end());
            setup_samplemap();
        }

        int sample(float value)
        {
            int idx = value / samplebin_size;
            if (straddle.at(idx))
            {
                if (value > samplemap.at(idx + 1))
                    return samplemap.at(idx + 1);
                else
                    return samplemap.at(idx);
            }
            else
            {
                return samplemap.at(idx);
            }
        }

        bool test_sample(float value)
        {
            bool eq_boundary;
            int idx = sample(value);
            int test_idx = get_range_idx_slow(value, eq_boundary);
            return eq_boundary || idx == test_idx;
        }

        bool test_sample()
        {
            std::default_random_engine gen;
            std::uniform_real_distribution<float> unif(ranges.front(), ranges.back());

            for (int i = 0; i < 1000; i++)
            {
                float value = unif(gen);
                if (!test_sample(value))
                    return false;
            }

            return true;
        }

private:
        void setup_samplemap()
        {
            samplebin_size = std::numeric_limits<float>::max();
            for (int i = 0; i < ranges.size() - 1; i++)
            {
                float diff = ranges.at(i + 1) - ranges.at(i);
                if (diff < samplebin_size)
                    samplebin_size = diff;
            }

            nsamplebins = (ranges.back() - ranges.front()) / samplebin_size;
            if (fmod(ranges.back() - ranges.front(), samplebin_size) > 1e-3f)
                nsamplebins++;

            for (int i = 0; i < nsamplebins - 1; i++)
            {
                float low = ranges.front() + i * samplebin_size;
                float high = ranges.front() + (i + 1) * samplebin_size;
                if (high > ranges.back())
                    high = ranges.back();
                bool eq1, eq2;
                int ridx1, ridx2;
                ridx1 = get_range_idx_slow(low, eq1);
                ridx2 = get_range_idx_slow(high, eq2);
                if (ridx2 - ridx1 > 1)
                {
                    throw std::runtime_error("Sample bins are too big, because an exact match straddles over more than one range bin");
                }
                if (ridx2 < ridx1)
                {
                    throw std::runtime_error("Higher calculated bin corresponds to a lower bin than the lower calculated bin");
                }
                if (eq1 && eq2)
                {
                    if (ridx1 != ridx2 - 1)
                    {
                        throw std::runtime_error("Sample bins are too big, because an exact match straddles over more than one range bin");
                    }
                    straddle.push_back(0);
                    samplemap.push_back(ridx1);
                    continue;
                }
                else if (eq1)
                {
                    straddle.push_back(0);
                    samplemap.push_back(ridx2);
                }
                else if (eq2)
                {
                    straddle.push_back(0);
                    samplemap.push_back(ridx1);
                }
                else if (ridx1 != ridx2)
                {
                    straddle.push_back(1);
                    samplemap.push_back(ridx1);
                }
                else if (ridx1 == ridx2)
                {
                    straddle.push_back(0);
                    samplemap.push_back(ridx1);
                }
                else
                {
                    throw std::runtime_error("case not handled");
                }
            }
        }


        int get_range_idx_slow(float value, bool &equaled)
        {
            equaled = false;
            if (value < ranges.at(0) - 1e-3f)
                return -1;
            for (int i = 0; i < ranges.size() - 1; i++)
            {
                if (fabs(value - ranges.at(i)) < 1e-3f)
                {
                    equaled = true;
                    return i;
                }
                if (fabs(value - ranges.at(i + 1)) < 1e-3f)
                {
                    equaled = true;
                    return i + 1;
                }
                if (value < ranges.at(i + 1))
                {
                    return i;
                }
            }
        }

private:
        std::vector<float> ranges;
        float samplebin_size;
        int nsamplebins;
        std::vector<unsigned char> straddle;
        std::vector<int> samplemap;
};
