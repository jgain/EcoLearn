#include "rw_sampler.h"
#include <iostream>

int main(int argc, char * argv [])
{
    rw_sampler rws = rw_sampler({2.0, 5.0, 8.2, 9.3});
    bool success = rws.test_sample();
    if (success)
    {
        std::cout << "SUCCESS: RW sampler passed test" << std::endl;
    }
    else
        std::cout << "FAIL: RW sampler failed test" << std::endl;
    return 0;
}
