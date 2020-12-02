#ifndef DICE_ROLLER_H
#define DICE_ROLLER_H

#include <chrono>
#include <random>

class DiceRoller {
public:
    DiceRoller(int from, int to);
    ~DiceRoller();

    int generate();
private:
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution;
};

#endif //DICE_ROLLER_H
